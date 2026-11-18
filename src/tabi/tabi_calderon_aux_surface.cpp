#include "tabi_calderon_aux_surface.h"
#include "tabi_shared_vertex_separation.h"

#include "tabi_analytic_ses.h"
#include "tabi_ses_bilevel_grid.h"
#include "tabi_ses_patch_ray.h"

#include "../titan_executor.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <tuple>
#include <unordered_map>
#include <iostream>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef TITAN_TABI_SES_TEST_HOOKS
#include <cstdlib>
#include <iostream>
#include <iterator>
#endif

namespace TitanPBE::Tabi::calderon {
namespace {

// Intersections computed from adjacent tetrahedra can differ by a few ulps.
// Welding at 1e-5 A is far below any supported grid spacing (>= 0.25 A), but
// prevents duplicate seam vertices and removes solver-hostile micro-slivers.
constexpr double kQuantize = 1.0e5;
constexpr std::size_t kMaxGridVertices = 4000000;

struct Vec3 {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

struct CellKey {
  int x{0};
  int y{0};
  int z{0};

  bool operator==(const CellKey &) const = default;
};

struct CellKeyHash {
  std::size_t operator()(const CellKey &key) const noexcept {
    std::size_t hash = static_cast<std::size_t>(key.x) * 73856093U;
    hash ^= static_cast<std::size_t>(key.y) * 19349663U;
    hash ^= static_cast<std::size_t>(key.z) * 83492791U;
    return hash;
  }
};

struct VertexKey {
  std::int64_t x{0};
  std::int64_t y{0};
  std::int64_t z{0};

  bool operator==(const VertexKey &) const = default;
};

struct VertexKeyHash {
  std::size_t operator()(const VertexKey &key) const noexcept {
    std::size_t hash = static_cast<std::size_t>(key.x);
    hash ^= static_cast<std::size_t>(key.y) + 0x9e3779b9U + (hash << 6U) +
            (hash >> 2U);
    hash ^= static_cast<std::size_t>(key.z) + 0x9e3779b9U + (hash << 6U) +
            (hash >> 2U);
    return hash;
  }
};

struct EdgeKey {
  std::size_t lo{0};
  std::size_t hi{0};

  bool operator==(const EdgeKey &) const = default;
};

struct EdgeKeyHash {
  std::size_t operator()(const EdgeKey &key) const noexcept {
    std::size_t hash = key.lo;
    hash ^= key.hi + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
    return hash;
  }
};

struct FieldSample {
  double value{1.0};
  std::size_t atom_index{0};
};

struct AtomGrid {
  Vec3 origin;
  double cell_size{1.0};
  double max_radius{0.0};
  std::unordered_map<CellKey, std::vector<std::size_t>, CellKeyHash> cells;
};

[[nodiscard]] Vec3 operator+(const Vec3 &left, const Vec3 &right) noexcept {
  return {left.x + right.x, left.y + right.y, left.z + right.z};
}

[[nodiscard]] Vec3 operator-(const Vec3 &left, const Vec3 &right) noexcept {
  return {left.x - right.x, left.y - right.y, left.z - right.z};
}

[[nodiscard]] Vec3 operator*(double scale, const Vec3 &value) noexcept {
  return {scale * value.x, scale * value.y, scale * value.z};
}

[[nodiscard]] double dot(const Vec3 &left, const Vec3 &right) noexcept {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

[[nodiscard]] Vec3 cross(const Vec3 &left, const Vec3 &right) noexcept {
  return {left.y * right.z - left.z * right.y,
          left.z * right.x - left.x * right.z,
          left.x * right.y - left.y * right.x};
}

[[nodiscard]] double norm(const Vec3 &value) noexcept {
  return std::sqrt(dot(value, value));
}

[[nodiscard]] Vec3 normalized(const Vec3 &value) noexcept {
  const double length = norm(value);
  return length > 1.0e-300 ? (1.0 / length) * value : Vec3{1.0, 0.0, 0.0};
}

[[nodiscard]] CellKey cell_for(const AtomGrid &grid,
                               const Vec3 &point) noexcept {
  return {
      static_cast<int>(std::floor((point.x - grid.origin.x) / grid.cell_size)),
      static_cast<int>(std::floor((point.y - grid.origin.y) / grid.cell_size)),
      static_cast<int>(std::floor((point.z - grid.origin.z) / grid.cell_size))};
}

[[nodiscard]] AtomGrid build_atom_grid(std::span<const Atom> atoms,
                                       const Vec3 &origin) {
  AtomGrid grid;
  grid.origin = origin;
  for (const Atom &atom : atoms) {
    grid.max_radius = std::max(grid.max_radius, std::max(0.0, atom.radius));
  }
  grid.cell_size = std::max(1.0e-6, grid.max_radius);
  grid.cells.reserve(atoms.size() * 2);
  for (std::size_t index = 0; index < atoms.size(); ++index) {
    const Atom &atom = atoms[index];
    if (!(atom.radius > 0.0) || !std::isfinite(atom.radius)) {
      continue;
    }
    grid.cells[cell_for(grid, {atom.x, atom.y, atom.z})].push_back(index);
  }
  return grid;
}

[[nodiscard]] FieldSample sample_union_sdf(const AtomGrid &grid,
                                           std::span<const Atom> atoms,
                                           const Vec3 &point) noexcept {
  const CellKey center = cell_for(grid, point);
  FieldSample best;
  best.value = std::numeric_limits<double>::infinity();
  // A sphere that contains a point must have its centre in this 3x3x3 cell
  // neighborhood because the cell width equals the largest radius.
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const auto found =
            grid.cells.find({center.x + dx, center.y + dy, center.z + dz});
        if (found == grid.cells.end()) {
          continue;
        }
        for (const std::size_t index : found->second) {
          const Atom &atom = atoms[index];
          const double rx = point.x - atom.x;
          const double ry = point.y - atom.y;
          const double rz = point.z - atom.z;
          const double value =
              std::sqrt(rx * rx + ry * ry + rz * rz) - atom.radius;
          if (value < best.value) {
            best = {value, index};
          }
        }
      }
    }
  }
  if (!std::isfinite(best.value)) {
    best.value = grid.max_radius;
    best.atom_index = 0;
  }
  return best;
}

// `radius_offset` grows or shrinks every atom for this query only.  A caller
// that wants a certificate rather than an answer passes a small negative
// offset to prove containment and a small positive one to prove exclusion,
// which leaves a thin shell around the boundary uncertified instead of
// resolving it by whichever way the last bit of a squared distance happened
// to fall.
[[nodiscard]] bool point_inside_union(const AtomGrid &grid,
                                      std::span<const Atom> atoms,
                                      const Vec3 &point,
                                      double radius_offset = 0.0) noexcept {
  const CellKey center = cell_for(grid, point);
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const auto found =
            grid.cells.find({center.x + dx, center.y + dy, center.z + dz});
        if (found == grid.cells.end())
          continue;
        for (const std::size_t index : found->second) {
          const Atom &atom = atoms[index];
          const double radius = atom.radius + radius_offset;
          if (!(radius > 0.0))
            continue;
          const double rx = point.x - atom.x;
          const double ry = point.y - atom.y;
          const double rz = point.z - atom.z;
          if (rx * rx + ry * ry + rz * rz < radius * radius)
            return true;
        }
      }
    }
  }
  return false;
}

[[nodiscard]] const std::array<Vec3, 27> &topology_sample_directions() {
  // Construct once rather than normalizing the same 26 directions at every
  // grid vertex. The zero entry represents the unshifted probe centre.
  static const std::array<Vec3, 27> directions = [] {
    std::array<Vec3, 27> result{};
    std::size_t index = 0;
    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0 && dz == 0) {
            result[index++] = {};
          } else {
            result[index++] =
                normalized({static_cast<double>(dx), static_cast<double>(dy),
                            static_cast<double>(dz)});
          }
        }
      }
    }
    return result;
  }();
  return directions;
}

// The current-family engines take their inside/outside sign by sampling the
// probe ball at these directions and asking whether every sample lands in the
// dilated atom set.  Sampling the sphere instead of testing the ball admits a
// vertex whose probe does not really fit, so the surface sits outside the SES
// by an amount set by the angular resolution of this set and not by the grid
// spacing -- which is why refining the grid does not reduce it.
//
// Half-width 1 is the 26 directions `current` has always used: the widest gap
// between neighbouring directions is 27.6 degrees and the average is 15.5,
// which at a 1.4 A probe is 0.16 A and 0.05 A of radial slack.  Half-width 2
// is 98 directions at 17.6 and 8.2 degrees, which is `current-dense`.
[[nodiscard]] std::vector<Vec3> build_closing_sample_directions(
    int half_width) {
  std::vector<Vec3> result;
  result.push_back({}); // the unshifted probe centre
  for (int dz = -half_width; dz <= half_width; ++dz)
    for (int dy = -half_width; dy <= half_width; ++dy)
      for (int dx = -half_width; dx <= half_width; ++dx) {
        if (dx == 0 && dy == 0 && dz == 0)
          continue;
        const Vec3 direction =
            normalized({static_cast<double>(dx), static_cast<double>(dy),
                        static_cast<double>(dz)});
        // A lattice box repeats the directions its inner shells already
        // cover, so the cost is the direction count rather than the lattice
        // size.  The test is an AND over every sample, so the order the
        // survivors end up in cannot change the answer.
        if (std::none_of(result.begin(), result.end(),
                         [&](const Vec3 &seen) {
                           return norm(seen - direction) < 1.0e-12;
                         }))
          result.push_back(direction);
      }
  return result;
}

[[nodiscard]] const std::vector<Vec3> &closing_sample_directions(
    int half_width) {
  static const std::vector<Vec3> narrow = build_closing_sample_directions(1);
  static const std::vector<Vec3> dense = build_closing_sample_directions(2);
  return half_width <= 1 ? narrow : dense;
}

// `current-dense` differs from `current` in that one number and nothing else,
// so everything that asks "is this the sampled-closing engine" asks here.
[[nodiscard]] bool is_current_family_engine(std::string_view engine) noexcept {
  return engine == "current" || engine == "current-dense";
}

[[nodiscard]] VertexKey vertex_key(const Vec3 &point) noexcept {
  return {static_cast<std::int64_t>(std::llround(point.x * kQuantize)),
          static_cast<std::int64_t>(std::llround(point.y * kQuantize)),
          static_cast<std::int64_t>(std::llround(point.z * kQuantize))};
}

std::size_t
add_vertex(SurfaceMesh &mesh,
           std::unordered_map<VertexKey, std::size_t, VertexKeyHash> &lookup,
           const AtomGrid &atom_grid, std::span<const Atom> atoms,
           const Vec3 &point) {
  const VertexKey key = vertex_key(point);
  if (const auto found = lookup.find(key); found != lookup.end()) {
    return found->second;
  }
  const FieldSample sample = sample_union_sdf(atom_grid, atoms, point);
  const Atom &atom = atoms[sample.atom_index];
  const Vec3 normal =
      normalized({point.x - atom.x, point.y - atom.y, point.z - atom.z});
  const std::size_t index = mesh.vertices.size();
  mesh.vertices.push_back({point.x, point.y, point.z, normal.x, normal.y,
                           normal.z, sample.atom_index});
  lookup.emplace(key, index);
  return index;
}

void add_triangle(
    SurfaceMesh &mesh,
    std::unordered_map<VertexKey, std::size_t, VertexKeyHash> &lookup,
    const AtomGrid &atom_grid, std::span<const Atom> atoms, Vec3 a, Vec3 b,
    Vec3 c) {
  const Vec3 centroid = (1.0 / 3.0) * (a + b + c);
  const FieldSample field = sample_union_sdf(atom_grid, atoms, centroid);
  const Atom &normal_atom = atoms[field.atom_index];
  const Vec3 outward =
      normalized({centroid.x - normal_atom.x, centroid.y - normal_atom.y,
                  centroid.z - normal_atom.z});
  Vec3 normal_vector = cross(b - a, c - a);
  if (dot(normal_vector, outward) < 0.0) {
    std::swap(b, c);
    normal_vector = cross(b - a, c - a);
  }
  const double twice_area = norm(normal_vector);
  if (!(twice_area > 1.0e-16) || !std::isfinite(twice_area)) {
    return;
  }
  const Vec3 normal = (1.0 / twice_area) * normal_vector;
  const std::size_t ia = add_vertex(mesh, lookup, atom_grid, atoms, a);
  const std::size_t ib = add_vertex(mesh, lookup, atom_grid, atoms, b);
  const std::size_t ic = add_vertex(mesh, lookup, atom_grid, atoms, c);
  if (ia == ib || ib == ic || ic == ia) {
    return;
  }
  const double area = 0.5 * twice_area;
  Triangle triangle;
  triangle.v0 = ia;
  triangle.v1 = ib;
  triangle.v2 = ic;
  triangle.atom_index = field.atom_index;
  triangle.cx = centroid.x;
  triangle.cy = centroid.y;
  triangle.cz = centroid.z;
  triangle.nx = normal.x;
  triangle.ny = normal.y;
  triangle.nz = normal.z;
  triangle.area = area;
  mesh.triangles.push_back(triangle);

  Panel panel;
  panel.x = centroid.x;
  panel.y = centroid.y;
  panel.z = centroid.z;
  panel.nx = normal.x;
  panel.ny = normal.y;
  panel.nz = normal.z;
  panel.area = area;
  panel.atom_index = field.atom_index;
  panel.v0x = a.x;
  panel.v0y = a.y;
  panel.v0z = a.z;
  panel.v1x = b.x;
  panel.v1y = b.y;
  panel.v1z = b.z;
  panel.v2x = c.x;
  panel.v2y = c.y;
  panel.v2z = c.z;
  panel.v0_index = ia;
  panel.v1_index = ib;
  panel.v2_index = ic;
  panel.has_triangle_geometry = true;
  mesh.panels.push_back(panel);
}

// `edge_margin` rescales the crossing parameter into [m, 1-m], holding every
// marching vertex that far from both endpoints of its edge as a fraction of
// the edge.  A crossing that lands a hair from an endpoint is the classic
// source of sliver triangles, and the rescale is continuous and monotone, so
// a small change in the field still moves the vertex by a small amount.  It
// never merges two vertices, so the combinatorics are exactly those of the
// unrescaled mesh and the surface stays closed.  Zero disables it, which is
// what the `current` engine passes: that engine's geometry is fixed by the
// runs already published with it.
[[nodiscard]] Vec3 interpolate(const Vec3 &a, const Vec3 &b, double fa,
                               double fb, double edge_margin) noexcept {
  const double denominator = fa - fb;
  double t = std::abs(denominator) > 1.0e-300
                 ? std::clamp(fa / denominator, 0.0, 1.0)
                 : 0.5;
  if (edge_margin > 0.0 && edge_margin < 0.5)
    t = edge_margin + (1.0 - 2.0 * edge_margin) * t;
  return a + t * (b - a);
}

void polygonize_tetrahedron(
    SurfaceMesh &mesh,
    std::unordered_map<VertexKey, std::size_t, VertexKeyHash> &vertex_lookup,
    const AtomGrid &atom_grid, std::span<const Atom> atoms,
    const std::array<Vec3, 4> &points, const std::array<double, 4> &values,
    double edge_margin) {
  std::array<int, 4> inside{};
  std::array<int, 4> outside{};
  int inside_count = 0;
  int outside_count = 0;
  for (int i = 0; i < 4; ++i) {
    if (values[static_cast<std::size_t>(i)] < 0.0) {
      inside[static_cast<std::size_t>(inside_count++)] = i;
    } else {
      outside[static_cast<std::size_t>(outside_count++)] = i;
    }
  }
  if (inside_count == 0 || inside_count == 4) {
    return;
  }
  const auto edge = [&](int a, int b) {
    return interpolate(points[static_cast<std::size_t>(a)],
                       points[static_cast<std::size_t>(b)],
                       values[static_cast<std::size_t>(a)],
                       values[static_cast<std::size_t>(b)], edge_margin);
  };
  if (inside_count == 1) {
    const int i = inside[0];
    add_triangle(mesh, vertex_lookup, atom_grid, atoms, edge(i, outside[0]),
                 edge(i, outside[1]), edge(i, outside[2]));
  } else if (inside_count == 3) {
    const int o = outside[0];
    add_triangle(mesh, vertex_lookup, atom_grid, atoms, edge(o, inside[0]),
                 edge(o, inside[2]), edge(o, inside[1]));
  } else {
    const int i0 = inside[0];
    const int i1 = inside[1];
    const int o0 = outside[0];
    const int o1 = outside[1];
    const Vec3 p_i0_o0 = edge(i0, o0);
    const Vec3 p_i0_o1 = edge(i0, o1);
    const Vec3 p_i1_o0 = edge(i1, o0);
    const Vec3 p_i1_o1 = edge(i1, o1);
    add_triangle(mesh, vertex_lookup, atom_grid, atoms, p_i0_o0, p_i0_o1,
                 p_i1_o1);
    add_triangle(mesh, vertex_lookup, atom_grid, atoms, p_i0_o0, p_i1_o1,
                 p_i1_o0);
  }
}

struct DirectedTriangleEdge {
  std::size_t triangle{0};
  int direction{1};
};

[[nodiscard]] EdgeKey edge_key(std::size_t a, std::size_t b) noexcept {
  return {std::min(a, b), std::max(a, b)};
}

[[nodiscard]] int edge_direction(std::size_t a, std::size_t b) noexcept {
  return a < b ? 1 : -1;
}

[[nodiscard]] Vec3 mesh_vertex(const SurfaceMesh &mesh,
                               std::size_t index) noexcept {
  const auto &vertex = mesh.vertices[index];
  return {vertex.x, vertex.y, vertex.z};
}

[[nodiscard]] std::size_t
mesh_capacity_bytes(const SurfaceMesh &mesh) noexcept {
  return mesh.vertices.capacity() * sizeof(Vertex) +
         mesh.triangles.capacity() * sizeof(Triangle) +
         mesh.panels.capacity() * sizeof(Panel);
}

[[nodiscard]] double stable_repair_metric(double value) noexcept {
  constexpr double scale = 1.0e12;
  return std::round(value * scale) / scale;
}

void update_triangle_geometry(SurfaceMesh &mesh, std::size_t triangle_index) {
  Triangle &triangle = mesh.triangles[triangle_index];
  Panel &panel = mesh.panels[triangle_index];
  const Vec3 a = mesh_vertex(mesh, triangle.v0);
  const Vec3 b = mesh_vertex(mesh, triangle.v1);
  const Vec3 c = mesh_vertex(mesh, triangle.v2);
  const Vec3 normal_vector = cross(b - a, c - a);
  const double twice_area = norm(normal_vector);
  if (!(twice_area > 1.0e-16)) {
    triangle.area = 0.0;
    triangle.nx = triangle.ny = triangle.nz = 0.0;
    panel.area = 0.0;
    panel.nx = panel.ny = panel.nz = 0.0;
    return;
  }
  const Vec3 normal = (1.0 / twice_area) * normal_vector;
  const Vec3 centroid = (1.0 / 3.0) * (a + b + c);
  triangle.cx = centroid.x;
  triangle.cy = centroid.y;
  triangle.cz = centroid.z;
  triangle.nx = normal.x;
  triangle.ny = normal.y;
  triangle.nz = normal.z;
  triangle.area = 0.5 * twice_area;
  panel.x = centroid.x;
  panel.y = centroid.y;
  panel.z = centroid.z;
  panel.nx = normal.x;
  panel.ny = normal.y;
  panel.nz = normal.z;
  panel.area = triangle.area;
  panel.v0x = a.x;
  panel.v0y = a.y;
  panel.v0z = a.z;
  panel.v1x = b.x;
  panel.v1y = b.y;
  panel.v1z = b.z;
  panel.v2x = c.x;
  panel.v2y = c.y;
  panel.v2z = c.z;
  panel.v0_index = triangle.v0;
  panel.v1_index = triangle.v1;
  panel.v2_index = triangle.v2;
}

void assign_analytic_triangle_owner(SurfaceMesh &mesh,
                                    std::size_t triangle_index) {
  Triangle &triangle = mesh.triangles[triangle_index];
  const std::size_t owner =
      std::min({mesh.vertices[triangle.v0].atom_index,
                mesh.vertices[triangle.v1].atom_index,
                mesh.vertices[triangle.v2].atom_index});
  triangle.atom_index = owner;
  mesh.panels[triangle_index].atom_index = owner;
}

void update_analytic_triangle_geometry_and_owner(
    SurfaceMesh &mesh, std::size_t triangle_index) {
  update_triangle_geometry(mesh, triangle_index);
  assign_analytic_triangle_owner(mesh, triangle_index);
}

void assign_all_analytic_triangle_owners(SurfaceMesh &mesh) {
  for (std::size_t triangle_index = 0U;
       triangle_index < mesh.triangles.size(); ++triangle_index)
    assign_analytic_triangle_owner(mesh, triangle_index);
}

[[nodiscard]] bool finite_vec3(const Vec3 &value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

struct FaceMoveValidation {
  bool geometry_valid{false};
  bool orientation_preserved{false};
};

[[nodiscard]] double face_edge_ratio(const Vec3 &a, const Vec3 &b,
                                     const Vec3 &c) noexcept {
  const double ab = norm(b - a);
  const double bc = norm(c - b);
  const double ca = norm(a - c);
  const double shortest = std::min({ab, bc, ca});
  const double longest = std::max({ab, bc, ca});
  if (!(shortest > 1.0e-14) || !std::isfinite(longest))
    return std::numeric_limits<double>::infinity();
  return longest / shortest;
}

// A refused vertex stays on its marching position, off the analytic surface
// by the move it did not make.  That is certifiable only while the largest
// outstanding move is small against the spacing the mesh is drawn at, and
// this is the fraction of a cell the certificate grants.  Every refusal the
// commit is allowed to invent for its own reasons is held to the same
// number, so no refusal can be the thing that fails the certificate.
constexpr double kRefusedVertexToleranceCells = 0.25;

// The marching mesh reaches the projection with a worst edge ratio near 16
// and no poor triangle; the projection alone takes it past 60.  A bound
// above what marching produces refuses only what the projection newly
// creates, and never blocks a move that improves a face already past it.
// NanoShaper reports a worst ratio under 10, which this cannot reach: what
// stands between is the marching mesh itself, not the projection.
constexpr double kProjectionShapeRatioBound = 20.0;


[[nodiscard]] FaceMoveValidation validate_face_move(
    const Vec3 &a, const Vec3 &b, const Vec3 &c,
    const Vec3 &reference_normal) noexcept {
  FaceMoveValidation result;
  if (!finite_vec3(a) || !finite_vec3(b) || !finite_vec3(c) ||
      !finite_vec3(reference_normal))
    return result;
  const double ab = norm(b - a);
  const double bc = norm(c - b);
  const double ca = norm(a - c);
  const Vec3 new_normal = cross(b - a, c - a);
  const double new_length = norm(new_normal);
  const double reference_length = norm(reference_normal);
  result.geometry_valid =
      std::isfinite(ab) && std::isfinite(bc) && std::isfinite(ca) &&
      std::isfinite(new_length) && std::isfinite(reference_length) &&
      std::min({ab, bc, ca}) > 1.0e-14 &&
      new_length > 1.0e-16 && reference_length > 1.0e-16;
  if (!result.geometry_valid)
    return result;
  result.orientation_preserved =
      dot(reference_normal, new_normal) >
      1.0e-8 * reference_length * new_length;
  return result;
}

struct WeldCellKey {
  std::int64_t x{0};
  std::int64_t y{0};
  std::int64_t z{0};

  bool operator==(const WeldCellKey &) const = default;
};

struct WeldCellKeyHash {
  std::size_t operator()(const WeldCellKey &key) const noexcept {
    std::size_t hash = static_cast<std::size_t>(key.x);
    hash ^= static_cast<std::size_t>(key.y) + 0x9e3779b97f4a7c15ULL +
            (hash << 6U) + (hash >> 2U);
    hash ^= static_cast<std::size_t>(key.z) + 0x9e3779b97f4a7c15ULL +
            (hash << 6U) + (hash >> 2U);
    return hash;
  }
};

struct WeldFaceKey {
  std::array<std::size_t, 3> vertices{};

  bool operator==(const WeldFaceKey &) const = default;
};

struct WeldFaceKeyHash {
  std::size_t operator()(const WeldFaceKey &key) const noexcept {
    std::size_t hash = key.vertices[0];
    hash ^= key.vertices[1] + 0x9e3779b97f4a7c15ULL + (hash << 6U) +
            (hash >> 2U);
    hash ^= key.vertices[2] + 0x9e3779b97f4a7c15ULL + (hash << 6U) +
            (hash >> 2U);
    return hash;
  }
};

[[nodiscard]] double seam_weld_tolerance(const SurfaceMesh &mesh,
                                         double grid_spacing) noexcept {
  double maximum_ulp = 0.0;
  for (const Vertex &vertex : mesh.vertices) {
    for (const double coordinate : {vertex.x, vertex.y, vertex.z}) {
      const double next =
          std::nextafter(coordinate, std::numeric_limits<double>::infinity());
      maximum_ulp = std::max(maximum_ulp, std::abs(next - coordinate));
    }
  }
  const double local_scale = std::max(1.0, grid_spacing);
  return std::max(64.0 * maximum_ulp,
                  2048.0 * std::numeric_limits<double>::epsilon() *
                      local_scale);
}

[[nodiscard]] std::size_t weld_near_coincident_vertices(
    const SurfaceMesh &source, SurfaceMesh &candidate, double tolerance,
    std::size_t &scratch_bytes) {
  if (source.vertices.empty() || source.triangles.empty() ||
      !(tolerance > 0.0) || !std::isfinite(tolerance))
    return 0U;
  Vec3 origin{std::numeric_limits<double>::infinity(),
              std::numeric_limits<double>::infinity(),
              std::numeric_limits<double>::infinity()};
  for (const Vertex &vertex : source.vertices) {
    origin.x = std::min(origin.x, vertex.x);
    origin.y = std::min(origin.y, vertex.y);
    origin.z = std::min(origin.z, vertex.z);
  }
  const auto cell_for_vertex = [&](const Vertex &vertex) {
    return WeldCellKey{
        static_cast<std::int64_t>(
            std::floor((vertex.x - origin.x) / tolerance)),
        static_cast<std::int64_t>(
            std::floor((vertex.y - origin.y) / tolerance)),
        static_cast<std::int64_t>(
            std::floor((vertex.z - origin.z) / tolerance))};
  };
  std::vector<std::size_t> parent(source.vertices.size());
  for (std::size_t index = 0U; index < parent.size(); ++index)
    parent[index] = index;
  const auto root_of = [&](std::size_t index) {
    while (parent[index] != index)
      index = parent[index];
    return index;
  };
  std::unordered_map<WeldCellKey, std::vector<std::size_t>, WeldCellKeyHash>
      cells;
  cells.reserve(source.vertices.size());
  const double tolerance2 = tolerance * tolerance;
  std::size_t welded = 0U;
  for (std::size_t index = 0U; index < source.vertices.size(); ++index) {
    const Vertex &vertex = source.vertices[index];
    const WeldCellKey cell = cell_for_vertex(vertex);
    std::size_t selected = index;
    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const auto found =
              cells.find({cell.x + dx, cell.y + dy, cell.z + dz});
          if (found == cells.end())
            continue;
          for (const std::size_t other_index : found->second) {
            const Vertex &other = source.vertices[other_index];
            const double px = vertex.x - other.x;
            const double py = vertex.y - other.y;
            const double pz = vertex.z - other.z;
            if (px * px + py * py + pz * pz > tolerance2)
              continue;
            const Vec3 normal{vertex.nx, vertex.ny, vertex.nz};
            const Vec3 other_normal{other.nx, other.ny, other.nz};
            if (dot(normalized(normal), normalized(other_normal)) < 0.25)
              continue;
            selected = std::min(selected, root_of(other_index));
          }
        }
      }
    }
    if (selected != index) {
      parent[index] = selected;
      ++welded;
    }
    cells[cell].push_back(index);
  }
  if (welded == 0U)
    return 0U;

  std::vector<std::size_t> compact_index(
      source.vertices.size(), std::numeric_limits<std::size_t>::max());
  candidate.vertices.reserve(source.vertices.size() - welded);
  for (std::size_t index = 0U; index < source.vertices.size(); ++index) {
    const std::size_t root = root_of(index);
    if (compact_index[root] == std::numeric_limits<std::size_t>::max()) {
      compact_index[root] = candidate.vertices.size();
      candidate.vertices.push_back(source.vertices[root]);
    }
    compact_index[index] = compact_index[root];
  }
  candidate.triangles.reserve(source.triangles.size());
  candidate.panels.reserve(source.panels.size());
  std::unordered_set<WeldFaceKey, WeldFaceKeyHash> faces;
  faces.reserve(source.triangles.size());
  for (std::size_t index = 0U; index < source.triangles.size(); ++index) {
    Triangle triangle = source.triangles[index];
    triangle.v0 = compact_index[triangle.v0];
    triangle.v1 = compact_index[triangle.v1];
    triangle.v2 = compact_index[triangle.v2];
    if (triangle.v0 == triangle.v1 || triangle.v1 == triangle.v2 ||
        triangle.v2 == triangle.v0)
      continue;
    WeldFaceKey key{{triangle.v0, triangle.v1, triangle.v2}};
    std::sort(key.vertices.begin(), key.vertices.end());
    if (!faces.insert(key).second)
      continue;
    candidate.triangles.push_back(triangle);
    candidate.panels.push_back(source.panels[index]);
  }
  for (std::size_t index = 0U; index < candidate.triangles.size(); ++index)
    update_triangle_geometry(candidate, index);
  scratch_bytes = std::max(
      scratch_bytes,
      mesh_capacity_bytes(candidate) +
          parent.capacity() * sizeof(std::size_t) +
          compact_index.capacity() * sizeof(std::size_t));
  return welded;
}

std::size_t orient_mesh_consistently(SurfaceMesh &mesh);

[[nodiscard]] std::size_t collapse_short_manifold_edges(SurfaceMesh &mesh,
                                                        double threshold) {
  if (mesh.vertices.empty() || mesh.triangles.empty() || !(threshold > 0.0))
    return 0;
  struct EdgeRecord {
    std::vector<std::size_t> triangles;
    double length{0.0};
  };
  std::unordered_map<EdgeKey, EdgeRecord, EdgeKeyHash> edges;
  std::vector<std::unordered_set<std::size_t>> neighbors(mesh.vertices.size());
  edges.reserve(mesh.triangles.size() * 3U);
  for (std::size_t triangle_index = 0; triangle_index < mesh.triangles.size();
       ++triangle_index) {
    const Triangle &triangle = mesh.triangles[triangle_index];
    const std::array<std::pair<std::size_t, std::size_t>, 3> local_edges = {
        std::pair{triangle.v0, triangle.v1},
        std::pair{triangle.v1, triangle.v2},
        std::pair{triangle.v2, triangle.v0}};
    for (const auto &[a, b] : local_edges) {
      EdgeRecord &record = edges[edge_key(a, b)];
      record.triangles.push_back(triangle_index);
      neighbors[a].insert(b);
      neighbors[b].insert(a);
    }
  }
  struct Candidate {
    EdgeKey edge;
    double length{0.0};
  };
  std::vector<Candidate> candidates;
  for (auto &[key, record] : edges) {
    const Vec3 a = mesh_vertex(mesh, key.lo);
    const Vec3 b = mesh_vertex(mesh, key.hi);
    record.length = norm(b - a);
    if (record.triangles.size() == 2U && record.length < threshold)
      candidates.push_back({key, record.length});
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &a, const Candidate &b) {
              const double a_length = stable_repair_metric(a.length);
              const double b_length = stable_repair_metric(b.length);
              if (a_length != b_length)
                return a_length < b_length;
              return a.edge.lo != b.edge.lo ? a.edge.lo < b.edge.lo
                                            : a.edge.hi < b.edge.hi;
            });
  std::vector<std::size_t> remap(mesh.vertices.size());
  for (std::size_t index = 0; index < remap.size(); ++index)
    remap[index] = index;
  std::vector<bool> touched(mesh.vertices.size(), false);
  std::size_t collapsed = 0;
  for (const Candidate &candidate : candidates) {
    const std::size_t keep = candidate.edge.lo;
    const std::size_t remove = candidate.edge.hi;
    if (touched[keep] || touched[remove])
      continue;
    std::size_t common_neighbors = 0;
    for (const std::size_t neighbor : neighbors[keep])
      common_neighbors += neighbors[remove].contains(neighbor) ? 1U : 0U;
    // Link condition for collapsing an interior edge of a closed 2-manifold.
    if (common_neighbors != 2U)
      continue;
    Vertex &kept = mesh.vertices[keep];
    const Vertex &removed = mesh.vertices[remove];
    kept.x = 0.5 * (kept.x + removed.x);
    kept.y = 0.5 * (kept.y + removed.y);
    kept.z = 0.5 * (kept.z + removed.z);
    const Vec3 blended_normal = normalized(
        {kept.nx + removed.nx, kept.ny + removed.ny, kept.nz + removed.nz});
    kept.nx = blended_normal.x;
    kept.ny = blended_normal.y;
    kept.nz = blended_normal.z;
    remap[remove] = keep;
    touched[keep] = true;
    touched[remove] = true;
    for (const std::size_t neighbor : neighbors[keep])
      touched[neighbor] = true;
    for (const std::size_t neighbor : neighbors[remove])
      touched[neighbor] = true;
    ++collapsed;
  }
  if (collapsed == 0)
    return 0;

  std::vector<Triangle> triangles;
  std::vector<Panel> panels;
  triangles.reserve(mesh.triangles.size());
  panels.reserve(mesh.panels.size());
  for (std::size_t index = 0; index < mesh.triangles.size(); ++index) {
    Triangle triangle = mesh.triangles[index];
    triangle.v0 = remap[triangle.v0];
    triangle.v1 = remap[triangle.v1];
    triangle.v2 = remap[triangle.v2];
    if (triangle.v0 == triangle.v1 || triangle.v1 == triangle.v2 ||
        triangle.v2 == triangle.v0)
      continue;
    triangles.push_back(triangle);
    panels.push_back(mesh.panels[index]);
  }
  std::vector<bool> referenced(mesh.vertices.size(), false);
  for (const Triangle &triangle : triangles) {
    referenced[triangle.v0] = true;
    referenced[triangle.v1] = true;
    referenced[triangle.v2] = true;
  }
  std::vector<std::size_t> compact(mesh.vertices.size(), 0U);
  std::vector<Vertex> vertices;
  vertices.reserve(mesh.vertices.size() - collapsed);
  for (std::size_t index = 0; index < mesh.vertices.size(); ++index) {
    if (!referenced[index])
      continue;
    compact[index] = vertices.size();
    vertices.push_back(mesh.vertices[index]);
  }
  for (Triangle &triangle : triangles) {
    triangle.v0 = compact[triangle.v0];
    triangle.v1 = compact[triangle.v1];
    triangle.v2 = compact[triangle.v2];
  }
  mesh.vertices = std::move(vertices);
  mesh.triangles = std::move(triangles);
  mesh.panels = std::move(panels);
  for (std::size_t index = 0; index < mesh.triangles.size(); ++index)
    update_triangle_geometry(mesh, index);
  return collapsed;
}

[[nodiscard]] double normalized_triangle_quality(const Vec3 &a, const Vec3 &b,
                                                 const Vec3 &c) noexcept {
  const double ab2 = dot(b - a, b - a);
  const double bc2 = dot(c - b, c - b);
  const double ca2 = dot(a - c, a - c);
  const double denominator = ab2 + bc2 + ca2;
  if (!(denominator > 1.0e-30))
    return 0.0;
  const double twice_area = norm(cross(b - a, c - a));
  return 2.0 * std::sqrt(3.0) * twice_area / denominator;
}

[[nodiscard]] std::size_t flip_edges_for_quality(SurfaceMesh &mesh) {
  struct Owner {
    std::size_t triangle{0};
    std::size_t from{0};
    std::size_t to{0};
    std::size_t opposite{0};
  };
  std::unordered_map<EdgeKey, std::vector<Owner>, EdgeKeyHash> owners;
  owners.reserve(mesh.triangles.size() * 3U);
  for (std::size_t index = 0; index < mesh.triangles.size(); ++index) {
    const Triangle &triangle = mesh.triangles[index];
    owners[edge_key(triangle.v0, triangle.v1)].push_back(
        {index, triangle.v0, triangle.v1, triangle.v2});
    owners[edge_key(triangle.v1, triangle.v2)].push_back(
        {index, triangle.v1, triangle.v2, triangle.v0});
    owners[edge_key(triangle.v2, triangle.v0)].push_back(
        {index, triangle.v2, triangle.v0, triangle.v1});
  }
  struct Candidate {
    EdgeKey edge;
    Owner first;
    Owner second;
    double old_quality{0.0};
  };
  std::vector<Candidate> candidates;
  for (const auto &[edge, edge_owners] : owners) {
    if (edge_owners.size() != 2U)
      continue;
    const Owner &first = edge_owners[0];
    const Owner &second = edge_owners[1];
    if (first.from != second.to || first.to != second.from ||
        first.opposite == second.opposite)
      continue;
    const Vec3 a = mesh_vertex(mesh, first.from);
    const Vec3 b = mesh_vertex(mesh, first.to);
    const Vec3 c = mesh_vertex(mesh, first.opposite);
    const Vec3 d = mesh_vertex(mesh, second.opposite);
    const double old_quality = std::min(normalized_triangle_quality(a, b, c),
                                        normalized_triangle_quality(b, a, d));
    candidates.push_back({edge, first, second, old_quality});
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &a, const Candidate &b) {
              const double a_quality = stable_repair_metric(a.old_quality);
              const double b_quality = stable_repair_metric(b.old_quality);
              if (a_quality != b_quality)
                return a_quality < b_quality;
              return a.edge.lo != b.edge.lo ? a.edge.lo < b.edge.lo
                                            : a.edge.hi < b.edge.hi;
            });
  std::vector<bool> touched(mesh.triangles.size(), false);
  std::size_t flipped = 0;
  for (const Candidate &candidate : candidates) {
    const Owner &first = candidate.first;
    const Owner &second = candidate.second;
    if (touched[first.triangle] || touched[second.triangle])
      continue;
    const EdgeKey replacement = edge_key(first.opposite, second.opposite);
    if (owners.contains(replacement))
      continue;
    const Vec3 a = mesh_vertex(mesh, first.from);
    const Vec3 b = mesh_vertex(mesh, first.to);
    const Vec3 c = mesh_vertex(mesh, first.opposite);
    const Vec3 d = mesh_vertex(mesh, second.opposite);
    const double new_first_quality = normalized_triangle_quality(c, d, b);
    const double new_second_quality = normalized_triangle_quality(d, c, a);
    const double new_quality = std::min(new_first_quality, new_second_quality);
    if (!(stable_repair_metric(new_quality) >
          stable_repair_metric(candidate.old_quality * 1.05 + 1.0e-8)))
      continue;
    const Vec3 normal_a = normalized(
        {mesh.vertices[first.from].nx, mesh.vertices[first.from].ny,
         mesh.vertices[first.from].nz});
    const Vec3 normal_b = normalized(
        {mesh.vertices[first.to].nx, mesh.vertices[first.to].ny,
         mesh.vertices[first.to].nz});
    const Vec3 normal_c = normalized(
        {mesh.vertices[first.opposite].nx,
         mesh.vertices[first.opposite].ny,
         mesh.vertices[first.opposite].nz});
    const Vec3 normal_d = normalized(
        {mesh.vertices[second.opposite].nx,
         mesh.vertices[second.opposite].ny,
         mesh.vertices[second.opposite].nz});
    const double old_normal_compatibility = dot(normal_a, normal_b);
    const double new_normal_compatibility = dot(normal_c, normal_d);
    // Do not replace an edge by a diagonal that crosses a materially sharper
    // analytic-normal transition.  Shape improvement remains the primary
    // criterion on smooth patches, while seams and high-curvature necks are
    // protected from connectivity shortcuts.
    if (new_normal_compatibility < 0.25 &&
        new_normal_compatibility + 0.10 < old_normal_compatibility)
      continue;
    const Vec3 old_first_normal = cross(b - a, c - a);
    const Vec3 old_second_normal = cross(a - b, d - b);
    const Vec3 reference_normal = old_first_normal + old_second_normal;
    const Vec3 new_first_normal = cross(d - c, b - c);
    const Vec3 new_second_normal = cross(c - d, a - d);
    const auto vertex_normal = [&](std::uint32_t index) {
      const Vertex &vertex = mesh.vertices[index];
      return Vec3{vertex.nx, vertex.ny, vertex.nz};
    };
    const Vec3 first_patch_normal =
        vertex_normal(first.opposite) + vertex_normal(second.opposite) +
        vertex_normal(first.to);
    const Vec3 second_patch_normal =
        vertex_normal(second.opposite) + vertex_normal(first.opposite) +
        vertex_normal(first.from);
    // A summed-normal check can hide one inverted replacement face.  Require
    // both new faces to stay on the same geometric side of the local quad and
    // of the analytic normals carried by their vertices.
    if (!(norm(reference_normal) > 1.0e-16) ||
        !(dot(reference_normal, new_first_normal) > 0.0) ||
        !(dot(reference_normal, new_second_normal) > 0.0) ||
        (norm(first_patch_normal) > 1.0e-12 &&
         !(dot(first_patch_normal, new_first_normal) > 0.0)) ||
        (norm(second_patch_normal) > 1.0e-12 &&
         !(dot(second_patch_normal, new_second_normal) > 0.0)))
      continue;
    Triangle &triangle_first = mesh.triangles[first.triangle];
    Triangle &triangle_second = mesh.triangles[second.triangle];
    triangle_first.v0 = first.opposite;
    triangle_first.v1 = second.opposite;
    triangle_first.v2 = first.to;
    triangle_second.v0 = second.opposite;
    triangle_second.v1 = first.opposite;
    triangle_second.v2 = first.from;
    update_triangle_geometry(mesh, first.triangle);
    update_triangle_geometry(mesh, second.triangle);
    touched[first.triangle] = true;
    touched[second.triangle] = true;
    ++flipped;
  }
  return flipped;
}

void repair_micro_slivers(SurfaceMesh &mesh, double grid_spacing) {
  orient_mesh_consistently(mesh);
  for (int round = 0; round < 8; ++round) {
    SurfaceMesh candidate = mesh;
    const std::size_t collapsed =
        collapse_short_manifold_edges(candidate, 1.0e-2 * grid_spacing);
    if (collapsed == 0)
      break;
    orient_mesh_consistently(candidate);
    const TraceSpaces audit = build_closed_surface_trace_spaces(candidate);
    if (!audit.quality.valid)
      break;
    mesh = std::move(candidate);
  }
  for (int round = 0; round < 8; ++round) {
    SurfaceMesh candidate = mesh;
    if (flip_edges_for_quality(candidate) == 0)
      return;
    orient_mesh_consistently(candidate);
    const TraceSpaces audit = build_closed_surface_trace_spaces(candidate);
    if (!audit.quality.valid)
      return;
    mesh = std::move(candidate);
  }
}

[[nodiscard]] double mesh_max_edge_ratio(const SurfaceMesh &mesh) noexcept {
  double result = 0.0;
  for (const Triangle &triangle : mesh.triangles) {
    const Vec3 a = mesh_vertex(mesh, triangle.v0);
    const Vec3 b = mesh_vertex(mesh, triangle.v1);
    const Vec3 c = mesh_vertex(mesh, triangle.v2);
    const double ab = norm(b - a);
    const double bc = norm(c - b);
    const double ca = norm(a - c);
    const double shortest = std::min({ab, bc, ca});
    if (shortest > 1.0e-14)
      result = std::max(result, std::max({ab, bc, ca}) / shortest);
  }
  return result;
}

void smooth_analytic_slivers(SurfaceMesh &mesh,
                             const analytic_ses::PatchComplex &patches,
                             double grid_spacing) {
  for (int round = 0; round < 4; ++round) {
    std::vector<std::unordered_set<std::size_t>> neighbors(
        mesh.vertices.size());
    std::vector<Vec3> normal_sums(mesh.vertices.size());
    std::vector<bool> poor(mesh.vertices.size(), false);
    for (const Triangle &triangle : mesh.triangles) {
      const Vec3 a = mesh_vertex(mesh, triangle.v0);
      const Vec3 b = mesh_vertex(mesh, triangle.v1);
      const Vec3 c = mesh_vertex(mesh, triangle.v2);
      const Vec3 face_normal = cross(b - a, c - a);
      normal_sums[triangle.v0] = normal_sums[triangle.v0] + face_normal;
      normal_sums[triangle.v1] = normal_sums[triangle.v1] + face_normal;
      normal_sums[triangle.v2] = normal_sums[triangle.v2] + face_normal;
      neighbors[triangle.v0].insert(triangle.v1);
      neighbors[triangle.v0].insert(triangle.v2);
      neighbors[triangle.v1].insert(triangle.v0);
      neighbors[triangle.v1].insert(triangle.v2);
      neighbors[triangle.v2].insert(triangle.v0);
      neighbors[triangle.v2].insert(triangle.v1);
      const double quality = normalized_triangle_quality(a, b, c);
      const double ab = norm(b - a);
      const double bc = norm(c - b);
      const double ca = norm(a - c);
      const double shortest = std::min({ab, bc, ca});
      const double ratio = shortest > 1.0e-14
                               ? std::max({ab, bc, ca}) / shortest
                               : std::numeric_limits<double>::infinity();
      if (quality < 0.02 || ratio > 50.0) {
        poor[triangle.v0] = true;
        poor[triangle.v1] = true;
        poor[triangle.v2] = true;
      }
    }
    SurfaceMesh candidate = mesh;
    std::size_t moved = 0;
    for (std::size_t index = 0; index < mesh.vertices.size(); ++index) {
      if (!poor[index] || neighbors[index].size() < 3U)
        continue;
      const Vec3 point = mesh_vertex(mesh, index);
      Vec3 average;
      for (const std::size_t neighbor : neighbors[index])
        average = average + mesh_vertex(mesh, neighbor);
      average = (1.0 / static_cast<double>(neighbors[index].size())) * average;
      const Vec3 normal = normalized(normal_sums[index]);
      Vec3 displacement = average - point;
      displacement = displacement - dot(displacement, normal) * normal;
      const double displacement_norm = norm(displacement);
      const double cap = 0.15 * grid_spacing;
      if (displacement_norm > cap)
        displacement = (cap / displacement_norm) * displacement;
      Vec3 proposed = point + 0.25 * displacement;
      const auto hits =
          analytic_ses::intersect_ray_x(patches, proposed.y, proposed.z);
      if (hits.empty())
        continue;
      const auto closest = std::min_element(
          hits.begin(), hits.end(), [&](const auto &left, const auto &right) {
            return std::abs(left.x - proposed.x) <
                   std::abs(right.x - proposed.x);
          });
      if (std::abs(closest->x - point.x) > 0.25 * grid_spacing)
        continue;
      proposed.x = closest->x;
      Vertex &vertex = candidate.vertices[index];
      vertex.x = proposed.x;
      vertex.y = proposed.y;
      vertex.z = proposed.z;
      vertex.nx = closest->normal.x;
      vertex.ny = closest->normal.y;
      vertex.nz = closest->normal.z;
      ++moved;
    }
    if (moved == 0)
      return;
    for (std::size_t index = 0; index < candidate.triangles.size(); ++index)
      update_triangle_geometry(candidate, index);
    orient_mesh_consistently(candidate);
    const TraceSpaces audit = build_closed_surface_trace_spaces(candidate);
    if (!audit.quality.valid)
      return;
    const double old_ratio = mesh_max_edge_ratio(mesh);
    const double new_ratio = mesh_max_edge_ratio(candidate);
    if (!(new_ratio < old_ratio * 0.995))
      return;
    mesh = std::move(candidate);
  }
}

struct CompactVertexAdjacency {
  std::vector<std::size_t> offsets;
  std::vector<std::uint32_t> neighbors;

  [[nodiscard]] std::size_t retained_bytes() const noexcept {
    return offsets.capacity() * sizeof(std::size_t) +
           neighbors.capacity() * sizeof(std::uint32_t);
  }
};

struct CompactVertexIncidence {
  std::vector<std::size_t> offsets;
  std::vector<std::uint32_t> triangles;

  [[nodiscard]] std::size_t retained_bytes() const noexcept {
    return offsets.capacity() * sizeof(std::size_t) +
           triangles.capacity() * sizeof(std::uint32_t);
  }
};

[[nodiscard]] CompactVertexIncidence
build_compact_vertex_incidence(const SurfaceMesh &mesh) {
  CompactVertexIncidence result;
  if (mesh.vertices.size() > std::numeric_limits<std::uint32_t>::max() ||
      mesh.triangles.size() > std::numeric_limits<std::uint32_t>::max())
    return result;
  result.offsets.assign(mesh.vertices.size() + 1U, 0U);
  for (const Triangle &face : mesh.triangles) {
    ++result.offsets[face.v0 + 1U];
    ++result.offsets[face.v1 + 1U];
    ++result.offsets[face.v2 + 1U];
  }
  for (std::size_t vertex = 0; vertex < mesh.vertices.size(); ++vertex)
    result.offsets[vertex + 1U] += result.offsets[vertex];
  result.triangles.resize(mesh.triangles.size() * 3U);
  std::vector<std::size_t> cursor = result.offsets;
  for (std::size_t triangle = 0; triangle < mesh.triangles.size(); ++triangle) {
    const Triangle &face = mesh.triangles[triangle];
    const std::uint32_t index = static_cast<std::uint32_t>(triangle);
    result.triangles[cursor[face.v0]++] = index;
    result.triangles[cursor[face.v1]++] = index;
    result.triangles[cursor[face.v2]++] = index;
  }
  return result;
}

// A link-condition pass only proves that an edge collapse preserves the
// combinatorial manifold.  It does not prevent the retained one-ring from
// folding through itself.  Certify every surviving incident face against its
// pre-collapse orientation before mutating a remap or a replacement vertex.
[[nodiscard]] bool collapse_preserves_incident_face_orientation(
    const SurfaceMesh &source, const CompactVertexIncidence &incidence,
    std::size_t keep, std::size_t remove,
    const Vec3 &replacement) noexcept {
  if (keep >= source.vertices.size() || remove >= source.vertices.size() ||
      keep == remove ||
      incidence.offsets.size() != source.vertices.size() + 1U ||
      !finite_vec3(replacement))
    return false;

  const auto validate_incident_faces = [&](std::size_t vertex) {
    for (std::size_t position = incidence.offsets[vertex];
         position < incidence.offsets[vertex + 1U]; ++position) {
      const std::size_t triangle_index = incidence.triangles[position];
      if (triangle_index >= source.triangles.size())
        return false;
      const Triangle &triangle = source.triangles[triangle_index];
      if (triangle.v0 >= source.vertices.size() ||
          triangle.v1 >= source.vertices.size() ||
          triangle.v2 >= source.vertices.size())
        return false;
      const bool contains_keep =
          triangle.v0 == keep || triangle.v1 == keep || triangle.v2 == keep;
      const bool contains_remove = triangle.v0 == remove ||
                                   triangle.v1 == remove ||
                                   triangle.v2 == remove;
      // These are exactly the two manifold faces removed by the collapse.
      if (contains_keep && contains_remove)
        continue;

      const Vec3 old_a = mesh_vertex(source, triangle.v0);
      const Vec3 old_b = mesh_vertex(source, triangle.v1);
      const Vec3 old_c = mesh_vertex(source, triangle.v2);
      const auto collapsed_point = [&](std::size_t index,
                                       const Vec3 &original) {
        return index == keep || index == remove ? replacement : original;
      };
      const FaceMoveValidation validation = validate_face_move(
          collapsed_point(triangle.v0, old_a),
          collapsed_point(triangle.v1, old_b),
          collapsed_point(triangle.v2, old_c),
          cross(old_b - old_a, old_c - old_a));
      if (!validation.geometry_valid || !validation.orientation_preserved)
        return false;
    }
    return true;
  };

  // A surviving face is incident to only one endpoint, so the two scans do
  // not duplicate work; faces incident to both endpoints are skipped above.
  return validate_incident_faces(keep) &&
         validate_incident_faces(remove);
}

[[nodiscard]] CompactVertexAdjacency
build_compact_vertex_adjacency(const SurfaceMesh &mesh) {
  CompactVertexAdjacency result;
  if (mesh.vertices.size() > std::numeric_limits<std::uint32_t>::max())
    return result;
  std::vector<EdgeKey> edges;
  edges.reserve(mesh.triangles.size() * 3U);
  const auto append = [&](std::size_t first, std::size_t second) {
    if (first == second)
      return;
    edges.push_back({std::min(first, second), std::max(first, second)});
  };
  for (const Triangle &triangle : mesh.triangles) {
    append(triangle.v0, triangle.v1);
    append(triangle.v1, triangle.v2);
    append(triangle.v2, triangle.v0);
  }
  std::sort(edges.begin(), edges.end(),
            [](const EdgeKey &a, const EdgeKey &b) {
              return std::tie(a.lo, a.hi) < std::tie(b.lo, b.hi);
            });
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  result.offsets.assign(mesh.vertices.size() + 1U, 0U);
  for (const EdgeKey &edge : edges) {
    ++result.offsets[edge.lo + 1U];
    ++result.offsets[edge.hi + 1U];
  }
  for (std::size_t index = 0; index < mesh.vertices.size(); ++index)
    result.offsets[index + 1U] += result.offsets[index];
  result.neighbors.resize(edges.size() * 2U);
  std::vector<std::size_t> cursor = result.offsets;
  for (const EdgeKey &edge : edges) {
    result.neighbors[cursor[edge.lo]++] =
        static_cast<std::uint32_t>(edge.hi);
    result.neighbors[cursor[edge.hi]++] =
        static_cast<std::uint32_t>(edge.lo);
  }
  return result;
}

struct CompactOwnedEdge {
  std::uint32_t lo{0};
  std::uint32_t hi{0};
  std::uint32_t triangle{0};
  std::uint32_t from{0};
  std::uint32_t to{0};
  std::uint32_t opposite{0};
};

[[nodiscard]] std::vector<CompactOwnedEdge>
build_compact_owned_edges(const SurfaceMesh &mesh) {
  std::vector<CompactOwnedEdge> edges;
  if (mesh.vertices.size() > std::numeric_limits<std::uint32_t>::max() ||
      mesh.triangles.size() > std::numeric_limits<std::uint32_t>::max())
    return edges;
  edges.reserve(mesh.triangles.size() * 3U);
  const auto append = [&](std::size_t triangle, std::size_t from,
                          std::size_t to, std::size_t opposite) {
    edges.push_back(
        {static_cast<std::uint32_t>(std::min(from, to)),
         static_cast<std::uint32_t>(std::max(from, to)),
         static_cast<std::uint32_t>(triangle),
         static_cast<std::uint32_t>(from), static_cast<std::uint32_t>(to),
         static_cast<std::uint32_t>(opposite)});
  };
  for (std::size_t index = 0; index < mesh.triangles.size(); ++index) {
    const Triangle &triangle = mesh.triangles[index];
    append(index, triangle.v0, triangle.v1, triangle.v2);
    append(index, triangle.v1, triangle.v2, triangle.v0);
    append(index, triangle.v2, triangle.v0, triangle.v1);
  }
  std::sort(edges.begin(), edges.end(),
            [](const CompactOwnedEdge &a, const CompactOwnedEdge &b) {
              return std::tie(a.lo, a.hi, a.triangle) <
                     std::tie(b.lo, b.hi, b.triangle);
            });
  return edges;
}

[[nodiscard]] CompactVertexAdjacency
build_compact_vertex_adjacency_from_owned_edges(
    std::size_t vertex_count,
    const std::vector<CompactOwnedEdge> &owned_edges) {
  CompactVertexAdjacency result;
  result.offsets.assign(vertex_count + 1U, 0U);
  std::size_t unique_edges = 0U;
  for (std::size_t begin = 0U; begin < owned_edges.size();) {
    std::size_t end = begin + 1U;
    while (end < owned_edges.size() &&
           owned_edges[end].lo == owned_edges[begin].lo &&
           owned_edges[end].hi == owned_edges[begin].hi)
      ++end;
    if (owned_edges[begin].lo != owned_edges[begin].hi) {
      ++result.offsets[static_cast<std::size_t>(owned_edges[begin].lo) + 1U];
      ++result.offsets[static_cast<std::size_t>(owned_edges[begin].hi) + 1U];
      ++unique_edges;
    }
    begin = end;
  }
  for (std::size_t vertex = 0U; vertex < vertex_count; ++vertex)
    result.offsets[vertex + 1U] += result.offsets[vertex];
  result.neighbors.resize(unique_edges * 2U);
  std::vector<std::size_t> cursor = result.offsets;
  for (std::size_t begin = 0U; begin < owned_edges.size();) {
    std::size_t end = begin + 1U;
    while (end < owned_edges.size() &&
           owned_edges[end].lo == owned_edges[begin].lo &&
           owned_edges[end].hi == owned_edges[begin].hi)
      ++end;
    const std::uint32_t lo = owned_edges[begin].lo;
    const std::uint32_t hi = owned_edges[begin].hi;
    if (lo != hi) {
      result.neighbors[cursor[lo]++] = hi;
      result.neighbors[cursor[hi]++] = lo;
    }
    begin = end;
  }
  return result;
}

[[nodiscard]] bool compact_edge_exists(
    const std::vector<CompactOwnedEdge> &edges, std::uint32_t lo,
    std::uint32_t hi) noexcept {
  if (lo > hi)
    std::swap(lo, hi);
  const auto found = std::lower_bound(
      edges.begin(), edges.end(), std::pair{lo, hi},
      [](const CompactOwnedEdge &edge,
         const std::pair<std::uint32_t, std::uint32_t> &key) {
        return std::pair{edge.lo, edge.hi} < key;
      });
  return found != edges.end() && found->lo == lo && found->hi == hi;
}

[[nodiscard]] std::size_t compact_common_neighbor_count(
    const CompactVertexAdjacency &adjacency, std::size_t a,
    std::size_t b) noexcept {
  std::size_t left = adjacency.offsets[a];
  std::size_t right = adjacency.offsets[b];
  const std::size_t left_end = adjacency.offsets[a + 1U];
  const std::size_t right_end = adjacency.offsets[b + 1U];
  std::size_t count = 0U;
  while (left < left_end && right < right_end) {
    const std::uint32_t x = adjacency.neighbors[left];
    const std::uint32_t y = adjacency.neighbors[right];
    if (x == y) {
      ++count;
      ++left;
      ++right;
    } else if (x < y) {
      ++left;
    } else {
      ++right;
    }
  }
  return count;
}

[[nodiscard]] std::size_t collapse_short_manifold_edges_compact(
    const SurfaceMesh &source, SurfaceMesh &candidate, double threshold,
    std::size_t &scratch_bytes, bool &topology_certified,
    std::span<const EdgeKey> allowed_edges = {},
    bool restrict_to_allowed_edges = false,
    bool preserve_kept_vertex = false,
    bool keep_higher_endpoint = false) {
  topology_certified = false;
  if (source.vertices.empty() || source.triangles.empty() ||
      source.panels.size() != source.triangles.size() ||
      !(threshold > 0.0) ||
      source.vertices.size() > std::numeric_limits<std::uint32_t>::max() ||
      source.triangles.size() > std::numeric_limits<std::uint32_t>::max())
    return 0U;
  for (const Triangle &triangle : source.triangles) {
    if (triangle.v0 >= source.vertices.size() ||
        triangle.v1 >= source.vertices.size() ||
        triangle.v2 >= source.vertices.size() ||
        triangle.v0 == triangle.v1 || triangle.v1 == triangle.v2 ||
        triangle.v2 == triangle.v0 || !(triangle.area > 0.0) ||
        !std::isfinite(triangle.area))
      return 0U;
  }
  std::vector<CompactOwnedEdge> edges = build_compact_owned_edges(source);
  CompactVertexAdjacency adjacency =
      build_compact_vertex_adjacency_from_owned_edges(source.vertices.size(),
                                                      edges);
  CompactVertexIncidence incidence =
      build_compact_vertex_incidence(source);
  if (adjacency.offsets.size() != source.vertices.size() + 1U ||
      incidence.offsets.size() != source.vertices.size() + 1U ||
      edges.empty())
    return 0U;
  bool source_closed_oriented = true;
  for (std::size_t vertex = 0U; vertex < source.vertices.size(); ++vertex) {
    if (adjacency.offsets[vertex] == adjacency.offsets[vertex + 1U]) {
      source_closed_oriented = false;
      break;
    }
  }
  struct Candidate {
    std::uint32_t lo{0};
    std::uint32_t hi{0};
    double length{0.0};
  };
  std::vector<Candidate> candidates;
  for (std::size_t begin = 0; begin < edges.size();) {
    std::size_t end = begin + 1U;
    while (end < edges.size() && edges[end].lo == edges[begin].lo &&
           edges[end].hi == edges[begin].hi)
      ++end;
    if (end - begin != 2U ||
        edges[begin].from != edges[begin + 1U].to ||
        edges[begin].to != edges[begin + 1U].from)
      source_closed_oriented = false;
    if (end - begin == 2U) {
      const Vec3 a = mesh_vertex(source, edges[begin].lo);
      const Vec3 b = mesh_vertex(source, edges[begin].hi);
      const double length = norm(b - a);
      const EdgeKey key{edges[begin].lo, edges[begin].hi};
      const bool allowed =
          !restrict_to_allowed_edges ||
          std::binary_search(
              allowed_edges.begin(), allowed_edges.end(), key,
              [](const EdgeKey &left, const EdgeKey &right) {
                return std::tie(left.lo, left.hi) <
                       std::tie(right.lo, right.hi);
              });
      if (allowed && length < threshold)
        candidates.push_back({edges[begin].lo, edges[begin].hi, length});
    }
    begin = end;
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &a, const Candidate &b) {
              const double a_length = stable_repair_metric(a.length);
              const double b_length = stable_repair_metric(b.length);
              if (a_length != b_length)
                return a_length < b_length;
              return std::tie(a.lo, a.hi) < std::tie(b.lo, b.hi);
            });
  std::vector<std::size_t> remap(source.vertices.size());
  for (std::size_t index = 0; index < remap.size(); ++index)
    remap[index] = index;
  std::vector<std::uint8_t> touched(source.vertices.size(), 0U);
  struct Replacement {
    std::size_t index{0U};
    Vertex vertex;
  };
  std::vector<Replacement> replacements;
  replacements.reserve(candidates.size());
  std::size_t collapsed = 0U;
  for (const Candidate &candidate : candidates) {
    const std::size_t keep =
        keep_higher_endpoint ? candidate.hi : candidate.lo;
    const std::size_t remove =
        keep_higher_endpoint ? candidate.lo : candidate.hi;
    if (touched[keep] != 0U || touched[remove] != 0U ||
        compact_common_neighbor_count(adjacency, keep, remove) != 2U)
      continue;
    Vertex kept = source.vertices[keep];
    const Vertex &removed = source.vertices[remove];
    if (!preserve_kept_vertex) {
      kept.x = 0.5 * (kept.x + removed.x);
      kept.y = 0.5 * (kept.y + removed.y);
      kept.z = 0.5 * (kept.z + removed.z);
      const Vec3 blended_normal = normalized(
          {kept.nx + removed.nx, kept.ny + removed.ny,
           kept.nz + removed.nz});
      kept.nx = blended_normal.x;
      kept.ny = blended_normal.y;
      kept.nz = blended_normal.z;
    }
    if (!collapse_preserves_incident_face_orientation(
            source, incidence, keep, remove,
            {kept.x, kept.y, kept.z}))
      continue;
    replacements.push_back({keep, kept});
    remap[remove] = keep;
    touched[keep] = 1U;
    touched[remove] = 1U;
    for (std::size_t position = adjacency.offsets[keep];
         position < adjacency.offsets[keep + 1U]; ++position)
      touched[adjacency.neighbors[position]] = 1U;
    for (std::size_t position = adjacency.offsets[remove];
         position < adjacency.offsets[remove + 1U]; ++position)
      touched[adjacency.neighbors[position]] = 1U;
    ++collapsed;
  }
  scratch_bytes = std::max(
      scratch_bytes,
      adjacency.retained_bytes() + incidence.retained_bytes() +
          edges.capacity() * sizeof(CompactOwnedEdge) +
          candidates.capacity() * sizeof(Candidate) +
          remap.capacity() * sizeof(std::size_t) +
          touched.capacity() * sizeof(std::uint8_t) +
          replacements.capacity() * sizeof(Replacement));
  if (collapsed == 0U) {
    topology_certified = source_closed_oriented;
    return 0U;
  }

  // The edge graph is not needed while the compact candidate is assembled.
  // Releasing it here keeps peak memory to one output mesh plus linear vertex
  // work arrays instead of retaining both the graph and a full mesh copy.
  std::vector<std::size_t>().swap(adjacency.offsets);
  std::vector<std::uint32_t>().swap(adjacency.neighbors);
  std::vector<std::size_t>().swap(incidence.offsets);
  std::vector<std::uint32_t>().swap(incidence.triangles);
  std::vector<CompactOwnedEdge>().swap(edges);
  std::vector<Candidate>().swap(candidates);
  std::vector<std::uint8_t>().swap(touched);

  candidate.vertices.clear();
  candidate.triangles.clear();
  candidate.panels.clear();
  candidate.triangles.reserve(source.triangles.size());
  candidate.panels.reserve(source.panels.size());
  for (std::size_t index = 0; index < source.triangles.size(); ++index) {
    Triangle triangle = source.triangles[index];
    triangle.v0 = remap[triangle.v0];
    triangle.v1 = remap[triangle.v1];
    triangle.v2 = remap[triangle.v2];
    if (triangle.v0 == triangle.v1 || triangle.v1 == triangle.v2 ||
        triangle.v2 == triangle.v0)
      continue;
    candidate.triangles.push_back(triangle);
    candidate.panels.push_back(source.panels[index]);
  }
  std::vector<std::uint8_t> referenced(source.vertices.size(), 0U);
  for (const Triangle &triangle : candidate.triangles) {
    referenced[triangle.v0] = 1U;
    referenced[triangle.v1] = 1U;
    referenced[triangle.v2] = 1U;
  }
  std::sort(replacements.begin(), replacements.end(),
            [](const Replacement &a, const Replacement &b) {
              return a.index < b.index;
            });
  candidate.vertices.reserve(source.vertices.size() - collapsed);
  std::size_t replacement = 0U;
  for (std::size_t index = 0; index < source.vertices.size(); ++index) {
    if (referenced[index] == 0U)
      continue;
    remap[index] = candidate.vertices.size();
    while (replacement < replacements.size() &&
           replacements[replacement].index < index)
      ++replacement;
    if (replacement < replacements.size() &&
        replacements[replacement].index == index) {
      candidate.vertices.push_back(replacements[replacement].vertex);
      ++replacement;
    } else {
      candidate.vertices.push_back(source.vertices[index]);
    }
  }
  for (Triangle &triangle : candidate.triangles) {
    triangle.v0 = remap[triangle.v0];
    triangle.v1 = remap[triangle.v1];
    triangle.v2 = remap[triangle.v2];
  }
  scratch_bytes = std::max(
      scratch_bytes,
      mesh_capacity_bytes(candidate) +
          remap.capacity() * sizeof(std::size_t) +
          replacements.capacity() * sizeof(Replacement) +
          referenced.capacity() * sizeof(std::uint8_t));
  bool candidate_geometry_valid = true;
  for (std::size_t index = 0; index < candidate.triangles.size(); ++index) {
    update_triangle_geometry(candidate, index);
    if (!(candidate.triangles[index].area > 0.0) ||
        !std::isfinite(candidate.triangles[index].area))
      candidate_geometry_valid = false;
  }
  topology_certified = source_closed_oriented && candidate_geometry_valid;
  return collapsed;
}

[[nodiscard]] bool compact_vertices_adjacent(
    const CompactVertexAdjacency &adjacency, std::size_t first,
    std::size_t second) noexcept {
  if (first + 1U >= adjacency.offsets.size())
    return false;
  const auto begin = adjacency.neighbors.begin() +
                     static_cast<std::ptrdiff_t>(adjacency.offsets[first]);
  const auto end = adjacency.neighbors.begin() +
                   static_cast<std::ptrdiff_t>(adjacency.offsets[first + 1U]);
  return std::binary_search(begin, end, static_cast<std::uint32_t>(second));
}

// Transactional fast path for the cold wide-fold fallback. The caller builds
// the source CSR once; each endpoint trial is then linear in the mesh size and
// avoids rebuilding and sorting the complete owned-edge list.
[[nodiscard]] std::size_t collapse_single_manifold_edge_compact(
    const SurfaceMesh &source, SurfaceMesh &candidate, const EdgeKey &edge,
    const CompactVertexAdjacency &adjacency, bool keep_higher_endpoint,
    const CompactVertexIncidence &incidence, std::size_t &scratch_bytes) {
  if (edge.lo >= source.vertices.size() || edge.hi >= source.vertices.size() ||
      edge.lo == edge.hi ||
      source.panels.size() != source.triangles.size() ||
      adjacency.offsets.size() != source.vertices.size() + 1U ||
      incidence.offsets.size() != source.vertices.size() + 1U ||
      !compact_vertices_adjacent(adjacency, edge.lo, edge.hi) ||
      compact_common_neighbor_count(adjacency, edge.lo, edge.hi) != 2U)
    return 0U;

  const std::size_t keep = keep_higher_endpoint ? edge.hi : edge.lo;
  const std::size_t remove = keep_higher_endpoint ? edge.lo : edge.hi;
  const Vec3 replacement = mesh_vertex(source, keep);
  if (!collapse_preserves_incident_face_orientation(
          source, incidence, keep, remove, replacement))
    return 0U;
  std::vector<std::size_t> remap(source.vertices.size());
  for (std::size_t index = 0U; index < remap.size(); ++index)
    remap[index] = index;
  remap[remove] = keep;

  candidate.vertices.clear();
  candidate.triangles.clear();
  candidate.panels.clear();
  candidate.triangles.reserve(source.triangles.size());
  candidate.panels.reserve(source.panels.size());
  for (std::size_t index = 0U; index < source.triangles.size(); ++index) {
    Triangle triangle = source.triangles[index];
    triangle.v0 = remap[triangle.v0];
    triangle.v1 = remap[triangle.v1];
    triangle.v2 = remap[triangle.v2];
    if (triangle.v0 == triangle.v1 || triangle.v1 == triangle.v2 ||
        triangle.v2 == triangle.v0)
      continue;
    candidate.triangles.push_back(triangle);
    candidate.panels.push_back(source.panels[index]);
  }

  std::vector<std::uint8_t> referenced(source.vertices.size(), 0U);
  for (const Triangle &triangle : candidate.triangles) {
    referenced[triangle.v0] = 1U;
    referenced[triangle.v1] = 1U;
    referenced[triangle.v2] = 1U;
  }
  candidate.vertices.reserve(source.vertices.size() - 1U);
  for (std::size_t index = 0U; index < source.vertices.size(); ++index) {
    if (referenced[index] == 0U)
      continue;
    remap[index] = candidate.vertices.size();
    candidate.vertices.push_back(source.vertices[index]);
  }
  for (Triangle &triangle : candidate.triangles) {
    triangle.v0 = remap[triangle.v0];
    triangle.v1 = remap[triangle.v1];
    triangle.v2 = remap[triangle.v2];
  }
  bool candidate_geometry_valid = true;
  for (std::size_t index = 0U; index < candidate.triangles.size(); ++index) {
    update_triangle_geometry(candidate, index);
    if (!(candidate.triangles[index].area > 0.0) ||
        !std::isfinite(candidate.triangles[index].area))
      candidate_geometry_valid = false;
  }
  scratch_bytes = std::max(
      scratch_bytes,
      adjacency.retained_bytes() + incidence.retained_bytes() +
          mesh_capacity_bytes(candidate) +
          remap.capacity() * sizeof(std::size_t) +
          referenced.capacity() * sizeof(std::uint8_t));
  return candidate_geometry_valid ? 1U : 0U;
}

// Sparse batch collapse for analytic-orientation repair. Only endpoints of
// the supplied inverted-face edges receive one-ring storage, so each repair
// round uses one linear triangle scan instead of sorting the full 3T edge
// graph. Candidate order and the link/touched rules match the generic path.
[[nodiscard]] std::size_t collapse_selected_manifold_edges_compact(
    const SurfaceMesh &source, SurfaceMesh &candidate, double threshold,
    std::span<const EdgeKey> allowed_edges, std::size_t &scratch_bytes) {
  if (source.vertices.empty() || source.triangles.empty() ||
      allowed_edges.empty() || !(threshold > 0.0) ||
      source.panels.size() != source.triangles.size() ||
      source.vertices.size() > std::numeric_limits<std::uint32_t>::max())
    return 0U;
  for (const Triangle &triangle : source.triangles) {
    if (triangle.v0 >= source.vertices.size() ||
        triangle.v1 >= source.vertices.size() ||
        triangle.v2 >= source.vertices.size() ||
        triangle.v0 == triangle.v1 || triangle.v1 == triangle.v2 ||
        triangle.v2 == triangle.v0)
      return 0U;
  }
  CompactVertexIncidence incidence =
      build_compact_vertex_incidence(source);
  if (incidence.offsets.size() != source.vertices.size() + 1U)
    return 0U;

  std::vector<std::uint32_t> endpoints;
  endpoints.reserve(allowed_edges.size() * 2U);
  for (const EdgeKey &edge : allowed_edges) {
    if (edge.lo >= source.vertices.size() || edge.hi >= source.vertices.size() ||
        edge.lo == edge.hi)
      continue;
    endpoints.push_back(static_cast<std::uint32_t>(edge.lo));
    endpoints.push_back(static_cast<std::uint32_t>(edge.hi));
  }
  std::sort(endpoints.begin(), endpoints.end());
  endpoints.erase(std::unique(endpoints.begin(), endpoints.end()),
                  endpoints.end());
  if (endpoints.empty())
    return 0U;

  const std::uint32_t no_slot = std::numeric_limits<std::uint32_t>::max();
  std::vector<std::uint32_t> endpoint_slot(source.vertices.size(), no_slot);
  for (std::size_t slot = 0U; slot < endpoints.size(); ++slot)
    endpoint_slot[endpoints[slot]] = static_cast<std::uint32_t>(slot);
  std::vector<std::vector<std::uint32_t>> neighbors(endpoints.size());
  for (const Triangle &triangle : source.triangles) {
    const std::array<std::uint32_t, 3> vertices = {
        static_cast<std::uint32_t>(triangle.v0),
        static_cast<std::uint32_t>(triangle.v1),
        static_cast<std::uint32_t>(triangle.v2)};
    for (std::size_t local = 0U; local < 3U; ++local) {
      const std::uint32_t slot = endpoint_slot[vertices[local]];
      if (slot == no_slot)
        continue;
      neighbors[slot].push_back(vertices[(local + 1U) % 3U]);
      neighbors[slot].push_back(vertices[(local + 2U) % 3U]);
    }
  }
  std::size_t neighbor_capacity = 0U;
  for (auto &one_ring : neighbors) {
    std::sort(one_ring.begin(), one_ring.end());
    one_ring.erase(std::unique(one_ring.begin(), one_ring.end()),
                   one_ring.end());
    neighbor_capacity += one_ring.capacity();
  }
  const auto common_neighbor_count = [&](std::uint32_t first,
                                          std::uint32_t second) {
    const auto &left = neighbors[endpoint_slot[first]];
    const auto &right = neighbors[endpoint_slot[second]];
    std::size_t i = 0U;
    std::size_t j = 0U;
    std::size_t count = 0U;
    while (i < left.size() && j < right.size()) {
      if (left[i] == right[j]) {
        ++count;
        ++i;
        ++j;
      } else if (left[i] < right[j]) {
        ++i;
      } else {
        ++j;
      }
    }
    return count;
  };

  struct Candidate {
    std::uint32_t lo{0U};
    std::uint32_t hi{0U};
    double length{0.0};
  };
  std::vector<Candidate> candidates;
  candidates.reserve(allowed_edges.size());
  for (const EdgeKey &edge : allowed_edges) {
    if (edge.lo >= source.vertices.size() || edge.hi >= source.vertices.size() ||
        edge.lo == edge.hi)
      continue;
    const std::uint32_t lo = static_cast<std::uint32_t>(edge.lo);
    const std::uint32_t hi = static_cast<std::uint32_t>(edge.hi);
    const auto &lo_neighbors = neighbors[endpoint_slot[lo]];
    if (!std::binary_search(lo_neighbors.begin(), lo_neighbors.end(), hi) ||
        common_neighbor_count(lo, hi) != 2U)
      continue;
    const double length =
        norm(mesh_vertex(source, lo) - mesh_vertex(source, hi));
    if (length < threshold)
      candidates.push_back({lo, hi, length});
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &a, const Candidate &b) {
              const double a_length = stable_repair_metric(a.length);
              const double b_length = stable_repair_metric(b.length);
              if (a_length != b_length)
                return a_length < b_length;
              return std::tie(a.lo, a.hi) < std::tie(b.lo, b.hi);
            });

  std::vector<std::size_t> remap(source.vertices.size());
  for (std::size_t index = 0U; index < remap.size(); ++index)
    remap[index] = index;
  std::vector<std::uint8_t> touched(source.vertices.size(), 0U);
  std::size_t collapsed = 0U;
  for (const Candidate &edge : candidates) {
    const std::size_t keep = edge.lo;
    const std::size_t remove = edge.hi;
    if (touched[keep] != 0U || touched[remove] != 0U)
      continue;
    if (!collapse_preserves_incident_face_orientation(
            source, incidence, keep, remove, mesh_vertex(source, keep)))
      continue;
    remap[remove] = keep;
    touched[keep] = 1U;
    touched[remove] = 1U;
    for (const std::uint32_t neighbor : neighbors[endpoint_slot[edge.lo]])
      touched[neighbor] = 1U;
    for (const std::uint32_t neighbor : neighbors[endpoint_slot[edge.hi]])
      touched[neighbor] = 1U;
    ++collapsed;
  }
  scratch_bytes = std::max(
      scratch_bytes,
      incidence.retained_bytes() +
          endpoints.capacity() * sizeof(std::uint32_t) +
          endpoint_slot.capacity() * sizeof(std::uint32_t) +
          neighbors.capacity() * sizeof(std::vector<std::uint32_t>) +
          neighbor_capacity * sizeof(std::uint32_t) +
          candidates.capacity() * sizeof(Candidate) +
          remap.capacity() * sizeof(std::size_t) +
          touched.capacity() * sizeof(std::uint8_t));
  if (collapsed == 0U)
    return 0U;

  std::vector<std::size_t>().swap(incidence.offsets);
  std::vector<std::uint32_t>().swap(incidence.triangles);
  candidate.vertices.clear();
  candidate.triangles.clear();
  candidate.panels.clear();
  candidate.triangles.reserve(source.triangles.size());
  candidate.panels.reserve(source.panels.size());
  for (std::size_t index = 0U; index < source.triangles.size(); ++index) {
    Triangle triangle = source.triangles[index];
    triangle.v0 = remap[triangle.v0];
    triangle.v1 = remap[triangle.v1];
    triangle.v2 = remap[triangle.v2];
    if (triangle.v0 == triangle.v1 || triangle.v1 == triangle.v2 ||
        triangle.v2 == triangle.v0)
      continue;
    candidate.triangles.push_back(triangle);
    candidate.panels.push_back(source.panels[index]);
  }
  std::vector<std::uint8_t> referenced(source.vertices.size(), 0U);
  for (const Triangle &triangle : candidate.triangles) {
    referenced[triangle.v0] = 1U;
    referenced[triangle.v1] = 1U;
    referenced[triangle.v2] = 1U;
  }
  candidate.vertices.reserve(source.vertices.size() - collapsed);
  for (std::size_t index = 0U; index < source.vertices.size(); ++index) {
    if (referenced[index] == 0U)
      continue;
    remap[index] = candidate.vertices.size();
    candidate.vertices.push_back(source.vertices[index]);
  }
  for (Triangle &triangle : candidate.triangles) {
    triangle.v0 = remap[triangle.v0];
    triangle.v1 = remap[triangle.v1];
    triangle.v2 = remap[triangle.v2];
  }
  bool candidate_geometry_valid = true;
  for (std::size_t index = 0U; index < candidate.triangles.size(); ++index) {
    update_triangle_geometry(candidate, index);
    if (!(candidate.triangles[index].area > 0.0) ||
        !std::isfinite(candidate.triangles[index].area))
      candidate_geometry_valid = false;
  }
  scratch_bytes = std::max(
      scratch_bytes,
      mesh_capacity_bytes(candidate) +
          referenced.capacity() * sizeof(std::uint8_t) +
          remap.capacity() * sizeof(std::size_t));
  return candidate_geometry_valid ? collapsed : 0U;
}

struct CompactEdgeFlipUndo {
  std::uint32_t first_index{0};
  std::uint32_t second_index{0};
  Triangle first_triangle;
  Triangle second_triangle;
  Panel first_panel;
  Panel second_panel;
};

[[nodiscard]] std::size_t flip_edges_for_quality_compact(
    SurfaceMesh &mesh, std::size_t &scratch_bytes,
    std::vector<CompactEdgeFlipUndo> &undo) {
  undo.clear();
  std::vector<CompactOwnedEdge> edges = build_compact_owned_edges(mesh);
  if (edges.empty())
    return 0U;
  struct Candidate {
    CompactOwnedEdge first;
    CompactOwnedEdge second;
    double old_quality{0.0};
  };
  std::vector<Candidate> candidates;
  for (std::size_t begin = 0; begin < edges.size();) {
    std::size_t end = begin + 1U;
    while (end < edges.size() && edges[end].lo == edges[begin].lo &&
           edges[end].hi == edges[begin].hi)
      ++end;
    if (end - begin == 2U) {
      const CompactOwnedEdge &first = edges[begin];
      const CompactOwnedEdge &second = edges[begin + 1U];
      if (first.from == second.to && first.to == second.from &&
          first.opposite != second.opposite) {
        const Vec3 a = mesh_vertex(mesh, first.from);
        const Vec3 b = mesh_vertex(mesh, first.to);
        const Vec3 c = mesh_vertex(mesh, first.opposite);
        const Vec3 d = mesh_vertex(mesh, second.opposite);
        candidates.push_back(
            {first, second,
             std::min(normalized_triangle_quality(a, b, c),
                      normalized_triangle_quality(b, a, d))});
      }
    }
    begin = end;
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &a, const Candidate &b) {
              const double aq = stable_repair_metric(a.old_quality);
              const double bq = stable_repair_metric(b.old_quality);
              if (aq != bq)
                return aq < bq;
              return std::tie(a.first.lo, a.first.hi) <
                     std::tie(b.first.lo, b.first.hi);
            });
  scratch_bytes = std::max(
      scratch_bytes,
      edges.capacity() * sizeof(CompactOwnedEdge) +
          candidates.capacity() * sizeof(Candidate) +
          mesh.triangles.size() * sizeof(std::uint8_t));
  std::vector<std::uint8_t> touched(mesh.triangles.size(), 0U);
  std::size_t flipped = 0U;
  for (const Candidate &candidate : candidates) {
    const CompactOwnedEdge &first = candidate.first;
    const CompactOwnedEdge &second = candidate.second;
    if (touched[first.triangle] != 0U || touched[second.triangle] != 0U ||
        compact_edge_exists(edges, first.opposite, second.opposite))
      continue;
    const Vec3 a = mesh_vertex(mesh, first.from);
    const Vec3 b = mesh_vertex(mesh, first.to);
    const Vec3 c = mesh_vertex(mesh, first.opposite);
    const Vec3 d = mesh_vertex(mesh, second.opposite);
    const double new_quality =
        std::min(normalized_triangle_quality(c, d, b),
                 normalized_triangle_quality(d, c, a));
    if (!(stable_repair_metric(new_quality) >
          stable_repair_metric(candidate.old_quality * 1.05 + 1.0e-8)))
      continue;
    const Vec3 old_normal = cross(b - a, c - a) + cross(a - b, d - b);
    const Vec3 new_normal = cross(d - c, b - c) + cross(c - d, a - d);
    if (dot(old_normal, new_normal) <= 0.0)
      continue;
    Triangle &triangle_first = mesh.triangles[first.triangle];
    Triangle &triangle_second = mesh.triangles[second.triangle];
    undo.push_back({first.triangle, second.triangle, triangle_first,
                    triangle_second, mesh.panels[first.triangle],
                    mesh.panels[second.triangle]});
    triangle_first.v0 = first.opposite;
    triangle_first.v1 = second.opposite;
    triangle_first.v2 = first.to;
    triangle_second.v0 = second.opposite;
    triangle_second.v1 = first.opposite;
    triangle_second.v2 = first.from;
    update_triangle_geometry(mesh, first.triangle);
    update_triangle_geometry(mesh, second.triangle);
    touched[first.triangle] = 1U;
    touched[second.triangle] = 1U;
    ++flipped;
  }
  scratch_bytes = std::max(
      scratch_bytes,
      edges.capacity() * sizeof(CompactOwnedEdge) +
          candidates.capacity() * sizeof(Candidate) +
          mesh.triangles.size() * sizeof(std::uint8_t) +
          undo.capacity() * sizeof(CompactEdgeFlipUndo));
  return flipped;
}

struct LocalPatchRepairMetrics {
  double projection_ms{0.0};
  double repair_ms{0.0};
  double orientation_certificate_ms{0.0};
  double self_intersection_audit_ms{0.0};
  double sampled_analytic_audit_ms{0.0};
  double initial_collapse_ms{0.0};
  double stable_sliver_ms{0.0};
  double post_projection_collapse_ms{0.0};
  double local_sliver_ms{0.0};
  double full_projection_ms{0.0};
  double seam_weld_ms{0.0};
  double refused_sliver_repair_ms{0.0};
  double certificate_cleanup_ms{0.0};
  double orientation_repair_ms{0.0};
  double wide_repair_ms{0.0};
  std::size_t projection_index_bytes{0};
  std::size_t scratch_bytes{0};
  std::size_t projected_vertices{0};
  std::size_t projection_queries{0};
  std::size_t projection_hits{0};
  std::size_t rejected_projection_vertices{0};
  std::size_t rejected_projection_rounds{0};
  std::size_t coupled_projection_components_attempted{0};
  std::size_t coupled_projection_components_accepted{0};
  std::size_t coupled_projection_vertices_rescued{0};
  std::size_t coupled_projection_max_component_size{0};
  std::size_t coupled_projection_unresolved_vertices{0};
  std::size_t collapsed_edges{0};
  std::size_t flipped_edges{0};
  std::size_t poor_triangles_before{0};
  std::size_t poor_triangles_after{0};
  double triangle_quality_min_before{0.0};
  double triangle_quality_min_after{0.0};
  double max_edge_ratio_before{0.0};
  double max_edge_ratio_after{0.0};
  std::string analytic_audit_status{"not_run"};
  std::size_t analytic_audit_samples{0};
  std::size_t analytic_audit_hits{0};
  std::size_t analytic_audit_misses{0};
  double analytic_projection_distance_rms{0.0};
  double analytic_projection_distance_max{0.0};
  double analytic_patch_residual_rms{0.0};
  double analytic_patch_residual_max{0.0};
  double analytic_normal_angle_rms_deg{0.0};
  double analytic_normal_angle_max_deg{0.0};
  double sampled_hausdorff_proxy{0.0};
  std::size_t analytic_vertex_projection_misses{0};
  double analytic_vertex_residual_max{0.0};
  // Vertices the last transactional projection could not move, with the
  // distance each still needs.  Both are small by construction (1-3 on a
  // protein surface of 10^4-10^5 vertices) and the targeted sliver repair
  // reads them to find the faces that blocked the move.
  std::vector<std::uint32_t> refused_vertex_indices;
  std::vector<double> refused_vertex_distances;
  std::vector<std::uint8_t> refused_vertex_shape_flags;
  double analytic_vertex_refused_distance_max{0.0};
  // The subset of those refusals the commit chose, to keep a one ring from
  // being sheared by a move it would otherwise have accepted.
  std::size_t analytic_vertex_refused_for_shape{0};
  std::size_t refused_sliver_collapses{0};
  std::size_t analytic_sheet_bridge_faces{0};
  std::size_t projection_orientation_reversals{0};
  std::size_t orientation_repair_rounds{0};
  std::size_t orientation_repair_faces{0};
  std::size_t wide_repair_attempts{0};
  std::size_t topology_certificate_accepts{0};
  std::size_t topology_fallback_audits{0};
  std::size_t welded_vertices{0};
  std::size_t analytic_orientation_certified_faces{0};
  std::size_t analytic_orientation_centroid_queries{0};
  std::size_t analytic_orientation_uncertified_faces{0};
  std::size_t analytic_inverted_faces{0};
  std::size_t analytic_self_intersections{0};
  std::size_t analytic_self_intersection_first{
      std::numeric_limits<std::size_t>::max()};
  std::size_t analytic_self_intersection_second{
      std::numeric_limits<std::size_t>::max()};
  double analytic_orientation_min_cosine{1.0};
  std::string analytic_orientation_status{"not_run"};
  std::string analytic_normal_quality_status{"not_run"};
  std::size_t analytic_orientation_smooth_faces{0};
  std::size_t analytic_orientation_seam_faces{0};
  std::size_t analytic_normal_cosine_below_025_faces{0};
  std::size_t analytic_normal_cosine_below_050_faces{0};
  double analytic_normal_area_weighted_rms_deg{0.0};
  double analytic_normal_p99_deg{0.0};
  double analytic_smooth_normal_p99_deg{0.0};
  double analytic_seam_normal_p99_deg{0.0};
  double analytic_normal_area_below_cosine_025_fraction{0.0};
  double analytic_normal_area_below_cosine_050_fraction{0.0};
  bool analytic_vertex_certificate{false};
#ifdef TITAN_TABI_SES_TEST_HOOKS
  // State of the last full transactional projection on the current compact
  // vertex indexing: 0=no certified proposal, 2=committed, 4=rejected.
  // This exists only in diagnostic/test binaries.
  std::vector<std::uint8_t> debug_last_projection_state;
#endif
};

struct InvertedEdgeCandidates {
  std::vector<EdgeKey> edges;
  std::size_t faces{0U};
  double maximum_length{0.0};
};

[[nodiscard]] bool certified_finite_projection_hit(
    const analytic_ses::LocalProjectionHit &hit) noexcept {
  const Vec3 point{hit.point.x, hit.point.y, hit.point.z};
  const Vec3 normal{hit.normal.x, hit.normal.y, hit.normal.z};
  const double normal_length = norm(normal);
  return hit.found && hit.bounded_closest_certified &&
         hit.normal_certified && std::isfinite(hit.distance) &&
         hit.distance >= 0.0 &&
         finite_vec3(point) && finite_vec3(normal) &&
         std::isfinite(normal_length) && normal_length > 1.0e-12 &&
         hit.atom_index != std::numeric_limits<std::size_t>::max();
}

[[nodiscard]] InvertedEdgeCandidates
collect_analytic_inverted_short_edges(
    const SurfaceMesh &mesh, const analytic_ses::PatchComplex &patches,
    const analytic_ses::LocalPatchProjectionIndex &projection,
    double grid_spacing, double maximum_edge_length,
    bool vertex_normals_certified, TitanExecutor *executor = nullptr,
    bool include_uncertified_faces = false) {
  if (mesh.triangles.empty())
    return {};
  const std::size_t workers =
      executor == nullptr
          ? 1U
          : std::min(executor->capacity(), mesh.triangles.size());
  std::vector<InvertedEdgeCandidates> per_worker(workers);
  const auto collect_range = [&](std::size_t worker, std::size_t begin,
                                 std::size_t end) {
    InvertedEdgeCandidates &local = per_worker[worker];
    for (std::size_t triangle_index = begin; triangle_index < end;
         ++triangle_index) {
      const Triangle &triangle = mesh.triangles[triangle_index];
      const Vec3 a = mesh_vertex(mesh, triangle.v0);
      const Vec3 b = mesh_vertex(mesh, triangle.v1);
      const Vec3 c = mesh_vertex(mesh, triangle.v2);
      const Vec3 face = normalized(cross(b - a, c - a));
      const Vertex &av = mesh.vertices[triangle.v0];
      const Vertex &bv = mesh.vertices[triangle.v1];
      const Vertex &cv = mesh.vertices[triangle.v2];
      const Vec3 an = normalized({av.nx, av.ny, av.nz});
      const Vec3 bn = normalized({bv.nx, bv.ny, bv.nz});
      const Vec3 cn = normalized({cv.nx, cv.ny, cv.nz});
      const double coherence =
          std::min({dot(an, bn), dot(bn, cn), dot(cn, an)});
      const Vec3 summed = an + bn + cn;
      const double summed_length = norm(summed);
      bool reference_found =
          vertex_normals_certified && coherence >= 0.50 &&
          summed_length > 1.0e-12;
      Vec3 analytic =
          reference_found ? (1.0 / summed_length) * summed : Vec3{};
      double cosine = reference_found ? dot(face, analytic) : -1.0;
      if (!reference_found || cosine <= 1.0e-8) {
        const Vec3 centroid = (1.0 / 3.0) * (a + b + c);
        const analytic_ses::LocalProjectionHit hit =
            analytic_ses::project_local_patch_closest(
                patches, projection, triangle.atom_index,
                {centroid.x, centroid.y, centroid.z}, 2.0 * grid_spacing);
        if (certified_finite_projection_hit(hit) &&
            hit.distance <= 0.75 * grid_spacing) {
          const Vec3 hit_normal{hit.normal.x, hit.normal.y, hit.normal.z};
          const double hit_normal_length = norm(hit_normal);
          if (hit_normal_length > 1.0e-12) {
            analytic = (1.0 / hit_normal_length) * hit_normal;
            reference_found = true;
            cosine = dot(face, analytic);
          }
        }
      }
      if (!reference_found) {
        // An unresolved analytic normal is not evidence of inversion.  Keep
        // the normal fast path conservative and let only the fully audited
        // wide fallback consider these faces as repair seeds.
        if (!include_uncertified_faces)
          continue;
      } else if (cosine > 1.0e-8) {
        continue;
      }
      ++local.faces;
      const std::array<std::pair<EdgeKey, double>, 3> candidates = {
          std::pair{edge_key(triangle.v0, triangle.v1), norm(b - a)},
          std::pair{edge_key(triangle.v1, triangle.v2), norm(c - b)},
          std::pair{edge_key(triangle.v2, triangle.v0), norm(a - c)}};
      for (const auto &[edge, length] : candidates) {
        if (!(length <= maximum_edge_length))
          continue;
        local.edges.push_back(edge);
        local.maximum_length = std::max(local.maximum_length, length);
      }
    }
  };
  if (executor != nullptr && workers > 1U) {
    executor->parallel_for(mesh.triangles.size(), workers, 512U,
                           collect_range);
  } else {
    collect_range(0U, 0U, mesh.triangles.size());
  }
  InvertedEdgeCandidates result;
  std::size_t edge_count = 0U;
  for (const InvertedEdgeCandidates &local : per_worker)
    edge_count += local.edges.size();
  result.edges.reserve(edge_count);
  for (InvertedEdgeCandidates &local : per_worker) {
    result.faces += local.faces;
    result.maximum_length =
        std::max(result.maximum_length, local.maximum_length);
    result.edges.insert(result.edges.end(), local.edges.begin(),
                        local.edges.end());
  }
  std::sort(result.edges.begin(), result.edges.end(),
            [](const EdgeKey &left, const EdgeKey &right) {
              return std::tie(left.lo, left.hi) <
                     std::tie(right.lo, right.hi);
            });
  result.edges.erase(
      std::unique(result.edges.begin(), result.edges.end()),
      result.edges.end());
  return result;
}

[[nodiscard]] analytic_ses::LocalProjectionHit nearest_local_patch(
    const analytic_ses::PatchComplex &patches,
    const analytic_ses::LocalPatchProjectionIndex &projection,
    std::size_t atom_index, const Vec3 &point,
    double max_distance) noexcept {
  return analytic_ses::project_local_patch_closest(
      patches, projection, atom_index, {point.x, point.y, point.z},
      max_distance);
}

struct AnalyticFaceOrientation {
  bool certified{false};
  bool used_centroid_query{false};
  double cosine{-1.0};
};

[[nodiscard]] AnalyticFaceOrientation evaluate_analytic_face_orientation(
    const SurfaceMesh &mesh, const Triangle &triangle,
    const analytic_ses::PatchComplex &patches,
    const analytic_ses::LocalPatchProjectionIndex &projection,
    double grid_spacing, bool vertex_normals_certified) {
  AnalyticFaceOrientation result;
  const Vec3 a = mesh_vertex(mesh, triangle.v0);
  const Vec3 b = mesh_vertex(mesh, triangle.v1);
  const Vec3 c = mesh_vertex(mesh, triangle.v2);
  const Vec3 face_vector = cross(b - a, c - a);
  const double face_length = norm(face_vector);
  if (!(face_length > 1.0e-16) || !std::isfinite(face_length))
    return result;
  const Vec3 face = (1.0 / face_length) * face_vector;
  const Vertex &av = mesh.vertices[triangle.v0];
  const Vertex &bv = mesh.vertices[triangle.v1];
  const Vertex &cv = mesh.vertices[triangle.v2];
  const Vec3 an = normalized({av.nx, av.ny, av.nz});
  const Vec3 bn = normalized({bv.nx, bv.ny, bv.nz});
  const Vec3 cn = normalized({cv.nx, cv.ny, cv.nz});
  const double coherence =
      std::min({dot(an, bn), dot(bn, cn), dot(cn, an)});
  const Vec3 summed = an + bn + cn;
  const double summed_length = norm(summed);
  if (vertex_normals_certified && coherence >= 0.50 &&
      summed_length > 1.0e-12) {
    result.certified = true;
    result.cosine = dot(face, (1.0 / summed_length) * summed);
  }
  if (!result.certified || result.cosine <= 1.0e-8) {
    result.used_centroid_query = true;
    const Vec3 centroid = (1.0 / 3.0) * (a + b + c);
    const analytic_ses::LocalProjectionHit hit = nearest_local_patch(
        patches, projection, triangle.atom_index, centroid,
        2.0 * grid_spacing);
    if (certified_finite_projection_hit(hit) &&
        hit.distance <= 0.75 * grid_spacing) {
      const Vec3 hit_normal{hit.normal.x, hit.normal.y, hit.normal.z};
      const double hit_normal_length = norm(hit_normal);
      if (hit_normal_length > 1.0e-12) {
        result.certified = true;
        result.cosine =
            dot(face, (1.0 / hit_normal_length) * hit_normal);
      }
    }
  }
  return result;
}

[[nodiscard]] std::size_t flip_edges_for_analytic_orientation(
    SurfaceMesh &mesh, const analytic_ses::PatchComplex &patches,
    const analytic_ses::LocalPatchProjectionIndex &projection,
    double grid_spacing, bool vertex_normals_certified,
    std::size_t &scratch_bytes) {
  std::vector<CompactOwnedEdge> edges = build_compact_owned_edges(mesh);
  if (edges.empty())
    return 0U;
  struct Candidate {
    CompactOwnedEdge first;
    CompactOwnedEdge second;
    double worst_cosine{1.0};
    double length{0.0};
  };
  std::vector<Candidate> candidates;
  for (std::size_t begin = 0U; begin < edges.size();) {
    std::size_t end = begin + 1U;
    while (end < edges.size() && edges[end].lo == edges[begin].lo &&
           edges[end].hi == edges[begin].hi)
      ++end;
    if (end - begin == 2U) {
      const CompactOwnedEdge &first = edges[begin];
      const CompactOwnedEdge &second = edges[begin + 1U];
      if (first.from == second.to && first.to == second.from &&
          first.opposite != second.opposite) {
        const AnalyticFaceOrientation first_orientation =
            evaluate_analytic_face_orientation(
                mesh, mesh.triangles[first.triangle], patches, projection,
                grid_spacing, vertex_normals_certified);
        const AnalyticFaceOrientation second_orientation =
            evaluate_analytic_face_orientation(
                mesh, mesh.triangles[second.triangle], patches, projection,
                grid_spacing, vertex_normals_certified);
        const bool inverted =
            (first_orientation.certified &&
             first_orientation.cosine <= 1.0e-8) ||
            (second_orientation.certified &&
             second_orientation.cosine <= 1.0e-8);
        if (inverted) {
          candidates.push_back(
              {first, second,
               std::min(first_orientation.cosine,
                        second_orientation.cosine),
               norm(mesh_vertex(mesh, first.to) -
                    mesh_vertex(mesh, first.from))});
        }
      }
    }
    begin = end;
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &left, const Candidate &right) {
              const double left_cosine =
                  stable_repair_metric(left.worst_cosine);
              const double right_cosine =
                  stable_repair_metric(right.worst_cosine);
              if (left_cosine != right_cosine)
                return left_cosine < right_cosine;
              const double left_length = stable_repair_metric(left.length);
              const double right_length = stable_repair_metric(right.length);
              if (left_length != right_length)
                return left_length < right_length;
              return std::tie(left.first.lo, left.first.hi) <
                     std::tie(right.first.lo, right.first.hi);
            });
  std::vector<std::uint8_t> touched(mesh.triangles.size(), 0U);
  std::vector<CompactEdgeFlipUndo> undo;
  for (const Candidate &candidate : candidates) {
    const CompactOwnedEdge &first = candidate.first;
    const CompactOwnedEdge &second = candidate.second;
    if (touched[first.triangle] != 0U || touched[second.triangle] != 0U ||
        compact_edge_exists(edges, first.opposite, second.opposite))
      continue;
    Triangle new_first = mesh.triangles[first.triangle];
    Triangle new_second = mesh.triangles[second.triangle];
    new_first.v0 = first.opposite;
    new_first.v1 = second.opposite;
    new_first.v2 = first.to;
    new_second.v0 = second.opposite;
    new_second.v1 = first.opposite;
    new_second.v2 = first.from;
    const AnalyticFaceOrientation first_orientation =
        evaluate_analytic_face_orientation(mesh, new_first, patches,
                                           projection, grid_spacing,
                                           vertex_normals_certified);
    const AnalyticFaceOrientation second_orientation =
        evaluate_analytic_face_orientation(mesh, new_second, patches,
                                           projection, grid_spacing,
                                           vertex_normals_certified);
    if (!first_orientation.certified || !second_orientation.certified ||
        !(first_orientation.cosine > 1.0e-8) ||
        !(second_orientation.cosine > 1.0e-8))
      continue;
    undo.push_back({first.triangle, second.triangle,
                    mesh.triangles[first.triangle],
                    mesh.triangles[second.triangle],
                    mesh.panels[first.triangle], mesh.panels[second.triangle]});
    mesh.triangles[first.triangle] = new_first;
    mesh.triangles[second.triangle] = new_second;
    update_analytic_triangle_geometry_and_owner(mesh, first.triangle);
    update_analytic_triangle_geometry_and_owner(mesh, second.triangle);
    touched[first.triangle] = 1U;
    touched[second.triangle] = 1U;
  }
  if (!undo.empty()) {
    const TraceSpaces topology = build_closed_surface_trace_spaces(mesh);
    if (!topology.quality.valid) {
      for (auto entry = undo.rbegin(); entry != undo.rend(); ++entry) {
        mesh.triangles[entry->first_index] = entry->first_triangle;
        mesh.triangles[entry->second_index] = entry->second_triangle;
        mesh.panels[entry->first_index] = entry->first_panel;
        mesh.panels[entry->second_index] = entry->second_panel;
      }
      undo.clear();
    }
  }
  scratch_bytes = std::max(
      scratch_bytes,
      edges.capacity() * sizeof(CompactOwnedEdge) +
          candidates.capacity() * sizeof(Candidate) +
          touched.capacity() * sizeof(std::uint8_t) +
          undo.capacity() * sizeof(CompactEdgeFlipUndo));
  return undo.size();
}

[[nodiscard]] double primitive_patch_residual(
    const analytic_ses::PatchComplex &patches, analytic_ses::PatchKind kind,
    std::size_t patch_index, const Vec3 &point) noexcept {
  const auto distance = [&](const analytic_ses::Vec3 &center) {
    const double dx = point.x - center.x;
    const double dy = point.y - center.y;
    const double dz = point.z - center.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  };
  switch (kind) {
  case analytic_ses::PatchKind::contact:
    if (patch_index < patches.contacts.size()) {
      const auto &patch = patches.contacts[patch_index];
      return std::abs(distance(patch.center) - patch.atom_radius);
    }
    break;
  case analytic_ses::PatchKind::reentrant:
    if (patch_index < patches.reentrant.size()) {
      const auto &patch = patches.reentrant[patch_index];
      return std::abs(distance(patch.probe_center) - patches.probe_radius);
    }
    break;
  case analytic_ses::PatchKind::toroidal:
    if (patch_index < patches.tori.size()) {
      const auto &patch = patches.tori[patch_index];
      const double qx = point.x - patch.circle_center.x;
      const double qy = point.y - patch.circle_center.y;
      const double qz = point.z - patch.circle_center.z;
      const double axial =
          qx * patch.axis.x + qy * patch.axis.y + qz * patch.axis.z;
      const double radial = std::sqrt(std::max(
          0.0, qx * qx + qy * qy + qz * qz - axial * axial));
      const double tube = std::hypot(radial - patch.circle_radius, axial);
      return std::abs(tube - patches.probe_radius);
    }
    break;
  }
  return std::numeric_limits<double>::infinity();
}

void audit_surface_against_analytic_patches(
    const SurfaceMesh &mesh, const analytic_ses::PatchComplex &patches,
    const analytic_ses::LocalPatchProjectionIndex &projection,
    double grid_spacing, LocalPatchRepairMetrics &metrics) {
  // A stratified 1024-point audit is stable on the validation meshes while
  // keeping this diagnostic below the surface-build hot path.
  constexpr std::size_t kMaxSamplesPerKind = 512U;
  double projection_squared_sum = 0.0;
  double residual_squared_sum = 0.0;
  double angle_squared_sum = 0.0;
  const auto audit_point = [&](const Vec3 &point, const Vec3 &mesh_normal,
                               std::size_t atom_index) {
    ++metrics.analytic_audit_samples;
    const analytic_ses::LocalProjectionHit hit = nearest_local_patch(
        patches, projection, atom_index, point, 2.0 * grid_spacing);
    if (!certified_finite_projection_hit(hit)) {
      ++metrics.analytic_audit_misses;
      return;
    }
    ++metrics.analytic_audit_hits;
    const double projection_distance = hit.distance;
    const double residual =
        primitive_patch_residual(patches, hit.kind, hit.patch_index, point);
    const Vec3 reference_normal{hit.normal.x, hit.normal.y, hit.normal.z};
    const double cosine = std::clamp(
        dot(normalized(mesh_normal), normalized(reference_normal)), -1.0, 1.0);
    const double angle = std::acos(cosine) * 57.2957795130823208768;
    projection_squared_sum += projection_distance * projection_distance;
    residual_squared_sum += residual * residual;
    angle_squared_sum += angle * angle;
    metrics.analytic_projection_distance_max =
        std::max(metrics.analytic_projection_distance_max,
                 projection_distance);
    metrics.analytic_patch_residual_max =
        std::max(metrics.analytic_patch_residual_max, residual);
    metrics.analytic_normal_angle_max_deg =
        std::max(metrics.analytic_normal_angle_max_deg, angle);
    metrics.sampled_hausdorff_proxy =
        std::max(metrics.sampled_hausdorff_proxy, projection_distance);
  };
  const auto sample_count = [=](std::size_t size) {
    return std::min(size, kMaxSamplesPerKind);
  };
  const auto sample_index = [](std::size_t sample, std::size_t count,
                               std::size_t size) {
    return count <= 1U ? 0U : sample * (size - 1U) / (count - 1U);
  };
  const std::size_t vertex_samples = sample_count(mesh.vertices.size());
  for (std::size_t sample = 0; sample < vertex_samples; ++sample) {
    const Vertex &vertex =
        mesh.vertices[sample_index(sample, vertex_samples, mesh.vertices.size())];
    audit_point({vertex.x, vertex.y, vertex.z},
                {vertex.nx, vertex.ny, vertex.nz}, vertex.atom_index);
  }
  const std::size_t triangle_samples = sample_count(mesh.triangles.size());
  for (std::size_t sample = 0; sample < triangle_samples; ++sample) {
    const Triangle &triangle = mesh.triangles[
        sample_index(sample, triangle_samples, mesh.triangles.size())];
    audit_point({triangle.cx, triangle.cy, triangle.cz},
                {triangle.nx, triangle.ny, triangle.nz}, triangle.atom_index);
  }
  if (metrics.analytic_audit_hits == 0U) {
    metrics.analytic_audit_status = "no_hits";
    return;
  }
  const double inverse_hits =
      1.0 / static_cast<double>(metrics.analytic_audit_hits);
  metrics.analytic_projection_distance_rms =
      std::sqrt(projection_squared_sum * inverse_hits);
  metrics.analytic_patch_residual_rms =
      std::sqrt(residual_squared_sum * inverse_hits);
  metrics.analytic_normal_angle_rms_deg =
      std::sqrt(angle_squared_sum * inverse_hits);
  const double miss_fraction =
      static_cast<double>(metrics.analytic_audit_misses) /
      static_cast<double>(metrics.analytic_audit_samples);
  metrics.analytic_audit_status =
      miss_fraction == 0.0 &&
              metrics.analytic_projection_distance_rms <= 0.30 * grid_spacing &&
              metrics.analytic_patch_residual_rms <= 0.30 * grid_spacing &&
              metrics.analytic_normal_angle_rms_deg <= 30.0
          ? "pass"
          : "warning";
}

struct ProjectionCommitResult {
  std::size_t accepted_vertices{0U};
  std::size_t rejected_vertices{0U};
  std::size_t orientation_rejections{0U};
  std::size_t components_attempted{0U};
  std::size_t components_accepted{0U};
  std::size_t vertices_rescued{0U};
  std::size_t max_component_size{0U};
  std::size_t retained_bytes{0U};
  // Moves refused for the shape of the one ring rather than its orientation.
  // Counted apart because the two failures want different repairs and only
  // this one is a choice the commit made.
  std::size_t shape_rejections{0U};
  // Vertices left in the rejected state.  Kept in production, not only in
  // test builds: the repair pipeline needs to know which one-rings blocked
  // the move, and the list is empty or tiny on every surface that gets here.
  std::vector<std::uint32_t> refused_vertices;
  // Parallel to refused_vertices: 1 where the commit declined the move to
  // keep the one ring's shape, 0 where a face would have reversed.  The
  // targeted sliver repair unblocks the second kind and must leave the
  // first alone, or it collapses an edge to enable the very move the gate
  // declined and hands the shear back.
  std::vector<std::uint8_t> refused_for_shape;
  bool all_faces_certified{false};
#ifdef TITAN_TABI_SES_TEST_HOOKS
  std::vector<std::uint8_t> final_state;
#endif
};

// Commit certified closest-point proposals without exposing the mesh to a
// transient fold.  The inexpensive first pass keeps the established
// deterministic vertex order.  A rejected move can nevertheless be part of a
// valid coupled move (for example, both endpoints of a narrow face must cross
// together), so rejected vertices connected through incident faces get one
// atomic component trial.  Every accepted path is checked against the same
// fixed pre-projection face normals; there is no tolerance relaxation or
// whole-mesh Jacobi fallback.
[[nodiscard]] ProjectionCommitResult commit_projection_proposals(
    SurfaceMesh &mesh, const CompactVertexIncidence &incidence,
    std::span<const Vertex> proposals,
    std::span<const std::uint8_t> proposal_valid,
    std::span<const Vec3> reference_normals,
    std::vector<std::uint8_t> &affected_faces,
    double shape_refusal_distance_limit = 0.0) {
  ProjectionCommitResult result;
  if (proposals.size() != mesh.vertices.size() ||
      proposal_valid.size() != mesh.vertices.size() ||
      reference_normals.size() != mesh.triangles.size() ||
      affected_faces.size() != mesh.triangles.size() ||
      incidence.offsets.size() != mesh.vertices.size() + 1U)
    return result;

  // 0 = no certified proposal, 1 = pending, 2 = committed,
  // 3 = current coupled component, 4 = rejected because a face would
  // have reversed, 5 = declined to keep the one ring's shape.
  std::vector<std::uint8_t> state(mesh.vertices.size(), 0U);
  for (std::size_t vertex = 0U; vertex < mesh.vertices.size(); ++vertex) {
    if (proposal_valid[vertex] != 0U)
      state[vertex] = 1U;
  }

  const auto validate_faces = [&](std::span<const std::size_t> faces,
                                  std::span<const std::uint8_t> selected,
                                  bool &orientation_failed) {
    orientation_failed = false;
    for (const std::size_t triangle_index : faces) {
      if (triangle_index >= mesh.triangles.size())
        return false;
      const Triangle &triangle = mesh.triangles[triangle_index];
      if (triangle.v0 >= mesh.vertices.size() ||
          triangle.v1 >= mesh.vertices.size() ||
          triangle.v2 >= mesh.vertices.size())
        return false;
      const auto point = [&](std::size_t vertex) {
        const Vertex &value =
            selected[vertex] != 0U ? proposals[vertex]
                                  : mesh.vertices[vertex];
        return Vec3{value.x, value.y, value.z};
      };
      const FaceMoveValidation validation =
          validate_face_move(point(triangle.v0), point(triangle.v1),
                             point(triangle.v2),
                             reference_normals[triangle_index]);
      if (!validation.geometry_valid)
        return false;
      if (!validation.orientation_preserved) {
        orientation_failed = true;
        return false;
      }
    }
    return true;
  };

  // A move survives validate_faces whenever every incident face keeps its
  // orientation, however much the move distorts it.  At a seam two
  // neighbouring vertices take their closest point on different patches -- a
  // contact sphere and a reentrant sphere -- and shear the triangle between
  // them; that is where this mesh loses its worst aspect ratio, and nothing
  // downstream recovers it.  Refusing such a move keeps the shape and leaves
  // the vertex off the surface by the distance it did not travel, so it is
  // offered only inside the tolerance the vertex certificate already grants
  // refusals.  A vertex that must travel further is committed whatever it
  // does to its one ring: position is the thing the certificate is about.
  const auto move_degrades_shape = [&](std::size_t vertex,
                                       std::span<const std::size_t> faces) {
    if (!(shape_refusal_distance_limit > 0.0))
      return false;
    const Vertex &proposal = proposals[vertex];
    const Vec3 target{proposal.x, proposal.y, proposal.z};
    if (!(norm(target - mesh_vertex(mesh, vertex)) <=
          shape_refusal_distance_limit))
      return false;
    for (const std::size_t triangle_index : faces) {
      const Triangle &triangle = mesh.triangles[triangle_index];
      const auto moved = [&](std::size_t index) {
        return index == vertex ? target : mesh_vertex(mesh, index);
      };
      const double after = face_edge_ratio(
          moved(triangle.v0), moved(triangle.v1), moved(triangle.v2));
      if (!(after > kProjectionShapeRatioBound))
        continue;
      const double before = face_edge_ratio(mesh_vertex(mesh, triangle.v0),
                                            mesh_vertex(mesh, triangle.v1),
                                            mesh_vertex(mesh, triangle.v2));
      if (after > before)
        return true;
    }
    return false;
  };

  std::vector<std::uint8_t> selected(mesh.vertices.size(), 0U);
  std::vector<std::size_t> incident_faces;
  for (std::size_t vertex = 0U; vertex < mesh.vertices.size(); ++vertex) {
    if (state[vertex] != 1U)
      continue;
    selected[vertex] = 1U;
    incident_faces.clear();
    incident_faces.reserve(incidence.offsets[vertex + 1U] -
                           incidence.offsets[vertex]);
    for (std::size_t position = incidence.offsets[vertex];
         position < incidence.offsets[vertex + 1U]; ++position)
      incident_faces.push_back(incidence.triangles[position]);
    bool orientation_failed = false;
    if (move_degrades_shape(vertex, incident_faces)) {
      // Final, not deferred to the coupled pass: the coupled pass exists to
      // rescue moves that only a neighbour's move makes safe, and this one
      // is safe already.  It was declined.
      state[vertex] = 5U;
      ++result.shape_rejections;
      selected[vertex] = 0U;
      continue;
    }
    if (validate_faces(incident_faces, selected, orientation_failed)) {
      mesh.vertices[vertex] = proposals[vertex];
      state[vertex] = 2U;
      ++result.accepted_vertices;
      for (const std::size_t triangle_index : incident_faces)
        affected_faces[triangle_index] = 1U;
    }
    selected[vertex] = 0U;
  }

  // A face can connect at most three pending vertices.  Components are
  // discovered from the lowest vertex index, and each triangle/vertex is
  // visited in its stored deterministic order.
  std::vector<std::uint8_t> face_selected(mesh.triangles.size(), 0U);
  std::vector<std::size_t> stack;
  std::vector<std::size_t> component;
  std::vector<std::size_t> component_faces;
  for (std::size_t root = 0U; root < mesh.vertices.size(); ++root) {
    if (state[root] != 1U)
      continue;
    stack.clear();
    component.clear();
    component_faces.clear();
    state[root] = 3U;
    stack.push_back(root);
    while (!stack.empty()) {
      const std::size_t vertex = stack.back();
      stack.pop_back();
      component.push_back(vertex);
      for (std::size_t position = incidence.offsets[vertex];
           position < incidence.offsets[vertex + 1U]; ++position) {
        const std::size_t triangle_index = incidence.triangles[position];
        if (triangle_index >= mesh.triangles.size())
          continue;
        if (face_selected[triangle_index] == 0U) {
          face_selected[triangle_index] = 1U;
          component_faces.push_back(triangle_index);
        }
        const Triangle &triangle = mesh.triangles[triangle_index];
        for (const std::size_t neighbor :
             {triangle.v0, triangle.v1, triangle.v2}) {
          if (neighbor < state.size() && state[neighbor] == 1U) {
            state[neighbor] = 3U;
            stack.push_back(neighbor);
          }
        }
      }
    }

    ++result.components_attempted;
    result.max_component_size =
        std::max(result.max_component_size, component.size());
    for (const std::size_t vertex : component)
      selected[vertex] = 1U;
    bool orientation_failed = false;
    const bool safe =
        validate_faces(component_faces, selected, orientation_failed);
    if (safe) {
      ++result.components_accepted;
      result.vertices_rescued += component.size();
      for (const std::size_t vertex : component) {
        mesh.vertices[vertex] = proposals[vertex];
        state[vertex] = 2U;
        ++result.accepted_vertices;
      }
      for (const std::size_t triangle_index : component_faces)
        affected_faces[triangle_index] = 1U;
    } else {
      for (const std::size_t vertex : component)
        state[vertex] = 4U;
      if (orientation_failed)
        result.orientation_rejections += component.size();
    }
    for (const std::size_t vertex : component)
      selected[vertex] = 0U;
    for (const std::size_t triangle_index : component_faces)
      face_selected[triangle_index] = 0U;
  }

  for (std::size_t vertex = 0U; vertex < state.size(); ++vertex) {
    if (state[vertex] != 4U && state[vertex] != 5U)
      continue;
    result.refused_vertices.push_back(static_cast<std::uint32_t>(vertex));
    result.refused_for_shape.push_back(state[vertex] == 5U ? 1U : 0U);
  }
  result.rejected_vertices = result.refused_vertices.size();
  result.all_faces_certified = true;
  for (std::size_t triangle_index = 0U;
       triangle_index < mesh.triangles.size(); ++triangle_index) {
    const Triangle &triangle = mesh.triangles[triangle_index];
    const FaceMoveValidation validation = validate_face_move(
        mesh_vertex(mesh, triangle.v0), mesh_vertex(mesh, triangle.v1),
        mesh_vertex(mesh, triangle.v2), reference_normals[triangle_index]);
    if (!validation.geometry_valid || !validation.orientation_preserved) {
      result.all_faces_certified = false;
      break;
    }
  }
  result.retained_bytes =
      state.capacity() * sizeof(std::uint8_t) +
      selected.capacity() * sizeof(std::uint8_t) +
      face_selected.capacity() * sizeof(std::uint8_t) +
      incident_faces.capacity() * sizeof(std::size_t) +
      stack.capacity() * sizeof(std::size_t) +
      component.capacity() * sizeof(std::size_t) +
      component_faces.capacity() * sizeof(std::size_t) +
      result.refused_vertices.capacity() * sizeof(std::uint32_t) +
      result.refused_for_shape.capacity() * sizeof(std::uint8_t);
#ifdef TITAN_TABI_SES_TEST_HOOKS
  result.final_state = std::move(state);
#endif
  return result;
}

// A vertex is refused by specific faces in its one ring: those whose smallest
// altitude is below the move it still needs, so the move sweeps the vertex
// past the opposite edge and reverses the face.  Every other incident face
// would have survived.  Collapsing the shortest edge of exactly those faces
// removes the obstruction and touches nothing else, which is what separates
// this from lowering the global short-edge threshold until something gives.
[[nodiscard]] std::vector<EdgeKey> collect_refused_vertex_sliver_edges(
    const SurfaceMesh &mesh, const CompactVertexIncidence &incidence,
    std::span<const std::uint32_t> refused,
    std::span<const double> distances,
    std::span<const std::uint8_t> shape_flags, double &maximum_length) {
  std::vector<EdgeKey> edges;
  maximum_length = 0.0;
  for (std::size_t entry = 0U; entry < refused.size(); ++entry) {
    const std::size_t vertex = refused[entry];
    if (vertex + 1U >= incidence.offsets.size())
      continue;
    // A shape refusal is not an obstruction to clear.  The move was safe and
    // was declined, so collapsing an edge to make room for it would spend a
    // sliver to buy back the shear the refusal prevented.
    if (entry < shape_flags.size() && shape_flags[entry] != 0U)
      continue;
    const double move = entry < distances.size()
                            ? distances[entry]
                            : std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(move) || !(move > 0.0))
      continue;
    for (std::size_t position = incidence.offsets[vertex];
         position < incidence.offsets[vertex + 1U]; ++position) {
      const std::size_t face = incidence.triangles[position];
      if (face >= mesh.triangles.size())
        continue;
      const Triangle &triangle = mesh.triangles[face];
      const Vec3 a = mesh_vertex(mesh, triangle.v0);
      const Vec3 b = mesh_vertex(mesh, triangle.v1);
      const Vec3 c = mesh_vertex(mesh, triangle.v2);
      const double ab = norm(b - a);
      const double bc = norm(c - b);
      const double ca = norm(a - c);
      const double longest = std::max({ab, bc, ca});
      if (!(longest > 0.0) || !std::isfinite(longest))
        continue;
      if (!(norm(cross(b - a, c - a)) / longest < move))
        continue;
      EdgeKey shortest{triangle.v0, triangle.v1};
      double shortest_length = ab;
      if (bc < shortest_length) {
        shortest = {triangle.v1, triangle.v2};
        shortest_length = bc;
      }
      if (ca < shortest_length) {
        shortest = {triangle.v2, triangle.v0};
        shortest_length = ca;
      }
      if (shortest.lo > shortest.hi)
        std::swap(shortest.lo, shortest.hi);
      edges.push_back(shortest);
      maximum_length = std::max(maximum_length, shortest_length);
    }
  }
  std::sort(edges.begin(), edges.end(),
            [](const EdgeKey &left, const EdgeKey &right) {
              return std::tie(left.lo, left.hi) <
                     std::tie(right.lo, right.hi);
            });
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  return edges;
}

[[nodiscard]] bool project_all_vertices_to_analytic_patches(
    SurfaceMesh &mesh, const analytic_ses::PatchComplex &patches,
    const analytic_ses::LocalPatchProjectionIndex &projection,
    double grid_spacing, LocalPatchRepairMetrics &metrics,
    TitanExecutor *executor) {
  metrics.analytic_vertex_certificate = false;
  metrics.analytic_vertex_projection_misses = 0U;
  metrics.analytic_vertex_residual_max = 0.0;
#ifdef TITAN_TABI_SES_TEST_HOOKS
  metrics.debug_last_projection_state.clear();
#endif
  // Refused vertices are indices into the current compact vertex set, so a
  // list left over from a coarser mesh must never survive into a failed
  // attempt on a newer one.
  metrics.refused_vertex_indices.clear();
  metrics.refused_vertex_distances.clear();
  metrics.refused_vertex_shape_flags.clear();
  metrics.analytic_vertex_refused_distance_max = 0.0;
  if (mesh.vertices.empty() || mesh.triangles.empty())
    return false;

  const CompactVertexIncidence incidence =
      build_compact_vertex_incidence(mesh);
  if (incidence.offsets.size() != mesh.vertices.size() + 1U) {
    metrics.analytic_vertex_projection_misses = mesh.vertices.size();
    return false;
  }

  // Exact projection queries are independent and may run in parallel.
  // Geometry changes are committed later in deterministic vertex order,
  // because a Jacobi swap can reverse a face even when every individual hit
  // is a certified closest point on the analytic SES.
  std::vector<Vertex> proposals = mesh.vertices;
  std::vector<std::uint8_t> proposal_valid(mesh.vertices.size(), 0U);
  const double projection_cap = 2.0 * grid_spacing;
  const double residual_tolerance =
      1.0e-8 * std::max(1.0, grid_spacing);
  struct ProjectionWorkerMetrics {
    std::size_t hits{0U};
    std::size_t misses{0U};
    double residual_max{0.0};
  };
  const std::size_t workers =
      executor == nullptr
          ? 1U
          : std::min(executor->capacity(), proposals.size());
  std::vector<ProjectionWorkerMetrics> per_worker(workers);
  const auto project_range = [&](std::size_t worker, std::size_t begin,
                                 std::size_t end) {
    ProjectionWorkerMetrics &local = per_worker[worker];
    for (std::size_t index = begin; index < end; ++index) {
      const Vertex &before = mesh.vertices[index];
      const Vec3 point{before.x, before.y, before.z};
      const analytic_ses::LocalProjectionHit hit = nearest_local_patch(
          patches, projection, before.atom_index, point, projection_cap);
      if (!certified_finite_projection_hit(hit) ||
          hit.atom_index >= patches.atoms.size()) {
        ++local.misses;
        continue;
      }
      const Vec3 projected{hit.point.x, hit.point.y, hit.point.z};
      const double residual = primitive_patch_residual(
          patches, hit.kind, hit.patch_index, projected);
      local.residual_max = std::max(local.residual_max, residual);
      if (!std::isfinite(residual) || residual > residual_tolerance) {
        ++local.misses;
        continue;
      }
      Vertex &after = proposals[index];
      after.x = hit.point.x;
      after.y = hit.point.y;
      after.z = hit.point.z;
      after.nx = hit.normal.x;
      after.ny = hit.normal.y;
      after.nz = hit.normal.z;
      after.atom_index = hit.atom_index;
      proposal_valid[index] = 1U;
      ++local.hits;
    }
  };
  if (executor != nullptr && workers > 1U) {
    executor->parallel_for(proposals.size(), workers, 256U, project_range);
  } else {
    project_range(0U, 0U, proposals.size());
  }
  metrics.projection_queries += proposals.size();
  for (const ProjectionWorkerMetrics &local : per_worker) {
    metrics.projection_hits += local.hits;
    metrics.analytic_vertex_projection_misses += local.misses;
    metrics.analytic_vertex_residual_max =
        std::max(metrics.analytic_vertex_residual_max, local.residual_max);
  }

  std::vector<Vec3> reference_normals(mesh.triangles.size());
  for (std::size_t triangle_index = 0U;
       triangle_index < mesh.triangles.size(); ++triangle_index) {
    const Triangle &triangle = mesh.triangles[triangle_index];
    const Vec3 a = mesh_vertex(mesh, triangle.v0);
    const Vec3 b = mesh_vertex(mesh, triangle.v1);
    const Vec3 c = mesh_vertex(mesh, triangle.v2);
    reference_normals[triangle_index] = cross(b - a, c - a);
  }

  std::vector<std::uint8_t> affected(mesh.triangles.size(), 0U);
  // The shape gate is a local rule serving a global number, so it can pay
  // its cost and buy nothing: on a lone sphere it declines three vertices of
  // 1428 and the worst edge ratio comes out at 23.0556 either way, because
  // the face that sets the maximum is made by moves the gate allows.  Paying
  // 0.04 A of position on a surface known in closed form, for no change in
  // the number the gate exists to hold down, is not a trade worth making.
  // So commit both ways and keep the gated mesh only when it is measurably
  // better shaped; ties go to the mesh that sits on the surface.  The
  // proposals are already computed and a commit is linear in the mesh, so
  // this costs one extra pass over what the projection itself cost.
  std::vector<Vertex> before_commit = mesh.vertices;
  ProjectionCommitResult commit = commit_projection_proposals(
      mesh, incidence, proposals, proposal_valid, reference_normals, affected,
      kRefusedVertexToleranceCells * grid_spacing);
  if (commit.shape_rejections != 0U) {
    const double gated_ratio = mesh_max_edge_ratio(mesh);
    std::vector<Vertex> gated_vertices = mesh.vertices;
    mesh.vertices = before_commit;
    std::vector<std::uint8_t> plain_affected(mesh.triangles.size(), 0U);
    ProjectionCommitResult plain = commit_projection_proposals(
        mesh, incidence, proposals, proposal_valid, reference_normals,
        plain_affected);
    if (!(mesh_max_edge_ratio(mesh) > gated_ratio)) {
      for (std::size_t face = 0U; face < affected.size(); ++face)
        affected[face] = static_cast<std::uint8_t>(
            affected[face] | plain_affected[face]);
      commit = std::move(plain);
    } else {
      mesh.vertices = std::move(gated_vertices);
    }
  }
  before_commit.clear();
  before_commit.shrink_to_fit();
#ifdef TITAN_TABI_SES_TEST_HOOKS
  metrics.debug_last_projection_state = commit.final_state;
#endif

  for (std::size_t triangle_index = 0U;
       triangle_index < mesh.triangles.size(); ++triangle_index) {
    if (affected[triangle_index] != 0U)
      update_analytic_triangle_geometry_and_owner(mesh, triangle_index);
  }

  metrics.projected_vertices += commit.accepted_vertices;
  metrics.rejected_projection_vertices += commit.rejected_vertices;
  metrics.projection_orientation_reversals +=
      commit.orientation_rejections;
  metrics.coupled_projection_components_attempted +=
      commit.components_attempted;
  metrics.coupled_projection_components_accepted +=
      commit.components_accepted;
  metrics.coupled_projection_vertices_rescued +=
      commit.vertices_rescued;
  metrics.coupled_projection_max_component_size =
      std::max(metrics.coupled_projection_max_component_size,
               commit.max_component_size);
  metrics.coupled_projection_unresolved_vertices =
      commit.rejected_vertices;
  if (commit.rejected_vertices != 0U)
    ++metrics.rejected_projection_rounds;
  metrics.scratch_bytes =
      std::max(metrics.scratch_bytes,
               incidence.retained_bytes() +
                   proposals.capacity() * sizeof(Vertex) +
                   proposal_valid.capacity() * sizeof(std::uint8_t) +
                   reference_normals.capacity() * sizeof(Vec3) +
                   affected.capacity() * sizeof(std::uint8_t) +
                   commit.retained_bytes);
  // Record which vertices stayed behind and how far each still had to go.
  // The distance is measured against the mesh as it stands now: an accepted
  // vertex already sits on its target, and a refused one was not moved, so
  // this is exactly the outstanding correction.
  metrics.refused_vertex_indices = commit.refused_vertices;
  metrics.refused_vertex_shape_flags = commit.refused_for_shape;
  metrics.refused_vertex_distances.clear();
  metrics.refused_vertex_distances.reserve(commit.refused_vertices.size());
  double refused_distance_max = 0.0;
  for (const std::uint32_t vertex : commit.refused_vertices) {
    const double distance =
        vertex < proposals.size()
            ? norm(Vec3{proposals[vertex].x, proposals[vertex].y,
                        proposals[vertex].z} -
                   mesh_vertex(mesh, vertex))
            : std::numeric_limits<double>::infinity();
    metrics.refused_vertex_distances.push_back(distance);
    refused_distance_max = std::max(refused_distance_max, distance);
  }
  metrics.analytic_vertex_refused_distance_max = refused_distance_max;
  // A refused vertex is not an uncertified one: its target was proved to be
  // the closest point on a bounded analytic patch, and the only reason it
  // stayed put is that one incident face would have reversed.  Leaving it
  // costs the surface that one outstanding move, which is meaningful only
  // against the spacing the rest of the mesh is drawn at.  Certify while the
  // largest outstanding move is a quarter cell or less, and report the count
  // and the distance either way rather than hiding them behind a bit.
  metrics.analytic_vertex_refused_for_shape += commit.shape_rejections;
  metrics.analytic_vertex_certificate =
      metrics.analytic_vertex_projection_misses == 0U &&
      commit.accepted_vertices + commit.rejected_vertices ==
          mesh.vertices.size() &&
      commit.all_faces_certified &&
      refused_distance_max <= kRefusedVertexToleranceCells * grid_spacing;
  return commit.accepted_vertices != 0U;
}

// Name the audit status after what actually failed.  One string used to cover
// three unrelated outcomes, and on a molecular surface the one that fires is
// almost never a projection that missed: every query is answered and
// certified, and what fails is the commit of a handful of those answers.
[[nodiscard]] std::string
analytic_vertex_failure_status(const LocalPatchRepairMetrics &metrics) {
  if (metrics.analytic_vertex_projection_misses != 0U)
    return "vertex_projection_missed";
  if (!metrics.refused_vertex_indices.empty())
    return "vertex_commit_refused";
  return "vertex_commit_uncertified";
}

void certify_all_face_orientations(
    const SurfaceMesh &mesh, const analytic_ses::PatchComplex &patches,
    const analytic_ses::LocalPatchProjectionIndex &projection,
    double grid_spacing, LocalPatchRepairMetrics &metrics,
    bool collect_normal_quality = false) {
  const bool vertex_normals_certified =
      metrics.analytic_vertex_certificate;
  metrics.analytic_orientation_certified_faces = 0U;
  metrics.analytic_orientation_centroid_queries = 0U;
  metrics.analytic_orientation_uncertified_faces = 0U;
  metrics.analytic_inverted_faces = 0U;
  metrics.analytic_sheet_bridge_faces = 0U;
  metrics.analytic_orientation_min_cosine = 1.0;
  metrics.analytic_normal_quality_status = "not_run";
  metrics.analytic_orientation_smooth_faces = 0U;
  metrics.analytic_orientation_seam_faces = 0U;
  metrics.analytic_normal_cosine_below_025_faces = 0U;
  metrics.analytic_normal_cosine_below_050_faces = 0U;
  metrics.analytic_normal_area_weighted_rms_deg = 0.0;
  metrics.analytic_normal_p99_deg = 0.0;
  metrics.analytic_smooth_normal_p99_deg = 0.0;
  metrics.analytic_seam_normal_p99_deg = 0.0;
  metrics.analytic_normal_area_below_cosine_025_fraction = 0.0;
  metrics.analytic_normal_area_below_cosine_050_fraction = 0.0;
  constexpr std::size_t kAngleHistogramBins = 1801U;
  constexpr double kAngleHistogramResolutionDeg = 0.1;
  std::vector<std::size_t> all_angle_histogram;
  std::vector<std::size_t> smooth_angle_histogram;
  std::vector<std::size_t> seam_angle_histogram;
  if (collect_normal_quality) {
    all_angle_histogram.assign(kAngleHistogramBins, 0U);
    smooth_angle_histogram.assign(kAngleHistogramBins, 0U);
    seam_angle_histogram.assign(kAngleHistogramBins, 0U);
    metrics.scratch_bytes = std::max(
        metrics.scratch_bytes,
        (all_angle_histogram.capacity() + smooth_angle_histogram.capacity() +
         seam_angle_histogram.capacity()) *
            sizeof(std::size_t));
  }
  double normal_area_weight = 0.0;
  double normal_area_angle_squared_sum = 0.0;
  double normal_area_below_cosine_025 = 0.0;
  double normal_area_below_cosine_050 = 0.0;
  constexpr double kOrientationTolerance = 1.0e-8;
  for (const Triangle &triangle : mesh.triangles) {
    const Vec3 a = mesh_vertex(mesh, triangle.v0);
    const Vec3 b = mesh_vertex(mesh, triangle.v1);
    const Vec3 c = mesh_vertex(mesh, triangle.v2);
    const Vec3 face_vector = cross(b - a, c - a);
    const double face_length = norm(face_vector);
    if (!(face_length > 1.0e-16) || !std::isfinite(face_length)) {
      ++metrics.analytic_orientation_uncertified_faces;
      metrics.analytic_orientation_min_cosine = -1.0;
      continue;
    }
    const Vec3 face = (1.0 / face_length) * face_vector;
    const Vertex &av = mesh.vertices[triangle.v0];
    const Vertex &bv = mesh.vertices[triangle.v1];
    const Vertex &cv = mesh.vertices[triangle.v2];
    const Vec3 an = normalized({av.nx, av.ny, av.nz});
    const Vec3 bn = normalized({bv.nx, bv.ny, bv.nz});
    const Vec3 cn = normalized({cv.nx, cv.ny, cv.nz});
    const double coherence =
        std::min({dot(an, bn), dot(bn, cn), dot(cn, an)});
    const Vec3 summed = an + bn + cn;
    const double summed_length = norm(summed);
    const bool smooth_face =
        vertex_normals_certified && coherence >= 0.50 &&
        summed_length > 1.0e-12;
    bool reference_found = smooth_face;
    Vec3 reference =
        reference_found ? (1.0 / summed_length) * summed : Vec3{};
    double cosine = reference_found ? dot(face, reference) : -1.0;

    // Smooth patch interiors are certified from their three already-projected
    // analytic vertex normals.  At a seam, or whenever that certificate says
    // the face is reversed, query the analytic patch at the centroid.  This
    // avoids both the old sampled-audit blind spot and false negatives from a
    // discontinuous patch owner at a welded seam.
    if (!reference_found || cosine <= kOrientationTolerance) {
      ++metrics.analytic_orientation_centroid_queries;
      const Vec3 centroid = (1.0 / 3.0) * (a + b + c);
      const analytic_ses::LocalProjectionHit hit = nearest_local_patch(
          patches, projection, triangle.atom_index, centroid,
          2.0 * grid_spacing);
      if (certified_finite_projection_hit(hit) &&
          hit.distance <= 0.75 * grid_spacing) {
        const Vec3 hit_normal{hit.normal.x, hit.normal.y, hit.normal.z};
        const double hit_normal_length = norm(hit_normal);
        reference_found = hit_normal_length > 1.0e-12;
        if (reference_found)
          reference = (1.0 / hit_normal_length) * hit_normal;
        cosine = reference_found ? dot(face, reference) : -1.0;
      }
    }
    if (!reference_found || !std::isfinite(cosine)) {
      ++metrics.analytic_orientation_uncertified_faces;
      metrics.analytic_orientation_min_cosine = -1.0;
      continue;
    }
    metrics.analytic_orientation_min_cosine =
        std::min(metrics.analytic_orientation_min_cosine, cosine);
    if (collect_normal_quality) {
      const double clamped_cosine = std::clamp(cosine, -1.0, 1.0);
      const double angle_deg =
          std::acos(clamped_cosine) * 180.0 / std::numbers::pi;
      const std::size_t angle_bin = std::min(
          kAngleHistogramBins - 1U,
          static_cast<std::size_t>(
              std::floor(angle_deg / kAngleHistogramResolutionDeg)));
      ++all_angle_histogram[angle_bin];
      if (smooth_face) {
        ++metrics.analytic_orientation_smooth_faces;
        ++smooth_angle_histogram[angle_bin];
      } else {
        ++metrics.analytic_orientation_seam_faces;
        ++seam_angle_histogram[angle_bin];
      }
      normal_area_weight += face_length;
      normal_area_angle_squared_sum += face_length * angle_deg * angle_deg;
      if (cosine < 0.25) {
        ++metrics.analytic_normal_cosine_below_025_faces;
        normal_area_below_cosine_025 += face_length;
      }
      if (cosine < 0.50) {
        ++metrics.analytic_normal_cosine_below_050_faces;
        normal_area_below_cosine_050 += face_length;
      }
    }
    if (!(cosine > kOrientationTolerance)) {
      ++metrics.analytic_inverted_faces;
      // Two of this face's own certified vertex normals point against each
      // other, so its corners are on facing sheets of the surface and the
      // triangle spans the gap between them.  Refining or reconnecting the
      // mesh cannot fix that: at this spacing one cell holds both sheets.
      if (coherence < 0.0)
        ++metrics.analytic_sheet_bridge_faces;
      continue;
    }
    ++metrics.analytic_orientation_certified_faces;
  }
  if (collect_normal_quality) {
    const auto percentile_99 = [&](const std::vector<std::size_t> &histogram,
                                   std::size_t count) {
      if (count == 0U)
        return 0.0;
      const std::size_t target = static_cast<std::size_t>(
          std::ceil(0.99 * static_cast<double>(count)));
      std::size_t cumulative = 0U;
      for (std::size_t index = 0U; index < histogram.size(); ++index) {
        cumulative += histogram[index];
        if (cumulative >= target)
          return std::min(180.0, (static_cast<double>(index) + 1.0) *
                                     kAngleHistogramResolutionDeg);
      }
      return 180.0;
    };
    const std::size_t evaluated_faces =
        metrics.analytic_orientation_smooth_faces +
        metrics.analytic_orientation_seam_faces;
    metrics.analytic_normal_p99_deg =
        percentile_99(all_angle_histogram, evaluated_faces);
    metrics.analytic_smooth_normal_p99_deg = percentile_99(
        smooth_angle_histogram, metrics.analytic_orientation_smooth_faces);
    metrics.analytic_seam_normal_p99_deg = percentile_99(
        seam_angle_histogram, metrics.analytic_orientation_seam_faces);
    if (normal_area_weight > 0.0) {
      metrics.analytic_normal_area_weighted_rms_deg =
          std::sqrt(normal_area_angle_squared_sum / normal_area_weight);
      metrics.analytic_normal_area_below_cosine_025_fraction =
          normal_area_below_cosine_025 / normal_area_weight;
      metrics.analytic_normal_area_below_cosine_050_fraction =
          normal_area_below_cosine_050 / normal_area_weight;
    }
    // Area-weighting prevents a handful of tiny seam slivers from dominating
    // the gate, while the global and smooth-patch p99 bounds still reject a
    // broad loss of tangency. These are geometry-quality gates, separate from
    // the stricter sign/orientation certificate above.
    const bool normal_quality_pass =
        evaluated_faces != 0U &&
        metrics.analytic_normal_area_weighted_rms_deg <= 30.0 &&
        metrics.analytic_normal_p99_deg <= 75.5 &&
        metrics.analytic_smooth_normal_p99_deg <= 60.0 &&
        metrics.analytic_normal_area_below_cosine_025_fraction <= 0.02 &&
        metrics.analytic_normal_area_below_cosine_050_fraction <= 0.05;
    metrics.analytic_normal_quality_status =
        normal_quality_pass ? "pass" : "warning";
  }
  if (metrics.analytic_inverted_faces != 0U)
    metrics.analytic_orientation_status =
        metrics.analytic_sheet_bridge_faces == metrics.analytic_inverted_faces
            ? "sheet_bridge_below_spacing"
            : "inverted_face";
  else if (metrics.analytic_orientation_uncertified_faces != 0U)
    metrics.analytic_orientation_status = "orientation_uncertified";
  else if (collect_normal_quality &&
           metrics.analytic_normal_quality_status != "pass")
    metrics.analytic_orientation_status = "normal_quality";
  else
    metrics.analytic_orientation_status = "pass";
}

#ifdef TITAN_TABI_SES_TEST_HOOKS
[[nodiscard]] const char *
projection_patch_kind_label(analytic_ses::PatchKind kind) noexcept {
  switch (kind) {
  case analytic_ses::PatchKind::contact:
    return "contact";
  case analytic_ses::PatchKind::toroidal:
    return "toroidal";
  case analytic_ses::PatchKind::reentrant:
    return "reentrant";
  }
  return "unknown";
}

[[nodiscard]] const char *
projection_commit_state_label(std::uint8_t state) noexcept {
  switch (state) {
  case 0U:
    return "no_certified_proposal";
  case 1U:
    return "pending";
  case 2U:
    return "committed";
  case 3U:
    return "coupled_pending";
  case 4U:
    return "rejected";
  default:
    return "unavailable";
  }
}

// Detailed failure evidence is compiled only into SES test binaries and is
// additionally environment-gated.  Production builds retain neither the
// projection-state vector nor getenv/debug-stream calls.
void emit_projection_failure_diagnostics(
    const SurfaceMesh &mesh, const analytic_ses::PatchComplex &patches,
    const analytic_ses::LocalPatchProjectionIndex &projection,
    double grid_spacing, const LocalPatchRepairMetrics &metrics) {
  const char *enabled = std::getenv("TITAN_TABI_SES_DEBUG_PROJECTION");
  if (enabled == nullptr || enabled[0] != '1')
    return;

  const auto projection_state = [&](std::size_t vertex) {
    return vertex < metrics.debug_last_projection_state.size()
               ? metrics.debug_last_projection_state[vertex]
               : std::uint8_t{255U};
  };
  const std::size_t unresolved = static_cast<std::size_t>(std::count(
      metrics.debug_last_projection_state.begin(),
      metrics.debug_last_projection_state.end(), std::uint8_t{4U}));
  std::cerr.precision(17);
  std::cerr << "ses_projection_diag,summary,vertices=" << mesh.vertices.size()
            << ",triangles=" << mesh.triangles.size()
            << ",projection_state_size="
            << metrics.debug_last_projection_state.size()
            << ",unresolved=" << unresolved
            << ",vertex_certificate="
            << (metrics.analytic_vertex_certificate ? 1 : 0)
            << ",reported_inverted=" << metrics.analytic_inverted_faces
            << '\n';
  const CompactVertexIncidence diagnostic_incidence =
      build_compact_vertex_incidence(mesh);
  const std::vector<CompactOwnedEdge> diagnostic_edges =
      build_compact_owned_edges(mesh);
  std::vector<std::uint8_t> logged_one_ring(mesh.triangles.size(), 0U);

  for (std::size_t vertex_index = 0U;
       vertex_index < mesh.vertices.size(); ++vertex_index) {
    if (projection_state(vertex_index) != 4U)
      continue;
    const Vertex &vertex = mesh.vertices[vertex_index];
    const Vec3 point{vertex.x, vertex.y, vertex.z};
    const analytic_ses::LocalProjectionHit hit = nearest_local_patch(
        patches, projection, vertex.atom_index, point, 2.0 * grid_spacing);
    std::cerr << "ses_projection_diag,unresolved_vertex,id="
              << vertex_index << ",position=" << point.x << ';' << point.y
              << ';' << point.z << ",stored_normal=" << vertex.nx << ';'
              << vertex.ny << ';' << vertex.nz
              << ",owner=" << vertex.atom_index
              << ",state=" << projection_commit_state_label(
                                    projection_state(vertex_index))
              << ",hit_found=" << (hit.found ? 1 : 0)
              << ",hit_bounded="
              << (hit.bounded_closest_certified ? 1 : 0)
              << ",hit_normal_cert=" << (hit.normal_certified ? 1 : 0)
              << ",hit_kind=" << projection_patch_kind_label(hit.kind)
              << ",hit_patch=" << hit.patch_index
              << ",hit_atom=" << hit.atom_index
              << ",hit_distance=" << hit.distance
              << ",hit_point=" << hit.point.x << ';' << hit.point.y << ';'
              << hit.point.z << ",hit_normal=" << hit.normal.x << ';'
              << hit.normal.y << ';' << hit.normal.z << '\n';
    // Which incident face refuses the move: report each one-ring face's
    // shape and the cosine between its normal before and after moving this
    // vertex to the certified hit.  A negative cosine on a face whose
    // smallest altitude is below the move distance is a sliver flip.
    if (!certified_finite_projection_hit(hit) ||
        vertex_index + 1U >= diagnostic_incidence.offsets.size())
      continue;
    const Vec3 target{hit.point.x, hit.point.y, hit.point.z};
    for (std::size_t position = diagnostic_incidence.offsets[vertex_index];
         position < diagnostic_incidence.offsets[vertex_index + 1U];
         ++position) {
      const std::size_t face = diagnostic_incidence.triangles[position];
      if (face >= mesh.triangles.size())
        continue;
      const Triangle &triangle = mesh.triangles[face];
      const Vec3 a = mesh_vertex(mesh, triangle.v0);
      const Vec3 b = mesh_vertex(mesh, triangle.v1);
      const Vec3 c = mesh_vertex(mesh, triangle.v2);
      const auto moved = [&](std::size_t index, const Vec3 &p) {
        return index == vertex_index ? target : p;
      };
      const Vec3 old_normal = cross(b - a, c - a);
      const Vec3 a2 = moved(triangle.v0, a);
      const Vec3 b2 = moved(triangle.v1, b);
      const Vec3 c2 = moved(triangle.v2, c);
      const Vec3 new_normal = cross(b2 - a2, c2 - a2);
      const double ab = norm(b - a);
      const double bc = norm(c - b);
      const double ca = norm(a - c);
      const double longest = std::max({ab, bc, ca});
      const double shortest = std::min({ab, bc, ca});
      const double area = 0.5 * norm(old_normal);
      const double old_length = norm(old_normal);
      const double new_length = norm(new_normal);
      std::cerr << "ses_projection_diag,unresolved_incident_face,vertex="
                << vertex_index << ",face=" << face << ",area=" << area
                << ",min_altitude=" << (longest > 0.0 ? 2.0 * area / longest : 0.0)
                << ",quality=" << normalized_triangle_quality(a, b, c)
                << ",edge_ratio="
                << (shortest > 1.0e-14
                        ? longest / shortest
                        : std::numeric_limits<double>::infinity())
                << ",cos_old_new="
                << (old_length > 0.0 && new_length > 0.0
                        ? dot(old_normal, new_normal) / (old_length * new_length)
                        : std::numeric_limits<double>::quiet_NaN())
                << ",new_area=" << 0.5 * new_length << '\n';
    }
  }

  constexpr double kOrientationTolerance = 1.0e-8;
  std::size_t emitted_inverted = 0U;
  for (std::size_t triangle_index = 0U;
       triangle_index < mesh.triangles.size(); ++triangle_index) {
    const Triangle &triangle = mesh.triangles[triangle_index];
    const Vec3 a = mesh_vertex(mesh, triangle.v0);
    const Vec3 b = mesh_vertex(mesh, triangle.v1);
    const Vec3 c = mesh_vertex(mesh, triangle.v2);
    const Vec3 face_vector = cross(b - a, c - a);
    const double face_length = norm(face_vector);
    if (!(face_length > 1.0e-16) || !std::isfinite(face_length))
      continue;
    const Vec3 face = (1.0 / face_length) * face_vector;
    const Vec3 centroid = (1.0 / 3.0) * (a + b + c);
    const analytic_ses::LocalProjectionHit centroid_hit =
        nearest_local_patch(patches, projection, triangle.atom_index,
                            centroid, 2.0 * grid_spacing);
    const bool centroid_certified =
        certified_finite_projection_hit(centroid_hit) &&
        centroid_hit.distance <= 0.75 * grid_spacing;
    if (!centroid_certified)
      continue;
    const Vec3 centroid_normal{centroid_hit.normal.x, centroid_hit.normal.y,
                               centroid_hit.normal.z};
    const double centroid_normal_length = norm(centroid_normal);
    if (!(centroid_normal_length > 1.0e-12))
      continue;
    const double cosine =
        dot(face, (1.0 / centroid_normal_length) * centroid_normal);
    if (cosine > kOrientationTolerance)
      continue;

    const double ab = norm(b - a);
    const double bc = norm(c - b);
    const double ca = norm(a - c);
    const double shortest = std::min({ab, bc, ca});
    const double edge_ratio =
        shortest > 1.0e-14
            ? std::max({ab, bc, ca}) / shortest
            : std::numeric_limits<double>::infinity();
    const bool incident_unresolved =
        projection_state(triangle.v0) == 4U ||
        projection_state(triangle.v1) == 4U ||
        projection_state(triangle.v2) == 4U;
    std::cerr << "ses_projection_diag,inverted_face,id=" << triangle_index
              << ",vertices=" << triangle.v0 << ';' << triangle.v1 << ';'
              << triangle.v2 << ",owner=" << triangle.atom_index
              << ",incident_unresolved=" << (incident_unresolved ? 1 : 0)
              << ",area=" << 0.5 * face_length
              << ",quality=" << normalized_triangle_quality(a, b, c)
              << ",edge_ratio=" << edge_ratio << ",face_normal=" << face.x
              << ';' << face.y << ';' << face.z
              << ",centroid=" << centroid.x << ';' << centroid.y << ';'
              << centroid.z << ",centroid_hit_kind="
              << projection_patch_kind_label(centroid_hit.kind)
              << ",centroid_hit_patch=" << centroid_hit.patch_index
              << ",centroid_hit_atom=" << centroid_hit.atom_index
              << ",centroid_hit_distance=" << centroid_hit.distance
              << ",centroid_hit_point=" << centroid_hit.point.x << ';'
              << centroid_hit.point.y << ';' << centroid_hit.point.z
              << ",centroid_hit_normal=" << centroid_hit.normal.x << ';'
              << centroid_hit.normal.y << ';' << centroid_hit.normal.z
              << ",cosine=" << cosine << '\n';

    for (const std::size_t vertex_index :
         {triangle.v0, triangle.v1, triangle.v2}) {
      const Vertex &vertex = mesh.vertices[vertex_index];
      const Vec3 point{vertex.x, vertex.y, vertex.z};
      const analytic_ses::LocalProjectionHit hit = nearest_local_patch(
          patches, projection, vertex.atom_index, point,
          2.0 * grid_spacing);
      const bool hit_certified = certified_finite_projection_hit(hit);
      const Vec3 hit_normal{hit.normal.x, hit.normal.y, hit.normal.z};
      const double hit_normal_length = norm(hit_normal);
      const double hit_face_cosine =
          hit_certified && hit_normal_length > 1.0e-12
              ? dot(face, (1.0 / hit_normal_length) * hit_normal)
              : std::numeric_limits<double>::quiet_NaN();
      std::cerr
          << "ses_projection_diag,inverted_face_vertex,face="
          << triangle_index << ",id=" << vertex_index
          << ",position=" << point.x << ';' << point.y << ';' << point.z
          << ",stored_normal=" << vertex.nx << ';' << vertex.ny << ';'
          << vertex.nz << ",owner=" << vertex.atom_index
          << ",state=" << projection_commit_state_label(
                                projection_state(vertex_index))
          << ",current_hit_cert=" << (hit_certified ? 1 : 0)
          << ",hit_found=" << (hit.found ? 1 : 0)
          << ",hit_bounded="
          << (hit.bounded_closest_certified ? 1 : 0)
          << ",hit_normal_cert=" << (hit.normal_certified ? 1 : 0)
          << ",hit_kind=" << projection_patch_kind_label(hit.kind)
          << ",hit_patch=" << hit.patch_index
          << ",hit_atom=" << hit.atom_index
          << ",hit_distance=" << hit.distance << ",hit_point="
          << hit.point.x << ';' << hit.point.y << ';' << hit.point.z
          << ",hit_normal=" << hit.normal.x << ';' << hit.normal.y << ';'
          << hit.normal.z << ",hit_face_cosine=" << hit_face_cosine
          << '\n';
    }

    for (const std::size_t vertex_index :
         {triangle.v0, triangle.v1, triangle.v2}) {
      if (diagnostic_incidence.offsets.size() != mesh.vertices.size() + 1U)
        break;
      for (std::size_t position = diagnostic_incidence.offsets[vertex_index];
           position < diagnostic_incidence.offsets[vertex_index + 1U];
           ++position) {
        const std::size_t neighbor_face =
            diagnostic_incidence.triangles[position];
        if (neighbor_face >= mesh.triangles.size() ||
            logged_one_ring[neighbor_face] != 0U)
          continue;
        logged_one_ring[neighbor_face] = 1U;
        const Triangle &neighbor = mesh.triangles[neighbor_face];
        const Vec3 na = mesh_vertex(mesh, neighbor.v0);
        const Vec3 nb = mesh_vertex(mesh, neighbor.v1);
        const Vec3 nc = mesh_vertex(mesh, neighbor.v2);
        const Vec3 neighbor_vector = cross(nb - na, nc - na);
        const double neighbor_length = norm(neighbor_vector);
        const Vec3 neighbor_centroid =
            (1.0 / 3.0) * (na + nb + nc);
        const analytic_ses::LocalProjectionHit neighbor_hit =
            nearest_local_patch(patches, projection, neighbor.atom_index,
                                neighbor_centroid, 2.0 * grid_spacing);
        double neighbor_cosine =
            std::numeric_limits<double>::quiet_NaN();
        if (certified_finite_projection_hit(neighbor_hit) &&
            neighbor_length > 1.0e-16) {
          const Vec3 hit_normal{neighbor_hit.normal.x, neighbor_hit.normal.y,
                                neighbor_hit.normal.z};
          const double hit_normal_length = norm(hit_normal);
          if (hit_normal_length > 1.0e-12)
            neighbor_cosine =
                dot((1.0 / neighbor_length) * neighbor_vector,
                    (1.0 / hit_normal_length) * hit_normal);
        }
        std::cerr << "ses_projection_diag,one_ring_face,seed_face="
                  << triangle_index << ",seed_vertex=" << vertex_index
                  << ",id=" << neighbor_face << ",vertices=" << neighbor.v0
                  << ';' << neighbor.v1 << ';' << neighbor.v2
                  << ",area=" << 0.5 * neighbor_length
                  << ",quality="
                  << normalized_triangle_quality(na, nb, nc)
                  << ",centroid_hit_kind="
                  << projection_patch_kind_label(neighbor_hit.kind)
                  << ",centroid_hit_patch=" << neighbor_hit.patch_index
                  << ",centroid_hit_distance=" << neighbor_hit.distance
                  << ",centroid_cosine=" << neighbor_cosine
                  << ",vertex_hits=";
        for (const std::size_t neighbor_vertex :
             {neighbor.v0, neighbor.v1, neighbor.v2}) {
          const Vertex &value = mesh.vertices[neighbor_vertex];
          const analytic_ses::LocalProjectionHit hit = nearest_local_patch(
              patches, projection, value.atom_index,
              {value.x, value.y, value.z}, 2.0 * grid_spacing);
          std::cerr << neighbor_vertex << ':'
                    << projection_patch_kind_label(hit.kind) << ':'
                    << hit.patch_index << ':'
                    << (certified_finite_projection_hit(hit) ? 1 : 0)
                    << ';';
        }
        std::cerr << '\n';
      }
    }

    for (const EdgeKey edge :
         {edge_key(triangle.v0, triangle.v1),
          edge_key(triangle.v1, triangle.v2),
          edge_key(triangle.v2, triangle.v0)}) {
      const std::pair<std::uint32_t, std::uint32_t> key{
          static_cast<std::uint32_t>(edge.lo),
          static_cast<std::uint32_t>(edge.hi)};
      const auto begin = std::lower_bound(
          diagnostic_edges.begin(), diagnostic_edges.end(), key,
          [](const CompactOwnedEdge &owned,
             const std::pair<std::uint32_t, std::uint32_t> &value) {
            return std::pair{owned.lo, owned.hi} < value;
          });
      auto end = begin;
      while (end != diagnostic_edges.end() && end->lo == key.first &&
             end->hi == key.second)
        ++end;
      const std::size_t owner_count =
          static_cast<std::size_t>(end - begin);
      if (owner_count != 2U) {
        std::cerr << "ses_projection_diag,edge_flip,seed_face="
                  << triangle_index << ",edge=" << edge.lo << ';' << edge.hi
                  << ",owners=" << owner_count
                  << ",manifold_oriented=0\n";
        continue;
      }
      const CompactOwnedEdge &first = begin[0];
      const CompactOwnedEdge &second = begin[1];
      const bool manifold_oriented =
          first.from == second.to && first.to == second.from &&
          first.opposite != second.opposite;
      const bool replacement_exists =
          compact_edge_exists(diagnostic_edges, first.opposite,
                              second.opposite);
      Triangle new_first = mesh.triangles[first.triangle];
      Triangle new_second = mesh.triangles[second.triangle];
      new_first.v0 = first.opposite;
      new_first.v1 = second.opposite;
      new_first.v2 = first.to;
      new_second.v0 = second.opposite;
      new_second.v1 = first.opposite;
      new_second.v2 = first.from;
      const AnalyticFaceOrientation first_orientation =
          evaluate_analytic_face_orientation(
              mesh, new_first, patches, projection, grid_spacing, false);
      const AnalyticFaceOrientation second_orientation =
          evaluate_analytic_face_orientation(
              mesh, new_second, patches, projection, grid_spacing, false);
      std::cerr << "ses_projection_diag,edge_flip,seed_face="
                << triangle_index << ",edge=" << edge.lo << ';' << edge.hi
                << ",owners=" << owner_count
                << ",manifold_oriented=" << (manifold_oriented ? 1 : 0)
                << ",opposites=" << first.opposite << ';'
                << second.opposite
                << ",replacement_exists="
                << (replacement_exists ? 1 : 0)
                << ",new_faces=" << new_first.v0 << ';' << new_first.v1
                << ';' << new_first.v2 << '|' << new_second.v0 << ';'
                << new_second.v1 << ';' << new_second.v2
                << ",new_quality="
                << normalized_triangle_quality(
                       mesh_vertex(mesh, new_first.v0),
                       mesh_vertex(mesh, new_first.v1),
                       mesh_vertex(mesh, new_first.v2))
                << ';'
                << normalized_triangle_quality(
                       mesh_vertex(mesh, new_second.v0),
                       mesh_vertex(mesh, new_second.v1),
                       mesh_vertex(mesh, new_second.v2))
                << ",new_certified=" << (first_orientation.certified ? 1 : 0)
                << ';' << (second_orientation.certified ? 1 : 0)
                << ",new_cosine=" << first_orientation.cosine << ';'
                << second_orientation.cosine << '\n';
    }
    ++emitted_inverted;
  }
  std::cerr << "ses_projection_diag,complete,emitted_inverted="
            << emitted_inverted << '\n';
}
#endif

struct TriangleBounds {
  Vec3 lower;
  Vec3 upper;
};

[[nodiscard]] TriangleBounds triangle_bounds(const SurfaceMesh &mesh,
                                             const Triangle &triangle) {
  const Vec3 a = mesh_vertex(mesh, triangle.v0);
  const Vec3 b = mesh_vertex(mesh, triangle.v1);
  const Vec3 c = mesh_vertex(mesh, triangle.v2);
  return {{std::min({a.x, b.x, c.x}), std::min({a.y, b.y, c.y}),
           std::min({a.z, b.z, c.z})},
          {std::max({a.x, b.x, c.x}), std::max({a.y, b.y, c.y}),
           std::max({a.z, b.z, c.z})}};
}

[[nodiscard]] bool bounds_overlap(const TriangleBounds &a,
                                  const TriangleBounds &b,
                                  double tolerance) noexcept {
  return a.lower.x <= b.upper.x + tolerance &&
         b.lower.x <= a.upper.x + tolerance &&
         a.lower.y <= b.upper.y + tolerance &&
         b.lower.y <= a.upper.y + tolerance &&
         a.lower.z <= b.upper.z + tolerance &&
         b.lower.z <= a.upper.z + tolerance;
}

struct SharedTriangleSimplex {
  std::array<std::size_t, 3> vertices{};
  std::size_t count{0U};
};

[[nodiscard]] SharedTriangleSimplex
shared_triangle_simplex(const Triangle &a, const Triangle &b) noexcept {
  const std::array<std::size_t, 3> av = {a.v0, a.v1, a.v2};
  const std::array<std::size_t, 3> bv = {b.v0, b.v1, b.v2};
  SharedTriangleSimplex shared;
  for (const std::size_t vertex : av) {
    if (std::find(bv.begin(), bv.end(), vertex) != bv.end())
      shared.vertices[shared.count++] = vertex;
  }
  return shared;
}

[[nodiscard]] bool segment_intersects_triangle(
    const Vec3 &p, const Vec3 &q, const Vec3 &a, const Vec3 &b,
    const Vec3 &c) noexcept {
  const Vec3 direction = q - p;
  const Vec3 edge1 = b - a;
  const Vec3 edge2 = c - a;
  const Vec3 p_vector = cross(direction, edge2);
  const double determinant = dot(edge1, p_vector);
  const double determinant_scale =
      std::max(1.0e-300, norm(direction) * norm(edge1) * norm(edge2));
  if (std::abs(determinant) <= 1.0e-12 * determinant_scale)
    return false;
  const double inverse = 1.0 / determinant;
  const Vec3 t_vector = p - a;
  const double u = dot(t_vector, p_vector) * inverse;
  const Vec3 q_vector = cross(t_vector, edge1);
  const double v = dot(direction, q_vector) * inverse;
  const double t = dot(edge2, q_vector) * inverse;
  constexpr double tolerance = 2.0e-10;
  return u >= -tolerance && v >= -tolerance &&
         u + v <= 1.0 + tolerance && t >= -tolerance &&
         t <= 1.0 + tolerance;
}

struct Vec2 {
  double x{0.0};
  double y{0.0};
};

[[nodiscard]] long double orient2d(const Vec2 &a, const Vec2 &b,
                                   const Vec2 &c) noexcept {
  return (static_cast<long double>(b.x) - a.x) *
             (static_cast<long double>(c.y) - a.y) -
         (static_cast<long double>(b.y) - a.y) *
             (static_cast<long double>(c.x) - a.x);
}

[[nodiscard]] bool point_in_triangle_2d(const Vec2 &point, const Vec2 &a,
                                        const Vec2 &b,
                                        const Vec2 &c) noexcept {
  const long double ab = orient2d(a, b, point);
  const long double bc = orient2d(b, c, point);
  const long double ca = orient2d(c, a, point);
  const bool negative = ab < 0.0L || bc < 0.0L || ca < 0.0L;
  const bool positive = ab > 0.0L || bc > 0.0L || ca > 0.0L;
  return !(negative && positive);
}

[[nodiscard]] bool segments_intersect_2d(const Vec2 &a, const Vec2 &b,
                                         const Vec2 &c,
                                         const Vec2 &d) noexcept {
  const long double ab_c = orient2d(a, b, c);
  const long double ab_d = orient2d(a, b, d);
  const long double cd_a = orient2d(c, d, a);
  const long double cd_b = orient2d(c, d, b);
  const auto on_segment = [](const Vec2 &p, const Vec2 &q, const Vec2 &r) {
    return r.x >= std::min(p.x, q.x) && r.x <= std::max(p.x, q.x) &&
           r.y >= std::min(p.y, q.y) && r.y <= std::max(p.y, q.y);
  };
  if (ab_c == 0.0L && on_segment(a, b, c))
    return true;
  if (ab_d == 0.0L && on_segment(a, b, d))
    return true;
  if (cd_a == 0.0L && on_segment(c, d, a))
    return true;
  if (cd_b == 0.0L && on_segment(c, d, b))
    return true;
  return ((ab_c < 0.0L && ab_d > 0.0L) ||
          (ab_c > 0.0L && ab_d < 0.0L)) &&
         ((cd_a < 0.0L && cd_b > 0.0L) ||
          (cd_a > 0.0L && cd_b < 0.0L));
}

[[nodiscard]] bool coplanar_triangles_intersect(
    const std::array<Vec3, 3> &a, const std::array<Vec3, 3> &b,
    const Vec3 &normal) noexcept {
  const Vec3 absolute{std::abs(normal.x), std::abs(normal.y),
                      std::abs(normal.z)};
  const int drop = absolute.x >= absolute.y && absolute.x >= absolute.z
                       ? 0
                       : (absolute.y >= absolute.z ? 1 : 2);
  const auto project = [drop](const Vec3 &point) {
    return drop == 0 ? Vec2{point.y, point.z}
                     : (drop == 1 ? Vec2{point.x, point.z}
                                  : Vec2{point.x, point.y});
  };
  std::array<Vec2, 3> pa{};
  std::array<Vec2, 3> pb{};
  for (std::size_t index = 0U; index < 3U; ++index) {
    pa[index] = project(a[index]);
    pb[index] = project(b[index]);
  }
  for (std::size_t i = 0U; i < 3U; ++i) {
    for (std::size_t j = 0U; j < 3U; ++j) {
      if (segments_intersect_2d(pa[i], pa[(i + 1U) % 3U], pb[j],
                                pb[(j + 1U) % 3U]))
        return true;
    }
  }
  return point_in_triangle_2d(pa[0], pb[0], pb[1], pb[2]) ||
         point_in_triangle_2d(pb[0], pa[0], pa[1], pa[2]);
}

[[nodiscard]] bool triangles_intersect(const SurfaceMesh &mesh,
                                       const Triangle &first,
                                       const Triangle &second,
                                       double tolerance) noexcept {
  const std::array<Vec3, 3> a = {mesh_vertex(mesh, first.v0),
                                 mesh_vertex(mesh, first.v1),
                                 mesh_vertex(mesh, first.v2)};
  const std::array<Vec3, 3> b = {mesh_vertex(mesh, second.v0),
                                 mesh_vertex(mesh, second.v1),
                                 mesh_vertex(mesh, second.v2)};
  const Vec3 normal_a = cross(a[1] - a[0], a[2] - a[0]);
  const Vec3 normal_b = cross(b[1] - b[0], b[2] - b[0]);
  const double normal_a_length = norm(normal_a);
  const double normal_b_length = norm(normal_b);
  if (!(normal_a_length > 1.0e-16) || !(normal_b_length > 1.0e-16))
    return true;
  const Vec3 normal_cross = cross(normal_a, normal_b);
  if (norm(normal_cross) <=
      1.0e-10 * normal_a_length * normal_b_length) {
    const double plane_distance =
        std::abs(dot(normal_a, b[0] - a[0])) / normal_a_length;
    if (plane_distance > tolerance)
      return false;
    return coplanar_triangles_intersect(a, b, normal_a);
  }
  for (std::size_t edge = 0U; edge < 3U; ++edge) {
    if (segment_intersects_triangle(a[edge], a[(edge + 1U) % 3U], b[0],
                                    b[1], b[2]) ||
        segment_intersects_triangle(b[edge], b[(edge + 1U) % 3U], a[0],
                                    a[1], a[2]))
      return true;
  }
  return false;
}

struct LineTriangleInterval {
  double lower{0.0};
  double upper{0.0};
  bool valid{false};
};

[[nodiscard]] LineTriangleInterval line_triangle_interval_at_vertex(
    const Vec3 &shared, const Vec3 &first, const Vec3 &second,
    const Vec3 &unit_direction) noexcept {
  const Vec3 e1 = first - shared;
  const Vec3 e2 = second - shared;
  const double g11 = dot(e1, e1);
  const double g12 = dot(e1, e2);
  const double g22 = dot(e2, e2);
  const double determinant = g11 * g22 - g12 * g12;
  const double determinant_scale = std::max(1.0e-300, g11 * g22);
  if (!(determinant > 128.0 * std::numeric_limits<double>::epsilon() *
                          determinant_scale))
    return {};
  const double d1 = dot(unit_direction, e1);
  const double d2 = dot(unit_direction, e2);
  const double alpha = (d1 * g22 - d2 * g12) / determinant;
  const double beta = (d2 * g11 - d1 * g12) / determinant;
  const Vec3 reconstructed = alpha * e1 + beta * e2;
  if (norm(reconstructed - unit_direction) > 1.0e-8)
    return {};

  double lower = -std::numeric_limits<double>::infinity();
  double upper = std::numeric_limits<double>::infinity();
  constexpr double kCoefficientTolerance = 64.0 *
      std::numeric_limits<double>::epsilon();
  const auto require_nonnegative = [&](double coefficient) {
    if (coefficient > kCoefficientTolerance)
      lower = std::max(lower, 0.0);
    else if (coefficient < -kCoefficientTolerance)
      upper = std::min(upper, 0.0);
  };
  require_nonnegative(alpha);
  require_nonnegative(beta);
  const double sum = alpha + beta;
  if (sum > kCoefficientTolerance)
    upper = std::min(upper, 1.0 / sum);
  else if (sum < -kCoefficientTolerance)
    lower = std::max(lower, 1.0 / sum);
  if (!(lower <= upper) || lower > 0.0 || upper < 0.0)
    return {};
  return {lower, upper, true};
}

[[nodiscard]] bool coplanar_shared_vertex_intersects_beyond(
    const std::array<Vec3, 3> &a, const std::array<Vec3, 3> &b,
    const Vec3 &normal, double tolerance) noexcept {
  const Vec3 absolute{std::abs(normal.x), std::abs(normal.y),
                      std::abs(normal.z)};
  const int drop = absolute.x >= absolute.y && absolute.x >= absolute.z
                       ? 0
                       : (absolute.y >= absolute.z ? 1 : 2);
  const auto project = [drop](const Vec3 &point) {
    return drop == 0 ? Vec2{point.y, point.z}
                     : (drop == 1 ? Vec2{point.x, point.z}
                                  : Vec2{point.x, point.y});
  };
  std::array<Vec2, 3> pa{};
  std::array<Vec2, 3> pb{};
  for (std::size_t index = 0U; index < 3U; ++index) {
    pa[index] = project(a[index]);
    pb[index] = project(b[index]);
  }
  for (std::size_t index = 1U; index < 3U; ++index) {
    if (norm(a[index] - a[0]) > tolerance &&
        point_in_triangle_2d(pa[index], pb[0], pb[1], pb[2]))
      return true;
    if (norm(b[index] - b[0]) > tolerance &&
        point_in_triangle_2d(pb[index], pa[0], pa[1], pa[2]))
      return true;
  }

  for (std::size_t ai = 0U; ai < 3U; ++ai) {
    const std::size_t an = (ai + 1U) % 3U;
    const bool a_incident = ai == 0U || an == 0U;
    for (std::size_t bi = 0U; bi < 3U; ++bi) {
      const std::size_t bn = (bi + 1U) % 3U;
      const bool b_incident = bi == 0U || bn == 0U;
      if (!segments_intersect_2d(pa[ai], pa[an], pb[bi], pb[bn]))
        continue;
      if (!(a_incident && b_incident))
        return true;
      const Vec2 a_other = ai == 0U ? pa[an] : pa[ai];
      const Vec2 b_other = bi == 0U ? pb[bn] : pb[bi];
      if (orient2d(pa[0], a_other, b_other) != 0.0L)
        continue;
      const double ax = a_other.x - pa[0].x;
      const double ay = a_other.y - pa[0].y;
      const double bx = b_other.x - pb[0].x;
      const double by = b_other.y - pb[0].y;
      if (ax * bx + ay * by > 0.0)
        return true;
    }
  }
  return false;
}

[[nodiscard]] bool shared_simplex_intersects_beyond(
    const SurfaceMesh &mesh, const Triangle &first, const Triangle &second,
    const SharedTriangleSimplex &shared, double tolerance) noexcept {
  if (shared.count == 0U)
    return triangles_intersect(mesh, first, second, tolerance);
  if (shared.count == 3U)
    return true;

  const std::array<std::size_t, 3> first_vertices = {first.v0, first.v1,
                                                     first.v2};
  const std::array<std::size_t, 3> second_vertices = {second.v0, second.v1,
                                                      second.v2};
  const auto is_shared = [&](std::size_t vertex) {
    return std::find(shared.vertices.begin(),
                     shared.vertices.begin() +
                         static_cast<std::ptrdiff_t>(shared.count),
                     vertex) !=
           shared.vertices.begin() +
               static_cast<std::ptrdiff_t>(shared.count);
  };
  std::array<std::size_t, 2> first_other{};
  std::array<std::size_t, 2> second_other{};
  std::size_t first_other_count = 0U;
  std::size_t second_other_count = 0U;
  for (const std::size_t vertex : first_vertices) {
    if (!is_shared(vertex))
      first_other[first_other_count++] = vertex;
  }
  for (const std::size_t vertex : second_vertices) {
    if (!is_shared(vertex))
      second_other[second_other_count++] = vertex;
  }

  if (shared.count == 2U) {
    if (first_other_count != 1U || second_other_count != 1U)
      return true;
    const Vec3 s0 = mesh_vertex(mesh, shared.vertices[0]);
    const Vec3 s1 = mesh_vertex(mesh, shared.vertices[1]);
    const Vec3 a = mesh_vertex(mesh, first_other[0]);
    const Vec3 b = mesh_vertex(mesh, second_other[0]);
    const Vec3 edge = s1 - s0;
    const Vec3 normal_a = cross(edge, a - s0);
    const Vec3 normal_b = cross(edge, b - s0);
    const double normal_a_length = norm(normal_a);
    const double normal_b_length = norm(normal_b);
    if (!(normal_a_length > 1.0e-16) || !(normal_b_length > 1.0e-16))
      return true;
    const double plane_distance =
        std::abs(dot(normal_a, b - s0)) / normal_a_length;
    if (plane_distance > tolerance)
      return false;
    const Vec3 absolute{std::abs(normal_a.x), std::abs(normal_a.y),
                        std::abs(normal_a.z)};
    const int drop = absolute.x >= absolute.y && absolute.x >= absolute.z
                         ? 0
                         : (absolute.y >= absolute.z ? 1 : 2);
    const auto project = [drop](const Vec3 &point) {
      return drop == 0 ? Vec2{point.y, point.z}
                       : (drop == 1 ? Vec2{point.x, point.z}
                                    : Vec2{point.x, point.y});
    };
    const Vec2 p0 = project(s0);
    const Vec2 p1 = project(s1);
    const long double side_a = orient2d(p0, p1, project(a));
    const long double side_b = orient2d(p0, p1, project(b));
    return side_a == 0.0L || side_b == 0.0L ||
           (side_a < 0.0L) == (side_b < 0.0L);
  }

  if (first_other_count != 2U || second_other_count != 2U)
    return true;
  const Vec3 common = mesh_vertex(mesh, shared.vertices[0]);
  const std::array<Vec3, 3> a = {
      common, mesh_vertex(mesh, first_other[0]),
      mesh_vertex(mesh, first_other[1])};
  const std::array<Vec3, 3> b = {
      common, mesh_vertex(mesh, second_other[0]),
      mesh_vertex(mesh, second_other[1])};
  const Vec3 normal_a = cross(a[1] - a[0], a[2] - a[0]);
  const Vec3 normal_b = cross(b[1] - b[0], b[2] - b[0]);
  const double normal_a_length = norm(normal_a);
  const double normal_b_length = norm(normal_b);
  if (!(normal_a_length > 1.0e-16) || !(normal_b_length > 1.0e-16))
    return true;
  const Vec3 normal_cross = cross(normal_a, normal_b);
  const double normal_cross_length = norm(normal_cross);
  if (normal_cross_length <=
      64.0 * std::numeric_limits<double>::epsilon() * normal_a_length *
          normal_b_length)
    return coplanar_shared_vertex_intersects_beyond(a, b, normal_a,
                                                     tolerance) &&
           !detail::certified_shared_vertex_separation(common, a[1], a[2],
                                                      b[1], b[2]);

  const Vec3 direction = (1.0 / normal_cross_length) * normal_cross;
  const LineTriangleInterval interval_a = line_triangle_interval_at_vertex(
      common, a[1], a[2], direction);
  const LineTriangleInterval interval_b = line_triangle_interval_at_vertex(
      common, b[1], b[2], direction);
  // An ill-conditioned line reconstruction (invalid interval) used to count
  // as an intersection. A certified separation now overrules it, and the
  // tolerance test below; a certificate never adds an intersection.
  if (!interval_a.valid || !interval_b.valid)
    return !detail::certified_shared_vertex_separation(common, a[1], a[2],
                                                       b[1], b[2]);
  const double lower = std::max(interval_a.lower, interval_b.lower);
  const double upper = std::min(interval_a.upper, interval_b.upper);
  return lower <= upper && (lower < -tolerance || upper > tolerance) &&
         !detail::certified_shared_vertex_separation(common, a[1], a[2],
                                                    b[1], b[2]);
}

struct SelfIntersectionAudit {
  std::size_t count{0U};
  std::size_t first{std::numeric_limits<std::size_t>::max()};
  std::size_t second{std::numeric_limits<std::size_t>::max()};
};

// `pairs`, when given, receives every intersecting (lower, higher) triangle
// index pair in ascending order.  The count and first pair are unchanged.
// active (optional, one flag per triangle): only pairs with at least one
// active triangle are tested.  When every intersecting pair has one (the
// remesh passes: the mesh before a pass is intersection-free and only the
// active triangles changed), the result is the full audit's.
[[nodiscard]] SelfIntersectionAudit detect_self_intersections(
    const SurfaceMesh &mesh, double grid_spacing,
    std::size_t &scratch_bytes, TitanExecutor *executor = nullptr,
    std::vector<std::array<std::uint32_t, 2>> *pairs = nullptr,
    const std::vector<std::uint8_t> *active = nullptr) {
  if (pairs != nullptr)
    pairs->clear();
  if (mesh.triangles.size() < 2U)
    return {};
  SelfIntersectionAudit result;
  Vec3 origin{std::numeric_limits<double>::infinity(),
              std::numeric_limits<double>::infinity(),
              std::numeric_limits<double>::infinity()};
  std::vector<TriangleBounds> bounds;
  bounds.reserve(mesh.triangles.size());
  double extent_sum = 0.0;
  for (const Triangle &triangle : mesh.triangles) {
    bounds.push_back(triangle_bounds(mesh, triangle));
    const TriangleBounds &box = bounds.back();
    origin.x = std::min(origin.x, box.lower.x);
    origin.y = std::min(origin.y, box.lower.y);
    origin.z = std::min(origin.z, box.lower.z);
    extent_sum += std::max({box.upper.x - box.lower.x, box.upper.y - box.lower.y,
                            box.upper.z - box.lower.z});
  }
  // Two boxes that overlap share a cell whatever the cell size, so the size
  // changes only the cost.  Two grid spacings suit a mesh whose edges are
  // about one spacing; a remesh graded towards charges has much smaller
  // triangles, which then crowd each cell, so the size follows the mean
  // triangle extent, within a quarter of the old size and the old size.
  const double mean_extent =
      extent_sum / static_cast<double>(mesh.triangles.size());
  const double cell_size =
      std::max(1.0e-6, std::clamp(2.0 * mean_extent, 0.5 * grid_spacing,
                                  2.0 * grid_spacing));
  const double tolerance = std::max(
      1.0e-12 * std::max(1.0, grid_spacing),
      64.0 * std::numeric_limits<double>::epsilon() *
          std::max({1.0, std::abs(origin.x), std::abs(origin.y),
                    std::abs(origin.z)}));
  const auto cell_coordinate = [&](double value, double lower) {
    return static_cast<std::int64_t>(
        std::floor((value - lower) / cell_size));
  };
  std::unordered_map<WeldCellKey, std::vector<std::uint32_t>,
                     WeldCellKeyHash>
      cells;
  cells.reserve(mesh.triangles.size());
  std::vector<std::array<WeldCellKey, 2>> cell_ranges;
  cell_ranges.reserve(mesh.triangles.size());
  for (std::size_t index = 0U; index < mesh.triangles.size(); ++index) {
    const TriangleBounds &box = bounds[index];
    const WeldCellKey first{cell_coordinate(box.lower.x, origin.x),
                            cell_coordinate(box.lower.y, origin.y),
                            cell_coordinate(box.lower.z, origin.z)};
    const WeldCellKey last{cell_coordinate(box.upper.x, origin.x),
                           cell_coordinate(box.upper.y, origin.y),
                           cell_coordinate(box.upper.z, origin.z)};
    cell_ranges.push_back({first, last});
    if (active != nullptr && !(*active)[index])
      continue;
    for (std::int64_t z = first.z; z <= last.z; ++z) {
      for (std::int64_t y = first.y; y <= last.y; ++y) {
        for (std::int64_t x = first.x; x <= last.x; ++x)
          cells[{x, y, z}].push_back(static_cast<std::uint32_t>(index));
      }
    }
  }
  struct IntersectionWorker {
    SelfIntersectionAudit audit;
    std::vector<std::uint32_t> candidates;
    std::vector<std::uint64_t> candidate_bits;
    std::vector<std::array<std::uint32_t, 2>> pairs;
  };
  const std::size_t candidate_bit_words =
      mesh.triangles.size() / 64U +
      (mesh.triangles.size() % 64U == 0U ? 0U : 1U);
  const std::size_t candidate_bit_bytes =
      candidate_bit_words * sizeof(std::uint64_t);
  constexpr std::size_t kCandidateBitBudgetBytes = 64U * 1024U * 1024U;
  const std::size_t memory_limited_workers =
      std::max<std::size_t>(
          1U, candidate_bit_bytes == 0U
                  ? 1U
                  : kCandidateBitBudgetBytes / candidate_bit_bytes);
  const std::size_t workers =
      executor == nullptr
          ? 1U
          : std::min({executor->capacity(), mesh.triangles.size(),
                      memory_limited_workers});
  std::vector<IntersectionWorker> per_worker(workers);
  for (IntersectionWorker &worker : per_worker) {
    worker.candidates.reserve(64U);
    worker.candidate_bits.assign(candidate_bit_words, std::uint64_t{0});
  }
  const auto &read_cells = cells;
  const auto inspect_range = [&](std::size_t worker_index,
                                 std::size_t begin, std::size_t end) {
    IntersectionWorker &worker = per_worker[worker_index];
    for (std::size_t index = begin; index < end; ++index) {
      const WeldCellKey &first = cell_ranges[index][0];
      const WeldCellKey &last = cell_ranges[index][1];
      worker.candidates.clear();
      for (std::int64_t z = first.z; z <= last.z; ++z) {
        for (std::int64_t y = first.y; y <= last.y; ++y) {
          for (std::int64_t x = first.x; x <= last.x; ++x) {
            const auto found = read_cells.find({x, y, z});
            if (found == read_cells.end())
              continue;
            for (const std::uint32_t other : found->second) {
              if (active == nullptr ? other >= index
                                    : (other == index ||
                                       ((*active)[index] && other > index)))
                continue;
              const std::size_t word = static_cast<std::size_t>(other) / 64U;
              const std::uint64_t bit =
                  std::uint64_t{1} << (static_cast<unsigned>(other) & 63U);
              if ((worker.candidate_bits[word] & bit) != 0U)
                continue;
              worker.candidate_bits[word] |= bit;
              worker.candidates.push_back(other);
            }
          }
        }
      }
      for (const std::uint32_t other : worker.candidates) {
        const std::size_t word = static_cast<std::size_t>(other) / 64U;
        const std::uint64_t bit =
            std::uint64_t{1} << (static_cast<unsigned>(other) & 63U);
        worker.candidate_bits[word] &= ~bit;
        // The pair as the full scan takes it: hi the larger index.
        const std::size_t hi = std::max(index, static_cast<std::size_t>(other));
        const std::size_t lo = std::min(index, static_cast<std::size_t>(other));
        if (!bounds_overlap(bounds[hi], bounds[lo], tolerance))
          continue;
        const SharedTriangleSimplex shared = shared_triangle_simplex(
            mesh.triangles[hi], mesh.triangles[lo]);
        if (!shared_simplex_intersects_beyond(
                mesh, mesh.triangles[hi], mesh.triangles[lo], shared,
                tolerance))
          continue;
        if (worker.audit.count == 0U || hi < worker.audit.second ||
            (hi == worker.audit.second && lo < worker.audit.first)) {
          worker.audit.first = lo;
          worker.audit.second = hi;
        }
        ++worker.audit.count;
        if (pairs != nullptr)
          worker.pairs.push_back({static_cast<std::uint32_t>(lo),
                                  static_cast<std::uint32_t>(hi)});
      }
    }
  };
  if (executor != nullptr && workers > 1U) {
    executor->parallel_for(mesh.triangles.size(), workers, 64U,
                           inspect_range);
  } else {
    inspect_range(0U, 0U, mesh.triangles.size());
  }
  std::size_t worker_scratch_bytes = 0U;
  for (const IntersectionWorker &worker : per_worker) {
    result.count += worker.audit.count;
    if (worker.audit.count != 0U &&
        (result.first == std::numeric_limits<std::size_t>::max() ||
         worker.audit.second < result.second ||
         (worker.audit.second == result.second &&
          worker.audit.first < result.first))) {
      result.first = worker.audit.first;
      result.second = worker.audit.second;
    }
    worker_scratch_bytes +=
        worker.candidates.capacity() * sizeof(std::uint32_t) +
        worker.candidate_bits.capacity() * sizeof(std::uint64_t);
    if (pairs != nullptr)
      pairs->insert(pairs->end(), worker.pairs.begin(), worker.pairs.end());
  }
  if (pairs != nullptr)
    std::sort(pairs->begin(), pairs->end());
  std::size_t nested_cell_capacity = 0U;
  for (const auto &entry : cells)
    nested_cell_capacity += entry.second.capacity();
  const std::size_t cell_scratch_bytes =
      cells.bucket_count() * sizeof(void *) +
      cells.size() * sizeof(decltype(cells)::value_type) +
      nested_cell_capacity * sizeof(std::uint32_t);
  scratch_bytes = std::max(
      scratch_bytes,
      bounds.capacity() * sizeof(TriangleBounds) +
          cell_ranges.capacity() * sizeof(std::array<WeldCellKey, 2>) +
          per_worker.capacity() * sizeof(IntersectionWorker) +
          worker_scratch_bytes + cell_scratch_bytes);
  return result;
}

#ifdef TITAN_TABI_SES_TEST_HOOKS
void emit_projection_star_triangulation_diagnostics(
    const SurfaceMesh &mesh, const analytic_ses::PatchComplex &patches,
    const analytic_ses::LocalPatchProjectionIndex &projection,
    double grid_spacing, const LocalPatchRepairMetrics &metrics,
    TitanExecutor *executor) {
  const char *enabled = std::getenv("TITAN_TABI_SES_DEBUG_PROJECTION");
  if (enabled == nullptr || enabled[0] != '1')
    return;

  constexpr double kOrientationTolerance = 1.0e-8;
  std::vector<std::size_t> inverted_faces;
  for (std::size_t face = 0U; face < mesh.triangles.size(); ++face) {
    const AnalyticFaceOrientation orientation =
        evaluate_analytic_face_orientation(mesh, mesh.triangles[face],
                                           patches, projection,
                                           grid_spacing, false);
    if (orientation.certified &&
        !(orientation.cosine > kOrientationTolerance))
      inverted_faces.push_back(face);
  }
  if (inverted_faces.empty()) {
    std::cerr << "ses_star_audit,status=no_inverted_faces\n";
    return;
  }

  std::array<std::size_t, 3> common = {
      mesh.triangles[inverted_faces.front()].v0,
      mesh.triangles[inverted_faces.front()].v1,
      mesh.triangles[inverted_faces.front()].v2};
  std::vector<std::size_t> common_vertices(common.begin(), common.end());
  for (const std::size_t face : inverted_faces) {
    const Triangle &triangle = mesh.triangles[face];
    common_vertices.erase(
        std::remove_if(common_vertices.begin(), common_vertices.end(),
                       [&](std::size_t vertex) {
                         return vertex != triangle.v0 &&
                                vertex != triangle.v1 &&
                                vertex != triangle.v2;
                       }),
        common_vertices.end());
  }
  if (common_vertices.size() != 1U) {
    std::cerr << "ses_star_audit,status=no_unique_shared_vertex,inverted="
              << inverted_faces.size() << ",common="
              << common_vertices.size() << '\n';
    return;
  }
  const std::size_t center = common_vertices.front();
  const CompactVertexIncidence incidence =
      build_compact_vertex_incidence(mesh);
  if (incidence.offsets.size() != mesh.vertices.size() + 1U) {
    std::cerr << "ses_star_audit,status=invalid_incidence\n";
    return;
  }
  std::vector<std::size_t> star_faces;
  for (std::size_t position = incidence.offsets[center];
       position < incidence.offsets[center + 1U]; ++position)
    star_faces.push_back(incidence.triangles[position]);
  std::sort(star_faces.begin(), star_faces.end());
  const auto in_star = [&](std::size_t face) {
    return std::binary_search(star_faces.begin(), star_faces.end(), face);
  };
  if (!std::all_of(inverted_faces.begin(), inverted_faces.end(), in_star)) {
    std::cerr << "ses_star_audit,status=inverted_outside_star,center="
              << center << '\n';
    return;
  }

  std::vector<std::pair<std::size_t, std::size_t>> directed_boundary;
  directed_boundary.reserve(star_faces.size());
  bool star_well_formed = true;
  for (const std::size_t face : star_faces) {
    if (face >= mesh.triangles.size()) {
      star_well_formed = false;
      break;
    }
    const Triangle &triangle = mesh.triangles[face];
    if (triangle.v0 == center)
      directed_boundary.emplace_back(triangle.v1, triangle.v2);
    else if (triangle.v1 == center)
      directed_boundary.emplace_back(triangle.v2, triangle.v0);
    else if (triangle.v2 == center)
      directed_boundary.emplace_back(triangle.v0, triangle.v1);
    else {
      star_well_formed = false;
      break;
    }
  }
  for (std::size_t first = 0U; first < directed_boundary.size(); ++first) {
    if (directed_boundary[first].first == center ||
        directed_boundary[first].second == center ||
        directed_boundary[first].first == directed_boundary[first].second) {
      star_well_formed = false;
      break;
    }
    for (std::size_t second = first + 1U;
         second < directed_boundary.size(); ++second) {
      if (directed_boundary[first].first ==
              directed_boundary[second].first ||
          directed_boundary[first].second ==
              directed_boundary[second].second)
        star_well_formed = false;
    }
  }
  if (!star_well_formed || directed_boundary.size() < 3U ||
      directed_boundary.size() > 8U) {
    std::cerr << "ses_star_audit,status=not_bounded_simple_disk,center="
              << center << ",star_faces=" << star_faces.size() << '\n';
    return;
  }

  const std::size_t cycle_start =
      std::min_element(directed_boundary.begin(), directed_boundary.end(),
                       [](const auto &left, const auto &right) {
                         return left.first < right.first;
                       })
          ->first;
  std::vector<std::size_t> boundary;
  boundary.reserve(directed_boundary.size());
  std::size_t current = cycle_start;
  for (std::size_t step = 0U; step < directed_boundary.size(); ++step) {
    boundary.push_back(current);
    const auto found = std::find_if(
        directed_boundary.begin(), directed_boundary.end(),
        [&](const auto &edge) { return edge.first == current; });
    if (found == directed_boundary.end()) {
      star_well_formed = false;
      break;
    }
    current = found->second;
  }
  if (!star_well_formed || current != cycle_start ||
      std::unordered_set<std::size_t>(boundary.begin(), boundary.end())
              .size() != boundary.size()) {
    std::cerr << "ses_star_audit,status=boundary_not_simple,center="
              << center << '\n';
    return;
  }

  const std::vector<CompactOwnedEdge> owned_edges =
      build_compact_owned_edges(mesh);
  const auto edge_owner_faces = [&](std::size_t first,
                                    std::size_t second) {
    const std::pair<std::uint32_t, std::uint32_t> key{
        static_cast<std::uint32_t>(std::min(first, second)),
        static_cast<std::uint32_t>(std::max(first, second))};
    const auto begin = std::lower_bound(
        owned_edges.begin(), owned_edges.end(), key,
        [](const CompactOwnedEdge &edge,
           const std::pair<std::uint32_t, std::uint32_t> &value) {
          return std::pair{edge.lo, edge.hi} < value;
        });
    std::vector<std::size_t> owners;
    for (auto entry = begin;
         entry != owned_edges.end() && entry->lo == key.first &&
         entry->hi == key.second;
         ++entry)
      owners.push_back(entry->triangle);
    return owners;
  };
  for (std::size_t edge = 0U; edge < boundary.size(); ++edge) {
    const std::vector<std::size_t> owners = edge_owner_faces(
        boundary[edge], boundary[(edge + 1U) % boundary.size()]);
    const std::size_t star_owners =
        static_cast<std::size_t>(std::count_if(
            owners.begin(), owners.end(), in_star));
    if (owners.size() != 2U || star_owners != 1U)
      star_well_formed = false;
    const std::vector<std::size_t> radial_owners =
        edge_owner_faces(center, boundary[edge]);
    if (radial_owners.size() != 2U ||
        !std::all_of(radial_owners.begin(), radial_owners.end(), in_star))
      star_well_formed = false;
  }
  if (!star_well_formed) {
    std::cerr << "ses_star_audit,status=star_not_oriented_disk,center="
              << center << '\n';
    return;
  }

  bool boundary_vertices_certified = true;
  for (const std::size_t vertex : boundary) {
    const Vertex &value = mesh.vertices[vertex];
    const analytic_ses::LocalProjectionHit hit = nearest_local_patch(
        patches, projection, value.atom_index,
        {value.x, value.y, value.z}, 2.0 * grid_spacing);
    boundary_vertices_certified =
        boundary_vertices_certified &&
        certified_finite_projection_hit(hit) &&
        hit.distance <=
            1.0e-8 * std::max(1.0, grid_spacing);
  }

  using ReplacementFace = std::array<std::size_t, 3>;
  using Triangulation = std::vector<ReplacementFace>;
  const auto enumerate_range =
      [&](auto &&self, std::size_t begin,
          std::size_t end) -> std::vector<Triangulation> {
    if (end <= begin + 1U)
      return {Triangulation{}};
    std::vector<Triangulation> result;
    for (std::size_t split = begin + 1U; split < end; ++split) {
      const std::vector<Triangulation> left =
          self(self, begin, split);
      const std::vector<Triangulation> right =
          self(self, split, end);
      for (const Triangulation &left_faces : left) {
        for (const Triangulation &right_faces : right) {
          Triangulation faces = left_faces;
          faces.insert(faces.end(), right_faces.begin(),
                       right_faces.end());
          faces.push_back(
              {boundary[begin], boundary[split], boundary[end]});
          result.push_back(std::move(faces));
        }
      }
    }
    return result;
  };
  std::vector<Triangulation> triangulations =
      enumerate_range(enumerate_range, 0U, boundary.size() - 1U);
  const auto canonical_key = [](const Triangulation &faces) {
    Triangulation key = faces;
    for (ReplacementFace &face : key)
      std::sort(face.begin(), face.end());
    std::sort(key.begin(), key.end());
    return key;
  };
  std::sort(triangulations.begin(), triangulations.end(),
            [&](const Triangulation &left, const Triangulation &right) {
              return canonical_key(left) < canonical_key(right);
            });

  std::cerr << "ses_star_audit,status=enumerating,center=" << center
            << ",star_faces=";
  for (const std::size_t face : star_faces)
    std::cerr << face << ';';
  std::cerr << ",boundary=";
  for (const std::size_t vertex : boundary)
    std::cerr << vertex << ';';
  std::cerr << ",boundary_certified="
            << (boundary_vertices_certified ? 1 : 0)
            << ",triangulations=" << triangulations.size() << '\n';

  const auto boundary_edge = [&](std::size_t first, std::size_t second) {
    for (std::size_t edge = 0U; edge < boundary.size(); ++edge) {
      const std::size_t a = boundary[edge];
      const std::size_t b = boundary[(edge + 1U) % boundary.size()];
      if ((a == first && b == second) ||
          (a == second && b == first))
        return true;
    }
    return false;
  };
  const auto canonical_face = [](const Triangle &triangle) {
    ReplacementFace key{triangle.v0, triangle.v1, triangle.v2};
    std::sort(key.begin(), key.end());
    return key;
  };

  for (std::size_t candidate_index = 0U;
       candidate_index < triangulations.size(); ++candidate_index) {
    const Triangulation &replacement = triangulations[candidate_index];
    bool outside_diagonal = false;
    bool duplicate_face = false;
    bool geometry_valid = true;
    bool analytic_certified = boundary_vertices_certified;
    double minimum_area = std::numeric_limits<double>::infinity();
    double minimum_quality = std::numeric_limits<double>::infinity();
    double minimum_cosine = 1.0;
    std::vector<Triangle> replacement_triangles;
    replacement_triangles.reserve(replacement.size());
    for (const ReplacementFace &face : replacement) {
      Triangle triangle;
      triangle.v0 = face[0];
      triangle.v1 = face[1];
      triangle.v2 = face[2];
      triangle.atom_index =
          std::min({mesh.vertices[face[0]].atom_index,
                    mesh.vertices[face[1]].atom_index,
                    mesh.vertices[face[2]].atom_index});
      replacement_triangles.push_back(triangle);
      const Vec3 a = mesh_vertex(mesh, face[0]);
      const Vec3 b = mesh_vertex(mesh, face[1]);
      const Vec3 c = mesh_vertex(mesh, face[2]);
      const double area = 0.5 * norm(cross(b - a, c - a));
      const double quality = normalized_triangle_quality(a, b, c);
      minimum_area = std::min(minimum_area, area);
      minimum_quality = std::min(minimum_quality, quality);
      geometry_valid = geometry_valid && std::isfinite(area) &&
                       std::isfinite(quality) && area > 1.0e-16 &&
                       quality > 0.0;
      const AnalyticFaceOrientation orientation =
          evaluate_analytic_face_orientation(
              mesh, triangle, patches, projection, grid_spacing, false);
      analytic_certified =
          analytic_certified && orientation.certified &&
          orientation.cosine > kOrientationTolerance;
      minimum_cosine = std::min(minimum_cosine, orientation.cosine);

      for (const EdgeKey edge :
           {edge_key(face[0], face[1]), edge_key(face[1], face[2]),
            edge_key(face[2], face[0])}) {
        if (boundary_edge(edge.lo, edge.hi))
          continue;
        const std::vector<std::size_t> owners =
            edge_owner_faces(edge.lo, edge.hi);
        if (std::any_of(owners.begin(), owners.end(),
                        [&](std::size_t owner) {
                          return !in_star(owner);
                        }))
          outside_diagonal = true;
      }
      const ReplacementFace key = canonical_face(triangle);
      for (std::size_t old_face = 0U;
           old_face < mesh.triangles.size(); ++old_face) {
        if (!in_star(old_face) &&
            canonical_face(mesh.triangles[old_face]) == key)
          duplicate_face = true;
      }
    }

    bool topology_valid = false;
    std::size_t self_intersections =
        std::numeric_limits<std::size_t>::max();
    if (!outside_diagonal && !duplicate_face && geometry_valid &&
        analytic_certified) {
      SurfaceMesh candidate;
      candidate.vertices = mesh.vertices;
      candidate.triangles.reserve(mesh.triangles.size() -
                                  star_faces.size() +
                                  replacement_triangles.size());
      candidate.panels.reserve(candidate.triangles.capacity());
      for (std::size_t face = 0U; face < mesh.triangles.size(); ++face) {
        if (in_star(face))
          continue;
        candidate.triangles.push_back(mesh.triangles[face]);
        candidate.panels.push_back(mesh.panels[face]);
      }
      for (const Triangle &triangle : replacement_triangles) {
        candidate.triangles.push_back(triangle);
        candidate.panels.emplace_back();
      }
      candidate.vertices.erase(candidate.vertices.begin() +
                               static_cast<std::ptrdiff_t>(center));
      for (Triangle &triangle : candidate.triangles) {
        if (triangle.v0 > center)
          --triangle.v0;
        if (triangle.v1 > center)
          --triangle.v1;
        if (triangle.v2 > center)
          --triangle.v2;
      }
      for (std::size_t face = 0U; face < candidate.triangles.size(); ++face)
        update_analytic_triangle_geometry_and_owner(candidate, face);
      topology_valid =
          build_closed_surface_trace_spaces(candidate).quality.valid;
      if (topology_valid) {
        std::size_t scratch_bytes = 0U;
        self_intersections =
            detect_self_intersections(candidate, grid_spacing,
                                      scratch_bytes, executor)
                .count;
      }
    }
    const std::size_t failures_after =
        metrics.analytic_inverted_faces >= inverted_faces.size()
            ? metrics.analytic_inverted_faces - inverted_faces.size() +
                  (analytic_certified ? 0U : 1U)
            : std::numeric_limits<std::size_t>::max();
    const bool accepted =
        inverted_faces.size() == metrics.analytic_inverted_faces &&
        metrics.analytic_orientation_uncertified_faces == 0U &&
        replacement.size() + 2U == boundary.size() &&
        !outside_diagonal && !duplicate_face && geometry_valid &&
        analytic_certified && topology_valid &&
        self_intersections == 0U && failures_after == 0U;
    std::cerr << "ses_star_audit,candidate=" << candidate_index
              << ",faces=";
    for (const ReplacementFace &face : replacement)
      std::cerr << face[0] << ';' << face[1] << ';' << face[2] << '|';
    std::cerr << ",outside_diagonal=" << (outside_diagonal ? 1 : 0)
              << ",duplicate_face=" << (duplicate_face ? 1 : 0)
              << ",geometry_valid=" << (geometry_valid ? 1 : 0)
              << ",analytic_certified=" << (analytic_certified ? 1 : 0)
              << ",min_cosine=" << minimum_cosine
              << ",min_area=" << minimum_area
              << ",min_quality=" << minimum_quality
              << ",topology_valid=" << (topology_valid ? 1 : 0)
              << ",self_intersections=" << self_intersections
              << ",failures_after=" << failures_after
              << ",accepted=" << (accepted ? 1 : 0) << '\n';
  }
}
#endif

struct MeshShapeAudit {
  std::size_t poor_triangles{0};
  double minimum_quality{0.0};
  double maximum_edge_ratio{0.0};
};

[[nodiscard]] MeshShapeAudit measure_mesh_shape(const SurfaceMesh &mesh) {
  MeshShapeAudit audit;
  if (mesh.triangles.empty())
    return audit;
  audit.minimum_quality = std::numeric_limits<double>::infinity();
  for (const Triangle &triangle : mesh.triangles) {
    const Vec3 a = mesh_vertex(mesh, triangle.v0);
    const Vec3 b = mesh_vertex(mesh, triangle.v1);
    const Vec3 c = mesh_vertex(mesh, triangle.v2);
    const double quality = normalized_triangle_quality(a, b, c);
    const double ab = norm(b - a);
    const double bc = norm(c - b);
    const double ca = norm(a - c);
    const double shortest = std::min({ab, bc, ca});
    const double ratio = shortest > 1.0e-14
                             ? std::max({ab, bc, ca}) / shortest
                             : std::numeric_limits<double>::infinity();
    audit.minimum_quality = std::min(audit.minimum_quality, quality);
    audit.maximum_edge_ratio = std::max(audit.maximum_edge_ratio, ratio);
    if (quality < 0.02 || ratio > 50.0)
      ++audit.poor_triangles;
  }
  return audit;
}

void smooth_analytic_slivers_stable_local(
    SurfaceMesh &mesh, const analytic_ses::PatchComplex &patches,
    const analytic_ses::LocalPatchProjectionIndex &projection,
    double grid_spacing, LocalPatchRepairMetrics &metrics) {
  const CompactVertexAdjacency adjacency =
      build_compact_vertex_adjacency(mesh);
  const CompactVertexIncidence incidence =
      build_compact_vertex_incidence(mesh);
  if (adjacency.offsets.size() != mesh.vertices.size() + 1U ||
      incidence.offsets.size() != mesh.vertices.size() + 1U)
    return;
  std::vector<Vec3> normal_sums(mesh.vertices.size());
  std::vector<Vec3> reference_normals(mesh.triangles.size());
  std::vector<std::uint8_t> poor(mesh.vertices.size(), 0U);
  struct VertexMove {
    std::size_t index{0};
    Vertex before;
    Vertex after;
  };
  std::vector<VertexMove> moves;
  for (int round = 0; round < 4; ++round) {
    std::fill(normal_sums.begin(), normal_sums.end(), Vec3{});
    std::fill(poor.begin(), poor.end(), std::uint8_t{0});
    for (std::size_t triangle_index = 0U;
         triangle_index < mesh.triangles.size(); ++triangle_index) {
      const Triangle &triangle = mesh.triangles[triangle_index];
      const Vec3 a = mesh_vertex(mesh, triangle.v0);
      const Vec3 b = mesh_vertex(mesh, triangle.v1);
      const Vec3 c = mesh_vertex(mesh, triangle.v2);
      const Vec3 face_normal = cross(b - a, c - a);
      reference_normals[triangle_index] = face_normal;
      normal_sums[triangle.v0] = normal_sums[triangle.v0] + face_normal;
      normal_sums[triangle.v1] = normal_sums[triangle.v1] + face_normal;
      normal_sums[triangle.v2] = normal_sums[triangle.v2] + face_normal;
      const double quality = normalized_triangle_quality(a, b, c);
      const double ab = norm(b - a);
      const double bc = norm(c - b);
      const double ca = norm(a - c);
      const double shortest = std::min({ab, bc, ca});
      const double ratio = shortest > 1.0e-14
                               ? std::max({ab, bc, ca}) / shortest
                               : std::numeric_limits<double>::infinity();
      if (quality < 0.02 || ratio > 50.0) {
        poor[triangle.v0] = 1U;
        poor[triangle.v1] = 1U;
        poor[triangle.v2] = 1U;
      }
    }
    moves.clear();
    for (std::size_t index = 0; index < mesh.vertices.size(); ++index) {
      const std::size_t begin = adjacency.offsets[index];
      const std::size_t end = adjacency.offsets[index + 1U];
      if (poor[index] == 0U || end - begin < 3U)
        continue;
      const Vec3 point = mesh_vertex(mesh, index);
      Vec3 average;
      for (std::size_t position = begin; position < end; ++position)
        average = average + mesh_vertex(mesh, adjacency.neighbors[position]);
      average = (1.0 / static_cast<double>(end - begin)) * average;
      const Vec3 normal = normalized(normal_sums[index]);
      Vec3 displacement = average - point;
      displacement = displacement - dot(displacement, normal) * normal;
      const double displacement_norm = norm(displacement);
      const double cap = 0.15 * grid_spacing;
      if (displacement_norm > cap)
        displacement = (cap / displacement_norm) * displacement;
      const Vec3 proposed = point + 0.25 * displacement;
      ++metrics.projection_queries;
      const double projection_cap =
          0.25 * grid_spacing + norm(proposed - point);
      const analytic_ses::LocalProjectionHit hit =
          nearest_local_patch(patches, projection,
                              mesh.vertices[index].atom_index, proposed,
                              projection_cap);
      const Vec3 projected{hit.point.x, hit.point.y, hit.point.z};
      const double projection_distance = norm(projected - point);
      if (!certified_finite_projection_hit(hit) ||
          hit.atom_index >= patches.atoms.size() ||
          !std::isfinite(projection_distance) ||
          projection_distance > 0.25 * grid_spacing)
        continue;
      ++metrics.projection_hits;
      Vertex after = mesh.vertices[index];
      after.x = hit.point.x;
      after.y = hit.point.y;
      after.z = hit.point.z;
      after.nx = hit.normal.x;
      after.ny = hit.normal.y;
      after.nz = hit.normal.z;
      after.atom_index = hit.atom_index;
      moves.push_back({index, mesh.vertices[index], after});
    }
    if (moves.empty())
      return;
    std::vector<std::uint8_t> affected_triangles(
        mesh.triangles.size(), 0U);
    std::size_t accepted_moves = 0U;
    std::size_t rejected_moves = 0U;
    std::size_t orientation_rejections = 0U;
    for (const VertexMove &move : moves) {
      const Vec3 projected{move.after.x, move.after.y, move.after.z};
      bool safe = true;
      for (std::size_t position = incidence.offsets[move.index];
           position < incidence.offsets[move.index + 1U]; ++position) {
        const std::size_t triangle_index =
            incidence.triangles[position];
        const Triangle &triangle = mesh.triangles[triangle_index];
        const Vec3 old_a = mesh_vertex(mesh, triangle.v0);
        const Vec3 old_b = mesh_vertex(mesh, triangle.v1);
        const Vec3 old_c = mesh_vertex(mesh, triangle.v2);
        const FaceMoveValidation validation = validate_face_move(
            triangle.v0 == move.index ? projected : old_a,
            triangle.v1 == move.index ? projected : old_b,
            triangle.v2 == move.index ? projected : old_c,
            reference_normals[triangle_index]);
        if (!validation.geometry_valid) {
          safe = false;
          break;
        }
        if (!validation.orientation_preserved) {
          safe = false;
          ++orientation_rejections;
          break;
        }
      }
      if (!safe) {
        ++rejected_moves;
        continue;
      }
      mesh.vertices[move.index] = move.after;
      ++accepted_moves;
      for (std::size_t position = incidence.offsets[move.index];
           position < incidence.offsets[move.index + 1U]; ++position)
        affected_triangles[incidence.triangles[position]] = 1U;
    }
    metrics.scratch_bytes = std::max(
        metrics.scratch_bytes,
        adjacency.retained_bytes() + incidence.retained_bytes() +
            normal_sums.capacity() * sizeof(Vec3) +
            reference_normals.capacity() * sizeof(Vec3) +
            poor.capacity() * sizeof(std::uint8_t) +
            moves.capacity() * sizeof(VertexMove) +
            affected_triangles.capacity() * sizeof(std::uint8_t));
    if (rejected_moves != 0U) {
      metrics.rejected_projection_vertices += rejected_moves;
      metrics.projection_orientation_reversals += orientation_rejections;
      ++metrics.rejected_projection_rounds;
    }
    if (accepted_moves == 0U)
      return;
    for (std::size_t triangle_index = 0U;
         triangle_index < affected_triangles.size(); ++triangle_index) {
      if (affected_triangles[triangle_index] != 0U)
        update_analytic_triangle_geometry_and_owner(mesh, triangle_index);
    }
    metrics.projected_vertices += accepted_moves;
  }
}

void smooth_analytic_slivers_local(
    SurfaceMesh &mesh, const analytic_ses::PatchComplex &patches,
    const analytic_ses::LocalPatchProjectionIndex &projection,
    double grid_spacing, LocalPatchRepairMetrics &metrics) {
  const CompactVertexAdjacency adjacency =
      build_compact_vertex_adjacency(mesh);
  const CompactVertexIncidence incidence =
      build_compact_vertex_incidence(mesh);
  if (adjacency.offsets.size() != mesh.vertices.size() + 1U ||
      incidence.offsets.size() != mesh.vertices.size() + 1U)
    return;
  metrics.scratch_bytes =
      std::max(metrics.scratch_bytes,
               adjacency.retained_bytes() + incidence.retained_bytes());
  std::vector<Vec3> normal_sums(mesh.vertices.size());
  std::vector<std::uint8_t> poor(mesh.vertices.size(), 0U);
  std::vector<std::uint8_t> selected(mesh.vertices.size(), 0U);
  std::vector<double> worst_quality(
      mesh.vertices.size(), std::numeric_limits<double>::infinity());
  std::vector<std::size_t> candidate_vertices;
  struct VertexMove {
    std::size_t index{0};
    Vertex before;
    Vertex after;
  };
  std::vector<VertexMove> moves;
  double total_area = 0.0;
  for (const Panel &panel : mesh.panels)
    total_area += panel.area;
  for (int round = 0; round < 4; ++round) {
    const double area_before_round = total_area;
    std::fill(normal_sums.begin(), normal_sums.end(), Vec3{});
    std::fill(poor.begin(), poor.end(), std::uint8_t{0});
    std::fill(selected.begin(), selected.end(), std::uint8_t{0});
    std::fill(worst_quality.begin(), worst_quality.end(),
              std::numeric_limits<double>::infinity());
    for (const Triangle &triangle : mesh.triangles) {
      const Vec3 a = mesh_vertex(mesh, triangle.v0);
      const Vec3 b = mesh_vertex(mesh, triangle.v1);
      const Vec3 c = mesh_vertex(mesh, triangle.v2);
      const Vec3 face_normal = cross(b - a, c - a);
      normal_sums[triangle.v0] = normal_sums[triangle.v0] + face_normal;
      normal_sums[triangle.v1] = normal_sums[triangle.v1] + face_normal;
      normal_sums[triangle.v2] = normal_sums[triangle.v2] + face_normal;
      const double quality = normalized_triangle_quality(a, b, c);
      const double ab = norm(b - a);
      const double bc = norm(c - b);
      const double ca = norm(a - c);
      const double shortest = std::min({ab, bc, ca});
      const double ratio = shortest > 1.0e-14
                               ? std::max({ab, bc, ca}) / shortest
                               : std::numeric_limits<double>::infinity();
      if (quality < 0.02 || ratio > 50.0) {
        poor[triangle.v0] = 1U;
        poor[triangle.v1] = 1U;
        poor[triangle.v2] = 1U;
        worst_quality[triangle.v0] =
            std::min(worst_quality[triangle.v0], quality);
        worst_quality[triangle.v1] =
            std::min(worst_quality[triangle.v1], quality);
        worst_quality[triangle.v2] =
            std::min(worst_quality[triangle.v2], quality);
      }
    }

    moves.clear();
    candidate_vertices.clear();
    for (std::size_t index = 0; index < mesh.vertices.size(); ++index) {
      if (poor[index] != 0U)
        candidate_vertices.push_back(index);
    }
    std::sort(candidate_vertices.begin(), candidate_vertices.end(),
              [&](std::size_t first, std::size_t second) {
                const double first_quality =
                    stable_repair_metric(worst_quality[first]);
                const double second_quality =
                    stable_repair_metric(worst_quality[second]);
                return first_quality != second_quality
                           ? first_quality < second_quality
                           : first < second;
              });
    // A few worst-first, mutually independent moves capture the useful
    // curvature correction without coherently shifting a large node-patch
    // operator.  The full energy gates showed that hundreds of individually
    // valid moves can still be solver-significant.
    constexpr std::size_t kMaximumMovesPerRound = 1U;
    bool guard_rejected_this_round = false;
    for (const std::size_t index : candidate_vertices) {
      if (moves.size() >= kMaximumMovesPerRound)
        break;
      const std::size_t begin = adjacency.offsets[index];
      const std::size_t end = adjacency.offsets[index + 1U];
      if (poor[index] == 0U || end - begin < 3U)
        continue;
      bool adjacent_to_selected = false;
      for (std::size_t position = begin; position < end; ++position) {
        if (selected[adjacency.neighbors[position]] != 0U) {
          adjacent_to_selected = true;
          break;
        }
      }
      if (adjacent_to_selected)
        continue;
      const Vec3 point = mesh_vertex(mesh, index);
      Vec3 average;
      for (std::size_t position = begin; position < end; ++position)
        average = average +
                  mesh_vertex(mesh, adjacency.neighbors[position]);
      average = (1.0 / static_cast<double>(end - begin)) * average;
      const Vec3 vertex_normal = normalized(
          {mesh.vertices[index].nx, mesh.vertices[index].ny,
           mesh.vertices[index].nz});
      const Vec3 normal = normalized(normal_sums[index] + vertex_normal);
      double normal_variation = 0.0;
      for (std::size_t position = begin; position < end; ++position) {
        const Vertex &neighbor = mesh.vertices[adjacency.neighbors[position]];
        const Vec3 neighbor_normal =
            normalized({neighbor.nx, neighbor.ny, neighbor.nz});
        normal_variation =
            std::max(normal_variation,
                     std::clamp(1.0 - dot(vertex_normal, neighbor_normal),
                                0.0, 2.0));
      }
      const double curvature_scale = 1.0 / (1.0 + 4.0 * normal_variation);
      Vec3 displacement = average - point;
      displacement = displacement - dot(displacement, normal) * normal;
      const double displacement_norm = norm(displacement);
      const double cap =
          0.15 * grid_spacing * std::max(0.2, curvature_scale);
      if (displacement_norm > cap)
        displacement = (cap / displacement_norm) * displacement;
      const Vec3 proposed = point + (0.25 * curvature_scale) * displacement;
      ++metrics.projection_queries;
      const double projection_cap =
          0.25 * grid_spacing + norm(proposed - point);
      const analytic_ses::LocalProjectionHit hit =
          nearest_local_patch(patches, projection,
                              mesh.vertices[index].atom_index, proposed,
                              projection_cap);
      const Vec3 projected{hit.point.x, hit.point.y, hit.point.z};
      const double projection_distance = norm(projected - point);
      if (!certified_finite_projection_hit(hit) ||
          hit.atom_index >= patches.atoms.size() ||
          !std::isfinite(projection_distance) ||
          projection_distance > 0.25 * grid_spacing)
        continue;
      Vertex after = mesh.vertices[index];
      after.x = hit.point.x;
      after.y = hit.point.y;
      after.z = hit.point.z;
      after.nx = hit.normal.x;
      after.ny = hit.normal.y;
      after.nz = hit.normal.z;
      after.atom_index = hit.atom_index;
      double old_worst_quality = std::numeric_limits<double>::infinity();
      double new_worst_quality = std::numeric_limits<double>::infinity();
      bool geometry_valid = true;
      bool orientation_preserved = true;
      for (std::size_t position = incidence.offsets[index];
           position < incidence.offsets[index + 1U]; ++position) {
        const Triangle &triangle =
            mesh.triangles[incidence.triangles[position]];
        const Vec3 old_a = mesh_vertex(mesh, triangle.v0);
        const Vec3 old_b = mesh_vertex(mesh, triangle.v1);
        const Vec3 old_c = mesh_vertex(mesh, triangle.v2);
        const Vec3 new_a = triangle.v0 == index ? projected : old_a;
        const Vec3 new_b = triangle.v1 == index ? projected : old_b;
        const Vec3 new_c = triangle.v2 == index ? projected : old_c;
        old_worst_quality =
            std::min(old_worst_quality,
                     normalized_triangle_quality(old_a, old_b, old_c));
        new_worst_quality =
            std::min(new_worst_quality,
                     normalized_triangle_quality(new_a, new_b, new_c));
        const FaceMoveValidation validation = validate_face_move(
            new_a, new_b, new_c,
            cross(old_b - old_a, old_c - old_a));
        if (!validation.geometry_valid) {
          geometry_valid = false;
          orientation_preserved = false;
          break;
        }
        if (!validation.orientation_preserved) {
          orientation_preserved = false;
          break;
        }
      }
      if (!orientation_preserved) {
        ++metrics.rejected_projection_vertices;
        if (geometry_valid)
          ++metrics.projection_orientation_reversals;
        guard_rejected_this_round = true;
        continue;
      }
      if (
          !(stable_repair_metric(new_worst_quality) >
            stable_repair_metric(old_worst_quality * 1.001 + 1.0e-10)))
        continue;
      ++metrics.projection_hits;
      selected[index] = 1U;
      moves.push_back({index, mesh.vertices[index], after});
    }
    if (guard_rejected_this_round)
      ++metrics.rejected_projection_rounds;
    if (moves.empty())
      return;
    std::vector<std::uint32_t> affected_triangles;
    affected_triangles.reserve(moves.size() * 7U);
    for (const VertexMove &move : moves) {
      mesh.vertices[move.index] = move.after;
      for (std::size_t position = incidence.offsets[move.index];
           position < incidence.offsets[move.index + 1U]; ++position)
        affected_triangles.push_back(incidence.triangles[position]);
    }
    std::sort(affected_triangles.begin(), affected_triangles.end());
    affected_triangles.erase(
        std::unique(affected_triangles.begin(), affected_triangles.end()),
        affected_triangles.end());
    metrics.scratch_bytes = std::max(
        metrics.scratch_bytes,
        adjacency.retained_bytes() + incidence.retained_bytes() +
            normal_sums.capacity() * sizeof(Vec3) +
            poor.capacity() * sizeof(std::uint8_t) +
            selected.capacity() * sizeof(std::uint8_t) +
            worst_quality.capacity() * sizeof(double) +
            candidate_vertices.capacity() * sizeof(std::size_t) +
            moves.capacity() * sizeof(VertexMove) +
            affected_triangles.capacity() * sizeof(std::uint32_t));
    bool valid = true;
    for (const std::uint32_t triangle_index : affected_triangles) {
      const Triangle &triangle = mesh.triangles[triangle_index];
      const Vec3 a = mesh_vertex(mesh, triangle.v0);
      const Vec3 b = mesh_vertex(mesh, triangle.v1);
      const Vec3 c = mesh_vertex(mesh, triangle.v2);
      const double ab = norm(b - a);
      const double bc = norm(c - b);
      const double ca = norm(a - c);
      if (!(norm(cross(b - a, c - a)) > 1.0e-16) ||
          !(std::min({ab, bc, ca}) > 1.0e-14)) {
        valid = false;
        break;
      }
    }
    if (!valid) {
      for (const VertexMove &move : moves)
        mesh.vertices[move.index] = move.before;
      ++metrics.rejected_projection_rounds;
      return;
    }
    double old_affected_area = 0.0;
    for (const std::uint32_t triangle_index : affected_triangles)
      old_affected_area += mesh.panels[triangle_index].area;
    for (const std::uint32_t triangle_index : affected_triangles)
      update_analytic_triangle_geometry_and_owner(mesh, triangle_index);
    double new_affected_area = 0.0;
    for (const std::uint32_t triangle_index : affected_triangles)
      new_affected_area += mesh.panels[triangle_index].area;
    const double candidate_area =
        area_before_round - old_affected_area + new_affected_area;
    const double area_drift = std::abs(candidate_area - area_before_round);
    // Node-patch energies are more sensitive to coherent surface drift than
    // aggregate triangle-shape statistics suggest.  Bound every optional
    // curvature batch to 1e-6 relative area; a larger move is rolled back even
    // when it remains a closed manifold and improves local aspect ratios.
    const double area_tolerance =
        1.0e-6 * std::max(1.0, area_before_round);
    if (area_drift > area_tolerance) {
      for (const VertexMove &move : moves)
        mesh.vertices[move.index] = move.before;
      for (const std::uint32_t triangle_index : affected_triangles)
        update_analytic_triangle_geometry_and_owner(mesh, triangle_index);
      ++metrics.rejected_projection_rounds;
      return;
    }
    total_area = candidate_area;
    metrics.projected_vertices += moves.size();
  }
}

void repair_patch_ray_surface_local(
    SurfaceMesh &mesh, const analytic_ses::PatchComplex &patches,
    const analytic_ses::LocalPatchProjectionIndex &projection,
    double grid_spacing, LocalPatchRepairMetrics &metrics,
    TitanExecutor *executor) {
  const MeshShapeAudit shape_before = measure_mesh_shape(mesh);
  metrics.poor_triangles_before = shape_before.poor_triangles;
  metrics.triangle_quality_min_before = shape_before.minimum_quality;
  metrics.max_edge_ratio_before = shape_before.maximum_edge_ratio;
  // Per-stage mesh shape, so the stage that spends it can be named rather
  // than guessed.  Diagnostic only; production compiles none of this.
  const auto stage_shape = [&](const char *stage) {
#ifdef TITAN_TABI_SES_TEST_HOOKS
    const MeshShapeAudit now = measure_mesh_shape(mesh);
    std::cerr << "ses_stage_shape,stage=" << stage
              << ",panels=" << mesh.triangles.size()
              << ",max_edge_ratio=" << now.maximum_edge_ratio
              << ",min_quality=" << now.minimum_quality
              << ",poor=" << now.poor_triangles << std::endl;
#else
    (void)stage;
#endif
  };
  stage_shape("marching");
  double repair_ms = 0.0;
  bool topology_certified = false;
  const auto collapse_rounds = [&](int maximum_rounds,
                                   double threshold_scale) {
    for (int round = 0; round < maximum_rounds; ++round) {
      const auto stage_begin = std::chrono::steady_clock::now();
      SurfaceMesh candidate;
      bool candidate_topology_certified = false;
      const std::size_t collapsed =
          collapse_short_manifold_edges_compact(
              mesh, candidate, threshold_scale * grid_spacing,
              metrics.scratch_bytes, candidate_topology_certified);
      bool accepted = false;
      if (collapsed > 0U) {
        if (!candidate_topology_certified) {
          ++metrics.topology_fallback_audits;
          orient_mesh_consistently(candidate);
          candidate_topology_certified =
              build_closed_surface_trace_spaces(candidate).quality.valid;
        } else {
          ++metrics.topology_certificate_accepts;
        }
        if (candidate_topology_certified) {
          mesh.vertices = std::move(candidate.vertices);
          mesh.triangles = std::move(candidate.triangles);
          mesh.panels = std::move(candidate.panels);
          assign_all_analytic_triangle_owners(mesh);
          metrics.collapsed_edges += collapsed;
          topology_certified = true;
          accepted = true;
        }
      } else if (candidate_topology_certified) {
        topology_certified = true;
      }
      repair_ms += std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - stage_begin)
                       .count();
      if (collapsed == 0U || !accepted)
        break;
    }
  };
  const double initial_collapse_begin = repair_ms;
  collapse_rounds(8, 1.0e-2);
  metrics.initial_collapse_ms = repair_ms - initial_collapse_begin;
  stage_shape("initial_collapse");

  const auto flip_rounds = [&](int maximum_rounds) {
    std::vector<CompactEdgeFlipUndo> undo;
    for (int round = 0; round < maximum_rounds; ++round) {
      const auto stage_begin = std::chrono::steady_clock::now();
      const std::size_t flipped =
          flip_edges_for_quality_compact(mesh, metrics.scratch_bytes, undo);
      bool accepted = false;
      if (flipped > 0U) {
        const TraceSpaces audit = build_closed_surface_trace_spaces(mesh);
        if (audit.quality.valid) {
          metrics.flipped_edges += flipped;
          accepted = true;
        } else {
          for (auto entry = undo.rbegin(); entry != undo.rend(); ++entry) {
            mesh.triangles[entry->first_index] = entry->first_triangle;
            mesh.triangles[entry->second_index] = entry->second_triangle;
            mesh.panels[entry->first_index] = entry->first_panel;
            mesh.panels[entry->second_index] = entry->second_panel;
          }
        }
      }
      repair_ms += std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - stage_begin)
                       .count();
      if (flipped == 0U || !accepted)
        break;
    }
  };

  const auto projection_begin = std::chrono::steady_clock::now();
  metrics.projection_index_bytes = projection.retained_bytes();
  smooth_analytic_slivers_stable_local(mesh, patches, projection,
                                       grid_spacing, metrics);
  metrics.stable_sliver_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() -
                                 projection_begin)
                                 .count();
  metrics.projection_ms = metrics.stable_sliver_ms;
  stage_shape("stable_sliver_smooth");

  // Projection can create a new, strictly local short edge.  One cleanup
  // round is enough because projected vertices are Jacobi-updated from a
  // non-inverted mesh and displacement is capped at 0.25 grid cells.
  const double post_projection_collapse_begin = repair_ms;
  collapse_rounds(4, 1.0e-2);
  metrics.post_projection_collapse_ms =
      repair_ms - post_projection_collapse_begin;
  stage_shape("post_projection_collapse");
  const auto refinement_begin = std::chrono::steady_clock::now();
  const auto local_sliver_begin = std::chrono::steady_clock::now();
  smooth_analytic_slivers_local(mesh, patches, projection, grid_spacing,
                                metrics);
  metrics.local_sliver_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() -
                                local_sliver_begin)
                                .count();
  stage_shape("local_sliver_smooth");
  const auto timed_project_all = [&]() {
    const auto project_begin = std::chrono::steady_clock::now();
    const bool accepted = project_all_vertices_to_analytic_patches(
        mesh, patches, projection, grid_spacing, metrics, executor);
    metrics.full_projection_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - project_begin)
            .count();
    return accepted;
  };
  bool projected = timed_project_all();
  stage_shape("full_projection");
  if (projected) {
    const auto weld_begin = std::chrono::steady_clock::now();
    SurfaceMesh welded_candidate;
    const std::size_t welded = weld_near_coincident_vertices(
        mesh, welded_candidate, seam_weld_tolerance(mesh, grid_spacing),
        metrics.scratch_bytes);
    if (welded > 0U) {
      orient_mesh_consistently(welded_candidate);
      const TraceSpaces topology =
          build_closed_surface_trace_spaces(welded_candidate);
      if (topology.quality.valid) {
        mesh.vertices = std::move(welded_candidate.vertices);
        mesh.triangles = std::move(welded_candidate.triangles);
        mesh.panels = std::move(welded_candidate.panels);
        assign_all_analytic_triangle_owners(mesh);
        metrics.welded_vertices += welded;
        projected = timed_project_all();
      }
    }
    metrics.seam_weld_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - weld_begin)
                               .count();
    repair_ms += metrics.seam_weld_ms;
  }
  // One refused vertex fails the certificate for the whole surface, and on a
  // molecular mesh the obstruction is a micro-sliver in that vertex's one
  // ring rather than anything wrong with its projection.  Collapse exactly
  // those slivers and reproject, before the global short-edge cleanup below,
  // which rewrites parts of the surface that were never in question.
  // Repair whenever a refusal is large enough to matter rather than only when
  // the certificate fails, so the tolerance above stays a backstop for what
  // cannot be repaired instead of the mechanism that grants the certificate.
  constexpr double kRefusedSliverRepairCells = 0.05;
  const auto refused_sliver_begin = std::chrono::steady_clock::now();
  for (int round = 0; round < 4 && projected &&
                      !metrics.refused_vertex_indices.empty() &&
                      metrics.analytic_vertex_refused_distance_max >
                          kRefusedSliverRepairCells * grid_spacing;
       ++round) {
    const CompactVertexIncidence incidence =
        build_compact_vertex_incidence(mesh);
    if (incidence.offsets.size() != mesh.vertices.size() + 1U)
      break;
    double maximum_length = 0.0;
    const std::vector<EdgeKey> sliver_edges =
        collect_refused_vertex_sliver_edges(
            mesh, incidence, metrics.refused_vertex_indices,
            metrics.refused_vertex_distances,
            metrics.refused_vertex_shape_flags, maximum_length);
    if (sliver_edges.empty() || !(maximum_length > 0.0))
      break;
    const auto stage_begin = std::chrono::steady_clock::now();
    SurfaceMesh candidate;
    const std::size_t collapsed = collapse_selected_manifold_edges_compact(
        mesh, candidate,
        std::nextafter(maximum_length,
                       std::numeric_limits<double>::infinity()),
        sliver_edges, metrics.scratch_bytes);
    bool accepted = false;
    if (collapsed > 0U) {
      // This repair may run on a surface that already holds a certificate, so
      // it audits the candidate outright instead of inheriting the standing
      // topology certificate the way the generic collapse rounds do.
      orient_mesh_consistently(candidate);
      ++metrics.topology_fallback_audits;
      if (build_closed_surface_trace_spaces(candidate).quality.valid) {
        // Keep the mesh that goes in, so a collapse that reprojects into a
        // worse state can be put back exactly.  The retained metrics belong
        // to that mesh and stay valid with it, so the rollback needs no
        // second projection.
        SurfaceMesh previous;
        previous.vertices = mesh.vertices;
        previous.triangles = mesh.triangles;
        previous.panels = mesh.panels;
        const bool previous_certificate = metrics.analytic_vertex_certificate;
        const std::size_t previous_refused =
            metrics.refused_vertex_indices.size();
        const double previous_distance =
            metrics.analytic_vertex_refused_distance_max;
        const std::vector<std::uint32_t> previous_indices =
            metrics.refused_vertex_indices;
        const std::vector<double> previous_distances =
            metrics.refused_vertex_distances;
        const std::vector<std::uint8_t> previous_shape_flags =
            metrics.refused_vertex_shape_flags;
        mesh.vertices = std::move(candidate.vertices);
        mesh.triangles = std::move(candidate.triangles);
        mesh.panels = std::move(candidate.panels);
        assign_all_analytic_triangle_owners(mesh);
        topology_certified = true;
        const bool reprojected = timed_project_all();
        const bool regressed =
            !reprojected ||
            (previous_certificate && !metrics.analytic_vertex_certificate) ||
            metrics.refused_vertex_indices.size() > previous_refused ||
            metrics.analytic_vertex_refused_distance_max > previous_distance;
        if (regressed) {
          mesh.vertices = std::move(previous.vertices);
          mesh.triangles = std::move(previous.triangles);
          mesh.panels = std::move(previous.panels);
          assign_all_analytic_triangle_owners(mesh);
          metrics.analytic_vertex_certificate = previous_certificate;
          metrics.refused_vertex_indices = previous_indices;
          metrics.refused_vertex_distances = previous_distances;
          metrics.refused_vertex_shape_flags = previous_shape_flags;
          metrics.analytic_vertex_refused_distance_max = previous_distance;
        } else {
          metrics.collapsed_edges += collapsed;
          metrics.refused_sliver_collapses += collapsed;
          accepted = true;
        }
      }
    }
    repair_ms += std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - stage_begin)
                     .count();
    if (!accepted)
      break;
  }
  metrics.refused_sliver_repair_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - refused_sliver_begin)
          .count();
  stage_shape("refused_sliver_repair");
  const auto certificate_cleanup_begin = std::chrono::steady_clock::now();
  for (int cleanup = 0;
       cleanup < 4 && projected && !metrics.analytic_vertex_certificate;
       ++cleanup) {
    // Exact projection can merge two near-coincident marching vertices.  The
    // zero-area marker above prevents a stale-area topology pass; compact the
    // new short edges transactionally, then reproject the final vertex set.
    ++metrics.rejected_projection_rounds;
    collapse_rounds(4, 0.05 * static_cast<double>(1U << cleanup));
    projected = timed_project_all();
  }
  metrics.certificate_cleanup_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - certificate_cleanup_begin)
          .count();
  stage_shape("certificate_cleanup");
  const auto orientation_repair_begin = std::chrono::steady_clock::now();
  bool orientation_topology_changed = false;
  for (int cleanup = 0; cleanup < 20 && projected; ++cleanup) {
    const InvertedEdgeCandidates inverted =
        collect_analytic_inverted_short_edges(
            mesh, patches, projection, grid_spacing, grid_spacing,
            metrics.analytic_vertex_certificate, executor);
    bool changed = false;
    if (!inverted.edges.empty()) {
      const auto stage_begin = std::chrono::steady_clock::now();
      SurfaceMesh candidate;
      const double threshold =
          std::nextafter(inverted.maximum_length,
                         std::numeric_limits<double>::infinity());
      const std::size_t collapsed = collapse_selected_manifold_edges_compact(
          mesh, candidate, threshold, inverted.edges, metrics.scratch_bytes);
      if (collapsed > 0U) {
        bool candidate_topology_certified = topology_certified;
        if (!candidate_topology_certified) {
          ++metrics.topology_fallback_audits;
          orient_mesh_consistently(candidate);
          candidate_topology_certified =
              build_closed_surface_trace_spaces(candidate).quality.valid;
        } else {
          ++metrics.topology_certificate_accepts;
        }
        if (candidate_topology_certified) {
          mesh.vertices = std::move(candidate.vertices);
          mesh.triangles = std::move(candidate.triangles);
          mesh.panels = std::move(candidate.panels);
          assign_all_analytic_triangle_owners(mesh);
          metrics.collapsed_edges += collapsed;
          metrics.orientation_repair_faces += inverted.faces;
          topology_certified = true;
          changed = true;
          orientation_topology_changed = true;
        }
      }
      repair_ms += std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - stage_begin)
                       .count();
      if (changed) {
        ++metrics.orientation_repair_rounds;
        continue;
      }
    }
    const auto flip_begin = std::chrono::steady_clock::now();
    const std::size_t flipped = flip_edges_for_analytic_orientation(
        mesh, patches, projection, grid_spacing,
        metrics.analytic_vertex_certificate, metrics.scratch_bytes);
    if (flipped != 0U) {
      metrics.flipped_edges += flipped;
      ++metrics.orientation_repair_rounds;
      metrics.orientation_repair_faces += flipped;
      orientation_topology_changed = true;
      repair_ms += std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - flip_begin)
                       .count();
      continue;
    }
    repair_ms += std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - flip_begin)
                     .count();
    break;
  }
  metrics.orientation_repair_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - orientation_repair_begin)
          .count();
  stage_shape("orientation_repair");
  if (projected && orientation_topology_changed) {
    // A topology transaction may remove the vertices whose safe projection
    // was rejected.  The collapse itself must not manufacture a certificate:
    // refresh the current compact vertex set, and promote the certificate
    // only when the full transactional projection now passes.
    projected = timed_project_all();
  }
  // Rare wide folds can have no topology-safe sub-h edge and no valid
  // diagonal flip.  First preserve the established one-edge repair, which
  // accepts only a strict reduction in the full-face failure count with no
  // self-intersection.  If that cannot remove a multi-triangle fold, seed a
  // cold deterministic fallback from the first certified intersection pair
  // and accept only Pareto progress: neither failed faces nor intersections
  // may increase and at least one must decrease.  Every candidate still gets
  // the full topology, orientation, and intersection certificates.
  const auto wide_repair_begin = std::chrono::steady_clock::now();
  for (int wide_round = 0; wide_round < 16 && projected; ++wide_round) {
    LocalPatchRepairMetrics baseline;
    baseline.analytic_vertex_certificate =
        metrics.analytic_vertex_certificate;
    certify_all_face_orientations(mesh, patches, projection, grid_spacing,
                                  baseline);
    const std::size_t baseline_failed_faces =
        baseline.analytic_inverted_faces +
        baseline.analytic_orientation_uncertified_faces;
    const bool repair_uncertified_faces =
        baseline.analytic_orientation_uncertified_faces != 0U;
    const InvertedEdgeCandidates wide = baseline_failed_faces == 0U
                                            ? InvertedEdgeCandidates{}
                                            : collect_analytic_inverted_short_edges(
                                                  mesh, patches, projection,
                                                  grid_spacing,
                                                  2.0 * grid_spacing,
                                                  baseline
                                                      .analytic_vertex_certificate,
                                                  executor,
                                                  repair_uncertified_faces);
    const auto wide_begin = std::chrono::steady_clock::now();
    const SurfaceMesh &source = mesh;
    const bool source_vertex_certificate =
        metrics.analytic_vertex_certificate;
    const CompactVertexAdjacency wide_adjacency =
        build_compact_vertex_adjacency(source);
    const CompactVertexIncidence wide_incidence =
        build_compact_vertex_incidence(source);
    metrics.scratch_bytes =
        std::max(metrics.scratch_bytes,
                 wide_adjacency.retained_bytes() +
                     wide_incidence.retained_bytes());
    bool accepted = false;
    const auto evaluate_candidate =
        [&](const EdgeKey &edge, bool keep_higher, SurfaceMesh &candidate,
            LocalPatchRepairMetrics &attempt,
            std::size_t &collapsed) -> bool {
      collapsed = collapse_single_manifold_edge_compact(
          source, candidate, edge, wide_adjacency, keep_higher,
          wide_incidence,
          metrics.scratch_bytes);
      if (collapsed == 0U)
        return false;
      bool candidate_topology_certified = topology_certified;
      if (!candidate_topology_certified) {
        ++metrics.topology_fallback_audits;
        orient_mesh_consistently(candidate);
        candidate_topology_certified =
            build_closed_surface_trace_spaces(candidate).quality.valid;
        if (!candidate_topology_certified)
          return false;
      } else {
        ++metrics.topology_certificate_accepts;
      }
      // Collapsing an edge retains one already-existing endpoint. It can
      // preserve a full vertex certificate, but it cannot manufacture one
      // when the source still contains a missed or rejected projection.
      attempt.analytic_vertex_certificate =
          source_vertex_certificate;
      certify_all_face_orientations(candidate, patches, projection,
                                    grid_spacing, attempt);
      return true;
    };
    const auto commit_candidate =
        [&](SurfaceMesh &candidate, const LocalPatchRepairMetrics &attempt,
            std::size_t collapsed) {
      mesh.vertices = std::move(candidate.vertices);
      mesh.triangles = std::move(candidate.triangles);
      mesh.panels = std::move(candidate.panels);
      assign_all_analytic_triangle_owners(mesh);
      metrics.collapsed_edges += collapsed;
      metrics.orientation_repair_faces += baseline_failed_faces;
      metrics.analytic_vertex_certificate =
          attempt.analytic_vertex_certificate;
      metrics.scratch_bytes =
          std::max(metrics.scratch_bytes, attempt.scratch_bytes);
      ++metrics.orientation_repair_rounds;
      topology_certified = true;
      accepted = true;
    };

    // Phase one is intentionally kept in the same edge/endpoint order and
    // with the same acceptance rule as the established fast path.
    std::size_t attempts = 0U;
    if (!repair_uncertified_faces) {
      for (const EdgeKey &edge : wide.edges) {
        for (const bool keep_higher : {false, true}) {
          if (attempts++ >= 48U)
            break;
          ++metrics.wide_repair_attempts;
          SurfaceMesh candidate;
          LocalPatchRepairMetrics attempt;
          std::size_t collapsed = 0U;
          if (!evaluate_candidate(edge, keep_higher, candidate, attempt,
                                  collapsed))
            continue;
          if (attempt.analytic_orientation_uncertified_faces != 0U ||
              attempt.analytic_inverted_faces >=
                  baseline.analytic_inverted_faces) {
            // Preserve the established inverted-face repair path exactly. An
            // otherwise attractive collapse must not exchange a certified
            // inversion for a less diagnosable normal-certification failure.
            continue;
          }
          SelfIntersectionAudit intersection = detect_self_intersections(
              candidate, grid_spacing, attempt.scratch_bytes, executor);
          if (intersection.count != 0U)
            continue;
          commit_candidate(candidate, attempt, collapsed);
          break;
        }
        if (accepted || attempts >= 48U)
          break;
      }
    }

    if (!accepted) {
      std::size_t baseline_scratch = 0U;
      const SelfIntersectionAudit baseline_intersection =
          detect_self_intersections(source, grid_spacing, baseline_scratch,
                                    executor);
      metrics.scratch_bytes =
          std::max(metrics.scratch_bytes, baseline_scratch);
      if (baseline_failed_faces != 0U ||
          baseline_intersection.count != 0U) {
        std::vector<EdgeKey> progressive_edges;
        progressive_edges.reserve(6U + wide.edges.size());
        const auto append_triangle_edges = [&](std::size_t triangle_index) {
          if (triangle_index >= source.triangles.size())
            return;
          const Triangle &triangle = source.triangles[triangle_index];
          const std::array<EdgeKey, 3> edges = {
              edge_key(triangle.v0, triangle.v1),
              edge_key(triangle.v1, triangle.v2),
              edge_key(triangle.v2, triangle.v0)};
          for (const EdgeKey &edge : edges) {
            const double length =
                norm(mesh_vertex(source, edge.hi) -
                     mesh_vertex(source, edge.lo));
            if (length <= 2.0 * grid_spacing)
              progressive_edges.push_back(edge);
          }
        };
        append_triangle_edges(baseline_intersection.first);
        append_triangle_edges(baseline_intersection.second);
        std::sort(progressive_edges.begin(), progressive_edges.end(),
                  [](const EdgeKey &left, const EdgeKey &right) {
                    return std::tie(left.lo, left.hi) <
                           std::tie(right.lo, right.hi);
                  });
        progressive_edges.erase(
            std::unique(progressive_edges.begin(), progressive_edges.end()),
            progressive_edges.end());
        // The certified intersection simplex is the most local repair seed.
        // Keep its deterministic order ahead of the wider orientation edges.
        for (const EdgeKey &edge : wide.edges) {
          if (std::find(progressive_edges.begin(), progressive_edges.end(),
                        edge) == progressive_edges.end())
            progressive_edges.push_back(edge);
        }

        bool best_found = false;
        std::size_t best_failed_faces =
            std::numeric_limits<std::size_t>::max();
        std::size_t best_intersections =
            std::numeric_limits<std::size_t>::max();
        std::size_t best_collapsed = 0U;
        SurfaceMesh best_candidate;
        LocalPatchRepairMetrics best_attempt;
        bool intersection_free_found = false;
        attempts = 0U;
        for (const EdgeKey &edge : progressive_edges) {
          for (const bool keep_higher : {false, true}) {
            if (attempts++ >= 64U)
              break;
            ++metrics.wide_repair_attempts;
            SurfaceMesh candidate;
            LocalPatchRepairMetrics attempt;
            SelfIntersectionAudit intersection;
            std::size_t collapsed = 0U;
            if (!evaluate_candidate(edge, keep_higher, candidate, attempt,
                                    collapsed))
              continue;
            intersection = detect_self_intersections(
                candidate, grid_spacing, attempt.scratch_bytes, executor);
            const std::size_t attempt_failed_faces =
                attempt.analytic_inverted_faces +
                attempt.analytic_orientation_uncertified_faces;
            const bool no_worse =
                attempt_failed_faces <= baseline_failed_faces &&
                intersection.count <= baseline_intersection.count;
            const bool strict_progress =
                attempt_failed_faces < baseline_failed_faces ||
                intersection.count < baseline_intersection.count;
            if (!no_worse || !strict_progress)
              continue;
            const bool better =
                !best_found || intersection.count < best_intersections ||
                (intersection.count == best_intersections &&
                 attempt_failed_faces < best_failed_faces);
            if (!better)
              continue;
            best_found = true;
            best_failed_faces = attempt_failed_faces;
            best_intersections = intersection.count;
            best_collapsed = collapsed;
            best_candidate = std::move(candidate);
            best_attempt = attempt;
            if (best_intersections == 0U) {
              intersection_free_found = true;
              break;
            }
          }
          if (intersection_free_found || attempts >= 64U)
            break;
        }
        if (best_found)
          commit_candidate(best_candidate, best_attempt, best_collapsed);
      }
    }
    repair_ms += std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - wide_begin)
                     .count();
    if (accepted) {
      // Wide repair also compacts the vertex indices.  Reprojection is the
      // only authoritative way to certify that the retained endpoints all
      // lie on bounded analytic patches with fold-safe incident faces.
      projected = timed_project_all();
    }
    if (!accepted)
      break;
  }
  metrics.wide_repair_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() -
                               wide_repair_begin)
                               .count();
  stage_shape("wide_repair");
  metrics.projection_ms += std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() -
                               refinement_begin)
                               .count();
  // Generic diagonal flipping is deliberately disabled for analytic patches.
  // A topologically legal flip can connect the wrong sides of a curved patch
  // seam and reverse a panel normal while still improving planar quality.
  // Keep the rollback implementation for a future patch-aware criterion, but
  // preserve the certified marching connectivity in production for now.
  flip_rounds(0);
  const auto audit_begin = std::chrono::steady_clock::now();
  const auto orientation_begin = audit_begin;
  certify_all_face_orientations(mesh, patches, projection, grid_spacing,
                                metrics, true);
  metrics.orientation_certificate_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - orientation_begin)
          .count();
  const auto intersection_begin = std::chrono::steady_clock::now();
  const SelfIntersectionAudit intersection = detect_self_intersections(
      mesh, grid_spacing, metrics.scratch_bytes, executor);
  metrics.self_intersection_audit_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - intersection_begin)
          .count();
  metrics.analytic_self_intersections = intersection.count;
  metrics.analytic_self_intersection_first = intersection.first;
  metrics.analytic_self_intersection_second = intersection.second;
  if (metrics.analytic_self_intersections != 0U)
    metrics.analytic_orientation_status = "self_intersection";
  const auto sampled_audit_begin = std::chrono::steady_clock::now();
  audit_surface_against_analytic_patches(mesh, patches, projection,
                                         grid_spacing, metrics);
  metrics.sampled_analytic_audit_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - sampled_audit_begin)
          .count();
  if (!metrics.analytic_vertex_certificate)
    metrics.analytic_audit_status = analytic_vertex_failure_status(metrics);
  metrics.projection_ms += std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - audit_begin)
                               .count();
  const MeshShapeAudit shape_after = measure_mesh_shape(mesh);
  metrics.poor_triangles_after = shape_after.poor_triangles;
  metrics.triangle_quality_min_after = shape_after.minimum_quality;
  metrics.max_edge_ratio_after = shape_after.maximum_edge_ratio;
  metrics.repair_ms = repair_ms;
}

[[nodiscard]] bool
point_inside_triangle_component(const SurfaceMesh &mesh, const Vec3 &point,
                                std::span<const std::size_t> triangle_indices) {
  const Vec3 direction =
      normalized({1.0, 0.3713906763541037, 0.2171827465516321});
  std::vector<double> intersections;
  intersections.reserve(16);
  for (const std::size_t triangle_index : triangle_indices) {
    const Triangle &triangle = mesh.triangles[triangle_index];
    const Vec3 a = mesh_vertex(mesh, triangle.v0);
    const Vec3 b = mesh_vertex(mesh, triangle.v1);
    const Vec3 c = mesh_vertex(mesh, triangle.v2);
    const Vec3 edge1 = b - a;
    const Vec3 edge2 = c - a;
    const Vec3 pvec = cross(direction, edge2);
    const double determinant = dot(edge1, pvec);
    if (std::abs(determinant) <= 1.0e-13)
      continue;
    const double inverse = 1.0 / determinant;
    const Vec3 tvec = point - a;
    const double u = dot(tvec, pvec) * inverse;
    if (u < -1.0e-10 || u > 1.0 + 1.0e-10)
      continue;
    const Vec3 qvec = cross(tvec, edge1);
    const double v = dot(direction, qvec) * inverse;
    if (v < -1.0e-10 || u + v > 1.0 + 1.0e-10)
      continue;
    const double distance = dot(edge2, qvec) * inverse;
    if (distance > 1.0e-9)
      intersections.push_back(distance);
  }
  std::sort(intersections.begin(), intersections.end());
  intersections.erase(
      std::unique(intersections.begin(), intersections.end(),
                  [](double a, double b) {
                    return std::abs(a - b) <=
                           1.0e-8 * std::max({1.0, std::abs(a), std::abs(b)});
                  }),
      intersections.end());
  return (intersections.size() & 1U) != 0U;
}

[[nodiscard]] std::size_t
remove_nonphysical_components(SurfaceMesh &mesh, std::span<const Atom> atoms) {
  if (mesh.triangles.empty())
    return 0;
  std::unordered_map<EdgeKey, std::vector<std::size_t>, EdgeKeyHash> owners;
  owners.reserve(mesh.triangles.size() * 3U);
  for (std::size_t index = 0; index < mesh.triangles.size(); ++index) {
    const Triangle &triangle = mesh.triangles[index];
    owners[edge_key(triangle.v0, triangle.v1)].push_back(index);
    owners[edge_key(triangle.v1, triangle.v2)].push_back(index);
    owners[edge_key(triangle.v2, triangle.v0)].push_back(index);
  }
  std::vector<std::vector<std::size_t>> adjacency(mesh.triangles.size());
  for (const auto &[edge, edge_owners] : owners) {
    (void)edge;
    if (edge_owners.size() != 2U)
      continue;
    adjacency[edge_owners[0]].push_back(edge_owners[1]);
    adjacency[edge_owners[1]].push_back(edge_owners[0]);
  }
  std::vector<int> component_of(mesh.triangles.size(), -1);
  std::vector<std::vector<std::size_t>> components;
  for (std::size_t root = 0; root < mesh.triangles.size(); ++root) {
    if (component_of[root] >= 0)
      continue;
    const int component = static_cast<int>(components.size());
    components.emplace_back();
    std::vector<std::size_t> stack = {root};
    component_of[root] = component;
    while (!stack.empty()) {
      const std::size_t current = stack.back();
      stack.pop_back();
      components.back().push_back(current);
      for (const std::size_t neighbor : adjacency[current]) {
        if (component_of[neighbor] < 0) {
          component_of[neighbor] = component;
          stack.push_back(neighbor);
        }
      }
    }
  }
  if (components.size() <= 1U)
    return 0;

  struct Bounds {
    Vec3 lower{std::numeric_limits<double>::infinity(),
               std::numeric_limits<double>::infinity(),
               std::numeric_limits<double>::infinity()};
    Vec3 upper{-std::numeric_limits<double>::infinity(),
               -std::numeric_limits<double>::infinity(),
               -std::numeric_limits<double>::infinity()};
    Vec3 sample;
  };
  std::vector<Bounds> bounds(components.size());
  for (std::size_t component = 0; component < components.size(); ++component) {
    for (const std::size_t triangle_index : components[component]) {
      const Triangle &triangle = mesh.triangles[triangle_index];
      for (const std::size_t vertex_index :
           {triangle.v0, triangle.v1, triangle.v2}) {
        const Vec3 point = mesh_vertex(mesh, vertex_index);
        bounds[component].lower.x =
            std::min(bounds[component].lower.x, point.x);
        bounds[component].lower.y =
            std::min(bounds[component].lower.y, point.y);
        bounds[component].lower.z =
            std::min(bounds[component].lower.z, point.z);
        bounds[component].upper.x =
            std::max(bounds[component].upper.x, point.x);
        bounds[component].upper.y =
            std::max(bounds[component].upper.y, point.y);
        bounds[component].upper.z =
            std::max(bounds[component].upper.z, point.z);
      }
    }
    const Triangle &triangle = mesh.triangles[components[component].front()];
    bounds[component].sample = (1.0 / 3.0) * (mesh_vertex(mesh, triangle.v0) +
                                              mesh_vertex(mesh, triangle.v1) +
                                              mesh_vertex(mesh, triangle.v2));
  }
  std::vector<bool> keep(components.size(), false);
  for (std::size_t component = 0; component < components.size(); ++component) {
    const Bounds &box = bounds[component];
    for (const Atom &atom : atoms) {
      const Vec3 center{atom.x, atom.y, atom.z};
      if (center.x <= box.lower.x || center.x >= box.upper.x ||
          center.y <= box.lower.y || center.y >= box.upper.y ||
          center.z <= box.lower.z || center.z >= box.upper.z)
        continue;
      if (point_inside_triangle_component(mesh, center,
                                          components[component])) {
        keep[component] = true;
        break;
      }
    }
  }
  for (std::size_t inner = 0; inner < components.size(); ++inner) {
    if (!keep[inner])
      continue;
    const Vec3 point = bounds[inner].sample;
    for (std::size_t outer = 0; outer < components.size(); ++outer) {
      if (inner == outer || !keep[outer])
        continue;
      const Bounds &box = bounds[outer];
      if (point.x <= box.lower.x || point.x >= box.upper.x ||
          point.y <= box.lower.y || point.y >= box.upper.y ||
          point.z <= box.lower.z || point.z >= box.upper.z)
        continue;
      if (point_inside_triangle_component(mesh, point, components[outer])) {
        keep[inner] = false;
        break;
      }
    }
  }
  const std::size_t removed_components =
      static_cast<std::size_t>(std::count(keep.begin(), keep.end(), false));
  if (removed_components == 0)
    return 0;
  std::vector<Triangle> triangles;
  std::vector<Panel> panels;
  for (std::size_t index = 0; index < mesh.triangles.size(); ++index) {
    if (keep[static_cast<std::size_t>(component_of[index])]) {
      triangles.push_back(mesh.triangles[index]);
      panels.push_back(mesh.panels[index]);
    }
  }
  std::vector<bool> referenced(mesh.vertices.size(), false);
  for (const Triangle &triangle : triangles) {
    referenced[triangle.v0] = true;
    referenced[triangle.v1] = true;
    referenced[triangle.v2] = true;
  }
  std::vector<std::size_t> compact(mesh.vertices.size(), 0U);
  std::vector<Vertex> vertices;
  for (std::size_t index = 0; index < mesh.vertices.size(); ++index) {
    if (!referenced[index])
      continue;
    compact[index] = vertices.size();
    vertices.push_back(mesh.vertices[index]);
  }
  for (Triangle &triangle : triangles) {
    triangle.v0 = compact[triangle.v0];
    triangle.v1 = compact[triangle.v1];
    triangle.v2 = compact[triangle.v2];
  }
  mesh.vertices = std::move(vertices);
  mesh.triangles = std::move(triangles);
  mesh.panels = std::move(panels);
  for (std::size_t index = 0; index < mesh.triangles.size(); ++index)
    update_triangle_geometry(mesh, index);
  return removed_components;
}

// Marching-tetrahedra cases are locally oriented, but a shared diagonal can
// still give neighboring tetrahedra the same edge direction. Repair the mesh
// deterministically before the closed-surface trace-space check.
std::size_t orient_mesh_consistently(SurfaceMesh &mesh) {
  if (mesh.triangles.empty() || mesh.panels.size() != mesh.triangles.size()) {
    return 0U;
  }
  std::unordered_map<EdgeKey, std::vector<DirectedTriangleEdge>, EdgeKeyHash>
      edge_owners;
  edge_owners.reserve(mesh.triangles.size() * 3);
  for (std::size_t index = 0; index < mesh.triangles.size(); ++index) {
    const Triangle &triangle = mesh.triangles[index];
    const std::array<std::pair<std::size_t, std::size_t>, 3> edges = {
        std::pair{triangle.v0, triangle.v1},
        std::pair{triangle.v1, triangle.v2},
        std::pair{triangle.v2, triangle.v0}};
    for (const auto &[a, b] : edges) {
      edge_owners[edge_key(a, b)].push_back({index, edge_direction(a, b)});
    }
  }

  struct Neighbor {
    std::size_t index{0};
    int direction_product{1};
  };
  std::vector<std::vector<Neighbor>> adjacency(mesh.triangles.size());
  for (const auto &[key, owners] : edge_owners) {
    (void)key;
    if (owners.size() != 2) {
      continue;
    }
    adjacency[owners[0].triangle].push_back(
        {owners[1].triangle, owners[0].direction * owners[1].direction});
    adjacency[owners[1].triangle].push_back(
        {owners[0].triangle, owners[0].direction * owners[1].direction});
  }

  std::vector<int> orientation(mesh.triangles.size(), 0);
  std::vector<std::vector<std::size_t>> components;
  for (std::size_t root = 0; root < mesh.triangles.size(); ++root) {
    if (orientation[root] != 0) {
      continue;
    }
    orientation[root] = 1;
    components.emplace_back();
    std::vector<std::size_t> stack = {root};
    while (!stack.empty()) {
      const std::size_t current = stack.back();
      stack.pop_back();
      components.back().push_back(current);
      for (const Neighbor neighbor : adjacency[current]) {
        const int required = -orientation[current] * neighbor.direction_product;
        if (orientation[neighbor.index] == 0) {
          orientation[neighbor.index] = required;
          stack.push_back(neighbor.index);
        }
      }
    }
  }

  for (const auto &component : components) {
    double signed_volume_six = 0.0;
    for (const std::size_t index : component) {
      const Triangle &triangle = mesh.triangles[index];
      const Vec3 a = mesh_vertex(mesh, triangle.v0);
      const Vec3 b = mesh_vertex(mesh, triangle.v1);
      const Vec3 c = mesh_vertex(mesh, triangle.v2);
      signed_volume_six +=
          static_cast<double>(orientation[index]) * dot(a, cross(b, c));
    }
    if (signed_volume_six < 0.0) {
      for (const std::size_t index : component) {
        orientation[index] = -orientation[index];
      }
    }
  }

  std::size_t flipped = 0U;
  for (std::size_t index = 0; index < mesh.triangles.size(); ++index) {
    if (orientation[index] < 0) {
      Triangle &triangle = mesh.triangles[index];
      std::swap(triangle.v1, triangle.v2);
      ++flipped;
    }
    update_triangle_geometry(mesh, index);
  }
  return flipped;
}

void polygonize_grid(SurfaceMesh &mesh, const AtomGrid &ownership_atom_grid,
                     std::span<const Atom> atoms, const Vec3 &lower,
                     double grid_spacing, std::size_t nx, std::size_t ny,
                     std::size_t nz, std::span<const double> values) {
  const auto grid_index = [nx, ny](std::size_t ix, std::size_t iy,
                                   std::size_t iz) {
    return ix + nx * (iy + ny * iz);
  };
  static constexpr std::array<std::array<int, 4>, 6> tetrahedra = {
      std::array<int, 4>{0, 5, 1, 6}, std::array<int, 4>{0, 1, 2, 6},
      std::array<int, 4>{0, 2, 3, 6}, std::array<int, 4>{0, 3, 7, 6},
      std::array<int, 4>{0, 7, 4, 6}, std::array<int, 4>{0, 4, 5, 6}};
  std::unordered_map<VertexKey, std::size_t, VertexKeyHash> vertex_lookup;
  vertex_lookup.reserve((nx - 1) * (ny - 1));
  for (std::size_t iz = 0; iz + 1 < nz; ++iz) {
    for (std::size_t iy = 0; iy + 1 < ny; ++iy) {
      for (std::size_t ix = 0; ix + 1 < nx; ++ix) {
        const std::array<Vec3, 8> cube_points = {
            Vec3{lower.x + grid_spacing * static_cast<double>(ix),
                 lower.y + grid_spacing * static_cast<double>(iy),
                 lower.z + grid_spacing * static_cast<double>(iz)},
            Vec3{lower.x + grid_spacing * static_cast<double>(ix + 1),
                 lower.y + grid_spacing * static_cast<double>(iy),
                 lower.z + grid_spacing * static_cast<double>(iz)},
            Vec3{lower.x + grid_spacing * static_cast<double>(ix + 1),
                 lower.y + grid_spacing * static_cast<double>(iy + 1),
                 lower.z + grid_spacing * static_cast<double>(iz)},
            Vec3{lower.x + grid_spacing * static_cast<double>(ix),
                 lower.y + grid_spacing * static_cast<double>(iy + 1),
                 lower.z + grid_spacing * static_cast<double>(iz)},
            Vec3{lower.x + grid_spacing * static_cast<double>(ix),
                 lower.y + grid_spacing * static_cast<double>(iy),
                 lower.z + grid_spacing * static_cast<double>(iz + 1)},
            Vec3{lower.x + grid_spacing * static_cast<double>(ix + 1),
                 lower.y + grid_spacing * static_cast<double>(iy),
                 lower.z + grid_spacing * static_cast<double>(iz + 1)},
            Vec3{lower.x + grid_spacing * static_cast<double>(ix + 1),
                 lower.y + grid_spacing * static_cast<double>(iy + 1),
                 lower.z + grid_spacing * static_cast<double>(iz + 1)},
            Vec3{lower.x + grid_spacing * static_cast<double>(ix),
                 lower.y + grid_spacing * static_cast<double>(iy + 1),
                 lower.z + grid_spacing * static_cast<double>(iz + 1)}};
        const std::array<double, 8> cube_values = {
            values[grid_index(ix, iy, iz)],
            values[grid_index(ix + 1, iy, iz)],
            values[grid_index(ix + 1, iy + 1, iz)],
            values[grid_index(ix, iy + 1, iz)],
            values[grid_index(ix, iy, iz + 1)],
            values[grid_index(ix + 1, iy, iz + 1)],
            values[grid_index(ix + 1, iy + 1, iz + 1)],
            values[grid_index(ix, iy + 1, iz + 1)]};
        for (const auto &tetrahedron : tetrahedra) {
          std::array<Vec3, 4> points{};
          std::array<double, 4> tetra_values{};
          for (std::size_t k = 0; k < tetrahedron.size(); ++k) {
            const std::size_t vertex = static_cast<std::size_t>(tetrahedron[k]);
            points[k] = cube_points[vertex];
            tetra_values[k] = cube_values[vertex];
          }
          // Zero keeps this engine's geometry exactly as the runs already
          // published with it produced; see
          // docs/TITAN_SES_ENGINE_FIX_2026-09-15.md.
          polygonize_tetrahedron(mesh, vertex_lookup, ownership_atom_grid,
                                 atoms, points, tetra_values, 0.0);
        }
      }
    }
  }
}

[[nodiscard]] bool crossing_parity(std::span<const double> crossings,
                                   double coordinate) noexcept {
  return (static_cast<std::size_t>(
              std::upper_bound(crossings.begin(), crossings.end(), coordinate) -
              crossings.begin()) &
          1U) != 0U;
}

[[nodiscard]] double nearest_crossing_distance(
    std::span<const double> crossings, double coordinate,
    double fallback) noexcept {
  const auto next =
      std::lower_bound(crossings.begin(), crossings.end(), coordinate);
  double distance = fallback;
  if (next != crossings.end())
    distance = std::min(distance, std::abs(*next - coordinate));
  if (next != crossings.begin())
    distance = std::min(distance, std::abs(*std::prev(next) - coordinate));
  return distance;
}

[[nodiscard]] std::array<std::span<const double>, 3>
vertex_crossings(const analytic_ses::PatchRayGrid &rays, std::size_t ix,
                 std::size_t iy, std::size_t iz, std::size_t nx,
                 std::size_t ny) noexcept {
  return {rays.axes[0].row(iy + ny * iz),
          rays.axes[1].row(ix + nx * iz),
          rays.axes[2].row(ix + nx * iy)};
}

[[nodiscard]] bool line_crosses_patch_bounds(
    const analytic_ses::PatchBounds &bounds, analytic_ses::RayAxis axis,
    const Vec3 &point, double tolerance) noexcept {
  switch (axis) {
  case analytic_ses::RayAxis::x:
    return point.y >= bounds.lower.y - tolerance &&
           point.y <= bounds.upper.y + tolerance &&
           point.z >= bounds.lower.z - tolerance &&
           point.z <= bounds.upper.z + tolerance;
  case analytic_ses::RayAxis::y:
    return point.x >= bounds.lower.x - tolerance &&
           point.x <= bounds.upper.x + tolerance &&
           point.z >= bounds.lower.z - tolerance &&
           point.z <= bounds.upper.z + tolerance;
  case analytic_ses::RayAxis::z:
    return point.x >= bounds.lower.x - tolerance &&
           point.x <= bounds.upper.x + tolerance &&
           point.y >= bounds.lower.y - tolerance &&
           point.y <= bounds.upper.y + tolerance;
  }
  return false;
}

[[nodiscard]] std::uint8_t certify_perturbed_point_parity(
    const analytic_ses::PatchComplex &patches,
    const analytic_ses::ReentrantProbeIndex &probe_index, const Vec3 &point,
    double duplicate_tolerance,
    std::array<std::vector<double>, 3> &scratch) {
  std::array<bool, 3> inside{};
  for (const analytic_ses::RayAxis axis :
       {analytic_ses::RayAxis::x, analytic_ses::RayAxis::y,
        analytic_ses::RayAxis::z}) {
    const std::size_t axis_index = static_cast<std::size_t>(axis);
    std::vector<double> &coordinates = scratch[axis_index];
    coordinates.clear();
    const auto consider = [&](analytic_ses::PatchKind kind,
                              std::size_t patch_index) {
      const analytic_ses::PatchBounds bounds =
          analytic_ses::patch_bounds(patches, kind, patch_index);
      if (!line_crosses_patch_bounds(bounds, axis, point,
                                     duplicate_tolerance))
        return true;
      double u = 0.0;
      double v = 0.0;
      switch (axis) {
      case analytic_ses::RayAxis::x:
        u = point.y;
        v = point.z;
        break;
      case analytic_ses::RayAxis::y:
        u = point.x;
        v = point.z;
        break;
      case analytic_ses::RayAxis::z:
        u = point.x;
        v = point.y;
        break;
      }
      const analytic_ses::PatchAxisRoots roots =
          analytic_ses::intersect_patch_axis(
              patches, probe_index, kind, patch_index, axis, u, v);
      if (!roots.certified)
        return false;
      for (std::size_t root = 0U; root < roots.count; ++root)
        coordinates.push_back(roots.coordinates[root]);
      return true;
    };
    for (std::size_t patch = 0U; patch < patches.contacts.size(); ++patch) {
      if (!consider(analytic_ses::PatchKind::contact, patch))
        return 0U;
    }
    for (std::size_t patch = 0U; patch < patches.tori.size(); ++patch) {
      if (!consider(analytic_ses::PatchKind::toroidal, patch))
        return 0U;
    }
    for (std::size_t patch = 0U; patch < patches.reentrant.size(); ++patch) {
      if (!consider(analytic_ses::PatchKind::reentrant, patch))
        return 0U;
    }
    std::sort(coordinates.begin(), coordinates.end());
    coordinates.erase(
        std::unique(coordinates.begin(), coordinates.end(),
                    [&](double left, double right) {
                      return std::abs(left - right) <= duplicate_tolerance;
                    }),
        coordinates.end());
    if ((coordinates.size() & 1U) != 0U)
      return 0U;
    const double coordinate =
        axis == analytic_ses::RayAxis::x
            ? point.x
            : (axis == analytic_ses::RayAxis::y ? point.y : point.z);
    inside[axis_index] = crossing_parity(coordinates, coordinate);
  }
  if (inside[0] != inside[1] || inside[0] != inside[2])
    return 0U;
  return inside[0] ? 2U : 1U;
}

struct SparseGridMagnitude {
  std::vector<std::uint32_t> indices;
  std::vector<float> values;

  [[nodiscard]] std::pair<bool, double>
  lookup(std::size_t index, double fallback) const noexcept {
    if (index > std::numeric_limits<std::uint32_t>::max())
      return {false, fallback};
    const auto target = static_cast<std::uint32_t>(index);
    const auto found = std::lower_bound(indices.begin(), indices.end(), target);
    if (found == indices.end() || *found != target)
      return {false, fallback};
    return {true, static_cast<double>(
                      values[static_cast<std::size_t>(found - indices.begin())])};
  }

  [[nodiscard]] std::size_t retained_bytes() const noexcept {
    return indices.capacity() * sizeof(std::uint32_t) +
           values.capacity() * sizeof(float);
  }
};

[[nodiscard]] double patch_ray_field_value(
    const analytic_ses::PatchRayGrid &rays,
    const analytic_ses::BilevelBitGrid &topology_inside, const Vec3 &lower,
    double grid_spacing, std::size_t ix, std::size_t iy, std::size_t iz,
    std::size_t nx, std::size_t ny,
    const SparseGridMagnitude &morphology_magnitude) noexcept {
  const Vec3 point{lower.x + grid_spacing * static_cast<double>(ix),
                   lower.y + grid_spacing * static_cast<double>(iy),
                   lower.z + grid_spacing * static_cast<double>(iz)};
  const auto crossings = vertex_crossings(rays, ix, iy, iz, nx, ny);
  const std::size_t grid_index = ix + nx * (iy + ny * iz);
  const auto local_magnitude =
      morphology_magnitude.lookup(grid_index, grid_spacing);
  double distance = local_magnitude.second;
  distance = nearest_crossing_distance(crossings[0], point.x, distance);
  if (!local_magnitude.first) {
    distance = nearest_crossing_distance(crossings[1], point.y, distance);
    distance = nearest_crossing_distance(crossings[2], point.z, distance);
  }
  distance = std::max(distance, 1.0e-9 * grid_spacing);
  return topology_inside.get(ix, iy, iz) ? -distance : distance;
}

struct ActivePatchRayCell {
  std::uint32_t linear_index{0};
  std::uint8_t inside_mask{0};
};

static_assert(sizeof(ActivePatchRayCell) <= 8U);

[[nodiscard]] std::vector<ActivePatchRayCell> collect_active_patch_ray_cells(
    const analytic_ses::BilevelBitGrid &topology_inside, std::size_t nx,
    std::size_t ny, std::size_t nz, TitanExecutor *executor) {
  const std::size_t cell_nx = nx - 1U;
  const std::size_t cell_ny = ny - 1U;
  const std::size_t cell_nz = nz - 1U;
  static constexpr std::array<std::array<std::size_t, 3>, 8> corners = {
      std::array<std::size_t, 3>{0, 0, 0},
      std::array<std::size_t, 3>{1, 0, 0},
      std::array<std::size_t, 3>{1, 1, 0},
      std::array<std::size_t, 3>{0, 1, 0},
      std::array<std::size_t, 3>{0, 0, 1},
      std::array<std::size_t, 3>{1, 0, 1},
      std::array<std::size_t, 3>{1, 1, 1},
      std::array<std::size_t, 3>{0, 1, 1}};
  const std::size_t workers =
      executor == nullptr ? 1U : std::min(executor->capacity(), cell_nz);
  std::vector<std::vector<ActivePatchRayCell>> per_worker(workers);
  const auto collect = [&](std::size_t worker, std::size_t begin_z,
                           std::size_t end_z) {
    std::vector<ActivePatchRayCell> &local = per_worker[worker];
    local.reserve((end_z - begin_z) * cell_nx * cell_ny / 16U);
    for (std::size_t iz = begin_z; iz < end_z; ++iz) {
      for (std::size_t iy = 0; iy < cell_ny; ++iy) {
        for (std::size_t ix = 0; ix < cell_nx; ++ix) {
          std::uint8_t mask = 0U;
          for (std::size_t corner = 0; corner < corners.size(); ++corner) {
            const auto &offset = corners[corner];
            if (topology_inside.get(ix + offset[0], iy + offset[1],
                                    iz + offset[2]))
              mask |= static_cast<std::uint8_t>(1U << corner);
          }
          if (mask == 0U || mask == 0xffU)
            continue;
          const std::size_t linear = ix + cell_nx * (iy + cell_ny * iz);
          local.push_back(
              {static_cast<std::uint32_t>(linear), mask});
        }
      }
    }
  };
  if (executor != nullptr && workers > 1U)
    executor->parallel_for(cell_nz, workers, 1U, collect);
  else
    collect(0U, 0U, cell_nz);

  std::size_t active_count = 0U;
  for (const auto &local : per_worker)
    active_count += local.size();
  std::vector<ActivePatchRayCell> active;
  active.reserve(active_count);
  for (auto &local : per_worker)
    active.insert(active.end(), local.begin(), local.end());
  return active;
}

[[nodiscard]] SparseGridMagnitude build_sparse_morphology_magnitude(
    std::span<const ActivePatchRayCell> active, const AtomGrid &expanded_grid,
    std::span<const Atom> expanded_atoms,
    const analytic_ses::BilevelBitGrid &cavity_fill,
    std::span<const Vec3> fallback_directions, const Vec3 &lower,
    double probe_radius, double grid_spacing, std::size_t nx, std::size_t ny,
    std::size_t nz, TitanExecutor *executor) {
  SparseGridMagnitude result;
  if (active.empty() || nx == 0U || ny == 0U || nz == 0U)
    return result;
  static constexpr std::array<std::array<std::size_t, 3>, 8> corners = {
      std::array<std::size_t, 3>{0, 0, 0},
      std::array<std::size_t, 3>{1, 0, 0},
      std::array<std::size_t, 3>{1, 1, 0},
      std::array<std::size_t, 3>{0, 1, 0},
      std::array<std::size_t, 3>{0, 0, 1},
      std::array<std::size_t, 3>{1, 0, 1},
      std::array<std::size_t, 3>{1, 1, 1},
      std::array<std::size_t, 3>{0, 1, 1}};
  const std::size_t cell_nx = nx - 1U;
  const std::size_t cell_ny = ny - 1U;
  result.indices.reserve(active.size() * corners.size());
  for (const ActivePatchRayCell &cell : active) {
    const std::size_t iz =
        static_cast<std::size_t>(cell.linear_index) / (cell_nx * cell_ny);
    const std::size_t remainder =
        static_cast<std::size_t>(cell.linear_index) - iz * cell_nx * cell_ny;
    const std::size_t iy = remainder / cell_nx;
    const std::size_t ix = remainder - iy * cell_nx;
    for (const auto &corner : corners) {
      const std::size_t linear = (ix + corner[0]) +
                                 nx * ((iy + corner[1]) +
                                       ny * (iz + corner[2]));
      result.indices.push_back(static_cast<std::uint32_t>(linear));
    }
  }
  std::sort(result.indices.begin(), result.indices.end());
  result.indices.erase(
      std::unique(result.indices.begin(), result.indices.end()),
      result.indices.end());
  std::erase_if(result.indices, [&](std::uint32_t index) {
    const std::size_t iz = static_cast<std::size_t>(index) / (nx * ny);
    const std::size_t remainder =
        static_cast<std::size_t>(index) - iz * nx * ny;
    const std::size_t iy = remainder / nx;
    const std::size_t ix = remainder - iy * nx;
    return cavity_fill.get(ix, iy, iz);
  });
  result.values.resize(result.indices.size(), static_cast<float>(grid_spacing));
  const auto evaluate = [&](std::size_t, std::size_t begin,
                            std::size_t end) {
    for (std::size_t position = begin; position < end; ++position) {
      const std::size_t index = result.indices[position];
      const std::size_t iz = index / (nx * ny);
      const std::size_t remainder = index - iz * nx * ny;
      const std::size_t iy = remainder / nx;
      const std::size_t ix = remainder - iy * nx;
      const Vec3 point{
          lower.x + grid_spacing * static_cast<double>(ix),
          lower.y + grid_spacing * static_cast<double>(iy),
          lower.z + grid_spacing * static_cast<double>(iz)};
      double eroded = -std::numeric_limits<double>::infinity();
      for (const Vec3 &direction : fallback_directions) {
        eroded = std::max(
            eroded,
            sample_union_sdf(expanded_grid, expanded_atoms,
                             point + probe_radius * direction)
                .value);
      }
      result.values[position] = static_cast<float>(std::abs(eroded));
    }
  };
  if (executor != nullptr && executor->capacity() > 1U &&
      result.indices.size() > 1U) {
    executor->parallel_for(result.indices.size(), executor->capacity(), 256U,
                           evaluate);
  } else {
    evaluate(0U, 0U, result.indices.size());
  }
  return result;
}

// Fraction of a marching edge that every crossing is held away from both of
// its endpoints, for the patch-ray engine only.  0.10 is the largest value
// that improves every fixture on every measure at once: worst aspect ratio
// 1.3x to 3.5x lower, poor triangles 10x to 16x fewer, normal fit better on
// all of them, 1 percent more panels and 0.06 percent more area.  Larger
// values improve shape further but approach a cliff where one fixture loses
// 40 percent of its panels, which is resolution the caller did not ask to
// give up.  The measurement is in docs/TITAN_SES_EDGE_MARGIN_2026-09-15.md;
// test builds may override the value to repeat it.
[[nodiscard]] double patch_ray_edge_margin() noexcept {
  constexpr double kPatchRayEdgeMargin = 0.10;
#ifdef TITAN_TABI_SES_TEST_HOOKS
  if (const char *override_value = std::getenv("TITAN_TABI_SES_SNAP")) {
    const double parsed = std::atof(override_value);
    if (std::isfinite(parsed) && parsed >= 0.0 && parsed < 0.5)
      return parsed;
  }
#endif
  return kPatchRayEdgeMargin;
}

[[nodiscard]] std::pair<std::size_t, std::size_t> polygonize_patch_ray_grid(
    SurfaceMesh &mesh, const AtomGrid &ownership_atom_grid,
    std::span<const Atom> atoms, const Vec3 &lower, double grid_spacing,
    std::size_t nx, std::size_t ny,
    const analytic_ses::PatchRayGrid &rays,
    const analytic_ses::BilevelBitGrid &topology_inside,
    const SparseGridMagnitude &morphology_magnitude,
    std::span<const ActivePatchRayCell> active) {
  static constexpr std::array<std::array<int, 4>, 6> tetrahedra = {
      std::array<int, 4>{0, 5, 1, 6}, std::array<int, 4>{0, 1, 2, 6},
      std::array<int, 4>{0, 2, 3, 6}, std::array<int, 4>{0, 3, 7, 6},
      std::array<int, 4>{0, 7, 4, 6}, std::array<int, 4>{0, 4, 5, 6}};
  static constexpr std::array<std::array<std::size_t, 3>, 8> corners = {
      std::array<std::size_t, 3>{0, 0, 0},
      std::array<std::size_t, 3>{1, 0, 0},
      std::array<std::size_t, 3>{1, 1, 0},
      std::array<std::size_t, 3>{0, 1, 0},
      std::array<std::size_t, 3>{0, 0, 1},
      std::array<std::size_t, 3>{1, 0, 1},
      std::array<std::size_t, 3>{1, 1, 1},
      std::array<std::size_t, 3>{0, 1, 1}};
  std::unordered_map<VertexKey, std::size_t, VertexKeyHash> vertex_lookup;
  vertex_lookup.reserve(active.size() * 2U);
  const double edge_margin = patch_ray_edge_margin();
  const std::size_t cell_nx = nx - 1U;
  const std::size_t cell_ny = ny - 1U;
  for (const ActivePatchRayCell &cell : active) {
    const std::size_t iz =
        static_cast<std::size_t>(cell.linear_index) / (cell_nx * cell_ny);
    const std::size_t remainder =
        static_cast<std::size_t>(cell.linear_index) - iz * cell_nx * cell_ny;
    const std::size_t iy = remainder / cell_nx;
    const std::size_t ix = remainder - iy * cell_nx;
    std::array<Vec3, 8> cube_points{};
    std::array<double, 8> cube_values{};
    for (std::size_t corner = 0; corner < corners.size(); ++corner) {
      const auto &offset = corners[corner];
      const std::size_t cx = ix + offset[0];
      const std::size_t cy = iy + offset[1];
      const std::size_t cz = iz + offset[2];
      cube_points[corner] = {
          lower.x + grid_spacing * static_cast<double>(cx),
          lower.y + grid_spacing * static_cast<double>(cy),
          lower.z + grid_spacing * static_cast<double>(cz)};
      cube_values[corner] = patch_ray_field_value(
          rays, topology_inside, lower, grid_spacing, cx, cy, cz, nx, ny,
          morphology_magnitude);
    }
    for (const auto &tetrahedron : tetrahedra) {
      std::array<Vec3, 4> points{};
      std::array<double, 4> tetra_values{};
      for (std::size_t k = 0; k < tetrahedron.size(); ++k) {
        const std::size_t vertex = static_cast<std::size_t>(tetrahedron[k]);
        points[k] = cube_points[vertex];
        tetra_values[k] = cube_values[vertex];
      }
      polygonize_tetrahedron(mesh, vertex_lookup, ownership_atom_grid, atoms,
                             points, tetra_values, edge_margin);
    }
  }
  return {active.size(), active.size_bytes()};
}

struct ResolutionProjectionMetrics {
  std::size_t projected{0};
  std::size_t missing{0};
  std::size_t star_refused{0};
  std::size_t sheet_refused{0};
  std::size_t partial{0};
  std::size_t reverted{0};
  std::size_t rounds{0};
  std::size_t initial_intersections{0};
  std::size_t final_intersections{0};
  double unprojected_distance_max{0.0};
  double projected_distance_rms{0.0};
  double ms{0.0};
};

// Resolution policy for the patch-ray engine.  Marching tetrahedra on the
// certified ray-parity signs gives a closed, consistently oriented mesh whose
// triangles each lie inside their own tetrahedron, so it cannot intersect
// itself.  Features thinner than the grid cell (two sheets closer than h, a
// neck, a sliver patch) are resolved only to that cell: this is the accepted
// topology.  Each vertex then moves onto its closest analytic patch point
// unless the move would turn an incident triangle against its marching
// orientation or nearly degenerate it, or the point is on another sheet than
// the vertex's own (kResolutionSheetCosine).  Any triangle pair that still
// intersects sends all six of its vertices back to their marching positions;
// since the pure marching mesh is intersection-free, the loop ends with a
// valid surface.  Connectivity is never changed.
//
// Where a plate or gap is thinner than about one cell, the closest patch
// point of a vertex on its rim can be on the facing sheet. Every incident
// triangle may still pass the orientation test, since rim triangles stand
// sideways already, but the vertex then sits on that sheet with a star facing
// the other way, next to the sheet's own vertices. On the 13,541-atom complex
// at h = 1 A the remeshed surface had 25 vertices whose area-weighted star
// normal was more than 78 degrees from their surface normal (20 more than
// 90), and non-adjacent vertex pairs 0.09 A apart, 0.02 A from each other's
// dual cell, where no quadrature of the BIE operator converges. So a point
// whose surface normal is more than acos(kResolutionSheetCosine) from the
// vertex's marching star normal is treated as not found, and a vertex moved
// onto the surface whose final star turns that far from it goes back. The
// complex then has 3 such vertices and no non-adjacent pair closer than
// 0.35 of the local edge.
constexpr double kResolutionSheetCosine = 0.2;
[[nodiscard]] bool resolution_sheet_agrees(
    const Vec3 &star_normal, const analytic_ses::LocalProjectionHit &hit) {
  const Vec3 normal{hit.normal.x, hit.normal.y, hit.normal.z};
  if (!hit.normal_certified || !finite_vec3(normal) || !(norm(normal) > 1e-12) ||
      !(norm(star_normal) > 0.0))
    return true;
  return dot(normalized(star_normal), normalized(normal)) >=
         kResolutionSheetCosine;
}
// A node-patch cell is collocated at its mesh vertex, and the smooth-surface
// jump coefficient one half assumes the normal there is the surface normal.
// A vertex the engine left on the analytic SES has that normal exactly, so
// the final position is queried once more and the certified normal stored.
// Projection and remeshing assign normals along the way, but a vertex that
// remeshing never revisited keeps a star normal, so the flag is set here from
// the final geometry rather than carried through every operation.  Vertices
// off the surface (marching or part-way positions) keep their star normal,
// and a certified normal whose final star faces another sheet is refused, as
// in the projection itself.
void mark_analytic_vertex_normals(
    SurfaceMesh &mesh, const analytic_ses::PatchComplex &patches,
    const analytic_ses::LocalPatchProjectionIndex &projection,
    double grid_spacing, TitanExecutor *executor, std::size_t &analytic,
    std::size_t &refused) {
  const std::size_t vertex_count = mesh.vertices.size();
  std::vector<Vec3> star(vertex_count);
  for (const Triangle &t : mesh.triangles)
    for (const std::size_t w : {t.v0, t.v1, t.v2})
      star[w] = star[w] + t.area * Vec3{t.nx, t.ny, t.nz};
  // A projected vertex sits exactly on its closest point; part-way and
  // marching vertices are a sizeable fraction of an edge away.
  constexpr double kOnSurfaceA = 1.0e-6;
  std::vector<std::uint8_t> state(vertex_count, 0U); // 1 analytic, 2 refused
  const auto mark = [&](std::size_t, std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      Vertex &v = mesh.vertices[i];
      v.normal_analytic = false;
      const analytic_ses::LocalProjectionHit hit = nearest_local_patch(
          patches, projection, v.atom_index, Vec3{v.x, v.y, v.z},
          grid_spacing);
      if (!certified_finite_projection_hit(hit) ||
          !(hit.distance <= kOnSurfaceA))
        continue;
      if (!resolution_sheet_agrees(star[i], hit)) {
        state[i] = 2U;
        continue;
      }
      const Vec3 unit =
          normalized(Vec3{hit.normal.x, hit.normal.y, hit.normal.z});
      v.nx = unit.x;
      v.ny = unit.y;
      v.nz = unit.z;
      v.normal_analytic = true;
      state[i] = 1U;
    }
  };
  if (executor != nullptr && executor->capacity() > 1U)
    executor->parallel_for(vertex_count, executor->capacity(), 256U, mark);
  else
    mark(0U, 0U, vertex_count);
  analytic = static_cast<std::size_t>(
      std::count(state.begin(), state.end(), std::uint8_t{1}));
  refused = static_cast<std::size_t>(
      std::count(state.begin(), state.end(), std::uint8_t{2}));
}
void resolution_project_patch_ray_surface(
    SurfaceMesh &mesh, const analytic_ses::PatchComplex &patches,
    const analytic_ses::LocalPatchProjectionIndex &projection,
    double grid_spacing, TitanExecutor *executor,
    ResolutionProjectionMetrics &metrics) {
  const auto begin = std::chrono::steady_clock::now();
  const std::size_t vertex_count = mesh.vertices.size();
  const std::size_t triangle_count = mesh.triangles.size();
  if (vertex_count == 0U || triangle_count == 0U)
    return;
  std::vector<std::size_t> offsets(vertex_count + 1U, 0U);
  for (const Triangle &triangle : mesh.triangles) {
    ++offsets[triangle.v0 + 1U];
    ++offsets[triangle.v1 + 1U];
    ++offsets[triangle.v2 + 1U];
  }
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex)
    offsets[vertex + 1U] += offsets[vertex];
  std::vector<std::size_t> incident(offsets.back());
  {
    std::vector<std::size_t> cursor(offsets.begin(), offsets.end() - 1);
    for (std::size_t index = 0; index < triangle_count; ++index) {
      const Triangle &triangle = mesh.triangles[index];
      incident[cursor[triangle.v0]++] = index;
      incident[cursor[triangle.v1]++] = index;
      incident[cursor[triangle.v2]++] = index;
    }
  }
  std::vector<Vec3> marching(vertex_count);
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex)
    marching[vertex] = mesh_vertex(mesh, vertex);
  std::vector<Vec3> reference(triangle_count);
  for (std::size_t index = 0; index < triangle_count; ++index) {
    const Triangle &triangle = mesh.triangles[index];
    reference[index] = normalized(
        cross(marching[triangle.v1] - marching[triangle.v0],
              marching[triangle.v2] - marching[triangle.v0]));
  }

  // A marching vertex lies on a grid edge whose endpoints straddle the
  // surface, so its closest surface point is within one edge length.
  const double search = 2.0 * grid_spacing;
  std::vector<analytic_ses::LocalProjectionHit> hits(vertex_count);
  const auto project_range = [&](std::size_t, std::size_t first,
                                 std::size_t last) {
    for (std::size_t vertex = first; vertex < last; ++vertex) {
      const Vertex &v = mesh.vertices[vertex];
      hits[vertex] = nearest_local_patch(patches, projection, v.atom_index,
                                         marching[vertex], search);
    }
  };
  if (executor != nullptr && executor->capacity() > 1U && vertex_count > 1U)
    executor->parallel_for(vertex_count, executor->capacity(), 256U,
                           project_range);
  else
    project_range(0U, 0U, vertex_count);

  // Sequential acceptance keeps the result independent of thread count.
  const double minimum_twice_area = 1.0e-3 * grid_spacing * grid_spacing;
  constexpr double kMinimumOrientationCosine = 0.2;
  std::vector<std::uint8_t> moved(vertex_count, 0U);
  const auto star_accepts = [&](std::size_t vertex, const Vec3 &candidate) {
    for (std::size_t slot = offsets[vertex]; slot < offsets[vertex + 1U];
         ++slot) {
      const std::size_t index = incident[slot];
      const Triangle &triangle = mesh.triangles[index];
      const auto position = [&](std::size_t corner) {
        return corner == vertex ? candidate : mesh_vertex(mesh, corner);
      };
      const Vec3 a = position(triangle.v0);
      const Vec3 normal_vector =
          cross(position(triangle.v1) - a, position(triangle.v2) - a);
      const double twice_area = norm(normal_vector);
      if (!(twice_area > minimum_twice_area) ||
          dot(normal_vector, reference[index]) <=
              kMinimumOrientationCosine * twice_area)
        return false;
    }
    return true;
  };
  double squared_sum = 0.0;
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    const analytic_ses::LocalProjectionHit &hit = hits[vertex];
    const Vec3 target{hit.point.x, hit.point.y, hit.point.z};
    if (!hit.found || !std::isfinite(hit.distance) || !finite_vec3(target) ||
        hit.distance > search) {
      ++metrics.missing;
      continue;
    }
    // The point must be on the vertex's own sheet as its marching star sees
    // it; otherwise the vertex does not move towards it at all.
    Vec3 marching_star{};
    for (std::size_t slot = offsets[vertex]; slot < offsets[vertex + 1U];
         ++slot) {
      const Triangle &triangle = mesh.triangles[incident[slot]];
      marching_star =
          marching_star + cross(marching[triangle.v1] - marching[triangle.v0],
                                marching[triangle.v2] - marching[triangle.v0]);
    }
    if (!resolution_sheet_agrees(marching_star, hit)) {
      ++metrics.sheet_refused;
      continue;
    }
    // A refused full move is retried half and a quarter of the way, so a
    // vertex in a tight spot still gets closer to the surface instead of
    // staying a full marching offset away.
    double fraction = 0.0;
    for (const double t : {1.0, 0.5, 0.25}) {
      if (star_accepts(vertex, marching[vertex] + t * (target - marching[vertex]))) {
        fraction = t;
        break;
      }
    }
    if (fraction == 0.0) {
      ++metrics.star_refused;
      continue;
    }
    const Vec3 accepted = marching[vertex] + fraction * (target - marching[vertex]);
    Vertex &v = mesh.vertices[vertex];
    v.x = accepted.x;
    v.y = accepted.y;
    v.z = accepted.z;
    // 1 = on the surface, 2 = half way, 4 = a quarter of the way.
    moved[vertex] = fraction == 1.0 ? 1U : (fraction == 0.5 ? 2U : 4U);
    if (fraction < 1.0)
      ++metrics.partial;
    ++metrics.projected;
    squared_sum += hit.distance * hit.distance;
  }
  // With every vertex in place, the star of a vertex moved onto the surface
  // must still face the sheet of its point: its neighbours have moved too.
  // One that does not goes back to its marching position; that changes its
  // neighbours' stars, so the sweep repeats until nothing changes. A vertex
  // moved part of the way is not on the sheet and keeps its star normal, so
  // it is left alone.
  for (int round = 0; round < 8; ++round) {
    bool changed = false;
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
      if (moved[vertex] != 1U)
        continue;
      Vec3 star{};
      for (std::size_t slot = offsets[vertex]; slot < offsets[vertex + 1U];
           ++slot) {
        const Triangle &triangle = mesh.triangles[incident[slot]];
        const Vec3 a = mesh_vertex(mesh, triangle.v0);
        star = star + cross(mesh_vertex(mesh, triangle.v1) - a,
                            mesh_vertex(mesh, triangle.v2) - a);
      }
      if (resolution_sheet_agrees(star, hits[vertex]))
        continue;
      Vertex &v = mesh.vertices[vertex];
      v.x = marching[vertex].x;
      v.y = marching[vertex].y;
      v.z = marching[vertex].z;
      moved[vertex] = 0U;
      ++metrics.sheet_refused;
      changed = true;
    }
    if (!changed)
      break;
  }

  std::vector<std::array<std::uint32_t, 2>> pairs;
  std::size_t scratch_bytes = 0U;
  constexpr std::size_t kMaximumRounds = 64U;
  for (std::size_t round = 0; round <= kMaximumRounds; ++round) {
    const SelfIntersectionAudit audit = detect_self_intersections(
        mesh, grid_spacing, scratch_bytes, executor, &pairs);
    if (round == 0U)
      metrics.initial_intersections = audit.count;
    metrics.final_intersections = audit.count;
    if (audit.count == 0U || round == kMaximumRounds)
      break;
    ++metrics.rounds;
    std::size_t reverted_this_round = 0U;
    for (const auto &pair : pairs) {
      for (const std::uint32_t index : pair) {
        const Triangle &triangle = mesh.triangles[index];
        for (const std::size_t vertex : {triangle.v0, triangle.v1, triangle.v2}) {
          if (moved[vertex] == 0U)
            continue;
          Vertex &v = mesh.vertices[vertex];
          v.x = marching[vertex].x;
          v.y = marching[vertex].y;
          v.z = marching[vertex].z;
          moved[vertex] = 0U;
          ++reverted_this_round;
        }
      }
    }
    metrics.reverted += reverted_this_round;
    if (reverted_this_round == 0U)
      break;
  }

  for (std::size_t index = 0; index < triangle_count; ++index)
    update_triangle_geometry(mesh, index);
  squared_sum = 0.0;
  std::size_t projected = 0U;
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    Vertex &v = mesh.vertices[vertex];
    const analytic_ses::LocalProjectionHit &hit = hits[vertex];
    if (moved[vertex] == 1U) {
      ++projected;
      squared_sum += hit.distance * hit.distance;
      const Vec3 normal{hit.normal.x, hit.normal.y, hit.normal.z};
      if (hit.normal_certified && finite_vec3(normal) && norm(normal) > 1e-12) {
        const Vec3 unit = normalized(normal);
        v.nx = unit.x;
        v.ny = unit.y;
        v.nz = unit.z;
        continue;
      }
    } else if (moved[vertex] != 0U) {
      metrics.unprojected_distance_max = std::max(
          metrics.unprojected_distance_max,
          (moved[vertex] == 2U ? 0.5 : 0.75) * hit.distance);
    } else if (hit.found && std::isfinite(hit.distance)) {
      metrics.unprojected_distance_max =
          std::max(metrics.unprojected_distance_max, hit.distance);
    } else {
      metrics.unprojected_distance_max =
          std::max(metrics.unprojected_distance_max, search);
    }
    Vec3 summed{};
    for (std::size_t slot = offsets[vertex]; slot < offsets[vertex + 1U];
         ++slot) {
      const Triangle &triangle = mesh.triangles[incident[slot]];
      summed = summed + triangle.area * Vec3{triangle.nx, triangle.ny,
                                             triangle.nz};
    }
    if (norm(summed) > 0.0) {
      const Vec3 unit = normalized(summed);
      v.nx = unit.x;
      v.ny = unit.y;
      v.nz = unit.z;
    }
  }
  metrics.projected = projected;
  metrics.projected_distance_rms =
      projected == 0U ? 0.0
                      : std::sqrt(squared_sum / static_cast<double>(projected));
  metrics.ms = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - begin)
                   .count();
}

[[nodiscard]] double triangle_min_angle(const Vec3 &a, const Vec3 &b,
                                        const Vec3 &c) noexcept {
  const double ab = norm(b - a), bc = norm(c - b), ca = norm(a - c);
  const auto angle = [](double opposite, double s1, double s2) {
    const double cosine = (s1 * s1 + s2 * s2 - opposite * opposite) /
                          std::max(1e-300, 2.0 * s1 * s2);
    return std::acos(std::clamp(cosine, -1.0, 1.0));
  };
  return std::min({angle(bc, ab, ca), angle(ca, ab, bc), angle(ab, bc, ca)});
}

struct ResolutionQualityMetrics {
  std::size_t angle_below_5{0}, angle_below_10{0}, angle_below_20{0};
  // Edge counts are per triangle side, so an interior edge counts twice.
  std::size_t edges_below_005{0}, edges_below_020{0};
  double edge_min{0.0}, edge_mean{0.0};
  double ms{0.0};
};

void measure_resolution_quality(const SurfaceMesh &mesh,
                                ResolutionQualityMetrics &metrics) {
  double edge_sum = 0.0;
  metrics.edge_min = std::numeric_limits<double>::infinity();
  metrics.angle_below_5 = metrics.angle_below_10 = metrics.angle_below_20 = 0U;
  metrics.edges_below_005 = metrics.edges_below_020 = 0U;
  for (const Triangle &triangle : mesh.triangles) {
    const Vec3 a = mesh_vertex(mesh, triangle.v0),
               b = mesh_vertex(mesh, triangle.v1),
               c = mesh_vertex(mesh, triangle.v2);
    edge_sum += norm(b - a) + norm(c - b) + norm(a - c);
    metrics.edge_min =
        std::min({metrics.edge_min, norm(b - a), norm(c - b), norm(a - c)});
    for (const double length : {norm(b - a), norm(c - b), norm(a - c)}) {
      metrics.edges_below_005 += length < 0.05;
      metrics.edges_below_020 += length < 0.20;
    }
    const double smallest = triangle_min_angle(a, b, c) * 180.0 / std::numbers::pi;
    metrics.angle_below_5 += smallest < 5.0;
    metrics.angle_below_10 += smallest < 10.0;
    metrics.angle_below_20 += smallest < 20.0;
  }
  metrics.edge_mean =
      mesh.triangles.empty()
          ? 0.0
          : edge_sum / static_cast<double>(3U * mesh.triangles.size());
}

struct ResolutionRemeshMetrics {
  std::size_t iterations{0};
  std::size_t splits{0};
  std::size_t collapses{0};
  std::size_t flips{0};
  std::size_t relaxed{0};
  std::size_t undone_operations{0};
  std::size_t vertices_before{0};
  std::size_t vertices_after{0};
  std::size_t off_surface_vertices{0};
  std::size_t final_intersections{0};
  // Snaps not taken because the point was on another sheet than the vertex's
  // star (kResolutionSheetCosine).
  std::size_t sheet_refused{0};
  // Charge-adaptive remeshing: final vertices whose local target edge is
  // below the uniform one.
  std::size_t refined_vertices{0};
  double target_edge{0.0};
  double ms{0.0};
};

// Local target edge of the resolution remeshing (ChargeRefinement).  Charges
// are hashed on a grid whose cell is the reach edge/alpha, beyond which the
// uniform edge applies, so a query visits 27 cells at most.
class RemeshSizing {
public:
  RemeshSizing(std::span<const Atom> atoms, double edge,
               const ChargeRefinement &refine)
      : edge_(edge), alpha_(refine.alpha),
        min_edge_(std::clamp(refine.min_edge, 1.0e-3, edge)) {
    if (!(alpha_ > 0.0) || !(edge_ > 0.0) || !std::isfinite(alpha_))
      return;
    reach_ = edge_ / alpha_;
    for (const Atom &atom : atoms)
      if (std::abs(atom.q) >= refine.min_charge &&
          std::isfinite(atom.x + atom.y + atom.z))
        charges_.push_back({atom.x, atom.y, atom.z});
    for (std::size_t k = 0; k < charges_.size(); ++k)
      cells_[cell_key(cell_of(charges_[k]))].push_back(k);
    build_dense_grid();
  }
  [[nodiscard]] bool active() const noexcept { return !charges_.empty(); }
  [[nodiscard]] std::size_t charges() const noexcept { return charges_.size(); }
  [[nodiscard]] double smallest() const noexcept {
    return active() ? min_edge_ : edge_;
  }
  [[nodiscard]] double operator()(const Vec3 &x) const {
    if (charges_.empty())
      return edge_;
    double best = reach_ * reach_;
    const std::array<std::int64_t, 3> c = cell_of(x);
    if (!dense_start_.empty()) {
      // The same charges as the hashed cells, from a dense CSR grid; the
      // minimum does not depend on the order they are visited in.
      for (std::int64_t dx = -1; dx <= 1; ++dx) {
        const std::int64_t gx = c[0] + dx - lo_[0];
        if (gx < 0 || gx >= dim_[0])
          continue;
        for (std::int64_t dy = -1; dy <= 1; ++dy) {
          const std::int64_t gy = c[1] + dy - lo_[1];
          if (gy < 0 || gy >= dim_[1])
            continue;
          for (std::int64_t dz = -1; dz <= 1; ++dz) {
            const std::int64_t gz = c[2] + dz - lo_[2];
            if (gz < 0 || gz >= dim_[2])
              continue;
            const std::size_t cell =
                static_cast<std::size_t>((gx * dim_[1] + gy) * dim_[2] + gz);
            for (std::uint32_t p = dense_start_[cell]; p < dense_start_[cell + 1];
                 ++p) {
              const Vec3 d = x - charges_[dense_items_[p]];
              best = std::min(best, dot(d, d));
            }
          }
        }
      }
      return std::clamp(alpha_ * std::sqrt(best), min_edge_, edge_);
    }
    for (std::int64_t dx = -1; dx <= 1; ++dx)
      for (std::int64_t dy = -1; dy <= 1; ++dy)
        for (std::int64_t dz = -1; dz <= 1; ++dz) {
          const auto found = cells_.find(cell_key({c[0] + dx, c[1] + dy, c[2] + dz}));
          if (found == cells_.end())
            continue;
          for (const std::size_t k : found->second) {
            const Vec3 d = x - charges_[k];
            best = std::min(best, dot(d, d));
          }
        }
    return std::clamp(alpha_ * std::sqrt(best), min_edge_, edge_);
  }

private:
  [[nodiscard]] std::array<std::int64_t, 3> cell_of(const Vec3 &x) const {
    return {static_cast<std::int64_t>(std::floor(x.x / reach_)),
            static_cast<std::int64_t>(std::floor(x.y / reach_)),
            static_cast<std::int64_t>(std::floor(x.z / reach_))};
  }
  [[nodiscard]] static std::uint64_t
  cell_key(const std::array<std::int64_t, 3> &c) noexcept {
    constexpr std::int64_t bias = std::int64_t{1} << 20;
    constexpr std::uint64_t mask = (std::uint64_t{1} << 21) - 1U;
    return ((static_cast<std::uint64_t>(c[0] + bias) & mask) << 42) |
           ((static_cast<std::uint64_t>(c[1] + bias) & mask) << 21) |
           (static_cast<std::uint64_t>(c[2] + bias) & mask);
  }
  // The charges' cells as a dense grid over their bounding box, when it is of
  // reasonable size; otherwise the hashed cells serve the queries.
  void build_dense_grid() {
    if (charges_.empty())
      return;
    std::array<std::int64_t, 3> lo = cell_of(charges_[0]), hi = lo;
    for (const Vec3 &q : charges_) {
      const auto c = cell_of(q);
      for (int a = 0; a < 3; ++a) {
        lo[a] = std::min(lo[a], c[a]);
        hi[a] = std::max(hi[a], c[a]);
      }
    }
    std::array<std::int64_t, 3> dim{};
    std::int64_t total = 1;
    for (int a = 0; a < 3; ++a) {
      dim[a] = hi[a] - lo[a] + 1;
      if (dim[a] <= 0 || dim[a] > (std::int64_t{1} << 20))
        return;
      total *= dim[a];
      if (total > (std::int64_t{1} << 24))
        return;
    }
    lo_ = lo;
    dim_ = dim;
    const auto index = [&](const Vec3 &q) {
      const auto c = cell_of(q);
      return static_cast<std::size_t>(
          ((c[0] - lo_[0]) * dim_[1] + (c[1] - lo_[1])) * dim_[2] + (c[2] - lo_[2]));
    };
    dense_start_.assign(static_cast<std::size_t>(total) + 1, 0U);
    for (const Vec3 &q : charges_)
      ++dense_start_[index(q) + 1];
    for (std::size_t c = 0; c + 1 < dense_start_.size(); ++c)
      dense_start_[c + 1] += dense_start_[c];
    dense_items_.resize(charges_.size());
    std::vector<std::uint32_t> fill(dense_start_.begin(), dense_start_.end() - 1);
    for (std::size_t k = 0; k < charges_.size(); ++k)
      dense_items_[fill[index(charges_[k])]++] = static_cast<std::uint32_t>(k);
  }
  double edge_{0.0}, alpha_{0.0}, min_edge_{0.0}, reach_{0.0};
  std::vector<Vec3> charges_;
  std::unordered_map<std::uint64_t, std::vector<std::size_t>> cells_;
  std::array<std::int64_t, 3> lo_{}, dim_{};
  std::vector<std::uint32_t> dense_start_, dense_items_;
};

// Isotropic remeshing (Botsch and Kobbelt 2004) of the resolution-policy
// mesh onto the analytic SES.  Each iteration splits edges longer than 4/3 L
// at their midpoint, collapses edges shorter than 4/5 L when the link
// condition holds, flips edges towards Delaunay on nearly flat pairs and
// relaxes vertices tangentially; every new or moved vertex is put on the
// closest analytic patch point when that point is within L/2.  Operations in
// one pass never share a triangle or a vertex star, and adjacency is rebuilt
// between passes.  Each local change must keep every touched triangle within
// 60 degrees of its previous orientation and above a minimum area, and a
// snapped point must be on the vertex's own sheet (kResolutionSheetCosine
// against the new star).  An iteration that ends with any self-intersection
// is rolled back and the remeshing stops there, so the result is always
// closed, consistently oriented and intersection-free.
void remesh_resolution_surface(
    SurfaceMesh &mesh, const analytic_ses::PatchComplex &patches,
    const analytic_ses::LocalPatchProjectionIndex &projection,
    double grid_spacing, double target_edge, int iterations,
    TitanExecutor *executor, ResolutionRemeshMetrics &metrics,
    const RemeshSizing *sizing = nullptr) {
  const auto begin = std::chrono::steady_clock::now();
  metrics.target_edge = target_edge;
  metrics.vertices_before = mesh.vertices.size();
  if (mesh.triangles.empty() || !(target_edge > 0.0))
    return;
  // Wall time of each phase, printed at the end (diagnostic only).
  using PhaseClock = std::chrono::steady_clock;
  const auto since = [](PhaseClock::time_point t0) {
    return std::chrono::duration<double, std::milli>(PhaseClock::now() - t0)
        .count();
  };
  double ms_rebuild = 0, ms_sort = 0, ms_select = 0, ms_snap = 0,
         ms_split = 0, ms_collapse = 0, ms_flip = 0, ms_propose = 0,
         ms_accept = 0, ms_export = 0, ms_detect = 0;
  std::size_t settle_rounds = 0, loop_snaps = 0, loop_batches = 0;
  double ms_loop_snap = 0;
  // The target edge at a point: uniform, or graded towards shallow charges
  // (ChargeRefinement).  Without charges every threshold below is the
  // uniform one, operation for operation.
  const bool graded = sizing != nullptr && sizing->active();
  const auto local_edge = [&](const Vec3 &x) {
    return graded ? (*sizing)(x) : target_edge;
  };
  const auto min_twice_area_at = [&](const Vec3 &x) {
    const double edge = local_edge(x);
    return 1.0e-3 * edge * edge;
  };
  const double smallest_edge = graded ? sizing->smallest() : target_edge;
  using Tri = std::array<std::size_t, 3>;
  struct State {
    std::vector<Vec3> p, n;
    std::vector<std::size_t> atom;
    std::vector<std::uint8_t> on_surface, vertex_alive;
    std::vector<Tri> tri;
    std::vector<std::uint8_t> tri_alive;
  } s;
  for (const Vertex &v : mesh.vertices) {
    s.p.push_back({v.x, v.y, v.z});
    s.n.push_back({v.nx, v.ny, v.nz});
    s.atom.push_back(v.atom_index);
    s.on_surface.push_back(0U);
    s.vertex_alive.push_back(1U);
  }
  for (const Triangle &t : mesh.triangles) {
    s.tri.push_back({t.v0, t.v1, t.v2});
    s.tri_alive.push_back(1U);
  }
  const double long_edge = 4.0 / 3.0 * target_edge;
  const double short_edge = 4.0 / 5.0 * target_edge;
  const double minimum_twice_area = 1.0e-3 * target_edge * target_edge;
  constexpr double kKeepOrientation = 0.5;
  const auto normal_of = [&](const Tri &t, double &twice_area) {
    const Vec3 c = cross(s.p[t[1]] - s.p[t[0]], s.p[t[2]] - s.p[t[0]]);
    twice_area = norm(c);
    return twice_area > 0.0 ? (1.0 / twice_area) * c : Vec3{};
  };
  // A snap is taken only within half an edge, so the closest-point search is
  // bounded there too: the closest patch point, when it is that near, is the
  // same either way, and the search prunes the patches beyond it.
  const auto snap = [&](const Vec3 &x, std::size_t atom, Vec3 &point,
                        Vec3 &normal) {
    const analytic_ses::LocalProjectionHit hit =
        nearest_local_patch(patches, projection, atom, x, 0.5 * target_edge);
    const Vec3 at{hit.point.x, hit.point.y, hit.point.z};
    if (!hit.found || !finite_vec3(at) || !(hit.distance <= 0.5 * target_edge))
      return false;
    point = at;
    const Vec3 hn{hit.normal.x, hit.normal.y, hit.normal.z};
    normal = hit.normal_certified && finite_vec3(hn) && norm(hn) > 1e-12
                 ? normalized(hn)
                 : Vec3{};
    return true;
  };
  // The split and collapse passes snap one midpoint per edge they act on, in
  // a serial loop.  An edge is acted on only while no operation of the pass
  // has touched its two triangles, so its endpoints, its frame and the owner
  // atom the snap is hinted with are still those at the start of the pass:
  // the snap can be computed ahead, in parallel, and the loop takes it.
  // Which edges the loop reaches is found first by running its locking
  // serially without the snaps (`select`), so few snaps are wasted; an edge
  // the loop reaches that was not planned (a collapse refused after its snap
  // leaves its neighbours unlocked) is snapped in the loop as before.
  struct PlannedSnap {
    Vec3 point{}, normal{};
    std::size_t atom{0};
    std::uint8_t planned{0U}, snapped{0U};
  };
  std::vector<std::size_t> picked;
  const auto plan_snaps = [&](std::vector<PlannedSnap> &plan, std::size_t count,
                              const auto &select) {
    const auto select_begin = PhaseClock::now();
    plan.assign(count, PlannedSnap{});
    picked.clear();
    for (std::size_t k = 0; k < count; ++k) {
      PlannedSnap &p = plan[k];
      if (!select(k, p.point, p.atom))
        continue;
      p.planned = 1U;
      picked.push_back(k);
    }
    ms_select += since(select_begin);
    const auto snap_begin = PhaseClock::now();
    const auto work = [&](std::size_t, std::size_t begin, std::size_t end) {
      for (std::size_t j = begin; j < end; ++j) {
        PlannedSnap &p = plan[picked[j]];
        const Vec3 midpoint = p.point;
        p.snapped = snap(midpoint, p.atom, p.point, p.normal) ? 1U : 0U;
      }
    };
    if (executor != nullptr && executor->capacity() > 1U && picked.size() > 64U)
      executor->parallel_for(picked.size(), executor->capacity(), 16U, work);
    else
      work(0U, 0U, picked.size());
    ms_snap += since(snap_begin);
  };
  std::vector<PlannedSnap> plan;
  // star: the summed unnormalised normals of the vertex's triangles at the
  // snapped point; normal: the snapped point's unit normal, or zero.
  const auto sheet_agrees = [&](const Vec3 &star, const Vec3 &normal) {
    if (!(norm(normal) > 0.0) || !(norm(star) > 0.0) ||
        dot(normalized(star), normal) >= kResolutionSheetCosine)
      return true;
    ++metrics.sheet_refused;
    return false;
  };

  // Undo log.  Operations of one pass never share a triangle or a vertex
  // star, so any subset of them can be undone independently.
  struct VertexOld {
    std::size_t index;
    Vec3 p, n;
    std::uint8_t on_surface, alive;
  };
  struct Operation {
    std::vector<std::pair<std::size_t, Tri>> triangles;
    std::vector<std::pair<std::size_t, std::uint8_t>> triangle_alive;
    std::vector<VertexOld> vertices;
    std::vector<std::size_t> created_triangles, created_vertices;
  };
  std::vector<Operation> operations;
  std::vector<std::int64_t> triangle_operation, vertex_operation;
  const auto start_pass = [&] {
    operations.clear();
    triangle_operation.assign(s.tri.size(), -1);
    vertex_operation.assign(s.p.size(), -1);
  };
  const auto touch_triangle = [&](Operation &op, std::size_t t) {
    op.triangles.push_back({t, s.tri[t]});
    op.triangle_alive.push_back({t, s.tri_alive[t]});
    if (t >= triangle_operation.size())
      triangle_operation.resize(t + 1U, -1);
    triangle_operation[t] = static_cast<std::int64_t>(operations.size());
  };
  const auto touch_vertex = [&](Operation &op, std::size_t i) {
    op.vertices.push_back({i, s.p[i], s.n[i], s.on_surface[i], s.vertex_alive[i]});
    if (i >= vertex_operation.size())
      vertex_operation.resize(i + 1U, -1);
    vertex_operation[i] = static_cast<std::int64_t>(operations.size());
  };
  const auto add_triangle_state = [&](Operation &op, const Tri &t) {
    const std::size_t index = s.tri.size();
    s.tri.push_back(t);
    s.tri_alive.push_back(1U);
    op.created_triangles.push_back(index);
    if (index >= triangle_operation.size())
      triangle_operation.resize(index + 1U, -1);
    triangle_operation[index] = static_cast<std::int64_t>(operations.size());
    return index;
  };
  const auto undo = [&](const Operation &op) {
    for (const std::size_t t : op.created_triangles)
      s.tri_alive[t] = 0U;
    for (const std::size_t i : op.created_vertices)
      s.vertex_alive[i] = 0U;
    for (const auto &[t, old] : op.triangles)
      s.tri[t] = old;
    for (const auto &[t, alive] : op.triangle_alive)
      s.tri_alive[t] = alive;
    for (const VertexOld &old : op.vertices) {
      s.p[old.index] = old.p;
      s.n[old.index] = old.n;
      s.on_surface[old.index] = old.on_surface;
      s.vertex_alive[old.index] = old.alive;
    }
  };

  // Adjacency for the current state.
  std::vector<std::vector<std::uint32_t>> vt;
  std::unordered_map<EdgeKey, std::array<std::size_t, 2>, EdgeKeyHash> et;
  const auto rebuild = [&] {
    const auto rebuild_begin = PhaseClock::now();
    vt.assign(s.p.size(), {});
    et.clear();
    et.reserve(s.tri.size() * 2U);
    for (std::size_t t = 0; t < s.tri.size(); ++t) {
      if (!s.tri_alive[t])
        continue;
      const Tri &tr = s.tri[t];
      for (int k = 0; k < 3; ++k) {
        vt[tr[k]].push_back(static_cast<std::uint32_t>(t));
        auto [it, inserted] = et.try_emplace(
            edge_key(tr[k], tr[(k + 1) % 3]),
            std::array<std::size_t, 2>{t, SIZE_MAX});
        if (!inserted)
          it->second[1] = t;
      }
    }
    ms_rebuild += since(rebuild_begin);
  };
  // For an edge owned by t1 and t2, (u, v, a, b) with t1 = (u, v, a) in
  // winding order and t2 = (v, u, b).
  const auto edge_frame = [&](std::size_t t1, std::size_t t2, std::size_t lo,
                              std::size_t hi, std::array<std::size_t, 4> &f) {
    const Tri &r1 = s.tri[t1];
    const Tri &r2 = s.tri[t2];
    for (int k = 0; k < 3; ++k) {
      const std::size_t x = r1[k], y = r1[(k + 1) % 3];
      if ((x == lo && y == hi) || (x == hi && y == lo)) {
        f = {x, y, r1[(k + 2) % 3], SIZE_MAX};
        for (int j = 0; j < 3; ++j)
          if (r2[j] == y && r2[(j + 1) % 3] == x)
            f[3] = r2[(j + 2) % 3];
        return f[3] != SIZE_MAX && f[2] != f[3];
      }
    }
    return false;
  };
  const auto sorted_edges = [&](bool longest_first) {
    const auto sort_begin = PhaseClock::now();
    std::vector<std::pair<double, EdgeKey>> edges;
    edges.reserve(et.size());
    for (const auto &[key, owners] : et)
      if (owners[1] != SIZE_MAX)
        edges.push_back({norm(s.p[key.lo] - s.p[key.hi]), key});
    std::sort(edges.begin(), edges.end(), [&](const auto &l, const auto &r) {
      if (l.first != r.first)
        return longest_first ? l.first > r.first : l.first < r.first;
      return std::tie(l.second.lo, l.second.hi) < std::tie(r.second.lo, r.second.hi);
    });
    ms_sort += since(sort_begin);
    return edges;
  };

  std::vector<std::array<std::uint32_t, 2>> pairs;
  std::vector<std::size_t> exported_triangle;
  std::vector<std::uint8_t> active_triangles;
  std::size_t scratch_bytes = 0U;
  SurfaceMesh audit_mesh;
  const auto export_mesh = [&](SurfaceMesh &out) {
    std::vector<std::size_t> remap(s.p.size(), SIZE_MAX);
    out.vertices.clear();
    out.triangles.clear();
    out.panels.clear();
    exported_triangle.clear();
    for (std::size_t i = 0; i < s.p.size(); ++i) {
      if (!s.vertex_alive[i])
        continue;
      remap[i] = out.vertices.size();
      Vertex v;
      v.x = s.p[i].x;
      v.y = s.p[i].y;
      v.z = s.p[i].z;
      v.nx = s.n[i].x;
      v.ny = s.n[i].y;
      v.nz = s.n[i].z;
      v.atom_index = s.atom[i];
      out.vertices.push_back(v);
    }
    for (std::size_t t = 0; t < s.tri.size(); ++t) {
      if (!s.tri_alive[t])
        continue;
      Triangle tr;
      tr.v0 = remap[s.tri[t][0]];
      tr.v1 = remap[s.tri[t][1]];
      tr.v2 = remap[s.tri[t][2]];
      tr.atom_index = s.atom[s.tri[t][0]];
      out.triangles.push_back(tr);
      exported_triangle.push_back(t);
      Panel panel;
      panel.atom_index = tr.atom_index;
      panel.has_triangle_geometry = true;
      out.panels.push_back(panel);
      update_triangle_geometry(out, out.triangles.size() - 1U);
    }
  };
  // Audits the pass and undoes the operations behind every intersecting pair.
  // The state before the pass was intersection-free, so undoing all of them
  // always ends the loop.
  const auto settle_pass = [&] {
    std::vector<std::uint8_t> undone(operations.size(), 0U);
    for (int round = 0; round < 16; ++round) {
      ++settle_rounds;
      auto phase_begin = PhaseClock::now();
      export_mesh(audit_mesh);
      ms_export += since(phase_begin);
      phase_begin = PhaseClock::now();
      // Active: a triangle an operation of the pass touched or one with a
      // vertex it moved; the rest are as before the pass, which was
      // intersection-free, so no intersecting pair is missed.
      active_triangles.assign(exported_triangle.size(), 0U);
      for (std::size_t local = 0; local < exported_triangle.size(); ++local) {
        const std::size_t t = exported_triangle[local];
        bool hot = t < triangle_operation.size() && triangle_operation[t] >= 0;
        for (const std::size_t w : s.tri[t])
          hot = hot || (w < vertex_operation.size() && vertex_operation[w] >= 0);
        active_triangles[local] = hot ? 1U : 0U;
      }
      const SelfIntersectionAudit audit = detect_self_intersections(
          audit_mesh, grid_spacing, scratch_bytes, executor, &pairs,
          &active_triangles);
      ms_detect += since(phase_begin);
      if (audit.count == 0U)
        return;
      bool any = false;
      const auto undo_index = [&](std::int64_t index) {
        if (index < 0 || undone[static_cast<std::size_t>(index)])
          return;
        undo(operations[static_cast<std::size_t>(index)]);
        undone[static_cast<std::size_t>(index)] = 1U;
        ++metrics.undone_operations;
        any = true;
      };
      for (const auto &pair : pairs)
        for (const std::uint32_t local : pair) {
          const std::size_t t = exported_triangle[local];
          if (t < triangle_operation.size())
            undo_index(triangle_operation[t]);
          for (const std::size_t w : s.tri[t])
            if (w < vertex_operation.size())
              undo_index(vertex_operation[w]);
        }
      if (!any)
        for (std::size_t k = 0; k < operations.size(); ++k)
          undo_index(static_cast<std::int64_t>(k));
      if (!any)
        return;
    }
    for (std::size_t k = 0; k < operations.size(); ++k)
      if (!undone[k]) {
        undo(operations[k]);
        ++metrics.undone_operations;
      }
  };

  for (int iteration = 0; iteration < iterations; ++iteration) {
    std::size_t changes = 0U;
    // Split.
    rebuild();
    start_pass();
    {
      std::vector<std::uint8_t> locked(s.tri.size(), 0U);
      const auto edges = sorted_edges(true);
      // Every edge the loop reaches is split, so this locking is the loop's.
      std::vector<std::uint8_t> will_lock(s.tri.size(), 0U);
      plan_snaps(plan, edges.size(), [&](std::size_t k, Vec3 &midpoint,
                                         std::size_t &atom) {
        const auto &[length, key] = edges[k];
        if (!(length > 4.0 / 3.0 * smallest_edge))
          return false;
        const double local =
            graded ? local_edge(0.5 * (s.p[key.lo] + s.p[key.hi])) : target_edge;
        if (!(length > 4.0 / 3.0 * local))
          return false;
        const auto owners = et.at(key);
        if (will_lock[owners[0]] || will_lock[owners[1]])
          return false;
        std::array<std::size_t, 4> f{};
        if (!edge_frame(owners[0], owners[1], key.lo, key.hi, f))
          return false;
        will_lock[owners[0]] = will_lock[owners[1]] = 1U;
        midpoint = 0.5 * (s.p[f[0]] + s.p[f[1]]);
        atom = s.atom[f[0]];
        return true;
      });
      const auto split_begin = PhaseClock::now();
      for (std::size_t k = 0; k < edges.size(); ++k) {
        const auto &[length, key] = edges[k];
        if (!(length > 4.0 / 3.0 * smallest_edge))
          break;
        const double local =
            graded ? local_edge(0.5 * (s.p[key.lo] + s.p[key.hi])) : target_edge;
        if (!(length > 4.0 / 3.0 * local))
          continue;
        const double split_min_twice_area = 1.0e-3 * local * local;
        const auto owners = et.at(key);
        const std::size_t t1 = owners[0], t2 = owners[1];
        if (locked[t1] || locked[t2])
          continue;
        std::array<std::size_t, 4> f{};
        if (!edge_frame(t1, t2, key.lo, key.hi, f))
          continue;
        const auto [u, v, a, b] = f;
        double area1 = 0, area2 = 0;
        const Vec3 n1 = normal_of(s.tri[t1], area1), n2 = normal_of(s.tri[t2], area2);
        Operation op;
        const Vec3 midpoint = 0.5 * (s.p[u] + s.p[v]);
        Vec3 m = midpoint, mn{};
        bool snapped = false;
        if (plan[k].planned) {
          m = plan[k].point;
          mn = plan[k].normal;
          snapped = plan[k].snapped != 0U;
        } else {
          const auto loop_snap_begin = PhaseClock::now();
          snapped = snap(midpoint, s.atom[u], m, mn);
          ms_loop_snap += since(loop_snap_begin);
          ++loop_snaps;
        }
        const std::size_t mi = s.p.size();
        s.p.push_back(m);
        s.n.push_back(norm(mn) > 0 ? mn : normalized(s.n[u] + s.n[v]));
        s.atom.push_back(s.atom[u]);
        s.on_surface.push_back(snapped ? 1U : 0U);
        s.vertex_alive.push_back(1U);
        op.created_vertices.push_back(mi);
        if (mi >= vertex_operation.size())
          vertex_operation.resize(mi + 1U, -1);
        vertex_operation[mi] = static_cast<std::int64_t>(operations.size());
        const std::array<Tri, 4> parts = {Tri{u, mi, a}, Tri{mi, v, a},
                                         Tri{v, mi, b}, Tri{mi, u, b}};
        const auto parts_valid = [&] {
          Vec3 star{};
          for (int k = 0; k < 4; ++k) {
            double twice = 0;
            const Vec3 nn = normal_of(parts[k], twice);
            if (!(twice > split_min_twice_area) ||
                dot(nn, k < 2 ? n1 : n2) <= kKeepOrientation)
              return false;
            star = star + twice * nn;
          }
          return sheet_agrees(star, mn);
        };
        if (snapped && !parts_valid()) {
          s.p[mi] = midpoint;
          s.n[mi] = normalized(s.n[u] + s.n[v]);
          s.on_surface[mi] = 0U;
        }
        touch_triangle(op, t1);
        touch_triangle(op, t2);
        s.tri[t1] = parts[0];
        s.tri[t2] = parts[2];
        add_triangle_state(op, parts[1]);
        add_triangle_state(op, parts[3]);
        locked[t1] = locked[t2] = 1U;
        locked.push_back(1U);
        locked.push_back(1U);
        operations.push_back(std::move(op));
        ++metrics.splits;
        ++changes;
      }
      ms_split += since(split_begin);
    }
    settle_pass();
    // Collapse.
    rebuild();
    start_pass();
    {
      std::vector<std::uint8_t> locked(s.p.size(), 0U);
      std::vector<std::size_t> neighbors_u, neighbors_v, common;
      const auto neighbors = [&](std::size_t x, std::vector<std::size_t> &out) {
        out.clear();
        for (const std::uint32_t t : vt[x])
          for (const std::size_t w : s.tri[t])
            if (w != x)
              out.push_back(w);
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
      };
      const auto edges = sorted_edges(false);
      // As if every collapse the loop reaches were accepted; one that is
      // refused leaves its neighbours free, and the loop snaps those itself.
      std::vector<std::uint8_t> will_lock(s.p.size(), 0U);
      plan_snaps(plan, edges.size(), [&](std::size_t k, Vec3 &midpoint,
                                         std::size_t &atom) {
        const auto &[length, key] = edges[k];
        if (!(length < short_edge))
          return false;
        const double local =
            graded ? local_edge(0.5 * (s.p[key.lo] + s.p[key.hi])) : target_edge;
        if (!(length < 4.0 / 5.0 * local))
          return false;
        const auto owners = et.at(key);
        std::array<std::size_t, 4> f{};
        if (!edge_frame(owners[0], owners[1], key.lo, key.hi, f))
          return false;
        const auto [u, v, a, b] = f;
        if (will_lock[u] || will_lock[v] || will_lock[a] || will_lock[b])
          return false;
        if (vt[a].size() <= 3U || vt[b].size() <= 3U)
          return false;
        neighbors(u, neighbors_u);
        neighbors(v, neighbors_v);
        common.clear();
        std::set_intersection(neighbors_u.begin(), neighbors_u.end(),
                              neighbors_v.begin(), neighbors_v.end(),
                              std::back_inserter(common));
        if (common.size() != 2U)
          return false;
        for (const std::size_t w : neighbors_u)
          will_lock[w] = 1U;
        for (const std::size_t w : neighbors_v)
          will_lock[w] = 1U;
        will_lock[u] = will_lock[v] = 1U;
        midpoint = 0.5 * (s.p[u] + s.p[v]);
        atom = s.atom[u];
        return true;
      });
      // An edge the plan did not foresee (a refused collapse freed it) is
      // snapped together with the next unforeseen edges the loop may still
      // reach, in parallel.  An edge the loop reaches has its pass-start
      // endpoints, frame and owner atom, so its snap is the one the loop
      // would compute; one that ends up locked was snapped for nothing.
      std::vector<std::size_t> ahead_u, ahead_v, ahead_common;
      std::size_t ahead_batches = 0;
      const auto snap_ahead = [&](std::size_t first) {
        constexpr std::size_t kAheadBatch = 256;
        picked.clear();
        for (std::size_t j = first;
             j < edges.size() && picked.size() < kAheadBatch; ++j) {
          if (plan[j].planned)
            continue;
          const auto &[length_j, key_j] = edges[j];
          if (!(length_j < short_edge))
            break;
          const double local_j =
              graded ? local_edge(0.5 * (s.p[key_j.lo] + s.p[key_j.hi]))
                     : target_edge;
          if (!(length_j < 4.0 / 5.0 * local_j))
            continue;
          const auto owners_j = et.at(key_j);
          std::array<std::size_t, 4> fj{};
          if (!edge_frame(owners_j[0], owners_j[1], key_j.lo, key_j.hi, fj))
            continue;
          const auto [uj, vj, aj, bj] = fj;
          if (locked[uj] || locked[vj] || locked[aj] || locked[bj])
            continue;
          if (vt[aj].size() <= 3U || vt[bj].size() <= 3U)
            continue;
          neighbors(uj, ahead_u);
          neighbors(vj, ahead_v);
          ahead_common.clear();
          std::set_intersection(ahead_u.begin(), ahead_u.end(), ahead_v.begin(),
                                ahead_v.end(), std::back_inserter(ahead_common));
          if (ahead_common.size() != 2U)
            continue;
          PlannedSnap &pj = plan[j];
          pj.point = 0.5 * (s.p[uj] + s.p[vj]);
          pj.atom = s.atom[uj];
          pj.planned = 1U;
          picked.push_back(j);
        }
        const auto work = [&](std::size_t, std::size_t begin, std::size_t end) {
          for (std::size_t q = begin; q < end; ++q) {
            PlannedSnap &pq = plan[picked[q]];
            const Vec3 midpoint_q = pq.point;
            pq.snapped = snap(midpoint_q, pq.atom, pq.point, pq.normal) ? 1U : 0U;
          }
        };
        if (executor != nullptr && executor->capacity() > 1U && picked.size() > 8U)
          executor->parallel_for(picked.size(), executor->capacity(), 4U, work);
        else
          work(0U, 0U, picked.size());
        loop_snaps += picked.size();
        ++ahead_batches;
      };
      const auto collapse_begin = PhaseClock::now();
      for (std::size_t k = 0; k < edges.size(); ++k) {
        const auto &[length, key] = edges[k];
        if (!(length < short_edge))
          break;
        const double local =
            graded ? local_edge(0.5 * (s.p[key.lo] + s.p[key.hi])) : target_edge;
        if (!(length < 4.0 / 5.0 * local))
          continue;
        const double collapse_long_edge = 4.0 / 3.0 * local;
        const double collapse_min_twice_area = 1.0e-3 * local * local;
        const auto owners = et.at(key);
        const std::size_t t1 = owners[0], t2 = owners[1];
        std::array<std::size_t, 4> f{};
        if (!edge_frame(t1, t2, key.lo, key.hi, f))
          continue;
        const auto [u, v, a, b] = f;
        if (locked[u] || locked[v] || locked[a] || locked[b])
          continue;
        if (vt[a].size() <= 3U || vt[b].size() <= 3U)
          continue;
        neighbors(u, neighbors_u);
        neighbors(v, neighbors_v);
        common.clear();
        std::set_intersection(neighbors_u.begin(), neighbors_u.end(),
                              neighbors_v.begin(), neighbors_v.end(),
                              std::back_inserter(common));
        if (common.size() != 2U)
          continue;
        const Vec3 midpoint = 0.5 * (s.p[u] + s.p[v]);
        Vec3 target = midpoint, tn{};
        bool snapped = false;
        if (!plan[k].planned) {
          const auto loop_snap_begin = PhaseClock::now();
          snap_ahead(k);
          ms_loop_snap += since(loop_snap_begin);
        }
        target = plan[k].point;
        tn = plan[k].normal;
        snapped = plan[k].snapped != 0U;
        bool valid = true;
        const Vec3 saved_u = s.p[u];
        Vec3 star{};
        for (const std::size_t w : {u, v}) {
          for (const std::uint32_t t : vt[w]) {
            if (t == t1 || t == t2)
              continue;
            Tri after = s.tri[t];
            for (auto &corner : after)
              if (corner == u || corner == v)
                corner = u;
            double old_twice = 0, new_twice = 0;
            const Vec3 old_n = normal_of(s.tri[t], old_twice);
            s.p[u] = target;
            const Vec3 new_n = normal_of(after, new_twice);
            s.p[u] = saved_u;
            if (!(new_twice > collapse_min_twice_area) ||
                dot(old_n, new_n) <= kKeepOrientation) {
              valid = false;
              break;
            }
            star = star + new_twice * new_n;
          }
          if (!valid)
            break;
        }
        if (valid && snapped && !sheet_agrees(star, tn))
          valid = false;
        for (const std::size_t w : neighbors_u)
          if (valid && w != v && norm(s.p[w] - target) > collapse_long_edge)
            valid = false;
        for (const std::size_t w : neighbors_v)
          if (valid && w != u && norm(s.p[w] - target) > collapse_long_edge)
            valid = false;
        if (!valid)
          continue;
        Operation op;
        touch_vertex(op, u);
        touch_vertex(op, v);
        touch_triangle(op, t1);
        touch_triangle(op, t2);
        for (const std::uint32_t t : vt[v])
          if (t != t1 && t != t2)
            touch_triangle(op, t);
        for (const std::uint32_t t : vt[u])
          if (t != t1 && t != t2)
            touch_triangle(op, t);
        s.p[u] = target;
        s.n[u] = snapped && norm(tn) > 0 ? tn : normalized(s.n[u] + s.n[v]);
        s.on_surface[u] = snapped ? 1U : 0U;
        s.tri_alive[t1] = s.tri_alive[t2] = 0U;
        for (const std::uint32_t t : vt[v])
          if (t != t1 && t != t2)
            for (auto &corner : s.tri[t])
              if (corner == v)
                corner = u;
        s.vertex_alive[v] = 0U;
        for (const std::size_t w : neighbors_u)
          locked[w] = 1U;
        for (const std::size_t w : neighbors_v)
          locked[w] = 1U;
        locked[u] = locked[v] = 1U;
        operations.push_back(std::move(op));
        ++metrics.collapses;
        ++changes;
      }
      ms_collapse += since(collapse_begin);
      loop_batches += ahead_batches;
    }
    settle_pass();
    // Flip towards Delaunay on nearly flat pairs.
    rebuild();
    start_pass();
    {
      const auto flip_begin = PhaseClock::now();
      std::vector<std::uint8_t> locked(s.tri.size(), 0U);
      std::vector<std::size_t> valence(s.p.size(), 0U);
      for (std::size_t i = 0; i < s.p.size(); ++i)
        valence[i] = vt[i].size();
      std::unordered_set<EdgeKey, EdgeKeyHash> created;
      for (const auto &[length, key] : sorted_edges(true)) {
        (void)length;
        const auto owners = et.at(key);
        const std::size_t t1 = owners[0], t2 = owners[1];
        if (locked[t1] || locked[t2])
          continue;
        std::array<std::size_t, 4> f{};
        if (!edge_frame(t1, t2, key.lo, key.hi, f))
          continue;
        const auto [u, v, a, b] = f;
        if (et.count(edge_key(a, b)) != 0U || created.count(edge_key(a, b)) != 0U ||
            valence[u] <= 3U || valence[v] <= 3U)
          continue;
        double s1 = 0, s2 = 0, s3 = 0, s4 = 0;
        const Vec3 n1 = normal_of(s.tri[t1], s1), n2 = normal_of(s.tri[t2], s2);
        if (!(s1 > 0) || !(s2 > 0) || dot(n1, n2) < 0.95)
          continue;
        const Tri f1{a, u, b}, f2{b, v, a};
        const Vec3 m1 = normal_of(f1, s3), m2 = normal_of(f2, s4);
        const Vec3 mean = normalized(n1 + n2);
        const double flip_min_twice_area =
            graded ? min_twice_area_at(0.5 * (s.p[u] + s.p[v]))
                   : minimum_twice_area;
        if (!(s3 > flip_min_twice_area) || !(s4 > flip_min_twice_area) ||
            dot(m1, mean) < 0.9 || dot(m2, mean) < 0.9)
          continue;
        const double before = std::min(triangle_min_angle(s.p[u], s.p[v], s.p[a]),
                                       triangle_min_angle(s.p[v], s.p[u], s.p[b]));
        const double after = std::min(triangle_min_angle(s.p[a], s.p[u], s.p[b]),
                                      triangle_min_angle(s.p[b], s.p[v], s.p[a]));
        if (!(after > before + 1.0e-3))
          continue;
        Operation op;
        touch_triangle(op, t1);
        touch_triangle(op, t2);
        s.tri[t1] = f1;
        s.tri[t2] = f2;
        locked[t1] = locked[t2] = 1U;
        created.insert(edge_key(a, b));
        --valence[u];
        --valence[v];
        ++valence[a];
        ++valence[b];
        operations.push_back(std::move(op));
        ++metrics.flips;
        ++changes;
      }
      ms_flip += since(flip_begin);
    }
    settle_pass();
    // Tangential relaxation onto the surface.
    rebuild();
    start_pass();
    {
      const std::size_t count = s.p.size();
      std::vector<Vec3> proposal(count);
      std::vector<Vec3> proposal_normal(count);
      std::vector<std::uint8_t> has(count, 0U);
      const auto propose = [&](std::size_t, std::size_t first, std::size_t last) {
        for (std::size_t i = first; i < last; ++i) {
          has[i] = 0U;
          if (!s.vertex_alive[i] || vt[i].empty())
            continue;
          Vec3 centroid{};
          double weight = 0.0;
          for (const std::uint32_t t : vt[i]) {
            double twice = 0;
            (void)normal_of(s.tri[t], twice);
            const Tri &tr = s.tri[t];
            centroid = centroid + (twice / 3.0) * (s.p[tr[0]] + s.p[tr[1]] + s.p[tr[2]]);
            weight += twice;
          }
          if (!(weight > 0))
            continue;
          centroid = (1.0 / weight) * centroid;
          const Vec3 normal = norm(s.n[i]) > 0 ? normalized(s.n[i]) : Vec3{};
          const Vec3 step = centroid - s.p[i];
          const Vec3 tangential = step - dot(step, normal) * normal;
          Vec3 point{}, pn{};
          if (snap(s.p[i] + tangential, s.atom[i], point, pn)) {
            proposal[i] = point;
            proposal_normal[i] = pn;
            has[i] = 1U;
          }
        }
      };
      auto relax_begin = PhaseClock::now();
      if (executor != nullptr && executor->capacity() > 1U)
        executor->parallel_for(count, executor->capacity(), 256U, propose);
      else
        propose(0U, 0U, count);
      ms_propose += since(relax_begin);
      relax_begin = PhaseClock::now();
      for (std::size_t i = 0; i < count; ++i) {
        if (!has[i])
          continue;
        const Vec3 here = s.p[i];
        double before = std::numbers::pi, after = std::numbers::pi;
        bool valid = true;
        Vec3 star{};
        const double relax_min_twice_area =
            graded ? min_twice_area_at(proposal[i]) : minimum_twice_area;
        for (const std::uint32_t t : vt[i]) {
          double old_twice = 0, new_twice = 0;
          const Vec3 old_n = normal_of(s.tri[t], old_twice);
          before = std::min(before, triangle_min_angle(s.p[s.tri[t][0]],
                                                       s.p[s.tri[t][1]],
                                                       s.p[s.tri[t][2]]));
          s.p[i] = proposal[i];
          const Vec3 new_n = normal_of(s.tri[t], new_twice);
          after = std::min(after, triangle_min_angle(s.p[s.tri[t][0]],
                                                     s.p[s.tri[t][1]],
                                                     s.p[s.tri[t][2]]));
          s.p[i] = here;
          if (!(new_twice > relax_min_twice_area) ||
              dot(old_n, new_n) <= kKeepOrientation) {
            valid = false;
            break;
          }
          star = star + new_twice * new_n;
        }
        if (valid && !sheet_agrees(star, proposal_normal[i]))
          valid = false;
        // Moving onto the surface is accepted unless it costs angle quality.
        if (!valid || (s.on_surface[i] && !(after > before - 1.0e-3)))
          continue;
        Operation op;
        touch_vertex(op, i);
        s.p[i] = proposal[i];
        if (norm(proposal_normal[i]) > 0)
          s.n[i] = proposal_normal[i];
        s.on_surface[i] = 1U;
        operations.push_back(std::move(op));
        ++metrics.relaxed;
      }
      ms_accept += since(relax_begin);
    }
    settle_pass();
    ++metrics.iterations;
    if (changes == 0U)
      break;
  }
  auto final_begin = PhaseClock::now();
  export_mesh(mesh);
  ms_export += since(final_begin);
  // Vertex normals off the surface: area-weighted triangle normals.
  std::vector<Vec3> summed(mesh.vertices.size());
  for (const Triangle &t : mesh.triangles)
    for (const std::size_t w : {t.v0, t.v1, t.v2})
      summed[w] = summed[w] + t.area * Vec3{t.nx, t.ny, t.nz};
  std::size_t index = 0;
  for (std::size_t i = 0; i < s.p.size(); ++i) {
    if (!s.vertex_alive[i])
      continue;
    Vertex &v = mesh.vertices[index];
    if (graded && local_edge(s.p[i]) < target_edge)
      ++metrics.refined_vertices;
    if (!s.on_surface[i] || norm(s.n[i]) == 0.0) {
      ++metrics.off_surface_vertices;
      if (norm(summed[index]) > 0) {
        const Vec3 unit = normalized(summed[index]);
        v.nx = unit.x;
        v.ny = unit.y;
        v.nz = unit.z;
      }
    }
    ++index;
  }
  std::size_t final_scratch = 0U;
  final_begin = PhaseClock::now();
  metrics.final_intersections =
      detect_self_intersections(mesh, grid_spacing, final_scratch, executor).count;
  ms_detect += since(final_begin);
  metrics.vertices_after = mesh.vertices.size();
  metrics.ms = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - begin)
                   .count();
  std::cerr << "TABI SES remesh timing: total " << metrics.ms / 1000.0
            << " s over " << metrics.iterations << " iteration(s): rebuild "
            << ms_rebuild / 1000.0 << ", sort " << ms_sort / 1000.0
            << ", select " << ms_select / 1000.0 << ", snap "
            << ms_snap / 1000.0 << ", split " << ms_split / 1000.0
            << ", collapse " << ms_collapse / 1000.0 << " (loop snaps "
            << loop_snaps << " in " << loop_batches << " batch(es), "
            << ms_loop_snap / 1000.0 << " s), flip "
            << ms_flip / 1000.0 << " (incl. its sort), relax propose "
            << ms_propose / 1000.0 << ", relax accept " << ms_accept / 1000.0
            << ", export " << ms_export / 1000.0 << ", self-intersection "
            << ms_detect / 1000.0 << " (" << settle_rounds
            << " settle round(s))\n";
}

AuxiliarySurfaceResult build_patch_ray_experimental_surface(
    analytic_ses::PatchComplex &patches, const Vec3 &lower,
    double grid_spacing, std::size_t nx, std::size_t ny, std::size_t nz,
    TitanExecutor *executor, bool validate_topology,
    bool resolution_policy = false, double remesh_edge = 0.0,
    std::span<const Atom> charges = {}, ChargeRefinement refinement = {},
    int remesh_iterations = 6) {
  const auto begin = std::chrono::steady_clock::now();
  AuxiliarySurfaceResult result;
  result.analytic_ses_engine =
      resolution_policy ? "patch-ray-resolution" : "patch-ray-experimental";
  result.cavity_policy = "external-only";
  result.grid_vertices = nx * ny * nz;
  const analytic_ses::PatchRayGrid rays = analytic_ses::build_patch_ray_grid(
      patches, {lower.x, lower.y, lower.z}, grid_spacing, nx, ny, nz, executor);
  result.ray_build_ms = rays.build_ms;
  result.patch_ray_events = rays.raw_events;
  result.patch_ray_duplicates = rays.duplicate_events;
  result.patch_ray_uncertified_intersection_queries =
      rays.uncertified_intersection_queries;
  result.patch_ray_uncertified_rows_repaired =
      rays.uncertified_rows_repaired;
  result.patch_ray_unresolved_uncertified_rows =
      rays.unresolved_uncertified_rows;
  result.patch_ray_degenerate_rows_repaired =
      rays.degenerate_rows_repaired;
  result.patch_ray_retained_bytes = rays.retained_bytes;
  result.patch_ray_peak_temporary_bytes = rays.peak_temporary_bytes;
  for (const auto &axis : rays.axes)
    result.odd_ray_rows += axis.odd_rows;
  // A ray row whose patch intersection stays uncertified after the perturbed
  // retries makes the certified engine refuse the surface. The resolution
  // policy instead leaves that row out of the parity vote, as it does for an
  // odd row, and its grid vertices take the symbolic-perturbation or
  // morphology path. On one 2,065-atom protein at h = 0.5 A, one such row
  // used to abort the build (and with it a BIE solve); h = 0.45 and 0.6 A
  // were fine.
  const bool uncertified_rows_voted_out =
      resolution_policy && rays.status == "uncertified_quartic";
  if (rays.status != "pass" && !uncertified_rows_voted_out) {
    result.status = "patch_ray_" + rays.status;
    return result;
  }
  // Build the atom-local analytic projection graph once.  Its reentrant probe
  // index is also reused by the rare symbolic-perturbation parity repair,
  // avoiding a second setup and keeping both stages on identical predicates.
  const analytic_ses::LocalPatchProjectionIndex projection =
      analytic_ses::build_local_patch_projection_index(patches);
  result.projection_index_bytes = projection.retained_bytes();

  const auto topology_begin = std::chrono::steady_clock::now();
  analytic_ses::BilevelBitGrid topology_inside(nx, ny, nz, false);
  std::vector<Atom> expanded_atoms = patches.atoms;
  for (Atom &atom : expanded_atoms)
    atom.radius += patches.probe_radius;
  const AtomGrid atom_grid = build_atom_grid(patches.atoms, lower);
  const AtomGrid expanded_grid = build_atom_grid(expanded_atoms, lower);
  const auto &fallback_directions = topology_sample_directions();
  const auto grid_index = [nx, ny](std::size_t ix, std::size_t iy,
                                   std::size_t iz) {
    return ix + nx * (iy + ny * iz);
  };
  // The scale every tolerance below is measured against.  It has to include
  // how far the molecule sits from the origin, not only how big it is: a
  // double holds a coordinate of 1000 A to 1e-13, so a squared distance or a
  // ray crossing computed there carries that much noise whatever the atom
  // radii are.
  double parity_scale = std::max(1.0, grid_spacing);
  for (const Atom &atom : patches.atoms) {
    parity_scale =
        std::max(parity_scale, atom.radius + patches.probe_radius);
    parity_scale = std::max({parity_scale, std::abs(atom.x), std::abs(atom.y),
                             std::abs(atom.z)});
  }
  // How close a boundary may come to a grid vertex before the cheap tests
  // below refuse to certify it either way.  Both of them decide an exact tie
  // by the last bit of an inexact quantity, and a grid vertex can sit exactly
  // on a boundary: at the origin with spacing 0.3 and radius 1.5 the vertex
  // (0.9, 1.2, 0) is exactly on the sphere, since 0.81 + 1.44 = 2.25 = 1.5^2
  // in binary too.  The same molecule a thousand angstrom away puts it 1e-13
  // off and the answer flips.  Leaving the shell uncertified sends those few
  // vertices to the symbolic-perturbation path, which exists for this case.
  const double certificate_margin = std::max(
      1024.0 * std::numeric_limits<double>::epsilon() * parity_scale,
      1.0e-12 * grid_spacing);
  // 0 = unresolved, 1 = outside, 2 = inside.  An axis with an odd number of
  // crossings, or an uncertified row (resolution policy only), is ignored
  // only for the grid vertices that use that row.  This keeps certified
  // parity authoritative everywhere else instead of turning a single
  // degenerate line into a whole-grid morphology fallback.
  // The row indices of vertex_crossings.
  const auto row_of = [nx, ny](std::size_t axis, std::size_t ix,
                               std::size_t iy, std::size_t iz) {
    return axis == 0U ? iy + ny * iz : (axis == 1U ? ix + nx * iz : ix + nx * iy);
  };
  std::vector<std::uint8_t> direct_classification(result.grid_vertices, 0U);
  const auto classify_parity_slices = [&](std::size_t begin_z,
                                           std::size_t end_z) {
    for (std::size_t iz = begin_z; iz < end_z; ++iz) {
      for (std::size_t iy = 0; iy < ny; ++iy) {
        for (std::size_t ix = 0; ix < nx; ++ix) {
          const Vec3 point{
              lower.x + grid_spacing * static_cast<double>(ix),
              lower.y + grid_spacing * static_cast<double>(iy),
              lower.z + grid_spacing * static_cast<double>(iz)};
          const auto crossings = vertex_crossings(rays, ix, iy, iz, nx, ny);
          const std::array<double, 3> coordinates = {point.x, point.y,
                                                      point.z};
          unsigned valid_votes = 0U;
          unsigned inside_votes = 0U;
          for (std::size_t axis = 0; axis < crossings.size(); ++axis) {
            if ((crossings[axis].size() & 1U) != 0U)
              continue;
            if (uncertified_rows_voted_out &&
                rays.axes[axis].row_uncertified(row_of(axis, ix, iy, iz)))
              continue;
            // A crossing that lands on the query coordinate makes this axis
            // parity a tie: upper_bound counts it as passed, which is a
            // choice, not a measurement.  Refuse the axis rather than make
            // the choice.
            if (nearest_crossing_distance(
                    crossings[axis], coordinates[axis],
                    std::numeric_limits<double>::infinity()) <=
                certificate_margin)
              continue;
            ++valid_votes;
            inside_votes += static_cast<unsigned>(
                crossing_parity(crossings[axis], coordinates[axis]));
          }
          // A closed patch complex must agree on all three axis parities at a
          // regular point.  A 2/3 majority is not a certificate: one missing
          // pair of patch events can make two axes agree on the same wrong
          // state while still producing a watertight but scientifically wrong
          // surface.  Keep only unanimous three-axis classifications here.
          if (valid_votes == 3U &&
              (inside_votes == 0U || inside_votes == 3U))
            direct_classification[grid_index(ix, iy, iz)] =
                inside_votes == 3U ? 2U : 1U;
        }
      }
    }
  };
  if (executor != nullptr && executor->capacity() > 1U && nz > 1U) {
    executor->parallel_for(
        nz, std::min(executor->capacity(), nz), 1U,
        [&](std::size_t, std::size_t begin_z, std::size_t end_z) {
          classify_parity_slices(begin_z, end_z);
        });
  } else {
    classify_parity_slices(0U, nz);
  }

  std::vector<std::uint8_t> classification(result.grid_vertices, 0U);
  const std::size_t workers =
      executor == nullptr ? 1U : std::min(executor->capacity(), nz);
  std::vector<std::size_t> geometric_counts(workers, 0U);
  std::vector<std::size_t> confident_counts(workers, 0U);
  std::vector<std::size_t> local_repair_counts(workers, 0U);
  std::vector<std::size_t> nearby_repair_counts(workers, 0U);
  std::vector<std::size_t> morphology_counts(workers, 0U);
  std::vector<std::array<std::vector<double>, 3>> parity_scratch(workers);
  const double parity_duplicate_tolerance = std::max(
      512.0 * std::numeric_limits<double>::epsilon() * parity_scale,
      1.0e-11 * grid_spacing);
  const double symbolic_delta = std::max(
      1.0e-7 * grid_spacing,
      1024.0 * std::numeric_limits<double>::epsilon() * parity_scale);
  const std::array<Vec3, 8> symbolic_offsets = {
      Vec3{symbolic_delta, 0.7548776662466927 * symbolic_delta,
           0.5698402909980532 * symbolic_delta},
      Vec3{-symbolic_delta, 0.7548776662466927 * symbolic_delta,
           0.5698402909980532 * symbolic_delta},
      Vec3{symbolic_delta, -0.7548776662466927 * symbolic_delta,
           0.5698402909980532 * symbolic_delta},
      Vec3{symbolic_delta, 0.7548776662466927 * symbolic_delta,
           -0.5698402909980532 * symbolic_delta},
      Vec3{-symbolic_delta, -0.7548776662466927 * symbolic_delta,
           0.5698402909980532 * symbolic_delta},
      Vec3{-symbolic_delta, 0.7548776662466927 * symbolic_delta,
           -0.5698402909980532 * symbolic_delta},
      Vec3{symbolic_delta, -0.7548776662466927 * symbolic_delta,
           -0.5698402909980532 * symbolic_delta},
      Vec3{-symbolic_delta, -0.7548776662466927 * symbolic_delta,
           -0.5698402909980532 * symbolic_delta}};
  const auto repair_ambiguous_slices = [&](std::size_t worker,
                                            std::size_t begin_z,
                                            std::size_t end_z) {
      for (std::size_t iz = begin_z; iz < end_z; ++iz) {
        for (std::size_t iy = 0; iy < ny; ++iy) {
          for (std::size_t ix = 0; ix < nx; ++ix) {
            const std::size_t index = grid_index(ix, iy, iz);
            const Vec3 point{
                lower.x + grid_spacing * static_cast<double>(ix),
                lower.y + grid_spacing * static_cast<double>(iy),
                lower.z + grid_spacing * static_cast<double>(iz)};
            // Two exact geometric certificates remove the large homogeneous
            // regions from morphology sampling.  The VDW union is contained
            // in the SES body, while points outside every probe-expanded atom
            // cannot belong to it.
            if (point_inside_union(atom_grid, patches.atoms, point,
                                   -certificate_margin)) {
              classification[index] = 2U;
              ++geometric_counts[worker];
              continue;
            }
            if (!point_inside_union(expanded_grid, expanded_atoms, point,
                                    certificate_margin)) {
              classification[index] = 1U;
              ++geometric_counts[worker];
              continue;
            }

            // Certified, unanimous three-axis parity is the topology source in
            // the solvent shell.  The sampled morphology predicate is only a
            // deterministic fallback for a vertex whose ray rows cannot
            // provide a certificate; it must never overwrite certified
            // analytic parity.
            const std::uint8_t parity_state = direct_classification[index];
            if (parity_state != 0U) {
              classification[index] = parity_state;
              ++confident_counts[worker];
              continue;
            }

            // Simulation-of-simplicity for an exact seam/tangency: evaluate
            // the full analytic patch complex at a fixed infinitesimal
            // off-grid point.  The first certified offset is deterministic,
            // dominates a one-ULP change in h, and never invokes the sampled
            // morphology predicate.
            std::uint8_t repaired_state = 0U;
            for (const Vec3 &offset : symbolic_offsets) {
              repaired_state = certify_perturbed_point_parity(
                  patches, projection.reentrant_probes, point + offset,
                  parity_duplicate_tolerance, parity_scratch[worker]);
              if (repaired_state != 0U)
                break;
            }
            if (repaired_state != 0U) {
              classification[index] = repaired_state;
              ++local_repair_counts[worker];
              continue;
            }
            // Next to a ray row whose quartic stayed uncertified, the
            // infinitesimal offsets meet the same near-tangency. A certified
            // point 1e-3 h or 1e-2 h away is still a far better witness than
            // the sampled morphology predicate: it can only differ from the
            // vertex if the surface passes within that distance, which moves
            // the marching surface by less than 0.01 h there. On a 2,065-atom
            // protein at h = 0.5 A, the 22 vertices that used to reach the
            // morphology predicate raised the area 0.3% above its trend in h.
            if (uncertified_rows_voted_out) {
              for (const double scale : {1.0e-3, 1.0e-2}) {
                for (const Vec3 &offset : symbolic_offsets) {
                  repaired_state = certify_perturbed_point_parity(
                      patches, projection.reentrant_probes,
                      point + (scale * grid_spacing / symbolic_delta) * offset,
                      parity_duplicate_tolerance, parity_scratch[worker]);
                  if (repaired_state != 0U)
                    break;
                }
                if (repaired_state != 0U)
                  break;
              }
              if (repaired_state != 0U) {
                classification[index] = repaired_state;
                ++nearby_repair_counts[worker];
                continue;
              }
            }

            bool inside = true;
            for (const Vec3 &direction : fallback_directions) {
              if (!point_inside_union(
                      expanded_grid, expanded_atoms,
                      point + patches.probe_radius * direction)) {
                inside = false;
                break;
              }
            }
            const std::uint8_t morphology_state = inside ? 2U : 1U;
            classification[index] = morphology_state;
            ++morphology_counts[worker];
          }
        }
      }
  };
  if (executor != nullptr && workers > 1U) {
    executor->parallel_for(nz, workers, 1U, repair_ambiguous_slices);
  } else {
    repair_ambiguous_slices(0U, 0U, nz);
  }
  for (std::size_t worker = 0; worker < workers; ++worker) {
    result.geometric_certificate_vertices += geometric_counts[worker];
    result.parity_confident_vertices += confident_counts[worker];
    result.parity_local_repair_vertices += local_repair_counts[worker];
    result.resolution_nearby_parity_vertices += nearby_repair_counts[worker];
    result.morphology_fallback_vertices += morphology_counts[worker];
  }
  result.parity_fallback_vertices = result.morphology_fallback_vertices;
  if (result.parity_fallback_vertices != 0U) {
    result.topology_source = "three_axis_ray_parity_morphology_fallback";
  } else if (result.parity_local_repair_vertices != 0U) {
    result.topology_source =
        "three_axis_ray_parity_symbolic_perturbation";
  } else {
    result.topology_source = "three_axis_ray_parity";
  }
  // Resolution policy: a solvent slab or a molecular plate one grid cell
  // thick along an axis would be polygonized as two facing sheets closer than
  // the spacing.  Fill such solvent slabs, then open such molecular plates,
  // so no feature thinner than one cell reaches the solver.
  if (resolution_policy) {
    const auto pass = [&](std::uint8_t from, std::uint8_t to) {
      std::vector<std::size_t> changed;
      for (std::size_t iz = 1; iz + 1 < nz; ++iz)
        for (std::size_t iy = 1; iy + 1 < ny; ++iy)
          for (std::size_t ix = 1; ix + 1 < nx; ++ix) {
            const std::size_t index = grid_index(ix, iy, iz);
            if (classification[index] != from)
              continue;
            const auto is_to = [&](std::size_t other) {
              return classification[other] == to;
            };
            if ((is_to(index - 1) && is_to(index + 1)) ||
                (is_to(index - nx) && is_to(index + nx)) ||
                (is_to(index - nx * ny) && is_to(index + nx * ny)))
              changed.push_back(index);
          }
      for (const std::size_t index : changed)
        classification[index] = to;
      return changed.size();
    };
    result.resolution_thin_solvent_filled = pass(1U, 2U);
    result.resolution_thin_plates_opened = pass(2U, 1U);
  }
  for (std::size_t iz = 0; iz < nz; ++iz) {
    for (std::size_t iy = 0; iy < ny; ++iy) {
      for (std::size_t ix = 0; ix < nx; ++ix) {
        if (classification[grid_index(ix, iy, iz)] == 2U)
          topology_inside.set(ix, iy, iz, true);
      }
    }
  }

  analytic_ses::BilevelBitGrid exterior(nx, ny, nz, false);
  analytic_ses::BilevelBitGrid cavity_fill(nx, ny, nz, false);
  std::vector<std::size_t> queue;
  queue.reserve(result.grid_vertices / 16U);
  const auto linear_index = [nx, ny](std::size_t ix, std::size_t iy,
                                     std::size_t iz) {
    return ix + nx * (iy + ny * iz);
  };
  const auto seed = [&](std::size_t ix, std::size_t iy, std::size_t iz) {
    if (!topology_inside.get(ix, iy, iz) && !exterior.get(ix, iy, iz)) {
      exterior.set(ix, iy, iz, true);
      queue.push_back(linear_index(ix, iy, iz));
    }
  };
  for (std::size_t iz = 0; iz < nz; ++iz) {
    for (std::size_t iy = 0; iy < ny; ++iy) {
      seed(0, iy, iz);
      seed(nx - 1U, iy, iz);
    }
  }
  for (std::size_t iz = 0; iz < nz; ++iz) {
    for (std::size_t ix = 0; ix < nx; ++ix) {
      seed(ix, 0, iz);
      seed(ix, ny - 1U, iz);
    }
  }
  for (std::size_t iy = 0; iy < ny; ++iy) {
    for (std::size_t ix = 0; ix < nx; ++ix) {
      seed(ix, iy, 0);
      seed(ix, iy, nz - 1U);
    }
  }
  for (std::size_t head = 0; head < queue.size(); ++head) {
    const std::size_t index = queue[head];
    const std::size_t iz = index / (nx * ny);
    const std::size_t remainder = index - iz * nx * ny;
    const std::size_t iy = remainder / nx;
    const std::size_t ix = remainder - iy * nx;
    const auto visit = [&](std::size_t x, std::size_t y, std::size_t z) {
      if (!topology_inside.get(x, y, z) && !exterior.get(x, y, z)) {
        exterior.set(x, y, z, true);
        queue.push_back(linear_index(x, y, z));
      }
    };
    if (ix > 0)
      visit(ix - 1U, iy, iz);
    if (ix + 1U < nx)
      visit(ix + 1U, iy, iz);
    if (iy > 0)
      visit(ix, iy - 1U, iz);
    if (iy + 1U < ny)
      visit(ix, iy + 1U, iz);
    if (iz > 0)
      visit(ix, iy, iz - 1U);
    if (iz + 1U < nz)
      visit(ix, iy, iz + 1U);
  }
  for (std::size_t iz = 0; iz < nz; ++iz) {
    for (std::size_t iy = 0; iy < ny; ++iy) {
      for (std::size_t ix = 0; ix < nx; ++ix) {
        if (topology_inside.get(ix, iy, iz) || exterior.get(ix, iy, iz) ||
            cavity_fill.get(ix, iy, iz))
          continue;
        ++result.filled_cavity_regions;
        queue.clear();
        const auto fill = [&](std::size_t x, std::size_t y, std::size_t z) {
          if (topology_inside.get(x, y, z) || exterior.get(x, y, z) ||
              cavity_fill.get(x, y, z))
            return;
          cavity_fill.set(x, y, z, true);
          topology_inside.set(x, y, z, true);
          queue.push_back(linear_index(x, y, z));
          ++result.filled_cavity_vertices;
        };
        fill(ix, iy, iz);
        for (std::size_t head = 0; head < queue.size(); ++head) {
          const std::size_t index = queue[head];
          const std::size_t z = index / (nx * ny);
          const std::size_t remainder = index - z * nx * ny;
          const std::size_t y = remainder / nx;
          const std::size_t x = remainder - y * nx;
          if (x > 0U)
            fill(x - 1U, y, z);
          if (x + 1U < nx)
            fill(x + 1U, y, z);
          if (y > 0U)
            fill(x, y - 1U, z);
          if (y + 1U < ny)
            fill(x, y + 1U, z);
          if (z > 0U)
            fill(x, y, z - 1U);
          if (z + 1U < nz)
            fill(x, y, z + 1U);
        }
      }
    }
  }
  topology_inside.compact();
  exterior.compact();
  cavity_fill.compact();
  result.topology_mixed_blocks = topology_inside.mixed_blocks();
  result.topology_retained_bytes =
      topology_inside.retained_bytes() + exterior.retained_bytes() +
      cavity_fill.retained_bytes();

  const std::vector<ActivePatchRayCell> active =
      collect_active_patch_ray_cells(topology_inside, nx, ny, nz, executor);
  SparseGridMagnitude morphology_magnitude;
  if (result.parity_fallback_vertices != 0U) {
    morphology_magnitude = build_sparse_morphology_magnitude(
        active, expanded_grid, expanded_atoms, cavity_fill,
        fallback_directions, lower, patches.probe_radius, grid_spacing, nx, ny,
        nz, executor);
    result.topology_retained_bytes += morphology_magnitude.retained_bytes();
  }
  result.topology_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - topology_begin)
                           .count();

  const auto polygonize_begin = std::chrono::steady_clock::now();
  const AtomGrid ownership_grid = build_atom_grid(patches.atoms, lower);
  const auto active_metrics = polygonize_patch_ray_grid(
      result.mesh, ownership_grid, patches.atoms, lower, grid_spacing, nx, ny,
      rays, topology_inside, morphology_magnitude, active);
  result.active_cells = active_metrics.first;
  result.active_cell_bytes = active_metrics.second;
  result.polygonize_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - polygonize_begin)
                             .count();

  const auto repair_begin = std::chrono::steady_clock::now();
  orient_mesh_consistently(result.mesh);
  const bool removed_components =
      remove_nonphysical_components(result.mesh, patches.atoms) > 0U;
  if (removed_components)
    orient_mesh_consistently(result.mesh);
  if (resolution_policy) {
    ResolutionProjectionMetrics resolution;
    resolution_project_patch_ray_surface(result.mesh, patches, projection,
                                         grid_spacing, executor, resolution);
    result.projection_ms = resolution.ms;
    result.projected_vertices = resolution.projected;
    result.projection_queries = result.mesh.vertices.size();
    result.rejected_projection_vertices =
        result.mesh.vertices.size() - resolution.projected;
    result.resolution_missing_projection_vertices = resolution.missing;
    result.resolution_star_refused_vertices = resolution.star_refused;
    result.resolution_sheet_refused_vertices = resolution.sheet_refused;
    result.resolution_partial_projection_vertices = resolution.partial;
    // Isotropic remeshing onto the analytic SES; remesh_edge 0 means one grid
    // spacing and a negative value keeps the marching connectivity.
    ResolutionRemeshMetrics remeshed;
    if (remesh_edge >= 0.0 && resolution.final_intersections == 0U) {
      const double edge = remesh_edge > 0.0 ? remesh_edge : grid_spacing;
      const RemeshSizing sizing(charges, edge, refinement);
      remesh_resolution_surface(result.mesh, patches, projection, grid_spacing,
                                edge, remesh_iterations, executor, remeshed,
                                &sizing);
      result.resolution_refine_charges = sizing.charges();
      result.resolution_refined_vertices = remeshed.refined_vertices;
      result.resolution_remesh_iterations = remeshed.iterations;
      result.resolution_remesh_reverted = remeshed.undone_operations;
      resolution.final_intersections = remeshed.final_intersections;
      result.resolution_remesh_splits = remeshed.splits;
      result.resolution_remesh_collapses = remeshed.collapses;
      result.resolution_remesh_flips = remeshed.flips;
      result.resolution_remesh_off_surface = remeshed.off_surface_vertices;
      result.resolution_remesh_sheet_refused = remeshed.sheet_refused;
      result.resolution_remesh_edge = remeshed.target_edge;
      result.resolution_remesh_ms = remeshed.ms;
    }
    ResolutionQualityMetrics quality;
    measure_resolution_quality(result.mesh, quality);
    result.resolution_angle_below_5 = quality.angle_below_5;
    result.resolution_angle_below_10 = quality.angle_below_10;
    result.resolution_angle_below_20 = quality.angle_below_20;
    result.resolution_edge_min = quality.edge_min;
    result.resolution_edge_mean = quality.edge_mean;
    result.resolution_short_edges_005 = quality.edges_below_005 / 2U;
    result.resolution_short_edges_020 = quality.edges_below_020 / 2U;
    result.resolution_reverted_vertices = resolution.reverted;
    result.resolution_revert_rounds = resolution.rounds;
    result.resolution_initial_intersections =
        resolution.initial_intersections;
    result.analytic_self_intersections = resolution.final_intersections;
    result.resolution_unprojected_distance_max =
        resolution.unprojected_distance_max;
    result.analytic_projection_distance_rms =
        resolution.projected_distance_rms;
    result.repair_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - repair_begin)
                           .count();
    if (result.mesh.triangles.empty()) {
      result.status = "empty_analytic_surface";
    } else {
      orient_mesh_consistently(result.mesh);
      mark_analytic_vertex_normals(result.mesh, patches, projection,
                                   grid_spacing, executor,
                                   result.resolution_analytic_normal_vertices,
                                   result.resolution_analytic_normal_refused);
      result.trace_spaces = build_closed_surface_trace_spaces(result.mesh);
      result.status = result.trace_spaces.quality.failure_reason;
      // Odd ray rows and morphology-fallback signs change only which side of
      // a cell a few grid vertices fall on; the marching mesh stays closed.
      // They are reported, not refused.  Intersections are still refused.
      if (result.status == "pass" && resolution.final_intersections != 0U)
        result.status = "resolution_self_intersection";
      if (!validate_topology && result.status == "pass")
        result.trace_spaces.quality.failure_reason = "pass";
    }
    result.build_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - begin)
                          .count();
    return result;
  }
  LocalPatchRepairMetrics local_metrics;
  repair_patch_ray_surface_local(result.mesh, patches, projection,
                                  grid_spacing, local_metrics, executor);
  const std::size_t final_orientation_flips =
      orient_mesh_consistently(result.mesh);
  if (final_orientation_flips != 0U) {
    // The final topology orientation pass changes triangle winding but not
    // geometry.  Any cached face-normal and mesh-to-patch audit is therefore
    // stale and must be recomputed before it is exposed as a certificate.
    const auto orientation_begin = std::chrono::steady_clock::now();
    certify_all_face_orientations(result.mesh, patches, projection,
                                  grid_spacing, local_metrics, true);
    const double orientation_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - orientation_begin)
            .count();
    local_metrics.orientation_certificate_ms += orientation_ms;
    if (local_metrics.analytic_self_intersections != 0U)
      local_metrics.analytic_orientation_status = "self_intersection";

    const auto sampled_audit_begin = std::chrono::steady_clock::now();
    audit_surface_against_analytic_patches(
        result.mesh, patches, projection, grid_spacing, local_metrics);
    const double sampled_audit_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - sampled_audit_begin)
            .count();
    local_metrics.sampled_analytic_audit_ms += sampled_audit_ms;
    if (!local_metrics.analytic_vertex_certificate)
      local_metrics.analytic_audit_status =
          analytic_vertex_failure_status(local_metrics);
    local_metrics.projection_ms += orientation_ms + sampled_audit_ms;
  }
#ifdef TITAN_TABI_SES_TEST_HOOKS
  emit_projection_failure_diagnostics(result.mesh, patches, projection,
                                      grid_spacing, local_metrics);
  emit_projection_star_triangulation_diagnostics(
      result.mesh, patches, projection, grid_spacing, local_metrics,
      executor);
#endif
  result.projection_ms = local_metrics.projection_ms;
  result.local_repair_ms = local_metrics.repair_ms;
  result.orientation_certificate_ms =
      local_metrics.orientation_certificate_ms;
  result.self_intersection_audit_ms =
      local_metrics.self_intersection_audit_ms;
  result.sampled_analytic_audit_ms =
      local_metrics.sampled_analytic_audit_ms;
  result.initial_collapse_ms = local_metrics.initial_collapse_ms;
  result.stable_sliver_ms = local_metrics.stable_sliver_ms;
  result.post_projection_collapse_ms =
      local_metrics.post_projection_collapse_ms;
  result.local_sliver_ms = local_metrics.local_sliver_ms;
  result.full_projection_ms = local_metrics.full_projection_ms;
  result.seam_weld_ms = local_metrics.seam_weld_ms;
  result.refused_sliver_repair_ms = local_metrics.refused_sliver_repair_ms;
  result.certificate_cleanup_ms = local_metrics.certificate_cleanup_ms;
  result.orientation_repair_ms = local_metrics.orientation_repair_ms;
  result.wide_repair_ms = local_metrics.wide_repair_ms;
  result.projection_index_bytes = local_metrics.projection_index_bytes;
  result.repair_scratch_bytes = local_metrics.scratch_bytes;
  result.projected_vertices = local_metrics.projected_vertices;
  result.projection_queries = local_metrics.projection_queries;
  result.projection_hits = local_metrics.projection_hits;
  result.rejected_projection_vertices =
      local_metrics.rejected_projection_vertices;
  result.rejected_projection_rounds =
      local_metrics.rejected_projection_rounds;
  result.projection_orientation_reversals =
      local_metrics.projection_orientation_reversals;
  result.coupled_projection_components_attempted =
      local_metrics.coupled_projection_components_attempted;
  result.coupled_projection_components_accepted =
      local_metrics.coupled_projection_components_accepted;
  result.coupled_projection_vertices_rescued =
      local_metrics.coupled_projection_vertices_rescued;
  result.coupled_projection_max_component_size =
      local_metrics.coupled_projection_max_component_size;
  result.coupled_projection_unresolved_vertices =
      local_metrics.coupled_projection_unresolved_vertices;
  result.orientation_repair_rounds =
      local_metrics.orientation_repair_rounds;
  result.orientation_repair_faces = local_metrics.orientation_repair_faces;
  result.wide_repair_attempts = local_metrics.wide_repair_attempts;
  result.topology_certificate_accepts =
      local_metrics.topology_certificate_accepts;
  result.topology_fallback_audits =
      local_metrics.topology_fallback_audits;
  result.welded_vertices = local_metrics.welded_vertices;
  result.collapsed_edges = local_metrics.collapsed_edges;
  result.flipped_edges = local_metrics.flipped_edges;
  result.poor_triangles_before = local_metrics.poor_triangles_before;
  result.poor_triangles_after = local_metrics.poor_triangles_after;
  result.triangle_quality_min_before =
      local_metrics.triangle_quality_min_before;
  result.triangle_quality_min_after =
      local_metrics.triangle_quality_min_after;
  result.max_edge_ratio_before = local_metrics.max_edge_ratio_before;
  result.max_edge_ratio_after = local_metrics.max_edge_ratio_after;
  result.analytic_audit_status = local_metrics.analytic_audit_status;
  result.analytic_audit_samples = local_metrics.analytic_audit_samples;
  result.analytic_audit_hits = local_metrics.analytic_audit_hits;
  result.analytic_audit_misses = local_metrics.analytic_audit_misses;
  result.analytic_projection_distance_rms =
      local_metrics.analytic_projection_distance_rms;
  result.analytic_projection_distance_max =
      local_metrics.analytic_projection_distance_max;
  result.analytic_patch_residual_rms =
      local_metrics.analytic_patch_residual_rms;
  result.analytic_patch_residual_max =
      local_metrics.analytic_patch_residual_max;
  result.analytic_normal_angle_rms_deg =
      local_metrics.analytic_normal_angle_rms_deg;
  result.analytic_normal_angle_max_deg =
      local_metrics.analytic_normal_angle_max_deg;
  result.sampled_hausdorff_proxy = local_metrics.sampled_hausdorff_proxy;
  result.analytic_vertex_projection_misses =
      local_metrics.analytic_vertex_projection_misses;
  result.analytic_vertex_residual_max =
      local_metrics.analytic_vertex_residual_max;
  result.analytic_vertex_refused =
      local_metrics.refused_vertex_indices.size();
  result.analytic_vertex_refused_distance_max =
      local_metrics.analytic_vertex_refused_distance_max;
  result.analytic_vertex_refused_for_shape =
      local_metrics.analytic_vertex_refused_for_shape;
  result.refused_sliver_collapses = local_metrics.refused_sliver_collapses;
  result.analytic_sheet_bridge_faces =
      local_metrics.analytic_sheet_bridge_faces;
  result.analytic_vertex_certificate =
      local_metrics.analytic_vertex_certificate;
  result.analytic_orientation_status =
      local_metrics.analytic_orientation_status;
  result.analytic_orientation_certified_faces =
      local_metrics.analytic_orientation_certified_faces;
  result.analytic_orientation_centroid_queries =
      local_metrics.analytic_orientation_centroid_queries;
  result.analytic_orientation_uncertified_faces =
      local_metrics.analytic_orientation_uncertified_faces;
  result.analytic_inverted_faces = local_metrics.analytic_inverted_faces;
  result.analytic_self_intersections =
      local_metrics.analytic_self_intersections;
  result.analytic_self_intersection_first =
      local_metrics.analytic_self_intersection_first;
  result.analytic_self_intersection_second =
      local_metrics.analytic_self_intersection_second;
  result.analytic_orientation_min_cosine =
      local_metrics.analytic_orientation_min_cosine;
  result.analytic_normal_quality_status =
      local_metrics.analytic_normal_quality_status;
  result.analytic_orientation_smooth_faces =
      local_metrics.analytic_orientation_smooth_faces;
  result.analytic_orientation_seam_faces =
      local_metrics.analytic_orientation_seam_faces;
  result.analytic_normal_cosine_below_025_faces =
      local_metrics.analytic_normal_cosine_below_025_faces;
  result.analytic_normal_cosine_below_050_faces =
      local_metrics.analytic_normal_cosine_below_050_faces;
  result.analytic_normal_area_weighted_rms_deg =
      local_metrics.analytic_normal_area_weighted_rms_deg;
  result.analytic_normal_p99_deg =
      local_metrics.analytic_normal_p99_deg;
  result.analytic_smooth_normal_p99_deg =
      local_metrics.analytic_smooth_normal_p99_deg;
  result.analytic_seam_normal_p99_deg =
      local_metrics.analytic_seam_normal_p99_deg;
  result.analytic_normal_area_below_cosine_025_fraction =
      local_metrics.analytic_normal_area_below_cosine_025_fraction;
  result.analytic_normal_area_below_cosine_050_fraction =
      local_metrics.analytic_normal_area_below_cosine_050_fraction;
  result.repair_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - repair_begin)
                         .count();
  if (result.mesh.triangles.empty()) {
    result.status = "empty_analytic_surface";
  } else {
    orient_mesh_consistently(result.mesh);
    result.trace_spaces = build_closed_surface_trace_spaces(result.mesh);
    result.status = result.trace_spaces.quality.failure_reason;
    if (result.status == "pass" && result.odd_ray_rows != 0U)
      result.status = "uncertified_odd_ray_rows";
    if (result.status == "pass" &&
        result.morphology_fallback_vertices != 0U)
      result.status = "uncertified_topology_fallback";
    if (result.status == "pass" &&
        result.analytic_orientation_status != "pass")
      result.status = "analytic_" + result.analytic_orientation_status;
    if (result.status == "pass" &&
        result.analytic_audit_status != "pass")
      result.status = "analytic_" + result.analytic_audit_status;
    if (!validate_topology && result.status == "pass")
      result.trace_spaces.quality.failure_reason = "pass";
  }
  result.build_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - begin)
                        .count();
  return result;
}

// Nearest point to x on the boundary of a union of balls (the atoms grown by
// the probe radius, whose boundary is the solvent-accessible surface), looked
// for within `reach`.  On that boundary the nearest point is the radial
// projection onto one sphere, the projection onto the circle where two
// spheres meet, or a point where three meet; each candidate counts only when
// no other ball covers it.  The distance from a sphere's surface bounds every
// candidate on that sphere from below, so spheres are taken nearest first and
// the search stops once the bound passes the best point found.  `scratch`
// holds the nearby balls between calls.  False when no boundary point lies
// within reach.
struct BallBoundaryCandidate {
  std::size_t index;
  double surface_distance;
};
bool nearest_ball_union_boundary(const AtomGrid &grid,
                                 std::span<const Atom> balls, const Vec3 &x,
                                 double reach,
                                 std::vector<BallBoundaryCandidate> &scratch,
                                 std::vector<std::size_t> &cover,
                                 Vec3 &nearest, double &distance) {
  scratch.clear();
  cover.clear();
  const CellKey center = cell_for(grid, x);
  const int span = std::max(
      1, static_cast<int>(std::ceil((grid.max_radius + reach) / grid.cell_size)));
  for (int dz = -span; dz <= span; ++dz)
    for (int dy = -span; dy <= span; ++dy)
      for (int dx = -span; dx <= span; ++dx) {
        const auto found =
            grid.cells.find({center.x + dx, center.y + dy, center.z + dz});
        if (found == grid.cells.end())
          continue;
        for (const std::size_t index : found->second) {
          const Atom &ball = balls[index];
          const double r = norm(x - Vec3{ball.x, ball.y, ball.z});
          // A ball covering any point within reach of x.
          if (!(r < ball.radius + reach))
            continue;
          cover.push_back(index);
          const double surface = std::abs(r - ball.radius);
          if (surface < reach)
            scratch.push_back({index, surface});
        }
      }
  std::sort(scratch.begin(), scratch.end(),
            [](const BallBoundaryCandidate &a, const BallBoundaryCandidate &b) {
              return a.surface_distance < b.surface_distance ||
                     (a.surface_distance == b.surface_distance &&
                      a.index < b.index);
            });
  const auto centre_of = [&](std::size_t i) {
    return Vec3{balls[i].x, balls[i].y, balls[i].z};
  };
  // Strictly inside another ball; a point on that ball's sphere is still on
  // the boundary.
  const auto covered = [&](const Vec3 &p, std::size_t a, std::size_t b,
                           std::size_t c) {
    for (const std::size_t j : cover) {
      if (j == a || j == b || j == c)
        continue;
      const double rj = balls[j].radius - 1.0e-10;
      const Vec3 d = p - centre_of(j);
      if (dot(d, d) < rj * rj)
        return true;
    }
    return false;
  };
  constexpr std::size_t none = static_cast<std::size_t>(-1);
  double best = reach;
  bool found_point = false;
  const auto offer = [&](const Vec3 &p, std::size_t a, std::size_t b,
                         std::size_t c) {
    const double d = norm(x - p);
    if (d < best && !covered(p, a, b, c)) {
      best = d;
      nearest = p;
      found_point = true;
    }
  };
  for (const auto &s : scratch) {
    if (!(s.surface_distance < best))
      break;
    const Vec3 c = centre_of(s.index);
    const double r = norm(x - c);
    if (r > 1.0e-12)
      offer(c + (balls[s.index].radius / r) * (x - c), s.index, none, none);
  }
  // Circles: the bound of a pair is the larger surface distance, which is the
  // second ball's in this order.
  const auto circle = [&](std::size_t a, std::size_t b, Vec3 &m, Vec3 &axis,
                          double &radius) {
    const Vec3 ca = centre_of(a), cb = centre_of(b);
    const double ra = balls[a].radius, rb = balls[b].radius;
    const double d = norm(cb - ca);
    if (!(d > 1.0e-12) || !(d < ra + rb) || !(d > std::abs(ra - rb)))
      return false;
    axis = (1.0 / d) * (cb - ca);
    const double t = (d * d + ra * ra - rb * rb) / (2.0 * d);
    const double r2 = ra * ra - t * t;
    if (!(r2 > 0.0))
      return false;
    m = ca + t * axis;
    radius = std::sqrt(r2);
    return true;
  };
  for (std::size_t jb = 1; jb < scratch.size(); ++jb) {
    if (!(scratch[jb].surface_distance < best))
      break;
    for (std::size_t ja = 0; ja < jb; ++ja) {
      const std::size_t a = scratch[ja].index, b = scratch[jb].index;
      Vec3 m, axis;
      double radius = 0.0;
      if (!circle(a, b, m, axis, radius))
        continue;
      const Vec3 w = (x - m) - dot(x - m, axis) * axis;
      const double wl = norm(w);
      // On the axis every circle point is equally near: take any.
      const Vec3 e = wl > 1.0e-12
                         ? (1.0 / wl) * w
                         : normalized(cross(axis, std::abs(axis.x) < 0.9
                                                      ? Vec3{1.0, 0.0, 0.0}
                                                      : Vec3{0.0, 1.0, 0.0}));
      offer(m + radius * e, a, b, none);
    }
  }
  // Points where three spheres meet.
  for (std::size_t jc = 2; jc < scratch.size(); ++jc) {
    if (!(scratch[jc].surface_distance < best))
      break;
    for (std::size_t jb = 1; jb < jc; ++jb)
      for (std::size_t ja = 0; ja < jb; ++ja) {
        const std::size_t a = scratch[ja].index, b = scratch[jb].index,
                          c = scratch[jc].index;
        const Vec3 ca = centre_of(a), cb = centre_of(b), cc = centre_of(c);
        const double ra = balls[a].radius, rb = balls[b].radius,
                     rc = balls[c].radius;
        const double d = norm(cb - ca);
        if (!(d > 1.0e-12))
          continue;
        const Vec3 ex = (1.0 / d) * (cb - ca);
        const double i = dot(ex, cc - ca);
        const Vec3 off = (cc - ca) - i * ex;
        const double j = norm(off);
        if (!(j > 1.0e-9))
          continue;
        const Vec3 ey = (1.0 / j) * off;
        const Vec3 ez = cross(ex, ey);
        const double px = (ra * ra - rb * rb + d * d) / (2.0 * d);
        const double py = (ra * ra - rc * rc + i * i + j * j) / (2.0 * j) -
                          (i / j) * px;
        const double pz2 = ra * ra - px * px - py * py;
        if (!(pz2 >= 0.0))
          continue;
        const double pz = std::sqrt(pz2);
        const Vec3 base = ca + px * ex + py * ey;
        offer(base + pz * ez, a, b, c);
        offer(base - pz * ez, a, b, c);
      }
  }
  distance = best;
  return found_point;
}

// The solvent-excluded surface as a level set: f(x) = probe - dist(x, P),
// with P the probe centres that touch no atom, i.e. everything outside the
// probe-grown atoms.  f is negative inside the SES, zero exactly on it and,
// near it, a signed distance to it, so interpolating its zero crossings
// converges with the grid spacing.  A point outside the grown atoms is itself
// in P and gets probe + its distance to them.  `reach` bounds the search:
// beyond it f is only bounded, which is enough for a vertex whose value no
// zero crossing of the marching cells depends on.
double exact_ses_field(const AtomGrid &grid, std::span<const Atom> balls,
                       const Vec3 &x, double probe_radius, double reach,
                       std::vector<BallBoundaryCandidate> &scratch,
                       std::vector<std::size_t> &cover) {
  const double sas = sample_union_sdf(grid, balls, x).value;
  if (sas >= 0.0)
    return probe_radius + sas;
  // -sas is the depth into the deepest ball, which never exceeds the
  // distance to the union's boundary.
  if (-sas >= reach)
    return probe_radius + sas;
  Vec3 nearest;
  double distance = 0.0;
  if (!nearest_ball_union_boundary(grid, balls, x, reach, scratch, cover,
                                   nearest, distance))
    return probe_radius - reach;
  return probe_radius - distance;
}

// Moves each marching vertex onto the SES along the line from its nearest
// probe position, where the normal is exact: the SES is the envelope of probe
// balls centred on the accessible surface, and the probe that touches the
// solute at a point lies along its normal.  A vertex keeps its marching
// position and star normal when the move is more than a quarter cell, when
// the projected point turns out to lie nearer another probe position (a cusp
// where two reentrant faces meet), when its normal faces away from the star
// of its triangles, or when the move reverses an incident triangle.
void project_grid_ses_vertices(SurfaceMesh &mesh, const AtomGrid &grid,
                               std::span<const Atom> balls,
                               double probe_radius, double grid_spacing,
                               TitanExecutor *executor, std::size_t &projected,
                               std::size_t &kept) {
  const std::size_t n = mesh.vertices.size();
  std::vector<Vec3> star(n);
  for (const Triangle &t : mesh.triangles)
    for (const std::size_t w : {t.v0, t.v1, t.v2})
      star[w] = star[w] + t.area * Vec3{t.nx, t.ny, t.nz};
  std::vector<Vec3> target(n), normal(n);
  std::vector<std::uint8_t> move(n, 0U);
  const double reach = probe_radius + 2.0 * grid_spacing;
  const auto solve = [&](std::size_t, std::size_t begin, std::size_t end) {
    std::vector<BallBoundaryCandidate> scratch;
    std::vector<std::size_t> cover;
    for (std::size_t i = begin; i < end; ++i) {
      const Vertex &v = mesh.vertices[i];
      const Vec3 x{v.x, v.y, v.z};
      Vec3 probe;
      double distance = 0.0;
      if (!nearest_ball_union_boundary(grid, balls, x, reach, scratch, cover,
                                       probe, distance) ||
          !(distance > 1.0e-9) ||
          !(std::abs(distance - probe_radius) <= 0.25 * grid_spacing))
        continue;
      const Vec3 outward = (1.0 / distance) * (probe - x);
      const Vec3 on_surface = probe - probe_radius * outward;
      const double star_length = norm(star[i]);
      if (!(star_length > 0.0) ||
          dot(outward, star[i]) < kResolutionSheetCosine * star_length)
        continue;
      Vec3 check;
      double check_distance = 0.0;
      if (!nearest_ball_union_boundary(grid, balls, on_surface, reach, scratch,
                                       cover, check, check_distance) ||
          !(std::abs(check_distance - probe_radius) <= 1.0e-6))
        continue;
      target[i] = on_surface;
      normal[i] = outward;
      move[i] = 1U;
    }
  };
  if (executor != nullptr && executor->capacity() > 1U)
    executor->parallel_for(n, executor->capacity(), 256U, solve);
  else
    solve(0U, 0U, n);
  const std::vector<Vertex> original = mesh.vertices;
  const auto position = [&](std::size_t i) {
    return move[i] ? target[i]
                   : Vec3{original[i].x, original[i].y, original[i].z};
  };
  // A triangle whose normal turns against its marching normal holds a
  // vertex back; repeat until none does.
  for (int round = 0; round < 8; ++round) {
    bool reverted = false;
    for (const Triangle &t : mesh.triangles) {
      if (!(move[t.v0] || move[t.v1] || move[t.v2]))
        continue;
      const Vec3 a = position(t.v0), b = position(t.v1), c = position(t.v2);
      if (dot(cross(b - a, c - a), Vec3{t.nx, t.ny, t.nz}) > 0.0)
        continue;
      for (const std::size_t w : {t.v0, t.v1, t.v2})
        if (move[w]) {
          move[w] = 0U;
          reverted = true;
        }
    }
    if (!reverted)
      break;
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (!move[i])
      continue;
    Vertex &v = mesh.vertices[i];
    v.x = target[i].x;
    v.y = target[i].y;
    v.z = target[i].z;
    v.nx = normal[i].x;
    v.ny = normal[i].y;
    v.nz = normal[i].z;
    v.normal_analytic = true;
  }
  for (std::size_t t = 0; t < mesh.triangles.size(); ++t)
    update_triangle_geometry(mesh, t);
  // Where the marching cells joined two sheets of the SES that pass within a
  // cell of each other, the moves pull the joined vertices onto different
  // sheets and the surface crosses itself without any triangle turning over.
  // Every moved vertex of an intersecting pair goes back to its marching
  // position, until no pair is left or none of its vertices moved.
  std::vector<std::array<std::uint32_t, 2>> pairs;
  for (int round = 0; round < 16; ++round) {
    std::size_t scratch_bytes = 0;
    if (detect_self_intersections(mesh, grid_spacing, scratch_bytes, executor,
                                  &pairs)
            .count == 0U)
      break;
    std::vector<std::uint8_t> touched(n, 0U);
    bool any = false;
    for (const auto &pair : pairs)
      for (const std::uint32_t t : pair) {
        const Triangle &tri = mesh.triangles[t];
        for (const std::size_t w : {tri.v0, tri.v1, tri.v2})
          if (move[w]) {
            move[w] = 0U;
            mesh.vertices[w] = original[w];
            touched[w] = 1U;
            any = true;
          }
      }
    if (!any)
      break;
    for (std::size_t t = 0; t < mesh.triangles.size(); ++t) {
      const Triangle &tri = mesh.triangles[t];
      if (touched[tri.v0] || touched[tri.v1] || touched[tri.v2])
        update_triangle_geometry(mesh, t);
    }
  }
  projected = static_cast<std::size_t>(
      std::count(move.begin(), move.end(), std::uint8_t{1}));
  kept = n - projected;
}

} // namespace

SurfaceSelfIntersectionAudit
audit_surface_self_intersections(const SurfaceMesh &mesh,
                                 double grid_spacing) {
  if (!(grid_spacing > 0.0) || !std::isfinite(grid_spacing))
    return {};
  std::size_t scratch_bytes = 0U;
  const SelfIntersectionAudit audit =
      detect_self_intersections(mesh, grid_spacing, scratch_bytes);
  return {audit.count, audit.first, audit.second};
}

#ifdef TITAN_TABI_SES_TEST_HOOKS
namespace testing {

bool collapse_single_manifold_edge_for_test(
    const SurfaceMesh &source, std::size_t first, std::size_t second,
    bool keep_higher_endpoint) {
  if (first >= source.vertices.size() || second >= source.vertices.size() ||
      first == second)
    return false;
  const EdgeKey edge{std::min(first, second), std::max(first, second)};
  const CompactVertexAdjacency adjacency =
      build_compact_vertex_adjacency(source);
  const CompactVertexIncidence incidence =
      build_compact_vertex_incidence(source);
  SurfaceMesh candidate;
  std::size_t scratch_bytes = 0U;
  return collapse_single_manifold_edge_compact(
             source, candidate, edge, adjacency, keep_higher_endpoint,
             incidence, scratch_bytes) == 1U;
}

bool commit_projection_proposals_for_test(
    SurfaceMesh &mesh, std::span<const Vertex> proposals) {
  if (mesh.vertices.empty() || mesh.triangles.empty() ||
      proposals.size() != mesh.vertices.size() ||
      mesh.panels.size() != mesh.triangles.size())
    return false;
  for (const Triangle &triangle : mesh.triangles) {
    if (triangle.v0 >= mesh.vertices.size() ||
        triangle.v1 >= mesh.vertices.size() ||
        triangle.v2 >= mesh.vertices.size())
      return false;
  }
  const CompactVertexIncidence incidence =
      build_compact_vertex_incidence(mesh);
  if (incidence.offsets.size() != mesh.vertices.size() + 1U)
    return false;
  std::vector<Vec3> reference_normals(mesh.triangles.size());
  for (std::size_t triangle_index = 0U;
       triangle_index < mesh.triangles.size(); ++triangle_index) {
    const Triangle &triangle = mesh.triangles[triangle_index];
    const Vec3 a = mesh_vertex(mesh, triangle.v0);
    const Vec3 b = mesh_vertex(mesh, triangle.v1);
    const Vec3 c = mesh_vertex(mesh, triangle.v2);
    reference_normals[triangle_index] = cross(b - a, c - a);
  }
  std::vector<std::uint8_t> valid(mesh.vertices.size(), 1U);
  std::vector<std::uint8_t> affected(mesh.triangles.size(), 0U);
  const ProjectionCommitResult commit = commit_projection_proposals(
      mesh, incidence, proposals, valid, reference_normals, affected);
  for (std::size_t triangle_index = 0U;
       triangle_index < mesh.triangles.size(); ++triangle_index) {
    if (affected[triangle_index] != 0U)
      update_analytic_triangle_geometry_and_owner(mesh, triangle_index);
  }
  return commit.accepted_vertices == mesh.vertices.size() &&
         commit.rejected_vertices == 0U &&
         commit.all_faces_certified;
}

} // namespace testing
#endif

AuxiliarySurfaceResult
build_implicit_surface(std::span<const Atom> atoms, double grid_spacing,
                       double probe_radius, TitanExecutor *executor,
                       std::size_t max_grid_vertices, bool validate_topology) {
  const auto begin = std::chrono::steady_clock::now();
  AuxiliarySurfaceResult result;
  // No analytic engine ran here.  The field defaults to the analytic default,
  // so leaving it would report a grid-ses run as an analytic `current` run,
  // which is exactly the provenance a reader would rely on.
  result.analytic_ses_engine = "not_applicable";
  if (atoms.empty() || !(grid_spacing > 0.0) || !std::isfinite(grid_spacing) ||
      !(probe_radius >= 0.0) || !std::isfinite(probe_radius)) {
    result.status = "invalid_input";
    return result;
  }

  Vec3 lower{std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity()};
  Vec3 upper{-std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity()};
  std::size_t valid_atoms = 0;
  for (const Atom &atom : atoms) {
    if (!(atom.radius > 0.0) || !std::isfinite(atom.radius)) {
      continue;
    }
    ++valid_atoms;
    const double expanded_radius = atom.radius + probe_radius;
    lower.x = std::min(lower.x, atom.x - expanded_radius);
    lower.y = std::min(lower.y, atom.y - expanded_radius);
    lower.z = std::min(lower.z, atom.z - expanded_radius);
    upper.x = std::max(upper.x, atom.x + expanded_radius);
    upper.y = std::max(upper.y, atom.y + expanded_radius);
    upper.z = std::max(upper.z, atom.z + expanded_radius);
  }
  if (valid_atoms == 0) {
    result.status = "no_valid_atoms";
    return result;
  }
  const double margin = 2.0 * grid_spacing;
  lower = lower - Vec3{margin, margin, margin};
  upper = upper + Vec3{margin, margin, margin};
  const auto axis_count = [&](double lo, double hi) {
    return static_cast<std::size_t>(std::ceil((hi - lo) / grid_spacing)) + 1;
  };
  const std::size_t nx = axis_count(lower.x, upper.x);
  const std::size_t ny = axis_count(lower.y, upper.y);
  const std::size_t nz = axis_count(lower.z, upper.z);
  const std::size_t grid_vertex_cap =
      std::max(kMaxGridVertices, max_grid_vertices);
  if (nx < 2 || ny < 2 || nz < 2 || nx > grid_vertex_cap / ny ||
      nx * ny > grid_vertex_cap / nz) {
    result.status = "grid_too_large";
    return result;
  }
  result.grid_vertices = nx * ny * nz;
  std::vector<Atom> field_atoms(atoms.begin(), atoms.end());
  if (probe_radius > 0.0) {
    for (Atom &atom : field_atoms) {
      atom.radius += probe_radius;
    }
  }
  const AtomGrid field_atom_grid = build_atom_grid(field_atoms, lower);
  const AtomGrid ownership_atom_grid = build_atom_grid(atoms, lower);
  std::vector<double> values(result.grid_vertices, 1.0);
  const auto grid_index = [nx, ny](std::size_t ix, std::size_t iy,
                                   std::size_t iz) {
    return ix + nx * (iy + ny * iz);
  };
  // The solvent-excluded volume is the morphological closing of the atoms by
  // the probe ball (grown by the probe, then eroded by it).  The erosion is
  // evaluated exactly, as the distance to the boundary of the grown atoms
  // (exact_ses_field), not by testing a few directions on the probe sphere:
  // a direction set lets a probe through gaps between its directions, which
  // leaves the surface outside the SES by an amount the grid spacing does not
  // reduce.  A zero crossing depends only on vertices within a cell diagonal
  // of it, so the exact value is needed only up to probe + 2 h from the
  // accessible surface.  The zero isosurface is then welded and
  // topology-audited, so no open patch reaches the solver.
  const double field_reach = probe_radius + 2.0 * grid_spacing;
  const auto evaluate_slices = [&](std::size_t begin_z, std::size_t end_z) {
    std::vector<BallBoundaryCandidate> scratch;
    std::vector<std::size_t> cover;
    for (std::size_t iz = begin_z; iz < end_z; ++iz) {
      for (std::size_t iy = 0; iy < ny; ++iy) {
        for (std::size_t ix = 0; ix < nx; ++ix) {
          const Vec3 point{lower.x + grid_spacing * static_cast<double>(ix),
                           lower.y + grid_spacing * static_cast<double>(iy),
                           lower.z + grid_spacing * static_cast<double>(iz)};
          values[grid_index(ix, iy, iz)] =
              probe_radius == 0.0
                  ? sample_union_sdf(field_atom_grid, field_atoms, point).value
                  : exact_ses_field(field_atom_grid, field_atoms, point,
                                    probe_radius, field_reach, scratch, cover);
        }
      }
    }
  };
  if (executor != nullptr && executor->capacity() > 1 && nz > 1) {
    executor->parallel_for(
        nz, std::min<std::size_t>(executor->capacity(), nz), 1,
        [&](std::size_t, std::size_t begin_z, std::size_t end_z) {
          evaluate_slices(begin_z, end_z);
        });
  } else {
    evaluate_slices(0, nz);
  }

  if (probe_radius > 0.0) {
    // Keep only solvent connected to the grid boundary. Positive pockets that
    // cannot be reached by a probe from the exterior are part of the molecular
    // volume, not additional dielectric interfaces.
    std::vector<std::uint8_t> exterior(result.grid_vertices, 0U);
    std::vector<std::size_t> queue;
    queue.reserve(result.grid_vertices / 8U);
    const auto seed = [&](std::size_t ix, std::size_t iy, std::size_t iz) {
      const std::size_t index = grid_index(ix, iy, iz);
      if (values[index] >= 0.0 && exterior[index] == 0U) {
        exterior[index] = 1U;
        queue.push_back(index);
      }
    };
    for (std::size_t iz = 0; iz < nz; ++iz) {
      for (std::size_t iy = 0; iy < ny; ++iy) {
        seed(0, iy, iz);
        seed(nx - 1, iy, iz);
      }
    }
    for (std::size_t iz = 0; iz < nz; ++iz) {
      for (std::size_t ix = 0; ix < nx; ++ix) {
        seed(ix, 0, iz);
        seed(ix, ny - 1, iz);
      }
    }
    for (std::size_t iy = 0; iy < ny; ++iy) {
      for (std::size_t ix = 0; ix < nx; ++ix) {
        seed(ix, iy, 0);
        seed(ix, iy, nz - 1);
      }
    }
    for (std::size_t head = 0; head < queue.size(); ++head) {
      const std::size_t index = queue[head];
      const std::size_t iz = index / (nx * ny);
      const std::size_t remainder = index - iz * nx * ny;
      const std::size_t iy = remainder / nx;
      const std::size_t ix = remainder - iy * nx;
      const auto visit = [&](std::size_t neighbor) {
        if (values[neighbor] >= 0.0 && exterior[neighbor] == 0U) {
          exterior[neighbor] = 1U;
          queue.push_back(neighbor);
        }
      };
      if (ix > 0)
        visit(index - 1);
      if (ix + 1 < nx)
        visit(index + 1);
      if (iy > 0)
        visit(index - nx);
      if (iy + 1 < ny)
        visit(index + nx);
      if (iz > 0)
        visit(index - nx * ny);
      if (iz + 1 < nz)
        visit(index + nx * ny);
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (values[index] >= 0.0 && exterior[index] == 0U) {
        values[index] = -grid_spacing;
      }
    }
  }

  polygonize_grid(result.mesh, ownership_atom_grid, atoms, lower, grid_spacing,
                  nx, ny, nz, values);
  repair_micro_slivers(result.mesh, grid_spacing);

  if (result.mesh.triangles.empty()) {
    result.status = "empty_implicit_surface";
  } else if (!validate_topology) {
    result.trace_spaces.quality.vertex_count = result.mesh.vertices.size();
    result.trace_spaces.quality.triangle_count = result.mesh.triangles.size();
    result.trace_spaces.quality.failure_reason = "surface_only_not_audited";
    result.status = "pass";
  } else {
    orient_mesh_consistently(result.mesh);
    if (probe_radius > 0.0) {
      result.projection_queries = result.mesh.vertices.size();
      std::size_t kept = 0;
      project_grid_ses_vertices(result.mesh, field_atom_grid, field_atoms,
                                probe_radius, grid_spacing, executor,
                                result.projected_vertices, kept);
    }
    result.trace_spaces = build_closed_surface_trace_spaces(result.mesh);
    result.status = result.trace_spaces.quality.failure_reason;
  }
  const auto end = std::chrono::steady_clock::now();
  result.build_ms =
      std::chrono::duration<double, std::milli>(end - begin).count();
  return result;
}

AuxiliarySurfaceResult build_implicit_vdw_auxiliary_surface(
    std::span<const Atom> atoms, double grid_spacing, TitanExecutor *executor) {
  return build_implicit_surface(atoms, grid_spacing, 0.0, executor,
                                kMaxGridVertices, true);
}

AuxiliarySurfaceResult build_implicit_ses_surface(std::span<const Atom> atoms,
                                                  double grid_spacing,
                                                  double probe_radius,
                                                  TitanExecutor *executor,
                                                  std::size_t max_grid_vertices,
                                                  bool validate_topology) {
  return build_implicit_surface(atoms, grid_spacing, probe_radius, executor,
                                max_grid_vertices, validate_topology);
}

AuxiliarySurfaceResult build_analytic_ses_surface(std::span<const Atom> atoms,
                                                  double grid_spacing,
                                                  double probe_radius,
                                                  TitanExecutor *executor,
                                                  std::size_t max_grid_vertices,
                                                  bool validate_topology,
                                                  std::string_view engine,
                                                  double remesh_edge,
                                                  ChargeRefinement charge_refinement,
                                                  int remesh_iterations) {
  const auto begin = std::chrono::steady_clock::now();
  AuxiliarySurfaceResult result;
  result.analytic_ses_engine = std::string(engine);
  result.cavity_policy = "external-only";
  if (atoms.empty() || !(grid_spacing > 0.0) || !std::isfinite(grid_spacing) ||
      !(probe_radius > 0.0) || !std::isfinite(probe_radius)) {
    result.status = "invalid_input";
    return result;
  }
  if (!is_current_family_engine(engine) &&
      engine != "patch-ray-experimental" &&
      engine != "patch-ray-resolution") {
    result.status = "invalid_analytic_ses_engine";
    return result;
  }
  const auto patch_begin = std::chrono::steady_clock::now();
  analytic_ses::PatchComplex patches = analytic_ses::build_patch_complex(
      atoms, probe_radius, executor, is_current_family_engine(engine));
  result.patch_build_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - patch_begin)
                              .count();
  if (patches.status != "pass") {
    result.status = "analytic_patch_" + patches.status;
    return result;
  }

  Vec3 lower{std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity()};
  Vec3 upper{-std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity()};
  for (const Atom &atom : patches.atoms) {
    const double radius = atom.radius + probe_radius;
    lower.x = std::min(lower.x, atom.x - radius);
    lower.y = std::min(lower.y, atom.y - radius);
    lower.z = std::min(lower.z, atom.z - radius);
    upper.x = std::max(upper.x, atom.x + radius);
    upper.y = std::max(upper.y, atom.y + radius);
    upper.z = std::max(upper.z, atom.z + radius);
  }
  const double margin = 2.0 * grid_spacing;
  lower = lower - Vec3{margin, margin, margin};
  upper = upper + Vec3{margin, margin, margin};
  const auto axis_count = [&](double lo, double hi) {
    return static_cast<std::size_t>(std::ceil((hi - lo) / grid_spacing)) + 1U;
  };
  const std::size_t nx = axis_count(lower.x, upper.x);
  const std::size_t ny = axis_count(lower.y, upper.y);
  const std::size_t nz = axis_count(lower.z, upper.z);
  const std::size_t cap = std::max<std::size_t>(1U, max_grid_vertices);
  if (nx < 2 || ny < 2 || nz < 2 || nx > cap / ny || nx * ny > cap / nz) {
    result.status = "grid_too_large";
    return result;
  }
  result.grid_vertices = nx * ny * nz;
  if (engine == "patch-ray-experimental" || engine == "patch-ray-resolution") {
    AuxiliarySurfaceResult experimental = build_patch_ray_experimental_surface(
        patches, lower, grid_spacing, nx, ny, nz, executor,
        validate_topology, engine == "patch-ray-resolution", remesh_edge,
        atoms, charge_refinement, remesh_iterations);
    experimental.patch_build_ms = result.patch_build_ms;
    experimental.build_ms += result.patch_build_ms;
    return experimental;
  }
  const auto grid_index = [nx, ny](std::size_t ix, std::size_t iy,
                                   std::size_t iz) {
    return ix + nx * (iy + ny * iz);
  };
  const auto topology_begin = std::chrono::steady_clock::now();
  // Use the flood-filled configuration-space closing only as an exterior
  // topology mask. Intersection locations and final vertex projection still
  // come from the analytic contact/toroidal/reentrant patches below.
  std::vector<Atom> expanded_atoms = patches.atoms;
  for (Atom &atom : expanded_atoms)
    atom.radius += probe_radius;
  const AtomGrid expanded_grid = build_atom_grid(expanded_atoms, lower);
  // This mask only carries topology/sign. Exact surface distances below come
  // from analytic ray crossings, so avoid 27 square roots and minimum-distance
  // reductions per grid vertex here.
  std::vector<std::uint8_t> topology_inside(result.grid_vertices, 0U);
  const auto &sample_directions =
      closing_sample_directions(engine == "current-dense" ? 2 : 1);
  const auto evaluate_topology_slices = [&](std::size_t begin_z,
                                            std::size_t end_z) {
    for (std::size_t iz = begin_z; iz < end_z; ++iz) {
      for (std::size_t iy = 0; iy < ny; ++iy) {
        for (std::size_t ix = 0; ix < nx; ++ix) {
          const Vec3 point{lower.x + grid_spacing * static_cast<double>(ix),
                           lower.y + grid_spacing * static_cast<double>(iy),
                           lower.z + grid_spacing * static_cast<double>(iz)};
          bool inside = true;
          for (const Vec3 &direction : sample_directions) {
            if (!point_inside_union(expanded_grid, expanded_atoms,
                                    point + probe_radius * direction)) {
              inside = false;
              break;
            }
          }
          topology_inside[grid_index(ix, iy, iz)] = inside ? 1U : 0U;
        }
      }
    }
  };
  if (executor != nullptr && executor->capacity() > 1 && nz > 1) {
    executor->parallel_for(
        nz, std::min<std::size_t>(executor->capacity(), nz), 1,
        [&](std::size_t, std::size_t begin_z, std::size_t end_z) {
          evaluate_topology_slices(begin_z, end_z);
        });
  } else {
    evaluate_topology_slices(0, nz);
  }
  std::vector<std::uint8_t> exterior(result.grid_vertices, 0U);
  std::vector<std::uint8_t> cavity_fill(result.grid_vertices, 0U);
  std::vector<std::size_t> queue;
  queue.reserve(result.grid_vertices / 8U);
  const auto seed = [&](std::size_t ix, std::size_t iy, std::size_t iz) {
    const std::size_t index = grid_index(ix, iy, iz);
    if (topology_inside[index] == 0U && exterior[index] == 0U) {
      exterior[index] = 1U;
      queue.push_back(index);
    }
  };
  for (std::size_t iz = 0; iz < nz; ++iz) {
    for (std::size_t iy = 0; iy < ny; ++iy) {
      seed(0, iy, iz);
      seed(nx - 1, iy, iz);
    }
  }
  for (std::size_t iz = 0; iz < nz; ++iz) {
    for (std::size_t ix = 0; ix < nx; ++ix) {
      seed(ix, 0, iz);
      seed(ix, ny - 1, iz);
    }
  }
  for (std::size_t iy = 0; iy < ny; ++iy) {
    for (std::size_t ix = 0; ix < nx; ++ix) {
      seed(ix, iy, 0);
      seed(ix, iy, nz - 1);
    }
  }
  for (std::size_t head = 0; head < queue.size(); ++head) {
    const std::size_t index = queue[head];
    const std::size_t iz = index / (nx * ny);
    const std::size_t remainder = index - iz * nx * ny;
    const std::size_t iy = remainder / nx;
    const std::size_t ix = remainder - iy * nx;
    const auto visit = [&](std::size_t neighbor) {
      if (topology_inside[neighbor] == 0U && exterior[neighbor] == 0U) {
        exterior[neighbor] = 1U;
        queue.push_back(neighbor);
      }
    };
    if (ix > 0)
      visit(index - 1);
    if (ix + 1 < nx)
      visit(index + 1);
    if (iy > 0)
      visit(index - nx);
    if (iy + 1 < ny)
      visit(index + nx);
    if (iz > 0)
      visit(index - nx * ny);
    if (iz + 1 < nz)
      visit(index + nx * ny);
  }
  for (std::size_t index = 0; index < topology_inside.size(); ++index) {
    if (topology_inside[index] == 0U && exterior[index] == 0U) {
      topology_inside[index] = 1U;
      cavity_fill[index] = 1U;
    }
  }
  result.topology_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - topology_begin)
                           .count();

  // Marching tetrahedra only uses field magnitudes on edges whose endpoints
  // have opposite signs. Reconstruct the original eroded-SDF magnitude just
  // for that one-vertex boundary band; all bulk vertices retain a cheap
  // grid-spacing fallback. This preserves the old interpolation geometry
  // without paying for 27 SDF evaluations throughout the volume.
  const auto boundary_magnitude_begin = std::chrono::steady_clock::now();
  std::vector<double> topology_magnitude(result.grid_vertices, grid_spacing);
  const auto evaluate_boundary_magnitudes = [&](std::size_t begin_z,
                                                std::size_t end_z) {
    for (std::size_t iz = begin_z; iz < end_z; ++iz) {
      for (std::size_t iy = 0; iy < ny; ++iy) {
        for (std::size_t ix = 0; ix < nx; ++ix) {
          const std::size_t index = grid_index(ix, iy, iz);
          if (cavity_fill[index] != 0U)
            continue;
          const bool inside = topology_inside[index] != 0U;
          bool boundary = false;
          for (int dz = -1; dz <= 1 && !boundary; ++dz) {
            for (int dy = -1; dy <= 1 && !boundary; ++dy) {
              for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0)
                  continue;
                const auto neighbor_x = static_cast<std::ptrdiff_t>(ix) + dx;
                const auto neighbor_y = static_cast<std::ptrdiff_t>(iy) + dy;
                const auto neighbor_z = static_cast<std::ptrdiff_t>(iz) + dz;
                if (neighbor_x < 0 || neighbor_y < 0 || neighbor_z < 0 ||
                    neighbor_x >= static_cast<std::ptrdiff_t>(nx) ||
                    neighbor_y >= static_cast<std::ptrdiff_t>(ny) ||
                    neighbor_z >= static_cast<std::ptrdiff_t>(nz)) {
                  continue;
                }
                const std::size_t neighbor =
                    grid_index(static_cast<std::size_t>(neighbor_x),
                               static_cast<std::size_t>(neighbor_y),
                               static_cast<std::size_t>(neighbor_z));
                if ((topology_inside[neighbor] != 0U) != inside) {
                  boundary = true;
                  break;
                }
              }
            }
          }
          if (!boundary)
            continue;
          const Vec3 point{lower.x + grid_spacing * static_cast<double>(ix),
                           lower.y + grid_spacing * static_cast<double>(iy),
                           lower.z + grid_spacing * static_cast<double>(iz)};
          double eroded = -std::numeric_limits<double>::infinity();
          for (const Vec3 &direction : sample_directions) {
            eroded = std::max(eroded,
                              sample_union_sdf(expanded_grid, expanded_atoms,
                                               point + probe_radius * direction)
                                  .value);
          }
          topology_magnitude[index] = std::abs(eroded);
        }
      }
    }
  };
  if (executor != nullptr && executor->capacity() > 1 && nz > 1) {
    executor->parallel_for(
        nz, std::min<std::size_t>(executor->capacity(), nz), 1,
        [&](std::size_t, std::size_t begin_z, std::size_t end_z) {
          evaluate_boundary_magnitudes(begin_z, end_z);
        });
  } else {
    evaluate_boundary_magnitudes(0, nz);
  }
  result.boundary_magnitude_ms = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() -
                                     boundary_magnitude_begin)
                                     .count();

  const auto ray_begin = std::chrono::steady_clock::now();
  std::vector<double> values(result.grid_vertices, grid_spacing);
  std::atomic_size_t odd_ray_rows{0};
  const auto evaluate_rows = [&](std::size_t begin_row, std::size_t end_row) {
    for (std::size_t row = begin_row; row < end_row; ++row) {
      const std::size_t iz = row / ny;
      const std::size_t iy = row - iz * ny;
      const double y = lower.y + grid_spacing * static_cast<double>(iy);
      const double z = lower.z + grid_spacing * static_cast<double>(iz);
      const std::vector<analytic_ses::RayHit> hits =
          analytic_ses::intersect_ray_x(patches, y, z);
      std::vector<double> crossings;
      crossings.reserve(hits.size());
      for (const auto &hit : hits) {
        if (std::abs(hit.normal.x) > 1.0e-8)
          crossings.push_back(hit.x);
      }
      if ((crossings.size() & 1U) != 0U)
        odd_ray_rows.fetch_add(1U, std::memory_order_relaxed);
      for (std::size_t ix = 0; ix < nx; ++ix) {
        const double x = lower.x + grid_spacing * static_cast<double>(ix);
        const auto next =
            std::lower_bound(crossings.begin(), crossings.end(), x);
        double distance = topology_magnitude[grid_index(ix, iy, iz)];
        if (next != crossings.end())
          distance = std::min(distance, std::abs(*next - x));
        if (next != crossings.begin())
          distance = std::min(distance, std::abs(*std::prev(next) - x));
        distance = std::max(distance, 1.0e-9 * grid_spacing);
        values[grid_index(ix, iy, iz)] =
            topology_inside[grid_index(ix, iy, iz)] != 0U ? -distance
                                                          : distance;
      }
    }
  };
  const std::size_t row_count = ny * nz;
  if (executor != nullptr && executor->capacity() > 1 && row_count > 1) {
    const std::size_t worker_count =
        std::min<std::size_t>(executor->capacity(), row_count);
    // Central rays intersect many more patches than peripheral rays. Static
    // contiguous z/y ranges therefore leave workers idle near the end of this
    // phase. Keep at least 64 tiles per worker, with bounds that amortize the
    // atomic fetch without making tiles too coarse for small molecules.
    const std::size_t target_tiles = 64U * worker_count;
    const std::size_t tile_rows =
        std::clamp((row_count + target_tiles - 1U) / target_tiles,
                   std::size_t{2U}, std::size_t{8U});
    std::atomic_size_t next_row{0U};
    executor->parallel_do(worker_count, [&](std::size_t) {
      for (;;) {
        const std::size_t row_begin =
            next_row.fetch_add(tile_rows, std::memory_order_relaxed);
        if (row_begin >= row_count)
          break;
        evaluate_rows(row_begin, std::min(row_count, row_begin + tile_rows));
      }
    });
  } else {
    evaluate_rows(0, row_count);
  }
  result.odd_ray_rows = odd_ray_rows.load(std::memory_order_relaxed);
  result.ray_build_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - ray_begin)
                            .count();

  const auto polygonize_begin = std::chrono::steady_clock::now();
  const AtomGrid ownership_grid = build_atom_grid(patches.atoms, lower);
  polygonize_grid(result.mesh, ownership_grid, patches.atoms, lower,
                  grid_spacing, nx, ny, nz, values);
  result.polygonize_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() -
                             polygonize_begin)
                             .count();
  const auto repair_begin = std::chrono::steady_clock::now();
  const auto pre_smooth_repair_begin = std::chrono::steady_clock::now();
  repair_micro_slivers(result.mesh, grid_spacing);
  result.current_pre_smooth_repair_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - pre_smooth_repair_begin)
          .count();
  const auto component_filter_begin = std::chrono::steady_clock::now();
  if (remove_nonphysical_components(result.mesh, patches.atoms) > 0U)
    orient_mesh_consistently(result.mesh);
  result.current_component_filter_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - component_filter_begin)
          .count();
  const auto smooth_analytic_slivers_begin = std::chrono::steady_clock::now();
  smooth_analytic_slivers(result.mesh, patches, grid_spacing);
  result.current_smooth_analytic_slivers_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - smooth_analytic_slivers_begin)
          .count();
  const auto post_smooth_repair_begin = std::chrono::steady_clock::now();
  repair_micro_slivers(result.mesh, grid_spacing);
  result.current_post_smooth_repair_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - post_smooth_repair_begin)
          .count();
  const auto final_orient_trace_space_begin =
      std::chrono::steady_clock::now();
  if (result.mesh.triangles.empty()) {
    result.status = "empty_analytic_surface";
  } else {
    orient_mesh_consistently(result.mesh);
    result.trace_spaces = build_closed_surface_trace_spaces(result.mesh);
    result.status = result.trace_spaces.quality.failure_reason;
    if (!validate_topology && result.status == "pass")
      result.trace_spaces.quality.failure_reason = "pass";
  }
  result.current_final_orient_trace_space_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - final_orient_trace_space_begin)
          .count();
  result.repair_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - repair_begin)
                         .count();
  result.build_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - begin)
                        .count();
  return result;
}

} // namespace TitanPBE::Tabi::calderon
