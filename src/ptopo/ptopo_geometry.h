#ifndef TITAN_PTOPO_GEOMETRY_H
#define TITAN_PTOPO_GEOMETRY_H

// Periodic geometry for the topology kernel (Backbone VI.3).
//
// Convention: Cell::B is stored row-major but the lattice vectors are its
// COLUMNS, x = B s. GROMACS stores box vectors as rows; use
// cell_from_box_rows. Geometry is floating point with a declared tolerance
// band; it is bounded exhaustive search, not interval-certified geometry.
// Integer outputs (image shifts) feed the exact integer kernel.

#include "ptopo_types.h"

#include <span>

class TitanExecutor;

namespace titan_ptopo {

struct Cell {
  std::array<Vec3, 3> B{};
};

// rows[k] is lattice vector k (GROMACS box layout); B takes them as columns.
Cell cell_from_box_rows(const std::array<Vec3, 3> &rows) noexcept;

struct GeometryPolicy {
  // Units of the supplied coordinates; not a contact cutoff.
  double absolute_tolerance = 1e-10;
  double relative_tolerance = 1e-12;
  double maximum_condition_bound = 1e10;
  std::size_t maximum_candidates = 1000000;
};

struct CellMetrics {
  Cell cell;
  std::array<Vec3, 3> inverse{};
  double determinant = 0.0;
  double norm_B = 0.0; // Frobenius norms
  double norm_inverse = 0.0;
  // ||row_k(B^-1)||: |s_k| <= ||row_k(B^-1)|| |x| for x = B s.
  Vec3 inverse_row_norm{};
  // min_k 1 / ||row_k(B^-1)||, the smallest perpendicular cell width. For
  // any nonzero integer n, |B n| >= minimum_width, so it bounds lambda_1
  // from below (tighter than [tr(G^-1)]^(-1/2)).
  double minimum_width = 0.0;
};

Result<CellMetrics> analyze_cell(const Cell &cell,
                                 const GeometryPolicy &policy = {});
Result<double> cell_volume(const Cell &cell);
Vec3 cartesian(const Cell &cell, const Vec3 &fractional) noexcept;
Vec3 to_fractional(const CellMetrics &metrics, const Vec3 &x) noexcept;

struct Image {
  std::array<std::int64_t, 3> shift{};
  Vec3 displacement{};
  double distance = 0.0;
};

struct NearestImages {
  // Every minimizer within the numerical tolerance is kept.
  std::vector<Image> minimizers;
  // +infinity when no runner-up was inside the bounded search.
  double separation_to_next = 0.0;
  bool unique_under_tolerance = false;
};

// Exhaustive bounded enumeration in 3D, NOT a fixed 27-image MIC.
Result<NearestImages> nearest_images(const Cell &cell,
                                     const Vec3 &delta_fractional,
                                     const GeometryPolicy &policy = {});
// Every image within `radius` (multiplicity kept). An image within the
// tolerance of the radius fails closed. For a self pair the caller drops
// shift 0 and keeps one of +-a.
Result<std::vector<Image>> images_within(const Cell &cell,
                                         const Vec3 &delta_fractional,
                                         double radius,
                                         const GeometryPolicy &policy = {});

// The unique shift a minimizing |B (delta + a)|; AmbiguousGeometry on a tie.
// Exact shortcut: if the rounded image x satisfies 2|x| + tol <
// minimum_width, every other image is at least minimum_width - |x| > |x| +
// tol away. Otherwise the bounded enumeration decides.
Result<Z3> unique_nearest_shift(const CellMetrics &metrics,
                                const Vec3 &delta_fractional,
                                const GeometryPolicy &policy = {});

// One periodic contact: displacement B (s_j - s_i + shift) for the INPUT
// fractional coordinates, i < j, or i == j with a lexicographically
// positive shift (self-image contact, counted once).
struct PeriodicContact {
  std::uint32_t i = 0;
  std::uint32_t j = 0;
  Z3 shift{};
  double distance = 0.0;
};

struct ContactSearchOptions {
  double cutoff = 0.0; // contact when distance < cutoff
  GeometryPolicy policy;
  std::size_t maximum_contacts = 50000000;
  std::size_t maximum_bins = std::size_t{1} << 22;
  // TitanExecutor workers (1 = serial). Results do not depend on it.
  std::size_t workers = 1;
  std::size_t min_grain = 1024; // points per worker, minimum
  bool allow_avx2 = true;
};

struct ContactSearchStats {
  std::array<std::size_t, 3> bins{};
  std::array<std::size_t, 3> reach{};
  std::size_t distance_evaluations = 0;
  std::size_t contacts = 0;
  bool used_avx2 = false;
};

// All periodic contacts within the cutoff, every image counted (a pair may
// touch several copies). Cell list in fractional space with a proven reach
// per axis; distances in an AVX2 kernel whose lanes use the scalar operation
// order, so the result is identical with and without AVX2 and for any worker
// count. A distance inside the tolerance band of the cutoff fails closed.
// Output sorted by (i, j, shift).
Result<std::vector<PeriodicContact>>
find_periodic_contacts(const Cell &cell, std::span<const Vec3> fractional,
                       const ContactSearchOptions &options,
                       TitanExecutor *executor = nullptr,
                       ContactSearchStats *stats = nullptr);

// O(N^2) reference over images_within; tests only.
Result<std::vector<PeriodicContact>>
find_periodic_contacts_reference(const Cell &cell,
                                 std::span<const Vec3> fractional,
                                 double cutoff,
                                 const GeometryPolicy &policy = {});

} // namespace titan_ptopo

#endif // TITAN_PTOPO_GEOMETRY_H
