#ifndef TITAN_CONTACT_KERNELS_H
#define TITAN_CONTACT_KERNELS_H

#include "BoxMatrix.h"

#include <cstddef>
#include <vector>

namespace titan_contact_kernel {

struct PointSoA {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;

  [[nodiscard]] std::size_t size() const noexcept { return x.size(); }
};

// A cubic cell of 2^40 A with exact diagonal matrix and inverse: the minimum
// image never moves a coordinate difference, bit for bit (power-of-two scale,
// zero off-diagonal terms), so the kernels below measure plain distances for
// a frame without a periodic box (MDB-13).
inline const BoxMatrix &open_boundary_matrix() {
  static const BoxMatrix matrix = [] {
    constexpr double side = 1099511627776.0; // 2^40
    double a = side, b = side, c = side, angle = 90.0;
    BoxMatrix open(box(a, b, c, angle, angle, angle));
    for (int row = 0; row < 3; ++row)
      for (int col = 0; col < 3; ++col) {
        open.m[row][col] = row == col ? side : 0.0;
        open.inv[row][col] = row == col ? 1.0 / side : 0.0;
      }
    return open;
  }();
  return matrix;
}

struct ContactCandidate {
  std::size_t row{};
  std::size_t col{};
  double distance_a{};
};

struct NativePairIndex {
  std::size_t row{};
  std::size_t col{};
};

// A native residue pair as two ranges of points (one point per selected
// atom in sidechain-heavy / bb mode): the pair distance is the minimum over
// all point pairs of the two ranges.
struct NativePairRange {
  std::size_t row_begin{};
  std::size_t row_count{};
  std::size_t col_begin{};
  std::size_t col_count{};
};

void scan_interface_contacts_scalar(const PointSoA &rows, const PointSoA &cols,
                                    const BoxMatrix &box_matrix,
                                    double cutoff_sq, std::size_t row_begin,
                                    std::size_t row_end,
                                    std::vector<ContactCandidate> &out);

void scan_interface_contacts_optimized(const PointSoA &rows,
                                       const PointSoA &cols,
                                       const BoxMatrix &box_matrix,
                                       double cutoff_sq, std::size_t row_begin,
                                       std::size_t row_end,
                                       std::vector<ContactCandidate> &out);

void scan_mol1_contacts_scalar(const PointSoA &points,
                               const BoxMatrix &box_matrix, double cutoff_sq,
                               std::size_t row_begin, std::size_t row_end,
                               std::vector<ContactCandidate> &out);

void scan_mol1_contacts_optimized(const PointSoA &points,
                                  const BoxMatrix &box_matrix,
                                  double cutoff_sq, std::size_t row_begin,
                                  std::size_t row_end,
                                  std::vector<ContactCandidate> &out);

void native_distances_scalar(const PointSoA &rows, const PointSoA &cols,
                             const std::vector<NativePairIndex> &pairs,
                             std::size_t begin, std::size_t end,
                             const BoxMatrix &box_matrix,
                             std::vector<double> &out_distances);

void native_distances_optimized(const PointSoA &rows, const PointSoA &cols,
                                const std::vector<NativePairIndex> &pairs,
                                std::size_t begin, std::size_t end,
                                const BoxMatrix &box_matrix,
                                std::vector<double> &out_distances);

// Minimum minimum-image distance between the two point ranges of each pair
// in [begin, end); +inf for an empty range.
void native_min_distances_ranges(const PointSoA &rows, const PointSoA &cols,
                                 const std::vector<NativePairRange> &pairs,
                                 std::size_t begin, std::size_t end,
                                 const BoxMatrix &box_matrix,
                                 std::vector<double> &out_distances);

} // namespace titan_contact_kernel

#endif // TITAN_CONTACT_KERNELS_H
