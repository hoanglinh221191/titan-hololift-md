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

struct ContactCandidate {
  std::size_t row{};
  std::size_t col{};
  double distance_a{};
};

struct NativePairIndex {
  std::size_t row{};
  std::size_t col{};
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

} // namespace titan_contact_kernel

#endif // TITAN_CONTACT_KERNELS_H
