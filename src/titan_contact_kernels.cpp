#include "titan_contact_kernels.h"

#include <algorithm>
#include <cmath>
#include <immintrin.h>

namespace titan_contact_kernel {
namespace {

double distance_with_mic(const PointSoA &rows, const PointSoA &cols,
                         std::size_t row, std::size_t col,
                         const BoxMatrix &box_matrix) {
  double dx = rows.x[row] - cols.x[col];
  double dy = rows.y[row] - cols.y[col];
  double dz = rows.z[row] - cols.z[col];
  box_matrix.apply_mic(dx, dy, dz);
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void append_contact_if_within(const PointSoA &rows, const PointSoA &cols,
                              const BoxMatrix &box_matrix, double cutoff_sq,
                              std::size_t row, std::size_t col,
                              std::vector<ContactCandidate> &out) {
  double dx = rows.x[row] - cols.x[col];
  double dy = rows.y[row] - cols.y[col];
  double dz = rows.z[row] - cols.z[col];
  box_matrix.apply_mic(dx, dy, dz);
  const double distance_sq = dx * dx + dy * dy + dz * dz;
  if (distance_sq <= cutoff_sq)
    out.push_back(ContactCandidate{row, col, std::sqrt(distance_sq)});
}

#if defined(__AVX2__)
bool is_orthorhombic(const BoxMatrix &box_matrix) {
  constexpr double eps = 1.0e-12;
  return std::fabs(box_matrix.m[0][1]) < eps &&
         std::fabs(box_matrix.m[0][2]) < eps &&
         std::fabs(box_matrix.m[1][0]) < eps &&
         std::fabs(box_matrix.m[1][2]) < eps &&
         std::fabs(box_matrix.m[2][0]) < eps &&
         std::fabs(box_matrix.m[2][1]) < eps &&
         std::fabs(box_matrix.m[0][0]) > eps &&
         std::fabs(box_matrix.m[1][1]) > eps &&
         std::fabs(box_matrix.m[2][2]) > eps;
}

void apply_mic_orthorhombic_avx(const BoxMatrix &box_matrix, __m256d &dx,
                                __m256d &dy, __m256d &dz) {
  const __m256d ax = _mm256_set1_pd(box_matrix.m[0][0]);
  const __m256d by = _mm256_set1_pd(box_matrix.m[1][1]);
  const __m256d cz = _mm256_set1_pd(box_matrix.m[2][2]);
  const __m256d inv_ax = _mm256_set1_pd(1.0 / box_matrix.m[0][0]);
  const __m256d inv_by = _mm256_set1_pd(1.0 / box_matrix.m[1][1]);
  const __m256d inv_cz = _mm256_set1_pd(1.0 / box_matrix.m[2][2]);
  const int round_mode = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

  dx = _mm256_sub_pd(
      dx, _mm256_mul_pd(ax, _mm256_round_pd(_mm256_mul_pd(dx, inv_ax),
                                            round_mode)));
  dy = _mm256_sub_pd(
      dy, _mm256_mul_pd(by, _mm256_round_pd(_mm256_mul_pd(dy, inv_by),
                                            round_mode)));
  dz = _mm256_sub_pd(
      dz, _mm256_mul_pd(cz, _mm256_round_pd(_mm256_mul_pd(dz, inv_cz),
                                            round_mode)));
}

void scan_cols_avx2(const PointSoA &rows, const PointSoA &cols,
                    const BoxMatrix &box_matrix, double cutoff_sq,
                    std::size_t row, std::size_t col_begin,
                    std::size_t col_end,
                    std::vector<ContactCandidate> &out) {
  const bool orthorhombic = is_orthorhombic(box_matrix);
  const __m256d row_x = _mm256_set1_pd(rows.x[row]);
  const __m256d row_y = _mm256_set1_pd(rows.y[row]);
  const __m256d row_z = _mm256_set1_pd(rows.z[row]);
  const __m256d cutoff = _mm256_set1_pd(cutoff_sq);
  alignas(32) double distances[4]{};

  std::size_t col = col_begin;
  for (; col + 4 <= col_end; col += 4) {
    __m256d dx = _mm256_sub_pd(
        row_x, _mm256_set_pd(cols.x[col + 3], cols.x[col + 2],
                             cols.x[col + 1], cols.x[col]));
    __m256d dy = _mm256_sub_pd(
        row_y, _mm256_set_pd(cols.y[col + 3], cols.y[col + 2],
                             cols.y[col + 1], cols.y[col]));
    __m256d dz = _mm256_sub_pd(
        row_z, _mm256_set_pd(cols.z[col + 3], cols.z[col + 2],
                             cols.z[col + 1], cols.z[col]));
    if (orthorhombic)
      apply_mic_orthorhombic_avx(box_matrix, dx, dy, dz);
    else
      box_matrix.apply_mic_avx(dx, dy, dz);
    const __m256d distance_sq = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
        _mm256_mul_pd(dz, dz));
    const int mask =
        _mm256_movemask_pd(_mm256_cmp_pd(distance_sq, cutoff, _CMP_LE_OQ));
    if (mask != 0) {
      _mm256_store_pd(distances, distance_sq);
      for (int lane = 0; lane < 4; ++lane) {
        if ((mask & (1 << lane)) != 0)
          out.push_back(ContactCandidate{
              row, col + static_cast<std::size_t>(lane),
              std::sqrt(distances[lane])});
      }
    }
  }
  for (; col < col_end; ++col)
    append_contact_if_within(rows, cols, box_matrix, cutoff_sq, row, col, out);
}

void native_distances_avx2(const PointSoA &rows, const PointSoA &cols,
                           const std::vector<NativePairIndex> &pairs,
                           std::size_t begin, std::size_t end,
                           const BoxMatrix &box_matrix,
                           std::vector<double> &out_distances) {
  const bool orthorhombic = is_orthorhombic(box_matrix);
  alignas(32) double distances[4]{};
  std::size_t i = begin;
  for (; i + 4 <= end; i += 4) {
    const NativePairIndex &p0 = pairs[i];
    const NativePairIndex &p1 = pairs[i + 1];
    const NativePairIndex &p2 = pairs[i + 2];
    const NativePairIndex &p3 = pairs[i + 3];
    __m256d dx = _mm256_sub_pd(
        _mm256_set_pd(rows.x[p3.row], rows.x[p2.row], rows.x[p1.row],
                      rows.x[p0.row]),
        _mm256_set_pd(cols.x[p3.col], cols.x[p2.col], cols.x[p1.col],
                      cols.x[p0.col]));
    __m256d dy = _mm256_sub_pd(
        _mm256_set_pd(rows.y[p3.row], rows.y[p2.row], rows.y[p1.row],
                      rows.y[p0.row]),
        _mm256_set_pd(cols.y[p3.col], cols.y[p2.col], cols.y[p1.col],
                      cols.y[p0.col]));
    __m256d dz = _mm256_sub_pd(
        _mm256_set_pd(rows.z[p3.row], rows.z[p2.row], rows.z[p1.row],
                      rows.z[p0.row]),
        _mm256_set_pd(cols.z[p3.col], cols.z[p2.col], cols.z[p1.col],
                      cols.z[p0.col]));
    if (orthorhombic)
      apply_mic_orthorhombic_avx(box_matrix, dx, dy, dz);
    else
      box_matrix.apply_mic_avx(dx, dy, dz);
    const __m256d distance_sq = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
        _mm256_mul_pd(dz, dz));
    _mm256_store_pd(distances, distance_sq);
    for (int lane = 0; lane < 4; ++lane)
      out_distances[i + static_cast<std::size_t>(lane)] =
          std::sqrt(distances[lane]);
  }
  for (; i < end; ++i)
    out_distances[i] = distance_with_mic(rows, cols, pairs[i].row,
                                         pairs[i].col, box_matrix);
}
#endif

} // namespace

void scan_interface_contacts_scalar(const PointSoA &rows, const PointSoA &cols,
                                    const BoxMatrix &box_matrix,
                                    double cutoff_sq, std::size_t row_begin,
                                    std::size_t row_end,
                                    std::vector<ContactCandidate> &out) {
  const std::size_t bounded_end = std::min(row_end, rows.size());
  for (std::size_t row = row_begin; row < bounded_end; ++row)
    for (std::size_t col = 0; col < cols.size(); ++col)
      append_contact_if_within(rows, cols, box_matrix, cutoff_sq, row, col,
                               out);
}

void scan_interface_contacts_optimized(const PointSoA &rows,
                                       const PointSoA &cols,
                                       const BoxMatrix &box_matrix,
                                       double cutoff_sq, std::size_t row_begin,
                                       std::size_t row_end,
                                       std::vector<ContactCandidate> &out) {
#if defined(__AVX2__)
  const std::size_t bounded_end = std::min(row_end, rows.size());
  for (std::size_t row = row_begin; row < bounded_end; ++row)
    scan_cols_avx2(rows, cols, box_matrix, cutoff_sq, row, 0, cols.size(), out);
#else
  scan_interface_contacts_scalar(rows, cols, box_matrix, cutoff_sq, row_begin,
                                 row_end, out);
#endif
}

void scan_mol1_contacts_scalar(const PointSoA &points,
                               const BoxMatrix &box_matrix, double cutoff_sq,
                               std::size_t row_begin, std::size_t row_end,
                               std::vector<ContactCandidate> &out) {
  const std::size_t bounded_end = std::min(row_end, points.size());
  for (std::size_t row = row_begin; row < bounded_end; ++row)
    for (std::size_t col = row + 1; col < points.size(); ++col)
      append_contact_if_within(points, points, box_matrix, cutoff_sq, row, col,
                               out);
}

void scan_mol1_contacts_optimized(const PointSoA &points,
                                  const BoxMatrix &box_matrix,
                                  double cutoff_sq, std::size_t row_begin,
                                  std::size_t row_end,
                                  std::vector<ContactCandidate> &out) {
#if defined(__AVX2__)
  const std::size_t bounded_end = std::min(row_end, points.size());
  for (std::size_t row = row_begin; row < bounded_end; ++row)
    scan_cols_avx2(points, points, box_matrix, cutoff_sq, row, row + 1,
                   points.size(), out);
#else
  scan_mol1_contacts_scalar(points, box_matrix, cutoff_sq, row_begin, row_end,
                            out);
#endif
}

void native_distances_scalar(const PointSoA &rows, const PointSoA &cols,
                             const std::vector<NativePairIndex> &pairs,
                             std::size_t begin, std::size_t end,
                             const BoxMatrix &box_matrix,
                             std::vector<double> &out_distances) {
  if (out_distances.size() < pairs.size())
    out_distances.resize(pairs.size());
  const std::size_t bounded_end = std::min(end, pairs.size());
  for (std::size_t i = begin; i < bounded_end; ++i)
    out_distances[i] = distance_with_mic(rows, cols, pairs[i].row,
                                         pairs[i].col, box_matrix);
}

void native_distances_optimized(const PointSoA &rows, const PointSoA &cols,
                                const std::vector<NativePairIndex> &pairs,
                                std::size_t begin, std::size_t end,
                                const BoxMatrix &box_matrix,
                                std::vector<double> &out_distances) {
  if (out_distances.size() < pairs.size())
    out_distances.resize(pairs.size());
  const std::size_t bounded_end = std::min(end, pairs.size());
#if defined(__AVX2__)
  native_distances_avx2(rows, cols, pairs, begin, bounded_end, box_matrix,
                        out_distances);
#else
  native_distances_scalar(rows, cols, pairs, begin, bounded_end, box_matrix,
                          out_distances);
#endif
}

} // namespace titan_contact_kernel
