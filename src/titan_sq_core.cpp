#include "titan_sq_core.h"

#include "BoxMatrix.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace titan_sq_detail {
namespace {

constexpr double kPi = 3.14159265358979323846;

double determinant3x3(const BoxMatrix &box_matrix) {
  return box_matrix.m[0][0] *
             (box_matrix.m[1][1] * box_matrix.m[2][2] -
              box_matrix.m[1][2] * box_matrix.m[2][1]) -
         box_matrix.m[0][1] *
             (box_matrix.m[1][0] * box_matrix.m[2][2] -
              box_matrix.m[1][2] * box_matrix.m[2][0]) +
         box_matrix.m[0][2] *
             (box_matrix.m[1][0] * box_matrix.m[2][1] -
              box_matrix.m[1][1] * box_matrix.m[2][0]);
}

std::string box_type_string(const box &frame_box) {
  box box_copy = frame_box;
  return box_copy.get_box_type();
}

} // namespace

bool has_positive_box_lengths(const box &frame_box) noexcept {
  return frame_box.get_a() > 0.0 && frame_box.get_b() > 0.0 &&
         frame_box.get_c() > 0.0;
}

bool is_orthorhombic_box(const box &frame_box) {
  const std::string box_type = box_type_string(frame_box);
  return box_type == "cubic" || box_type == "rectangular";
}

double frame_box_volume_a3(const box &frame_box) {
  if (!has_positive_box_lengths(frame_box))
    return std::numeric_limits<double>::quiet_NaN();
  const BoxMatrix box_matrix(frame_box);
  const double volume_a3 = std::abs(determinant3x3(box_matrix));
  if (!(volume_a3 > 0.0))
    return std::numeric_limits<double>::quiet_NaN();
  return volume_a3;
}

double shell_volume_a3(double r_inner, double r_outer) {
  return (4.0 / 3.0) * kPi *
         (r_outer * r_outer * r_outer - r_inner * r_inner * r_inner);
}

double lorch_window(double r, double r_max) {
  if (!(r_max > 0.0) || !(r >= 0.0))
    return 1.0;
  if (r >= r_max)
    return 0.0;
  const double x = kPi * r / r_max;
  if (std::abs(x) < 1.0e-12)
    return 1.0;
  return std::sin(x) / x;
}

void for_each_active_pair_within_cutoff(
    std::span<const double> xs, std::span<const double> ys,
    std::span<const double> zs, const box &frame_box,
    const std::vector<unsigned char> &active_mask, double cutoff_a,
    titan_neighbor::HalfShellGrid &neighbor_grid,
    const std::function<void(std::size_t, std::size_t, double)> &visit) {
  const std::size_t atom_limit =
      std::min({xs.size(), ys.size(), zs.size(), active_mask.size()});
  if (atom_limit < 2 || !(cutoff_a > 0.0))
    return;

  const BoxMatrix box_matrix(frame_box);
  const double cutoff_sq = cutoff_a * cutoff_a;

  if (is_orthorhombic_box(frame_box)) {
    titan_neighbor::OrthorhombicGridSpec spec;
    spec.cutoff = cutoff_a;
    spec.len_x = frame_box.get_a();
    spec.len_y = frame_box.get_b();
    spec.len_z = frame_box.get_c();
    spec.shift_x = 0.0;
    spec.shift_y = 0.0;
    spec.shift_z = 0.0;
    spec.periodic = true;

    neighbor_grid.for_each_pair(
        atom_limit, spec,
        [&](std::size_t idx, double &x, double &y, double &z) {
          x = xs[idx];
          y = ys[idx];
          z = zs[idx];
        },
        [&](std::size_t idx) {
          return idx < atom_limit && active_mask[idx] != 0;
        },
        [&](std::size_t lhs, std::size_t rhs) {
          double dx = xs[lhs] - xs[rhs];
          double dy = ys[lhs] - ys[rhs];
          double dz = zs[lhs] - zs[rhs];
          box_matrix.apply_mic(dx, dy, dz);
          const double dist_sq = dx * dx + dy * dy + dz * dz;
          if (dist_sq <= cutoff_sq)
            visit(lhs, rhs, std::sqrt(dist_sq));
        });
    return;
  }

  std::vector<std::size_t> active_indices;
  active_indices.reserve(atom_limit);
  for (std::size_t atom_idx = 0; atom_idx < atom_limit; ++atom_idx) {
    if (active_mask[atom_idx] != 0)
      active_indices.push_back(atom_idx);
  }
  if (active_indices.size() < 2)
    return;

  for (std::size_t i = 0; i < active_indices.size(); ++i) {
    const std::size_t lhs = active_indices[i];
    for (std::size_t j = i + 1; j < active_indices.size(); ++j) {
      const std::size_t rhs = active_indices[j];
      double dx = xs[lhs] - xs[rhs];
      double dy = ys[lhs] - ys[rhs];
      double dz = zs[lhs] - zs[rhs];
      box_matrix.apply_mic(dx, dy, dz);
      const double dist_sq = dx * dx + dy * dy + dz * dz;
      if (dist_sq <= cutoff_sq)
        visit(lhs, rhs, std::sqrt(dist_sq));
    }
  }
}

std::vector<double>
build_structure_factor_from_gr(const std::vector<double> &g_profile,
                               double rdf_bin_a, double rdf_max_a,
                               double q_bin_inv_a, std::size_t q_bin_count,
                               double rho_partner_a3,
                               bool include_self_baseline) {
  std::vector<double> sq_profile(q_bin_count,
                                 include_self_baseline ? 1.0 : 0.0);
  if (!(rho_partner_a3 > 0.0) || !(rdf_bin_a > 0.0) || !(q_bin_inv_a > 0.0))
    return sq_profile;

  for (std::size_t q_bin = 0; q_bin < q_bin_count; ++q_bin) {
    const double q = static_cast<double>(q_bin + 1) * q_bin_inv_a;
    double sq_value = include_self_baseline ? 1.0 : 0.0;
    for (std::size_t r_bin = 0; r_bin < g_profile.size(); ++r_bin) {
      const double r = (static_cast<double>(r_bin) + 0.5) * rdf_bin_a;
      const double qr = q * r;
      const double sinc = std::abs(qr) < 1.0e-12 ? 1.0 : std::sin(qr) / qr;
      const double window = lorch_window(r, rdf_max_a);
      sq_value += 4.0 * kPi * rho_partner_a3 * r * r *
                  (g_profile[r_bin] - 1.0) * sinc * window * rdf_bin_a;
    }
    sq_profile[q_bin] = sq_value;
  }

  return sq_profile;
}

} // namespace titan_sq_detail
