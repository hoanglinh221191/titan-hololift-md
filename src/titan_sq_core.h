#ifndef TITAN_SQ_CORE_H
#define TITAN_SQ_CORE_H

#include "box_type.h"
#include "titan_neighbor_grid.h"

#include <cstddef>
#include <functional>
#include <span>
#include <vector>

namespace titan_sq_detail {

bool has_positive_box_lengths(const box &frame_box) noexcept;
bool is_orthorhombic_box(const box &frame_box);
double frame_box_volume_a3(const box &frame_box);
double shell_volume_a3(double r_inner, double r_outer);
double lorch_window(double r, double r_max);

void for_each_active_pair_within_cutoff(
    std::span<const double> xs, std::span<const double> ys,
    std::span<const double> zs, const box &frame_box,
    const std::vector<unsigned char> &active_mask, double cutoff_a,
    titan_neighbor::HalfShellGrid &neighbor_grid,
    const std::function<void(std::size_t, std::size_t, double)> &visit);

std::vector<double>
build_structure_factor_from_gr(const std::vector<double> &g_profile,
                               double rdf_bin_a, double rdf_max_a,
                               double q_bin_inv_a, std::size_t q_bin_count,
                               double rho_partner_a3,
                               bool include_self_baseline);

} // namespace titan_sq_detail

#endif // TITAN_SQ_CORE_H
