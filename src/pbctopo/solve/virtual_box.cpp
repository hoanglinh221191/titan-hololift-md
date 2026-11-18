#include "pbctopo/solve/solve_common.h"

namespace pbctopo_solve {

bool find_virtual_box_shift(double wrapped_coord, double center, int &shift_idx) {
  const double low = center - 0.5;
  const double high = center + 0.5;
  for (int candidate = 0; candidate <= 2; ++candidate) {
    const double coord = wrapped_coord + static_cast<double>(candidate);
    if (coord + kDistanceEps >= low && coord < high - kDistanceEps) {
      shift_idx = candidate;
      return true;
    }
  }
  const double mid = 0.5 * (low + high);
  double best_dist = numeric_limits<double>::infinity();
  int best_candidate = 0;
  for (int candidate = 0; candidate <= 2; ++candidate) {
    const double coord = wrapped_coord + static_cast<double>(candidate);
    const double dist = fabs(coord - mid);
    if (dist < best_dist) {
      best_dist = dist;
      best_candidate = candidate;
    }
  }
  shift_idx = best_candidate;
  return true;
}

void push_unique_xy_seed(vector<PbctopoSeed2D> &best, const PbctopoSeed2D &seed,
                         size_t max_keep) {
  for (const auto &existing : best) {
    if (fabs(existing.cx - seed.cx) <= 1.0e-9 &&
        fabs(existing.cy - seed.cy) <= 1.0e-9) {
      return;
    }
  }
  best.push_back(seed);
  sort(best.begin(), best.end(), [](const PbctopoSeed2D &lhs,
                                    const PbctopoSeed2D &rhs) {
    int cmp = compare_pbctopo_double_metric(lhs.score, rhs.score, 1.0e-9);
    if (cmp != 0)
      return cmp > 0;
    cmp = compare_pbctopo_double_metric(lhs.cx, rhs.cx, 1.0e-9);
    if (cmp != 0)
      return cmp < 0;
    return compare_pbctopo_double_metric(lhs.cy, rhs.cy, 1.0e-9) < 0;
  });
  if (best.size() > max_keep)
    best.resize(max_keep);
}

void push_unique_axis_seed(vector<PbctopoCutOption> &best,
                           const PbctopoCutOption &seed, size_t max_keep) {
  for (const auto &existing : best) {
    if (fabs(existing.cut - seed.cut) <= 1.0e-9)
      return;
  }
  best.push_back(seed);
  sort(best.begin(), best.end(),
       [](const PbctopoCutOption &lhs, const PbctopoCutOption &rhs) {
         const int gap_cmp =
             compare_pbctopo_double_metric(lhs.gap, rhs.gap, 1.0e-9);
         if (gap_cmp != 0)
           return gap_cmp > 0;
         return compare_pbctopo_double_metric(lhs.cut, rhs.cut, 1.0e-9) < 0;
       });
  if (best.size() > max_keep)
    best.resize(max_keep);
}

void push_unique_xyz_seed(vector<PbctopoSeed3D> &best, const PbctopoSeed3D &seed,
                          size_t max_keep) {
  for (const auto &existing : best) {
    if (fabs(existing.cx - seed.cx) <= 1.0e-9 &&
        fabs(existing.cy - seed.cy) <= 1.0e-9 &&
        fabs(existing.cz - seed.cz) <= 1.0e-9) {
      return;
    }
  }
  best.push_back(seed);
  sort(best.begin(), best.end(), [](const PbctopoSeed3D &lhs,
                                    const PbctopoSeed3D &rhs) {
    int cmp = compare_pbctopo_double_metric(lhs.score, rhs.score, 1.0e-9);
    if (cmp != 0)
      return cmp > 0;
    cmp = compare_pbctopo_double_metric(lhs.cx, rhs.cx, 1.0e-9);
    if (cmp != 0)
      return cmp < 0;
    cmp = compare_pbctopo_double_metric(lhs.cy, rhs.cy, 1.0e-9);
    if (cmp != 0)
      return cmp < 0;
    return compare_pbctopo_double_metric(lhs.cz, rhs.cz, 1.0e-9) < 0;
  });
  if (best.size() > max_keep)
    best.resize(max_keep);
}

vector<double> build_projection_density_grid(const vector<PbctopoAtom> &atoms,
                                             size_t mesh_bins, double radius,
                                             char axis_a, char axis_b) {
  vector<double> scores(mesh_bins * mesh_bins, 0.0);
  if (atoms.empty() || mesh_bins == 0)
    return scores;

  const double span = 2.0;
  const double step =
      (mesh_bins == 1) ? 0.0 : span / static_cast<double>(mesh_bins - 1);
  auto clamp_index = [mesh_bins](long long value) -> size_t {
    if (value < 0)
      return 0;
    const long long max_index = static_cast<long long>(mesh_bins) - 1;
    if (value > max_index)
      return static_cast<size_t>(max_index);
    return static_cast<size_t>(value);
  };

  for (const auto &atom : atoms) {
    const double wrapped_a = projection_axis_coord(atom, axis_a);
    const double wrapped_b = projection_axis_coord(atom, axis_b);
    for (int shift_a = 0; shift_a <= 2; ++shift_a) {
      const double shifted_a = wrapped_a + static_cast<double>(shift_a);
      if (shifted_a + radius < 0.5 || shifted_a - radius > 2.5)
        continue;
      const size_t ia_begin =
          (step <= kDistanceEps)
              ? 0
              : clamp_index(static_cast<long long>(
                    ceil((shifted_a - radius - 0.5) / step)));
      const size_t ia_end =
          (step <= kDistanceEps)
              ? 0
              : clamp_index(static_cast<long long>(
                    floor((shifted_a + radius - 0.5) / step)));
      for (int shift_b = 0; shift_b <= 2; ++shift_b) {
        const double shifted_b = wrapped_b + static_cast<double>(shift_b);
        if (shifted_b + radius < 0.5 || shifted_b - radius > 2.5)
          continue;
        const size_t ib_begin =
            (step <= kDistanceEps)
                ? 0
                : clamp_index(static_cast<long long>(
                      ceil((shifted_b - radius - 0.5) / step)));
        const size_t ib_end =
            (step <= kDistanceEps)
                ? 0
                : clamp_index(static_cast<long long>(
                      floor((shifted_b + radius - 0.5) / step)));
        for (size_t ia = ia_begin; ia <= ia_end; ++ia) {
          const double ca = 0.5 + step * static_cast<double>(ia);
          const double da = fabs(shifted_a - ca);
          if (da > radius)
            continue;
          for (size_t ib = ib_begin; ib <= ib_end; ++ib) {
            const double cb = 0.5 + step * static_cast<double>(ib);
            const double db = fabs(shifted_b - cb);
            if (db > radius)
              continue;
            scores[ia * mesh_bins + ib] += (radius - da) * (radius - db);
          }
        }
      }
    }
  }
  return scores;
}

bool projection_density_is_local_maximum(const vector<double> &scores,
                                         size_t mesh_bins, size_t ix,
                                         size_t iy) {
  const double score = scores[ix * mesh_bins + iy];
  if (!(score > 0.0))
    return false;

  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      if (dx == 0 && dy == 0)
        continue;
      const long long nx_ll = static_cast<long long>(ix) + dx;
      const long long ny_ll = static_cast<long long>(iy) + dy;
      if (nx_ll < 0 || ny_ll < 0)
        continue;
      const size_t nx = static_cast<size_t>(nx_ll);
      const size_t ny = static_cast<size_t>(ny_ll);
      if (nx >= mesh_bins || ny >= mesh_bins)
        continue;
      const double neighbor = scores[nx * mesh_bins + ny];
      if (neighbor > score + 1.0e-12)
        return false;
      if (fabs(neighbor - score) <= 1.0e-12 &&
          (nx < ix || (nx == ix && ny < iy))) {
        return false;
      }
    }
  }
  return true;
}

bool projection_seed2d_better(const PbctopoSeed2D &lhs,
                              const PbctopoSeed2D &rhs) {
  const int score_cmp =
      compare_pbctopo_double_metric(lhs.score, rhs.score, 1.0e-12);
  if (score_cmp != 0)
    return score_cmp > 0;
  if (lhs.ix != rhs.ix)
    return lhs.ix < rhs.ix;
  return lhs.iy < rhs.iy;
}

void collect_projection_plane_hotspots(const vector<double> &scores,
                                       size_t mesh_bins, size_t top_k,
                                       vector<PbctopoSeed2D> &hotspots) {
  hotspots.clear();
  if (mesh_bins == 0 || top_k == 0 || scores.size() != mesh_bins * mesh_bins)
    return;

  vector<PbctopoSeed2D> local_maxima;
  vector<PbctopoSeed2D> ranked_cells;
  local_maxima.reserve(mesh_bins * mesh_bins / 4 + 1);
  ranked_cells.reserve(mesh_bins * mesh_bins);

  for (size_t ix = 0; ix < mesh_bins; ++ix) {
    const double cx = projection_grid_center(mesh_bins, ix);
    for (size_t iy = 0; iy < mesh_bins; ++iy) {
      const double score = scores[ix * mesh_bins + iy];
      if (!(score > 0.0))
        continue;
      PbctopoSeed2D seed{cx, projection_grid_center(mesh_bins, iy), score, ix,
                         iy};
      ranked_cells.push_back(seed);
      if (projection_density_is_local_maximum(scores, mesh_bins, ix, iy))
        local_maxima.push_back(seed);
    }
  }

  sort(local_maxima.begin(), local_maxima.end(), projection_seed2d_better);
  sort(ranked_cells.begin(), ranked_cells.end(), projection_seed2d_better);

  auto append_from = [&](const vector<PbctopoSeed2D> &source,
                         long long min_bin_gap) {
    for (const auto &seed : source) {
      if (hotspots.size() >= top_k)
        break;
      bool too_close = false;
      for (const auto &existing : hotspots) {
        const long long dx =
            llabs(static_cast<long long>(existing.ix) -
                  static_cast<long long>(seed.ix));
        const long long dy =
            llabs(static_cast<long long>(existing.iy) -
                  static_cast<long long>(seed.iy));
        if (max(dx, dy) <= min_bin_gap) {
          too_close = true;
          break;
        }
      }
      if (!too_close)
        hotspots.push_back(seed);
    }
  };

  const long long nms_bin_gap = (mesh_bins >= 10) ? 1ll : 0ll;
  append_from(local_maxima, nms_bin_gap);
  if (hotspots.size() < min(top_k, static_cast<size_t>(4)))
    append_from(ranked_cells, nms_bin_gap);
  if (hotspots.size() < top_k)
    append_from(ranked_cells, 0);
}

double projection_grid_score(const vector<double> &scores, size_t mesh_bins,
                             size_t ia, size_t ib) {
  if (ia >= mesh_bins || ib >= mesh_bins || scores.size() != mesh_bins * mesh_bins)
    return 0.0;
  return scores[ia * mesh_bins + ib];
}

void append_dense_projection_volume_seeds(const vector<double> &xy_scores,
                                          const vector<double> &xz_scores,
                                          const vector<double> &yz_scores,
                                          size_t mesh_bins, size_t top_k,
                                          vector<PbctopoSeed3D> &seeds) {
  for (size_t ix = 0; ix < mesh_bins; ++ix) {
    const double cx = projection_grid_center(mesh_bins, ix);
    for (size_t iy = 0; iy < mesh_bins; ++iy) {
      const double cy = projection_grid_center(mesh_bins, iy);
      const double xy_score = projection_grid_score(xy_scores, mesh_bins, ix, iy);
      for (size_t iz = 0; iz < mesh_bins; ++iz) {
        const double score =
            xy_score + projection_grid_score(xz_scores, mesh_bins, ix, iz) +
            projection_grid_score(yz_scores, mesh_bins, iy, iz);
        push_unique_xyz_seed(
            seeds,
            {cx, cy, projection_grid_center(mesh_bins, iz), score, ix, iy, iz},
            top_k);
      }
    }
  }
}

void collect_projection_volume_seeds(const vector<PbctopoAtom> &atoms,
                                     size_t mesh_bins, size_t top_k,
                                     double radius,
                                     vector<PbctopoSeed3D> &seeds) {
  seeds.clear();
  if (atoms.empty() || mesh_bins == 0 || top_k == 0)
    return;

  const auto xy_scores =
      build_projection_density_grid(atoms, mesh_bins, radius, 'x', 'y');
  const auto xz_scores =
      build_projection_density_grid(atoms, mesh_bins, radius, 'x', 'z');
  const auto yz_scores =
      build_projection_density_grid(atoms, mesh_bins, radius, 'y', 'z');

  vector<PbctopoSeed2D> xy_hotspots;
  vector<PbctopoSeed2D> xz_hotspots;
  vector<PbctopoSeed2D> yz_hotspots;
  const size_t plane_top_k =
      min(mesh_bins * mesh_bins, max<size_t>(top_k * 2, static_cast<size_t>(12)));
  collect_projection_plane_hotspots(xy_scores, mesh_bins, plane_top_k,
                                    xy_hotspots);
  collect_projection_plane_hotspots(xz_scores, mesh_bins, plane_top_k,
                                    xz_hotspots);
  collect_projection_plane_hotspots(yz_scores, mesh_bins, plane_top_k,
                                    yz_hotspots);

  seeds.reserve(top_k);
  auto push_seed = [&](size_t ix, size_t iy, size_t iz, double score) {
    push_unique_xyz_seed(
        seeds,
        {projection_grid_center(mesh_bins, ix), projection_grid_center(mesh_bins, iy),
         projection_grid_center(mesh_bins, iz), score, ix, iy, iz},
        top_k);
  };

  for (const auto &xy_seed : xy_hotspots) {
    for (const auto &xz_seed : xz_hotspots) {
      if (xy_seed.ix != xz_seed.ix)
        continue;
      for (const auto &yz_seed : yz_hotspots) {
        if (xy_seed.iy != yz_seed.ix || xz_seed.iy != yz_seed.iy)
          continue;
        push_seed(xy_seed.ix, xy_seed.iy, xz_seed.iy,
                  xy_seed.score + xz_seed.score + yz_seed.score);
      }
    }
  }

  if (seeds.size() < top_k) {
    for (const auto &xy_seed : xy_hotspots) {
      for (const auto &xz_seed : xz_hotspots) {
        if (xy_seed.ix != xz_seed.ix)
          continue;
        push_seed(xy_seed.ix, xy_seed.iy, xz_seed.iy,
                  xy_seed.score + xz_seed.score +
                      projection_grid_score(yz_scores, mesh_bins, xy_seed.iy,
                                            xz_seed.iy));
      }
    }
  }

  if (seeds.size() < top_k) {
    for (const auto &xy_seed : xy_hotspots) {
      for (const auto &yz_seed : yz_hotspots) {
        if (xy_seed.iy != yz_seed.ix)
          continue;
        push_seed(xy_seed.ix, xy_seed.iy, yz_seed.iy,
                  xy_seed.score + yz_seed.score +
                      projection_grid_score(xz_scores, mesh_bins, xy_seed.ix,
                                            yz_seed.iy));
      }
    }
  }

  if (seeds.size() < top_k) {
    for (const auto &xz_seed : xz_hotspots) {
      for (const auto &yz_seed : yz_hotspots) {
        if (xz_seed.iy != yz_seed.iy)
          continue;
        push_seed(xz_seed.ix, yz_seed.ix, xz_seed.iy,
                  xz_seed.score + yz_seed.score +
                      projection_grid_score(xy_scores, mesh_bins, xz_seed.ix,
                                            yz_seed.ix));
      }
    }
  }

  if (seeds.size() < top_k)
    append_dense_projection_volume_seeds(xy_scores, xz_scores, yz_scores,
                                         mesh_bins, top_k, seeds);
}

[[maybe_unused]] vector<PbctopoSeed2D>
collect_xy_density_seeds(const vector<PbctopoAtom> &atoms, size_t mesh_bins,
                         size_t top_k, double radius) {
  vector<PbctopoSeed2D> seeds;
  if (atoms.empty() || mesh_bins == 0 || top_k == 0)
    return seeds;

  const double span = 2.0;
  const double step =
      (mesh_bins == 1) ? 0.0 : span / static_cast<double>(mesh_bins - 1);
  vector<double> scores(mesh_bins * mesh_bins, 0.0);
  auto clamp_index = [mesh_bins](long long value) -> size_t {
    if (value < 0)
      return 0;
    const long long max_index = static_cast<long long>(mesh_bins) - 1;
    if (value > max_index)
      return static_cast<size_t>(max_index);
    return static_cast<size_t>(value);
  };

  for (const auto &atom : atoms) {
    for (int shift_x = 0; shift_x <= 2; ++shift_x) {
      const double shifted_x = atom.sx + static_cast<double>(shift_x);
      if (shifted_x + radius < 0.5 || shifted_x - radius > 2.5)
        continue;
      const size_t ix_begin =
          (step <= kDistanceEps)
              ? 0
              : clamp_index(static_cast<long long>(
                    ceil((shifted_x - radius - 0.5) / step)));
      const size_t ix_end =
          (step <= kDistanceEps)
              ? 0
              : clamp_index(static_cast<long long>(
                    floor((shifted_x + radius - 0.5) / step)));
      for (int shift_y = 0; shift_y <= 2; ++shift_y) {
        const double shifted_y = atom.sy + static_cast<double>(shift_y);
        if (shifted_y + radius < 0.5 || shifted_y - radius > 2.5)
          continue;
        const size_t iy_begin =
            (step <= kDistanceEps)
                ? 0
                : clamp_index(static_cast<long long>(
                      ceil((shifted_y - radius - 0.5) / step)));
        const size_t iy_end =
            (step <= kDistanceEps)
                ? 0
                : clamp_index(static_cast<long long>(
                      floor((shifted_y + radius - 0.5) / step)));
        for (size_t ix = ix_begin; ix <= ix_end; ++ix) {
          const double cx = 0.5 + step * static_cast<double>(ix);
          const double dx = fabs(shifted_x - cx);
          if (dx > radius)
            continue;
          for (size_t iy = iy_begin; iy <= iy_end; ++iy) {
            const double cy = 0.5 + step * static_cast<double>(iy);
            const double dy = fabs(shifted_y - cy);
            if (dy > radius)
              continue;
            scores[ix * mesh_bins + iy] += (radius - dx) * (radius - dy);
          }
        }
      }
    }
  }

  for (size_t ix = 0; ix < mesh_bins; ++ix) {
    const double cx = 0.5 + step * static_cast<double>(ix);
    for (size_t iy = 0; iy < mesh_bins; ++iy) {
      const double cy = 0.5 + step * static_cast<double>(iy);
      push_unique_xy_seed(seeds, {cx, cy, scores[ix * mesh_bins + iy]}, top_k);
    }
  }
  return seeds;
}

[[maybe_unused]] vector<PbctopoCutOption>
collect_z_density_seeds(const vector<PbctopoAtom> &atoms, size_t mesh_bins,
                        size_t top_k, double radius) {
  vector<PbctopoCutOption> seeds;
  if (atoms.empty() || mesh_bins == 0 || top_k == 0)
    return seeds;

  const double span = 2.0;
  const double step =
      (mesh_bins == 1) ? 0.0 : span / static_cast<double>(mesh_bins - 1);
  vector<double> scores(mesh_bins, 0.0);
  auto clamp_index = [mesh_bins](long long value) -> size_t {
    if (value < 0)
      return 0;
    const long long max_index = static_cast<long long>(mesh_bins) - 1;
    if (value > max_index)
      return static_cast<size_t>(max_index);
    return static_cast<size_t>(value);
  };

  for (const auto &atom : atoms) {
    for (int shift_z = 0; shift_z <= 2; ++shift_z) {
      const double shifted_z = atom.sz + static_cast<double>(shift_z);
      if (shifted_z + radius < 0.5 || shifted_z - radius > 2.5)
        continue;
      const size_t iz_begin =
          (step <= kDistanceEps)
              ? 0
              : clamp_index(static_cast<long long>(
                    ceil((shifted_z - radius - 0.5) / step)));
      const size_t iz_end =
          (step <= kDistanceEps)
              ? 0
              : clamp_index(static_cast<long long>(
                    floor((shifted_z + radius - 0.5) / step)));
      for (size_t iz = iz_begin; iz <= iz_end; ++iz) {
        const double cz = 0.5 + step * static_cast<double>(iz);
        const double dz = fabs(shifted_z - cz);
        if (dz > radius)
          continue;
        scores[iz] += radius - dz;
      }
    }
  }

  for (size_t iz = 0; iz < mesh_bins; ++iz) {
    const double cz = 0.5 + step * static_cast<double>(iz);
    push_unique_axis_seed(seeds, {cz, scores[iz]}, top_k);
  }
  return seeds;
}

void push_top_virtual_box_candidate(vector<PbctopoCutCandidate> &top_candidates,
                                    const PbctopoCutCandidate &candidate,
                                    size_t max_keep) {
  if (!candidate.valid)
    return;
  top_candidates.push_back(candidate);
  sort(top_candidates.begin(), top_candidates.end(), better_cut_candidate);
  if (top_candidates.size() > max_keep)
    top_candidates.resize(max_keep);
}

[[maybe_unused]] bool
build_virtual_box_positions(const vector<PbctopoAtom> &atoms, box Box,
                            double center_x, double center_y, double center_z,
                            vector<Vec3> &placed_points, PointStats &stats,
                            double &shift_norm2) {
  placed_points.resize(atoms.size());
  stats = PointStats{};
  shift_norm2 = 0.0;

  const int global_shift_x = static_cast<int>(floor(center_x)) - 1;
  const int global_shift_y = static_cast<int>(floor(center_y)) - 1;
  const int global_shift_z = static_cast<int>(floor(center_z)) - 1;

  for (size_t idx = 0; idx < atoms.size(); ++idx) {
    const auto &atom = atoms[idx];
    int search_shift_x = -1;
    int search_shift_y = -1;
    int search_shift_z = -1;
    if (!find_virtual_box_shift(atom.sx, center_x, search_shift_x) ||
        !find_virtual_box_shift(atom.sy, center_y, search_shift_y) ||
        !find_virtual_box_shift(atom.sz, center_z, search_shift_z)) {
      return false;
    }

    const int actual_shift_x = search_shift_x - 1 - global_shift_x;
    const int actual_shift_y = search_shift_y - 1 - global_shift_y;
    const int actual_shift_z = search_shift_z - 1 - global_shift_z;

    Vec3 point{};
    scaled_to_cartesian(Box, atom.sx + static_cast<double>(actual_shift_x),
                        atom.sy + static_cast<double>(actual_shift_y),
                        atom.sz + static_cast<double>(actual_shift_z), point.x,
                        point.y, point.z);
    placed_points[idx] = point;
    stats.add(point.x, point.y, point.z);
    shift_norm2 += static_cast<double>(actual_shift_x * actual_shift_x +
                                       actual_shift_y * actual_shift_y +
                                       actual_shift_z * actual_shift_z);
  }

  return stats.count == atoms.size();
}

PbctopoCutCandidate evaluate_virtual_box_candidate(
    const vector<PbctopoAtom> &atoms, const vector<PbctopoLocalEdge> &local_edges,
    const unordered_set<uint64_t> &edge_set,
    const PbctopoTranslationCache &translation_cache, box Box, double cut_x,
    double cut_y, double cut_z, double gap_score, vector<Vec3> &placed_points,
    double clash_cutoff, const vector<PbctopoContinuityPair> *continuity_pairs,
    const vector<size_t> *anchor_indices,
    const vector<pair<int, int>> *contacting_owner_pairs,
    double continuity_cutoff, const PbctopoCutCandidate *incumbent,
    bool skip_clash, double frame_time_ps, const char *reject_context,
    const PbctopoAnchorGraphContext *anchor_context,
    PbctopoVirtualSeedSource seed_source,
    const vector<size_t> *atom_component_index) {
  PbctopoCutCandidate candidate;
  candidate.cut_x = clamp_virtual_box_center(cut_x);
  candidate.cut_y = clamp_virtual_box_center(cut_y);
  candidate.cut_z = clamp_virtual_box_center(cut_z);
  candidate.gap_score = gap_score;
  candidate.virtual_seed_source = seed_source;
  candidate.broken_edges = 0;
  candidate.edge_stretch = 0.0;
  candidate.shift_norm2 = 0.0;
  const PbctopoCutCandidate *active_incumbent =
      (incumbent != nullptr && incumbent->valid && !incumbent->ultra_clash)
          ? incumbent
          : nullptr;
  const PbctopoCutCandidate *metric_incumbent =
      (active_incumbent != nullptr && active_incumbent->broken_edges == 0)
          ? active_incumbent
          : nullptr;

  if (atoms.empty())
    return candidate;

  placed_points.resize(atoms.size());
  PointStats stats;
  vector<Int3> atom_image_offsets(atoms.size());

  const int global_shift_x = static_cast<int>(floor(candidate.cut_x)) - 1;
  const int global_shift_y = static_cast<int>(floor(candidate.cut_y)) - 1;
  const int global_shift_z = static_cast<int>(floor(candidate.cut_z)) - 1;

  for (size_t idx = 0; idx < atoms.size(); ++idx) {
    const auto &atom = atoms[idx];
    int search_shift_x = -1;
    int search_shift_y = -1;
    int search_shift_z = -1;
    if (!find_virtual_box_shift(atom.sx, candidate.cut_x, search_shift_x) ||
        !find_virtual_box_shift(atom.sy, candidate.cut_y, search_shift_y) ||
        !find_virtual_box_shift(atom.sz, candidate.cut_z, search_shift_z)) {
      return PbctopoCutCandidate{};
    }

    const int actual_shift_x = search_shift_x - 1 - global_shift_x;
    const int actual_shift_y = search_shift_y - 1 - global_shift_y;
    const int actual_shift_z = search_shift_z - 1 - global_shift_z;
    const Vec3 &translation =
        translation_cache.by_shift[actual_shift_x + 2][actual_shift_y + 2]
                                  [actual_shift_z + 2];
    Vec3 point{atom.x + translation.x, atom.y + translation.y,
               atom.z + translation.z};
    placed_points[idx] = point;
    stats.add(point.x, point.y, point.z);
    atom_image_offsets[idx] = {actual_shift_x, actual_shift_y, actual_shift_z};
  }

  if (stats.count != atoms.size())
    return PbctopoCutCandidate{};

  const vector<Int3> *metric_offsets = &atom_image_offsets;
  vector<Int3> component_image_offsets;
  vector<unsigned char> component_offset_seen;
  bool component_assignment_consistent =
      atom_component_index != nullptr &&
      atom_component_index->size() == atoms.size();
  if (component_assignment_consistent) {
    size_t component_count = 0;
    for (size_t comp_idx : *atom_component_index) {
      if (comp_idx >= atoms.size()) {
        component_assignment_consistent = false;
        break;
      }
      component_count = max(component_count, comp_idx + 1);
    }
    if (component_assignment_consistent && component_count > 0) {
      component_image_offsets.resize(component_count);
      component_offset_seen.assign(component_count, 0);
      for (size_t atom_idx = 0; atom_idx < atom_image_offsets.size(); ++atom_idx) {
        const size_t comp_idx = (*atom_component_index)[atom_idx];
        if (component_offset_seen[comp_idx] == 0) {
          component_image_offsets[comp_idx] = atom_image_offsets[atom_idx];
          component_offset_seen[comp_idx] = 1;
        } else if (!same_delta(component_image_offsets[comp_idx],
                               atom_image_offsets[atom_idx])) {
          component_assignment_consistent = false;
          break;
        }
      }
      if (component_assignment_consistent &&
          find(component_offset_seen.begin(), component_offset_seen.end(), 0) ==
              component_offset_seen.end()) {
        metric_offsets = &component_image_offsets;
      }
    }
  }
  candidate.shift_norm2 = pbctopo_component_shift_metric(Box, *metric_offsets);

  if (contacting_owner_pairs != nullptr) {
    PbctopoOwnerPairSplitDetail split_detail;
    const bool split_observed = candidate_has_split_contacting_owners(
        placed_points, atoms, Box, *contacting_owner_pairs,
        candidate.split_owner_pairs,
        candidate.max_owner_raw_minus_minimg_delta, &split_detail);
    if (split_observed) {
      if (reject_context != nullptr && split_detail.valid &&
          isfinite(frame_time_ps)) {
        append_pbctopo_log_line(
            "[pbctopo] observe owner_com_image_gap at frame " +
            to_string(static_cast<long long>(frame_time_ps)) + " via " +
            reject_context + ": owners=" + to_string(split_detail.owner_a) + "-" +
            to_string(split_detail.owner_b) +
            ", raw_delta_scaled=" + format_vec3(split_detail.raw_delta_scaled) +
            ", min_image_delta_scaled=" +
            format_vec3(split_detail.minimg_delta_scaled) +
            ", policy=diagnostic_only");
      }
    }
  }

  if (continuity_pairs != nullptr) {
    double continuity_path = numeric_limits<double>::infinity();
    double continuity_max_d2 = numeric_limits<double>::infinity();
    double continuity_min_margin_A = numeric_limits<double>::quiet_NaN();
    size_t continuity_margin_atom_a = numeric_limits<size_t>::max();
    size_t continuity_margin_atom_b = numeric_limits<size_t>::max();
    if (!continuity_metrics_ok(placed_points, *continuity_pairs, continuity_cutoff,
                               continuity_path, continuity_max_d2,
                               continuity_min_margin_A,
                               continuity_margin_atom_a,
                               continuity_margin_atom_b,
                               metric_incumbent)) {
      return PbctopoCutCandidate{};
    }
    candidate.continuity_path = continuity_path;
    candidate.continuity_max_d2 = continuity_max_d2;
    candidate.continuity_min_margin_A = continuity_min_margin_A;
    candidate.continuity_margin_atom_a = continuity_margin_atom_a;
    candidate.continuity_margin_atom_b = continuity_margin_atom_b;
  }
  if (anchor_indices != nullptr) {
    double anchor_mst2 = numeric_limits<double>::infinity();
    double anchor_max_d2 = numeric_limits<double>::infinity();
    PbctopoAnchorGraphScore anchor_graph_score;
    const vector<PbctopoContinuityPair> empty_continuity_pairs;
    const vector<PbctopoContinuityPair> &anchor_continuity_pairs =
        (continuity_pairs != nullptr) ? *continuity_pairs
                                      : empty_continuity_pairs;
    if (!anchor_graph_metrics_enhanced(
            placed_points, *anchor_indices, anchor_continuity_pairs,
            anchor_mst2, anchor_max_d2, anchor_graph_score, metric_incumbent,
            PbctopoRefineScorePhase::Exact, PbctopoAnchorBackend::Prim,
            nullptr, anchor_context)) {
      return PbctopoCutCandidate{};
    }
    candidate.anchor_mst2 = anchor_mst2;
    candidate.anchor_max_d2 = anchor_max_d2;
    candidate.anchor_graph = anchor_graph_score;
  }

  candidate.bbox_volume = compute_bbox_volume(stats);
  candidate.rg2 = compute_rg2_from_stats(stats);
  const array<double, 3> principal_rg2 = compute_principal_rg2(stats);
  candidate.principal_rg2_max = principal_rg2[0];
  candidate.principal_rg2_mid = principal_rg2[1];
  candidate.principal_rg2_min = principal_rg2[2];
  const auto shape_metrics = compute_gyration_shape_metrics(principal_rg2);
  candidate.shape_kappa2 = shape_metrics.kappa2;
  candidate.shape_asphericity = shape_metrics.asphericity;
  candidate.shape_acylindricity = shape_metrics.acylindricity;

  for (const auto &edge : local_edges) {
    const Vec3 delta = placed_points[edge.a] - placed_points[edge.b];
    const double d2 = norm2(delta);
    if (d2 > edge.break_cutoff_d2)
      ++candidate.broken_edges;
    candidate.edge_stretch +=
        pbctopo_symmetric_reference_strain(d2, edge.ref_d2);
  }

  candidate.valid = true;
  if (!skip_clash &&
      preclash_cut_candidate_cannot_beat_incumbent(candidate,
                                                   active_incumbent)) {
    return PbctopoCutCandidate{};
  }
  if (!skip_clash)
    analyze_cut_candidate_ultraclash(placed_points, edge_set, candidate,
                                     clash_cutoff, &atoms,
                                     atom_component_index);
  return candidate;
}

} // namespace pbctopo_solve
