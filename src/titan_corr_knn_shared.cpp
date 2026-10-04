#include "titan_corr_knn_shared.h"

#include "titan_neighbor_grid.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace titan_corr_knn {
namespace {

inline std::size_t upper_triangle_pair_index(std::size_t i, std::size_t j,
                                             std::size_t node_count) {
  return i * (node_count - 1) - (i * (i - 1)) / 2 + (j - i - 1);
}

std::vector<std::pair<std::size_t, std::size_t>>
build_dense_upper_triangle_pairs(std::size_t node_count) {
  std::vector<std::pair<std::size_t, std::size_t>> pairs;
  if (node_count < 2)
    return pairs;
  pairs.reserve(node_count * (node_count - 1) / 2);
  for (std::size_t i = 0; i < node_count; ++i)
    for (std::size_t j = i + 1; j < node_count; ++j)
      pairs.emplace_back(i, j);
  return pairs;
}

std::uint64_t pack_pair_key(std::size_t i, std::size_t j) noexcept {
  const std::uint64_t a = static_cast<std::uint64_t>(std::min(i, j));
  const std::uint64_t b = static_cast<std::uint64_t>(std::max(i, j));
  return (a << 32) | b;
}

std::vector<std::pair<std::size_t, std::size_t>>
build_frame0_spatial_sparse_pairs(const TrajectoryRows &traj_x,
                                  const TrajectoryRows &traj_y,
                                  const TrajectoryRows &traj_z,
                                  std::size_t node_count,
                                  double cutoff_ang, int topk) {
  struct Neighbor {
    float dist2{0.0f};
    std::size_t partner{0};
  };
  std::vector<std::pair<std::size_t, std::size_t>> pairs;
  if (node_count < 2 || cutoff_ang <= 0.0 || traj_x.empty() ||
      traj_y.empty() || traj_z.empty() || traj_x[0].size() < node_count ||
      traj_y[0].size() < node_count || traj_z[0].size() < node_count ||
      node_count > static_cast<std::size_t>(
                       std::numeric_limits<std::uint32_t>::max())) {
    return pairs;
  }

  const auto &x_row = traj_x[0];
  const auto &y_row = traj_y[0];
  const auto &z_row = traj_z[0];
  const auto [xmin, xmax] =
      std::minmax_element(x_row.begin(), x_row.begin() + node_count);
  const auto [ymin, ymax] =
      std::minmax_element(y_row.begin(), y_row.begin() + node_count);
  const auto [zmin, zmax] =
      std::minmax_element(z_row.begin(), z_row.begin() + node_count);

  titan_neighbor::OrthorhombicGridSpec spec;
  spec.cutoff = cutoff_ang;
  spec.shift_x = -(*xmin) + cutoff_ang;
  spec.shift_y = -(*ymin) + cutoff_ang;
  spec.shift_z = -(*zmin) + cutoff_ang;
  spec.len_x = (*xmax - *xmin) + 2.0 * cutoff_ang;
  spec.len_y = (*ymax - *ymin) + 2.0 * cutoff_ang;
  spec.len_z = (*zmax - *zmin) + 2.0 * cutoff_ang;
  spec.periodic = false;

  std::vector<std::uint64_t> keys;
  keys.reserve(node_count * static_cast<std::size_t>(std::max(4, topk)));
  std::vector<std::vector<Neighbor>> ranked_neighbors;
  const bool use_knn = topk > 0;
  if (use_knn)
    ranked_neighbors.resize(node_count);
  const double cutoff2 = cutoff_ang * cutoff_ang;
  titan_neighbor::HalfShellGrid neighbor_grid;
  neighbor_grid.for_each_pair(
      node_count, spec,
      [&](std::size_t atom_index, double &x, double &y, double &z) {
        x = x_row[atom_index];
        y = y_row[atom_index];
        z = z_row[atom_index];
      },
      [&](std::size_t atom_index) { return atom_index < node_count; },
      [&](std::size_t lhs, std::size_t rhs) {
        const double dx = x_row[lhs] - x_row[rhs];
        const double dy = y_row[lhs] - y_row[rhs];
        const double dz = z_row[lhs] - z_row[rhs];
        const double dist2 = dx * dx + dy * dy + dz * dz;
        if (dist2 > cutoff2)
          return;
        keys.push_back(pack_pair_key(lhs, rhs));
        if (use_knn) {
          const float d2 = static_cast<float>(dist2);
          ranked_neighbors[lhs].push_back(Neighbor{d2, rhs});
          ranked_neighbors[rhs].push_back(Neighbor{d2, lhs});
        }
      });

  if (use_knn) {
    auto better = [](const Neighbor &lhs, const Neighbor &rhs) {
      if (lhs.dist2 != rhs.dist2)
        return lhs.dist2 < rhs.dist2;
      return lhs.partner < rhs.partner;
    };
    const std::size_t keep_topk =
        std::min<std::size_t>(static_cast<std::size_t>(topk), node_count - 1);
    for (std::size_t i = 0; i < node_count; ++i) {
      auto &neighbors = ranked_neighbors[i];
      if (neighbors.empty())
        continue;
      if (neighbors.size() > keep_topk) {
        std::nth_element(
            neighbors.begin(),
            neighbors.begin() + static_cast<std::ptrdiff_t>(keep_topk),
            neighbors.end(), better);
        neighbors.resize(keep_topk);
      }
      for (const auto &neighbor : neighbors)
        keys.push_back(pack_pair_key(i, neighbor.partner));
    }
  }

  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  pairs.reserve(keys.size());
  for (const std::uint64_t key : keys) {
    pairs.emplace_back(static_cast<std::size_t>(key >> 32),
                       static_cast<std::size_t>(key & 0xffffffffULL));
  }
  return pairs;
}

bool has_complete_rows(const TrajectoryRows &x_rows, const TrajectoryRows &y_rows,
                       const TrajectoryRows &z_rows, std::size_t node_count,
                       std::size_t frame_limit) {
  if (x_rows.size() < frame_limit || y_rows.size() < frame_limit ||
      z_rows.size() < frame_limit) {
    return false;
  }
  for (std::size_t frame = 0; frame < frame_limit; ++frame) {
    if (x_rows[frame].size() < node_count || y_rows[frame].size() < node_count ||
        z_rows[frame].size() < node_count) {
      return false;
    }
  }
  return true;
}

} // namespace

std::string build_node_label(const std::vector<std::string> &resnames,
                             const std::vector<int> &resnrs,
                             const std::vector<std::string> &chains,
                             std::size_t index) {
  return resnames[index] + std::to_string(resnrs[index]) + "_" + chains[index];
}

std::vector<std::string> build_node_labels(
    const std::vector<std::string> &resnames, const std::vector<int> &resnrs,
    const std::vector<std::string> &chains) {
  const std::size_t node_count = resnames.size();
  std::vector<std::string> labels(node_count);
  for (std::size_t i = 0; i < node_count; ++i)
    labels[i] = build_node_label(resnames, resnrs, chains, i);
  return labels;
}

std::vector<std::pair<std::size_t, std::size_t>> build_candidate_pairs(
    const TrajectoryRows &traj_x, const TrajectoryRows &traj_y,
    const TrajectoryRows &traj_z, const TrajectoryRows &anchor_x,
    const TrajectoryRows &anchor_y, const TrajectoryRows &anchor_z,
    std::size_t node_count, std::size_t total_frames,
    const PairPruningConfig &cfg, std::size_t *dense_pair_count_out) {
  struct RankedNeighbor {
    std::size_t partner{0};
    std::uint32_t score{0};
    float min_dist2{std::numeric_limits<float>::infinity()};
  };

  const std::size_t dense_pair_count =
      node_count < 2 ? 0 : node_count * (node_count - 1) / 2;
  if (dense_pair_count_out != nullptr)
    *dense_pair_count_out = dense_pair_count;
  constexpr std::size_t kSafeDensePairVectorLimit = 50'000'000;
  if (dense_pair_count > kSafeDensePairVectorLimit) {
    const double cutoff =
        cfg.pair_cutoff_ang > 0.0 ? cfg.pair_cutoff_ang : 30.0;
    const int topk = cfg.pair_topk > 0 ? cfg.pair_topk : 24;
    return build_frame0_spatial_sparse_pairs(traj_x, traj_y, traj_z,
                                             node_count, cutoff, topk);
  }
  if (cfg.pair_cutoff_ang <= 0.0 && cfg.pair_topk <= 0)
    return build_dense_upper_triangle_pairs(node_count);

  std::vector<std::pair<std::size_t, std::size_t>> pairs;
  std::vector<unsigned char> seen(dense_pair_count, 0);
  auto mark_pair = [&](std::size_t i, std::size_t j) {
    const std::size_t a = std::min(i, j);
    const std::size_t b = std::max(i, j);
    const std::size_t pair_index = upper_triangle_pair_index(a, b, node_count);
    if (seen[pair_index] != 0)
      return false;
    seen[pair_index] = 1;
    pairs.emplace_back(a, b);
    return true;
  };
  auto append_top_ranked_pairs =
      [&](const std::vector<std::uint32_t> &scores,
          const std::vector<float> &min_dist2, std::size_t keep_topk) {
        std::vector<std::vector<RankedNeighbor>> ranked_neighbors(node_count);
        for (auto &neighbors : ranked_neighbors)
          neighbors.reserve(keep_topk * 2);
        for (std::size_t i = 0; i < node_count; ++i) {
          for (std::size_t j = i + 1; j < node_count; ++j) {
            const std::size_t pair_index =
                upper_triangle_pair_index(i, j, node_count);
            if (scores[pair_index] == 0)
              continue;
            ranked_neighbors[i].push_back(
                RankedNeighbor{j, scores[pair_index], min_dist2[pair_index]});
            ranked_neighbors[j].push_back(
                RankedNeighbor{i, scores[pair_index], min_dist2[pair_index]});
          }
        }

        auto better_neighbor = [](const RankedNeighbor &lhs,
                                  const RankedNeighbor &rhs) {
          if (lhs.score != rhs.score)
            return lhs.score > rhs.score;
          if (lhs.min_dist2 != rhs.min_dist2)
            return lhs.min_dist2 < rhs.min_dist2;
          return lhs.partner < rhs.partner;
        };

        bool added_any = false;
        for (std::size_t i = 0; i < node_count; ++i) {
          auto &neighbors = ranked_neighbors[i];
          if (neighbors.empty())
            continue;
          if (neighbors.size() > keep_topk) {
            std::nth_element(
                neighbors.begin(),
                neighbors.begin() + static_cast<std::ptrdiff_t>(keep_topk),
                neighbors.end(), better_neighbor);
            std::sort(neighbors.begin(),
                      neighbors.begin() + static_cast<std::ptrdiff_t>(keep_topk),
                      better_neighbor);
          } else {
            std::sort(neighbors.begin(), neighbors.end(), better_neighbor);
          }
          const std::size_t keep = std::min(keep_topk, neighbors.size());
          for (std::size_t n = 0; n < keep; ++n) {
            added_any |= mark_pair(i, neighbors[n].partner);
          }
        }
        return added_any;
      };
  auto append_frame_voted_knn = [&](std::size_t keep_topk,
                                    std::size_t frame_limit) {
    if (!has_complete_rows(traj_x, traj_y, traj_z, node_count, frame_limit))
      return false;

    std::vector<std::uint32_t> vote_counts(dense_pair_count, 0);
    std::vector<float> min_dist2(dense_pair_count,
                                 std::numeric_limits<float>::infinity());
    std::vector<std::pair<float, std::size_t>> distances;
    distances.reserve(node_count > 0 ? node_count - 1 : 0);
    auto rank_cmp = [](const auto &lhs, const auto &rhs) {
      if (lhs.first != rhs.first)
        return lhs.first < rhs.first;
      return lhs.second < rhs.second;
    };

    for (std::size_t frame = 0; frame < frame_limit; ++frame) {
      const auto &x_row = traj_x[frame];
      const auto &y_row = traj_y[frame];
      const auto &z_row = traj_z[frame];

      for (std::size_t i = 0; i < node_count; ++i) {
        distances.clear();
        const double xi = x_row[i];
        const double yi = y_row[i];
        const double zi = z_row[i];
        for (std::size_t j = 0; j < node_count; ++j) {
          if (i == j)
            continue;
          const double dx = xi - x_row[j];
          const double dy = yi - y_row[j];
          const double dz = zi - z_row[j];
          distances.emplace_back(
              static_cast<float>(dx * dx + dy * dy + dz * dz), j);
        }

        if (distances.size() > keep_topk) {
          std::nth_element(
              distances.begin(),
              distances.begin() + static_cast<std::ptrdiff_t>(keep_topk),
              distances.end(), rank_cmp);
          std::sort(
              distances.begin(),
              distances.begin() + static_cast<std::ptrdiff_t>(keep_topk),
              rank_cmp);
        } else {
          std::sort(distances.begin(), distances.end(), rank_cmp);
        }

        for (std::size_t n = 0; n < keep_topk; ++n) {
          const std::size_t partner = distances[n].second;
          const std::size_t a = std::min(i, partner);
          const std::size_t b = std::max(i, partner);
          const std::size_t pair_index =
              upper_triangle_pair_index(a, b, node_count);
          if (vote_counts[pair_index] !=
              std::numeric_limits<std::uint32_t>::max()) {
            ++vote_counts[pair_index];
          }
          min_dist2[pair_index] =
              std::min(min_dist2[pair_index], distances[n].first);
        }
      }
    }

    return append_top_ranked_pairs(vote_counts, min_dist2, keep_topk);
  };
  auto append_contact_frequency_knn = [&](std::size_t keep_topk,
                                          std::size_t frame_limit) {
    if (!has_complete_rows(anchor_x, anchor_y, anchor_z, node_count,
                           frame_limit)) {
      return false;
    }

    const double cutoff = cfg.pair_contact_cutoff_ang;
    const double cutoff2 = cutoff * cutoff;
    titan_neighbor::HalfShellGrid neighbor_grid;
    std::vector<std::uint32_t> contact_counts(dense_pair_count, 0);
    std::vector<float> min_dist2(dense_pair_count,
                                 std::numeric_limits<float>::infinity());
    bool saw_contact = false;

    for (std::size_t frame = 0; frame < frame_limit; ++frame) {
      const auto &x_row = anchor_x[frame];
      const auto &y_row = anchor_y[frame];
      const auto &z_row = anchor_z[frame];
      const auto [xmin, xmax] =
          std::minmax_element(x_row.begin(), x_row.begin() + node_count);
      const auto [ymin, ymax] =
          std::minmax_element(y_row.begin(), y_row.begin() + node_count);
      const auto [zmin, zmax] =
          std::minmax_element(z_row.begin(), z_row.begin() + node_count);

      titan_neighbor::OrthorhombicGridSpec spec;
      spec.cutoff = cutoff;
      spec.shift_x = -(*xmin) + cutoff;
      spec.shift_y = -(*ymin) + cutoff;
      spec.shift_z = -(*zmin) + cutoff;
      spec.len_x = (*xmax - *xmin) + 2.0 * cutoff;
      spec.len_y = (*ymax - *ymin) + 2.0 * cutoff;
      spec.len_z = (*zmax - *zmin) + 2.0 * cutoff;
      spec.periodic = false;

      neighbor_grid.for_each_pair(
          node_count, spec,
          [&](std::size_t atom_index, double &x, double &y, double &z) {
            x = x_row[atom_index];
            y = y_row[atom_index];
            z = z_row[atom_index];
          },
          [&](std::size_t atom_index) { return atom_index < node_count; },
          [&](std::size_t lhs, std::size_t rhs) {
            const double dx = x_row[lhs] - x_row[rhs];
            const double dy = y_row[lhs] - y_row[rhs];
            const double dz = z_row[lhs] - z_row[rhs];
            const float dist2 =
                static_cast<float>(dx * dx + dy * dy + dz * dz);
            if (dist2 > cutoff2)
              return;

            saw_contact = true;
            const std::size_t i = std::min(lhs, rhs);
            const std::size_t j = std::max(lhs, rhs);
            const std::size_t pair_index =
                upper_triangle_pair_index(i, j, node_count);
            if (contact_counts[pair_index] !=
                std::numeric_limits<std::uint32_t>::max()) {
              ++contact_counts[pair_index];
            }
            min_dist2[pair_index] = std::min(min_dist2[pair_index], dist2);
          });
    }

    if (!saw_contact)
      return false;
    return append_top_ranked_pairs(contact_counts, min_dist2, keep_topk);
  };

  if (cfg.pair_cutoff_ang > 0.0 &&
      has_complete_rows(traj_x, traj_y, traj_z, node_count, total_frames)) {
    std::vector<unsigned char> pair_hits(dense_pair_count, 0);
    std::vector<float> pair_distances(
        dense_pair_count, std::numeric_limits<float>::infinity());
    const double cutoff2 = cfg.pair_cutoff_ang * cfg.pair_cutoff_ang;
    const std::size_t frame_limit = cfg.use_avg_cutoff ? total_frames : 1;
    for (std::size_t frame = 0; frame < frame_limit; ++frame) {
      const auto &x_row = traj_x[frame];
      const auto &y_row = traj_y[frame];
      const auto &z_row = traj_z[frame];
      for (std::size_t i = 0; i < node_count; ++i) {
        for (std::size_t j = i + 1; j < node_count; ++j) {
          const double dx = x_row[i] - x_row[j];
          const double dy = y_row[i] - y_row[j];
          const double dz = z_row[i] - z_row[j];
          const float dist2 =
              static_cast<float>(dx * dx + dy * dy + dz * dz);
          if (dist2 > cutoff2)
            continue;
          const std::size_t pair_index =
              upper_triangle_pair_index(i, j, node_count);
          pair_hits[pair_index] = 1;
          pair_distances[pair_index] = std::min(pair_distances[pair_index], dist2);
        }
      }
    }
    for (std::size_t i = 0; i < node_count; ++i) {
      for (std::size_t j = i + 1; j < node_count; ++j) {
        const std::size_t pair_index =
            upper_triangle_pair_index(i, j, node_count);
        if (pair_hits[pair_index] != 0)
          mark_pair(i, j);
      }
    }
  }

  if (cfg.pair_topk > 0 && node_count > 1 && total_frames > 0) {
    const std::size_t keep_topk =
        std::min<std::size_t>(static_cast<std::size_t>(cfg.pair_topk),
                              node_count - 1);
    const std::size_t frame_limit = cfg.use_avg_cutoff ? total_frames : 1;
    const bool can_rank =
        has_complete_rows(traj_x, traj_y, traj_z, node_count, frame_limit);
    if (!can_rank) {
      if (pairs.empty())
        return build_dense_upper_triangle_pairs(node_count);
      std::sort(pairs.begin(), pairs.end());
      return pairs;
    }
    bool added_knn_pairs = false;
    if (cfg.pair_use_contact_prior) {
      added_knn_pairs = append_contact_frequency_knn(keep_topk, frame_limit);
    }
    if (!added_knn_pairs) {
      added_knn_pairs = append_frame_voted_knn(keep_topk, frame_limit);
    }
    if (!added_knn_pairs && pairs.empty())
      return build_dense_upper_triangle_pairs(node_count);
  }

  std::sort(pairs.begin(), pairs.end());
  return pairs;
}

} // namespace titan_corr_knn
