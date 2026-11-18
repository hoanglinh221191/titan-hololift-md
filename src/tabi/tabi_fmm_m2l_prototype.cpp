#include "tabi/tabi_fmm_m2l_prototype.h"
#include "tabi/tabi_fmm_m2l_cartesian.h"
#include "tabi/tabi_fmm_m2l_fallback.h"
#include "tabi/tabi_fmm_m2l_local.h"
#include "tabi/tabi_near.h"
#include "tabi/tabi_node_patch_curved.h"
#include "tabi/tabi_types.h"

#include "aligned_allocator.h"
#include "titan_executor.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace titan::tabi::fmm_m2l {
namespace {

using Clock = std::chrono::steady_clock;

struct PointSoA {
  AlignedDoubleVector x;
  AlignedDoubleVector y;
  AlignedDoubleVector z;
  AlignedDoubleVector q;
  std::vector<std::size_t> panel_id;
};

struct Cell {
  std::size_t begin = 0;
  std::size_t end = 0;
  int left = -1;
  int right = -1;
  int depth = 0;
  double cx = 0.0;
  double cy = 0.0;
  double cz = 0.0;
  double radius = 0.0;
  double min_x = 0.0;
  double min_y = 0.0;
  double min_z = 0.0;
  double max_x = 0.0;
  double max_y = 0.0;
  double max_z = 0.0;
  double charge = 0.0;
  double qx = 0.0;
  double qy = 0.0;
  double qz = 0.0;
  double expansion_radius = 0.0;
  double center_offset = 0.0;
  double px = 0.0;
  double py = 0.0;
  double pz = 0.0;

  [[nodiscard]] bool is_leaf() const noexcept { return left < 0 && right < 0; }
  [[nodiscard]] std::size_t count() const noexcept { return end - begin; }
};

struct TargetLeafWork {
  int target_node = -1;
  std::vector<int> near_sources;
  std::vector<int> far_sources;
};

struct PrototypeEvaluation {
  PrototypeMetrics metrics;
  AlignedDoubleVector reference;
  AlignedDoubleVector approx;
};

struct Tree {
  std::vector<std::size_t> order;
  std::vector<std::size_t> panel_id;
  std::vector<Cell> cells;
  std::vector<int> target_leaf_nodes;
  std::vector<TargetLeafWork> work;
};

double elapsed_seconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double>(end - begin).count();
}

double kernel_value(KernelKind kernel, double kappa, double r2) {
  constexpr double kTiny = 1.0e-28;
  if (!(r2 > kTiny)) {
    return 0.0;
  }
  const double r = std::sqrt(r2);
  if (kernel == KernelKind::Yukawa) {
    return std::exp(-std::max(0.0, kappa) * r) / r;
  }
  return 1.0 / r;
}

PointSoA make_synthetic_points(std::size_t n, unsigned seed) {
  PointSoA points;
  points.x.resize(n);
  points.y.resize(n);
  points.z.resize(n);
  points.q.resize(n);
  points.panel_id.resize(n);

  std::uint64_t state =
      (static_cast<std::uint64_t>(seed) << 32) ^ 0x9e3779b97f4a7c15ull;
  auto next01 = [&]() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    const std::uint64_t bits = (state >> 11) | 0x3ff0000000000000ull;
    double value = 0.0;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    return value - 1.0;
  };

  for (std::size_t i = 0; i < n; ++i) {
    const double cluster = (i < n / 2) ? -1.0 : 1.0;
    const double shell = static_cast<double>((i % 17) + 1) / 18.0;
    const double theta = 6.2831853071795864769 * next01();
    const double phi = std::acos(std::clamp(2.0 * next01() - 1.0, -1.0, 1.0));
    const double jitter = 0.08 * (next01() - 0.5);
    const double r = 0.35 + 0.35 * shell + jitter;
    points.x[i] = cluster * 2.0 + r * std::sin(phi) * std::cos(theta);
    points.y[i] = 0.35 * cluster + r * std::sin(phi) * std::sin(theta);
    points.z[i] = r * std::cos(phi);
    points.q[i] = ((i % 2) == 0 ? 1.0 : -0.7) + 0.05 * (next01() - 0.5);
    points.panel_id[i] = i;
  }
  return points;
}

PointSoA from_public_point_cloud(const PrototypePointCloud &input) {
  const std::size_t n = input.x.size();
  if (input.y.size() != n || input.z.size() != n || input.q.size() != n) {
    throw std::invalid_argument(
        "FMM/M2L prototype point cloud has inconsistent coordinate sizes");
  }
  if (n == 0) {
    throw std::invalid_argument(
        "FMM/M2L prototype requires at least one point");
  }
  PointSoA points;
  points.x.assign(input.x.begin(), input.x.end());
  points.y.assign(input.y.begin(), input.y.end());
  points.z.assign(input.z.begin(), input.z.end());
  points.q.assign(input.q.begin(), input.q.end());
  if (input.panel_id.empty()) {
    points.panel_id.resize(n);
    std::iota(points.panel_id.begin(), points.panel_id.end(), std::size_t{0});
  } else {
    if (input.panel_id.size() != n) {
      throw std::invalid_argument(
          "FMM/M2L prototype point cloud has inconsistent panel ids");
    }
    points.panel_id = input.panel_id;
  }
  return points;
}

void compute_cell_geometry(Tree &tree, const PointSoA &points, int node_index) {
  Cell &cell = tree.cells[static_cast<std::size_t>(node_index)];
  double min_x = std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double min_z = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  double max_z = -std::numeric_limits<double>::infinity();
  double sx = 0.0;
  double sy = 0.0;
  double sz = 0.0;
  double q_sum = 0.0;
  double qx = 0.0;
  double qy = 0.0;
  double qz = 0.0;

  for (std::size_t pos = cell.begin; pos < cell.end; ++pos) {
    const std::size_t idx = tree.order[pos];
    const double x = points.x[idx];
    const double y = points.y[idx];
    const double z = points.z[idx];
    min_x = std::min(min_x, x);
    min_y = std::min(min_y, y);
    min_z = std::min(min_z, z);
    max_x = std::max(max_x, x);
    max_y = std::max(max_y, y);
    max_z = std::max(max_z, z);
    sx += x;
    sy += y;
    sz += z;
    q_sum += points.q[idx];
    qx += points.q[idx] * x;
    qy += points.q[idx] * y;
    qz += points.q[idx] * z;
  }

  const double inv_count =
      cell.count() > 0 ? 1.0 / static_cast<double>(cell.count()) : 0.0;
  cell.cx = sx * inv_count;
  cell.cy = sy * inv_count;
  cell.cz = sz * inv_count;
  cell.min_x = min_x;
  cell.min_y = min_y;
  cell.min_z = min_z;
  cell.max_x = max_x;
  cell.max_y = max_y;
  cell.max_z = max_z;
  cell.radius = 0.0;
  for (std::size_t pos = cell.begin; pos < cell.end; ++pos) {
    const std::size_t idx = tree.order[pos];
    const double dx = points.x[idx] - cell.cx;
    const double dy = points.y[idx] - cell.cy;
    const double dz = points.z[idx] - cell.cz;
    cell.radius = std::max(cell.radius, std::sqrt(dx * dx + dy * dy + dz * dz));
  }
  cell.charge = q_sum;
  cell.qx = 0.5 * (min_x + max_x);
  cell.qy = 0.5 * (min_y + max_y);
  cell.qz = 0.5 * (min_z + max_z);
  const double center_dx = cell.qx - cell.cx;
  const double center_dy = cell.qy - cell.cy;
  const double center_dz = cell.qz - cell.cz;
  cell.center_offset = std::sqrt(center_dx * center_dx + center_dy * center_dy +
                                 center_dz * center_dz);
  cell.expansion_radius = 0.0;
  for (std::size_t pos = cell.begin; pos < cell.end; ++pos) {
    const std::size_t idx = tree.order[pos];
    const double edx = points.x[idx] - cell.qx;
    const double edy = points.y[idx] - cell.qy;
    const double edz = points.z[idx] - cell.qz;
    cell.expansion_radius = std::max(
        cell.expansion_radius, std::sqrt(edx * edx + edy * edy + edz * edz));
    cell.px += points.q[idx] * (points.x[idx] - cell.qx);
    cell.py += points.q[idx] * (points.y[idx] - cell.qy);
    cell.pz += points.q[idx] * (points.z[idx] - cell.qz);
  }
  (void)qx;
  (void)qy;
  (void)qz;
}

int build_node(Tree &tree, const PointSoA &points, std::size_t begin,
               std::size_t end, std::size_t max_leaf, int depth) {
  const int node_index = static_cast<int>(tree.cells.size());
  Cell cell;
  cell.begin = begin;
  cell.end = end;
  cell.depth = depth;
  tree.cells.push_back(cell);
  compute_cell_geometry(tree, points, node_index);

  if (end - begin <= max_leaf) {
    tree.target_leaf_nodes.push_back(node_index);
    return node_index;
  }

  std::array<double, 3> min_v{std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::infinity()};
  std::array<double, 3> max_v{-std::numeric_limits<double>::infinity(),
                              -std::numeric_limits<double>::infinity(),
                              -std::numeric_limits<double>::infinity()};
  for (std::size_t pos = begin; pos < end; ++pos) {
    const std::size_t idx = tree.order[pos];
    min_v[0] = std::min(min_v[0], points.x[idx]);
    min_v[1] = std::min(min_v[1], points.y[idx]);
    min_v[2] = std::min(min_v[2], points.z[idx]);
    max_v[0] = std::max(max_v[0], points.x[idx]);
    max_v[1] = std::max(max_v[1], points.y[idx]);
    max_v[2] = std::max(max_v[2], points.z[idx]);
  }
  const std::array<double, 3> span{max_v[0] - min_v[0], max_v[1] - min_v[1],
                                   max_v[2] - min_v[2]};
  const int axis = (span[1] > span[0] && span[1] >= span[2])  ? 1
                   : (span[2] > span[0] && span[2] > span[1]) ? 2
                                                              : 0;
  const std::size_t mid = begin + (end - begin) / 2;
  auto coord = [&](std::size_t idx) {
    if (axis == 0) {
      return points.x[idx];
    }
    if (axis == 1) {
      return points.y[idx];
    }
    return points.z[idx];
  };
  std::stable_sort(tree.order.begin() + static_cast<std::ptrdiff_t>(begin),
                   tree.order.begin() + static_cast<std::ptrdiff_t>(end),
                   [&](std::size_t lhs, std::size_t rhs) {
                     const double cl = coord(lhs);
                     const double cr = coord(rhs);
                     if (cl != cr) {
                       return cl < cr;
                     }
                     return lhs < rhs;
                   });

  const int left = build_node(tree, points, begin, mid, max_leaf, depth + 1);
  const int right = build_node(tree, points, mid, end, max_leaf, depth + 1);
  tree.cells[static_cast<std::size_t>(node_index)].left = left;
  tree.cells[static_cast<std::size_t>(node_index)].right = right;
  return node_index;
}

Tree build_tree(const PointSoA &points, std::size_t max_leaf) {
  Tree tree;
  tree.order.resize(points.x.size());
  std::iota(tree.order.begin(), tree.order.end(), std::size_t{0});
  tree.panel_id = points.panel_id;
  tree.cells.reserve(points.x.size() / std::max<std::size_t>(1, max_leaf) * 2 +
                     8);
  build_node(tree, points, 0, points.x.size(),
             std::max<std::size_t>(1, max_leaf), 0);
  tree.work.resize(tree.cells.size());
  for (int node : tree.target_leaf_nodes) {
    tree.work[static_cast<std::size_t>(node)].target_node = node;
  }
  return tree;
}

bool accept_far(const Tree &target_tree, const Tree &source_tree,
                int target_node, int source_node, double theta,
                bool exclude_same_index) {
  if (!(theta > 0.0)) {
    return false;
  }
  const Cell &t = target_tree.cells[static_cast<std::size_t>(target_node)];
  const Cell &s = source_tree.cells[static_cast<std::size_t>(source_node)];
  if (exclude_same_index) {
    for (std::size_t tpos = t.begin; tpos < t.end; ++tpos) {
      const std::size_t ti = target_tree.order[tpos];
      for (std::size_t spos = s.begin; spos < s.end; ++spos) {
        const std::size_t si = source_tree.order[spos];
        if (target_tree.panel_id[ti] == source_tree.panel_id[si]) {
          return false;
        }
      }
    }
  }
  const double dx = t.cx - s.cx;
  const double dy = t.cy - s.cy;
  const double dz = t.cz - s.cz;
  const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
  if (!(dist > 1.0e-12)) {
    return false;
  }
  return (t.radius + s.radius) / dist <= theta;
}

double axis_gap(double lhs_min, double lhs_max, double rhs_min,
                double rhs_max) noexcept {
  if (lhs_max < rhs_min)
    return rhs_min - lhs_max;
  if (rhs_max < lhs_min)
    return lhs_min - rhs_max;
  return 0.0;
}

double aabb_gap(const Cell &target, const Cell &source) noexcept {
  const double dx =
      axis_gap(target.min_x, target.max_x, source.min_x, source.max_x);
  const double dy =
      axis_gap(target.min_y, target.max_y, source.min_y, source.max_y);
  const double dz =
      axis_gap(target.min_z, target.max_z, source.min_z, source.max_z);
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double strict_geometry_ratio(const Cell &target, const Cell &source) noexcept {
  const double gap = aabb_gap(target, source);
  const double effective_radius = target.radius + source.radius;
  return effective_radius / std::max(1.0e-12, gap);
}

void build_work(Tree &target_tree, const Tree &source_tree, int target_node,
                int source_node, double theta, bool exclude_same_index) {
  const Cell &target = target_tree.cells[static_cast<std::size_t>(target_node)];
  if (!target.is_leaf()) {
    build_work(target_tree, source_tree, target.left, source_node, theta,
               exclude_same_index);
    build_work(target_tree, source_tree, target.right, source_node, theta,
               exclude_same_index);
    return;
  }

  TargetLeafWork &work =
      target_tree.work[static_cast<std::size_t>(target_node)];
  const Cell &source = source_tree.cells[static_cast<std::size_t>(source_node)];
  if (accept_far(target_tree, source_tree, target_node, source_node, theta,
                 exclude_same_index)) {
    work.far_sources.push_back(source_node);
    return;
  }
  if (source.is_leaf()) {
    work.near_sources.push_back(source_node);
    return;
  }
  build_work(target_tree, source_tree, target_node, source.left, theta,
             exclude_same_index);
  build_work(target_tree, source_tree, target_node, source.right, theta,
             exclude_same_index);
}

void sort_work(Tree &tree) {
  for (int node : tree.target_leaf_nodes) {
    auto &work = tree.work[static_cast<std::size_t>(node)];
    std::sort(work.near_sources.begin(), work.near_sources.end());
    work.near_sources.erase(
        std::unique(work.near_sources.begin(), work.near_sources.end()),
        work.near_sources.end());
    std::sort(work.far_sources.begin(), work.far_sources.end());
    work.far_sources.erase(
        std::unique(work.far_sources.begin(), work.far_sources.end()),
        work.far_sources.end());
  }
}

std::string operator_key(const Tree &target_tree, const Tree &source_tree,
                         int target_node, int source_node,
                         const PrototypeConfig &config) {
  const Cell &t = target_tree.cells[static_cast<std::size_t>(target_node)];
  const Cell &s = source_tree.cells[static_cast<std::size_t>(source_node)];
  const double scale = std::max(1.0e-12, t.radius + s.radius);
  const auto bucket = [&](double value) {
    return static_cast<long long>(std::llround(value / scale * 16.0));
  };
  const long long ratio = static_cast<long long>(
      std::llround((s.radius / std::max(1.0e-12, t.radius)) * 8.0));
  return std::to_string(static_cast<int>(config.kernel)) + ":" +
         std::to_string(config.expansion_order) + ":" +
         std::to_string(bucket(s.cx - t.cx)) + ":" +
         std::to_string(bucket(s.cy - t.cy)) + ":" +
         std::to_string(bucket(s.cz - t.cz)) + ":" + std::to_string(ratio);
}

void compute_counts_and_operators(const Tree &target_tree,
                                  const Tree &source_tree,
                                  const PrototypeConfig &config,
                                  PrototypeMetrics &metrics) {
  std::set<std::string> keys;
  for (int tnode : target_tree.target_leaf_nodes) {
    const TargetLeafWork &work =
        target_tree.work[static_cast<std::size_t>(tnode)];
    const Cell &target = target_tree.cells[static_cast<std::size_t>(tnode)];
    for (int source_node : work.near_sources) {
      metrics.near_pairs +=
          target.count() *
          source_tree.cells[static_cast<std::size_t>(source_node)].count();
    }
    for (int source_node : work.far_sources) {
      metrics.far_pairs +=
          target.count() *
          source_tree.cells[static_cast<std::size_t>(source_node)].count();
      keys.insert(
          operator_key(target_tree, source_tree, tnode, source_node, config));
    }
  }
  metrics.m2l_operators = keys.size();
}

void apply_direct_reference(const PointSoA &sources, const PointSoA &targets,
                            const PrototypeConfig &config,
                            AlignedDoubleVector &out,
                            bool exclude_same_index = false) {
  out.assign(targets.x.size(), 0.0);
  for (std::size_t i = 0; i < targets.x.size(); ++i) {
    double sum = 0.0;
    for (std::size_t j = 0; j < sources.x.size(); ++j) {
      if (exclude_same_index && targets.panel_id[i] == sources.panel_id[j]) {
        continue;
      }
      const double dx = targets.x[i] - sources.x[j];
      const double dy = targets.y[i] - sources.y[j];
      const double dz = targets.z[i] - sources.z[j];
      const double r2 = dx * dx + dy * dy + dz * dz;
      sum += sources.q[j] * kernel_value(config.kernel, config.kappa, r2);
    }
    out[i] = sum;
  }
}

void apply_prototype(const PointSoA &sources, const Tree &source_tree,
                     const PointSoA &targets, const Tree &target_tree,
                     const PrototypeConfig &config, AlignedDoubleVector &out,
                     bool exclude_same_index = false) {
  out.assign(targets.x.size(), 0.0);
  TitanExecutor executor(static_cast<std::size_t>(std::max(1, config.threads)));
  const std::size_t workers = std::max<std::size_t>(1, executor.capacity());
  executor.parallel_for(
      target_tree.target_leaf_nodes.size(), workers, 1,
      [&](std::size_t, std::size_t begin, std::size_t end) {
        for (std::size_t leaf_pos = begin; leaf_pos < end; ++leaf_pos) {
          const int target_node = target_tree.target_leaf_nodes[leaf_pos];
          const Cell &target =
              target_tree.cells[static_cast<std::size_t>(target_node)];
          const TargetLeafWork &work =
              target_tree.work[static_cast<std::size_t>(target_node)];
          for (std::size_t tpos = target.begin; tpos < target.end; ++tpos) {
            const std::size_t ti = target_tree.order[tpos];
            double sum = 0.0;
            for (int source_node : work.near_sources) {
              const Cell &source =
                  source_tree.cells[static_cast<std::size_t>(source_node)];
              for (std::size_t spos = source.begin; spos < source.end; ++spos) {
                const std::size_t si = source_tree.order[spos];
                if (exclude_same_index &&
                    targets.panel_id[ti] == sources.panel_id[si]) {
                  continue;
                }
                const double dx = targets.x[ti] - sources.x[si];
                const double dy = targets.y[ti] - sources.y[si];
                const double dz = targets.z[ti] - sources.z[si];
                const double r2 = dx * dx + dy * dy + dz * dz;
                sum += sources.q[si] *
                       kernel_value(config.kernel, config.kappa, r2);
              }
            }
            for (int source_node : work.far_sources) {
              const Cell &source =
                  source_tree.cells[static_cast<std::size_t>(source_node)];
              const double dx = targets.x[ti] - source.qx;
              const double dy = targets.y[ti] - source.qy;
              const double dz = targets.z[ti] - source.qz;
              const double r2 = dx * dx + dy * dy + dz * dz;
              const double g = kernel_value(config.kernel, config.kappa, r2);
              double far = source.charge * g;
              if (config.expansion_order >= 2 && r2 > 1.0e-28) {
                const double r = std::sqrt(r2);
                const double dot =
                    source.px * dx + source.py * dy + source.pz * dz;
                const double dipole_scale =
                    (config.kernel == KernelKind::Yukawa)
                        ? std::exp(-std::max(0.0, config.kappa) * r) *
                              (std::max(0.0, config.kappa) / (r * r) +
                               1.0 / (r * r * r))
                        : 1.0 / (r * r * r);
                far += dot * dipole_scale;
              }
              sum += far;
            }
            out[ti] = sum;
          }
        }
      });
}

void compute_error(const AlignedDoubleVector &reference,
                   const AlignedDoubleVector &approx,
                   PrototypeMetrics &metrics) {
  double err2 = 0.0;
  double ref2 = 0.0;
  double max_abs = 0.0;
  bool finite = true;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double err = approx[i] - reference[i];
    err2 += err * err;
    ref2 += reference[i] * reference[i];
    max_abs = std::max(max_abs, std::abs(err));
    finite = finite && std::isfinite(reference[i]) && std::isfinite(approx[i]);
  }
  const double inv =
      reference.empty() ? 0.0 : 1.0 / static_cast<double>(reference.size());
  metrics.rms_abs_error = std::sqrt(err2 * inv);
  metrics.max_abs_error = max_abs;
  metrics.rel_rms_error = std::sqrt(err2 / std::max(ref2, 1.0e-30));
  metrics.finite_outputs = finite;
}

bool finite_vector(const AlignedDoubleVector &values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

// The degree-one (p1) moments of a source cell about its centre q: of the
// single-layer density A dphi, the monopole and dipole; of the double-layer
// density A phi n, the vector monopole and its first moment
// phi_n_p[3 b + c] = sum A phi n_b (y_c - q_c), which K2 and K4 need for
// degree one as the dipole serves K1 and K3.
struct PanelSourceMoments {
  double q_phin = 0.0;
  double phin_px = 0.0;
  double phin_py = 0.0;
  double phin_pz = 0.0;
  double phi_nx = 0.0;
  double phi_ny = 0.0;
  double phi_nz = 0.0;
  std::array<double, 9> phi_n_p{};
};

using JcpChannelValues = std::array<double, 4>;

void add_channels(JcpChannelValues &target,
                  const JcpChannelValues &source) noexcept {
  for (std::size_t channel = 0; channel < target.size(); ++channel)
    target[channel] += source[channel];
}

double channel_phi(const JcpChannelValues &channels) noexcept {
  return channels[0] + channels[1];
}

double channel_phin(const JcpChannelValues &channels) noexcept {
  return channels[2] + channels[3];
}

double jcp_phin_for_panel(const PrototypePanel &panel, std::size_t index) {
  const double t = static_cast<double>(index + 1);
  return 0.65 * std::sin(0.017 * t) + 0.35 * panel.nz;
}

PointSoA make_panel_geometry_points(const std::vector<PrototypePanel> &panels) {
  PointSoA points;
  points.x.reserve(panels.size());
  points.y.reserve(panels.size());
  points.z.reserve(panels.size());
  points.q.assign(panels.size(), 1.0);
  points.panel_id.reserve(panels.size());
  for (std::size_t i = 0; i < panels.size(); ++i) {
    points.x.push_back(panels[i].x);
    points.y.push_back(panels[i].y);
    points.z.push_back(panels[i].z);
    points.panel_id.push_back(i);
  }
  return points;
}

double source_x(const PrototypePanel &panel) noexcept {
  return panel.has_source_quadrature_center ? panel.source_x : panel.x;
}

double source_y(const PrototypePanel &panel) noexcept {
  return panel.has_source_quadrature_center ? panel.source_y : panel.y;
}

double source_z(const PrototypePanel &panel) noexcept {
  return panel.has_source_quadrature_center ? panel.source_z : panel.z;
}

double source_nx(const PrototypePanel &panel) noexcept {
  return panel.has_source_quadrature_center ? panel.source_nx : panel.nx;
}

double source_ny(const PrototypePanel &panel) noexcept {
  return panel.has_source_quadrature_center ? panel.source_ny : panel.ny;
}

double source_nz(const PrototypePanel &panel) noexcept {
  return panel.has_source_quadrature_center ? panel.source_nz : panel.nz;
}

PointSoA make_panel_source_points(const std::vector<PrototypePanel> &panels) {
  PointSoA points;
  points.x.reserve(panels.size());
  points.y.reserve(panels.size());
  points.z.reserve(panels.size());
  points.q.assign(panels.size(), 1.0);
  points.panel_id.reserve(panels.size());
  for (std::size_t i = 0; i < panels.size(); ++i) {
    points.x.push_back(source_x(panels[i]));
    points.y.push_back(source_y(panels[i]));
    points.z.push_back(source_z(panels[i]));
    points.panel_id.push_back(i);
  }
  return points;
}

void compute_panel_source_moments_into(
    const Tree &tree, const std::vector<PrototypePanel> &panels,
    const std::vector<double> &phi_values,
    const std::vector<double> &phin_values,
    std::vector<PanelSourceMoments> &moments) {
  moments.assign(tree.cells.size(), PanelSourceMoments{});
  // The moments of a cell summed over its own sources.
  const auto from_sources = [&](const Cell &cell) {
    PanelSourceMoments acc;
    for (std::size_t pos = cell.begin; pos < cell.end; ++pos) {
      const std::size_t source_index = tree.order[pos];
      const PrototypePanel &source = panels[source_index];
      const std::array<double, 3> offset{source_x(source) - cell.qx,
                                         source_y(source) - cell.qy,
                                         source_z(source) - cell.qz};
      const double q_phin = source.area * phin_values[source_index];
      acc.q_phin += q_phin;
      acc.phin_px += q_phin * offset[0];
      acc.phin_py += q_phin * offset[1];
      acc.phin_pz += q_phin * offset[2];
      const double q_phi = source.area * phi_values[source_index];
      const std::array<double, 3> qn{q_phi * source_nx(source),
                                     q_phi * source_ny(source),
                                     q_phi * source_nz(source)};
      acc.phi_nx += qn[0];
      acc.phi_ny += qn[1];
      acc.phi_nz += qn[2];
      for (int b = 0; b < 3; ++b)
        for (int c = 0; c < 3; ++c)
          acc.phi_n_p[3 * b + c] += qn[b] * offset[c];
    }
    return acc;
  };
  for (std::size_t reverse = tree.cells.size(); reverse > 0; --reverse) {
    const std::size_t node_index = reverse - 1u;
    const Cell &cell = tree.cells[node_index];
    PanelSourceMoments acc;
    if (cell.is_leaf()) {
      acc = from_sources(cell);
    } else {
      const std::array<int, 2> children{cell.left, cell.right};
      bool valid_children = true;
      for (const int child_index : children) {
        if (child_index < 0 ||
            static_cast<std::size_t>(child_index) >= moments.size()) {
          valid_children = false;
          break;
        }
        const std::size_t child = static_cast<std::size_t>(child_index);
        const Cell &child_cell = tree.cells[child];
        const PanelSourceMoments &source = moments[child];
        acc.q_phin += source.q_phin;
        acc.phin_px += source.phin_px +
                       source.q_phin * (child_cell.qx - cell.qx);
        acc.phin_py += source.phin_py +
                       source.q_phin * (child_cell.qy - cell.qy);
        acc.phin_pz += source.phin_pz +
                       source.q_phin * (child_cell.qz - cell.qz);
        acc.phi_nx += source.phi_nx;
        acc.phi_ny += source.phi_ny;
        acc.phi_nz += source.phi_nz;
        // The first moment shifted from the child's centre to this one.
        const std::array<double, 3> qn{source.phi_nx, source.phi_ny,
                                       source.phi_nz};
        const std::array<double, 3> shift{child_cell.qx - cell.qx,
                                          child_cell.qy - cell.qy,
                                          child_cell.qz - cell.qz};
        for (int b = 0; b < 3; ++b)
          for (int c = 0; c < 3; ++c)
            acc.phi_n_p[3 * b + c] +=
                source.phi_n_p[3 * b + c] + qn[b] * shift[c];
      }
      if (!valid_children)
        acc = from_sources(cell);
    }
    moments[node_index] = acc;
  }
}

std::vector<PanelSourceMoments>
compute_panel_source_moments(const Tree &tree,
                             const std::vector<PrototypePanel> &panels,
                             const std::vector<double> &phi_values,
                             const std::vector<double> &phin_values) {
  std::vector<PanelSourceMoments> moments;
  compute_panel_source_moments_into(tree, panels, phi_values, phin_values,
                                    moments);
  return moments;
}

JcpChannelValues jcp_direct_pair_channels(const PrototypePanel &target,
                                          const PrototypePanel &source,
                                          double source_phi, double source_phin,
                                          double eps_ratio, double kappa) {
  JcpChannelValues channels{};
  constexpr double inv_four_pi = 0.079577471545947667884;
  const double safe_eps_ratio = std::max(1.0e-12, eps_ratio);
  const double inv_eps_ratio = 1.0 / safe_eps_ratio;
  const double kk = std::max(0.0, kappa);
  const double ux = target.x - source_x(source);
  const double uy = target.y - source_y(source);
  const double uz = target.z - source_z(source);
  const double r2 = ux * ux + uy * uy + uz * uz;
  if (!(r2 > 1.0e-28)) {
    return channels;
  }
  const double r = std::sqrt(r2);
  const double inv_r = 1.0 / r;
  const double inv_r2 = inv_r * inv_r;
  const double inv_r3 = inv_r2 * inv_r;
  const double expkr = std::exp(-kk * r);
  const double one_plus_kr = 1.0 + kk * r;
  const double snx = source_nx(source), sny = source_ny(source),
               snz = source_nz(source);
  const double source_projection = snx * ux + sny * uy + snz * uz;
  const double target_projection =
      target.nx * ux + target.ny * uy + target.nz * uz;
  const double phin = source_phin;
  const double g0 = inv_four_pi * inv_r;
  const double gk = expkr * g0;
  const double source_d0 = inv_four_pi * source_projection * inv_r3;
  const double source_dk = expkr * one_plus_kr * source_d0;
  const double target_d0 = -inv_four_pi * target_projection * inv_r3;
  const double target_dk = expkr * one_plus_kr * target_d0;
  const double nxu = target_projection * inv_r;
  const double nyu = source_projection * inv_r;
  const double nxny = target.nx * snx + target.ny * sny + target.nz * snz;
  const double nxu_nyu = nxu * nyu;
  const double tangent_metric = nxny - nxu_nyu;
  const double mixed0 = inv_four_pi * inv_r3 * (nxny - 3.0 * nxu_nyu);
  const double fprime_k_over_r = -expkr * one_plus_kr * inv_four_pi * inv_r3;
  const double fsecond_k = expkr * inv_four_pi *
                           (kk * kk * inv_r + 2.0 * kk * inv_r2 + 2.0 * inv_r3);
  const double mixedk = -fsecond_k * nxu_nyu - fprime_k_over_r * tangent_metric;
  const double phi = source_phi;
  channels[0] = source.area * (g0 - gk) * phin;
  channels[1] = source.area * (safe_eps_ratio * source_dk - source_d0) * phi;
  channels[2] = source.area * (target_d0 - inv_eps_ratio * target_dk) * phin;
  channels[3] = source.area * (mixedk - mixed0) * phi;
  return channels;
}

void add_jcp_direct_pair(const PrototypePanel &target,
                         const PrototypePanel &source, double source_phi,
                         double source_phin, double eps_ratio, double kappa,
                         double &phi_out, double &phin_out) {
  const JcpChannelValues channels = jcp_direct_pair_channels(
      target, source, source_phi, source_phin, eps_ratio, kappa);
  phi_out += channel_phi(channels);
  phin_out += channel_phin(channels);
}

JcpChannelValues jcp_near_kernel_pair_channels(
    const TitanPBE::Tabi::Panel &target, std::size_t target_index,
    const TitanPBE::Tabi::Panel &source, std::size_t source_index,
    double source_phi, double source_phin, double eps_ratio, double kappa,
    bool coupled, const std::string &near_integration,
    const std::string &quad_buffer, TitanPBE::Tabi::TreecodeStats *stats) {
  JcpChannelValues channels{};
  const TitanPBE::Tabi::JcpNearPairKernel selected =
      TitanPBE::Tabi::evaluate_jcp_near_pair_kernel(
          target, target_index, source, source_index, eps_ratio, kappa, coupled,
          near_integration, quad_buffer, stats);
  if (!selected.valid)
    return channels;
  channels[0] = selected.k1 * source_phin;
  channels[1] = selected.k2 * source_phi;
  if (coupled) {
    channels[2] = selected.k3 * source_phin;
    channels[3] = selected.k4 * source_phi;
  }
  return channels;
}

void add_jcp_near_kernel_pair(
    const TitanPBE::Tabi::Panel &target, std::size_t target_index,
    const TitanPBE::Tabi::Panel &source, std::size_t source_index,
    double source_phi, double source_phin, double eps_ratio, double kappa,
    bool coupled, const std::string &near_integration,
    const std::string &quad_buffer, double &phi_out, double &phin_out,
    TitanPBE::Tabi::TreecodeStats *stats) {
  const JcpChannelValues channels = jcp_near_kernel_pair_channels(
      target, target_index, source, source_index, source_phi, source_phin,
      eps_ratio, kappa, coupled, near_integration, quad_buffer, stats);
  phi_out += channel_phi(channels);
  phin_out += channel_phin(channels);
}

void add_jcp_direct_pair(const PrototypePanel &target,
                         const PrototypePanel &source, double source_phin,
                         double eps_ratio, double kappa, double &phi_out,
                         double &phin_out) {
  add_jcp_direct_pair(target, source, source.scalar_density, source_phin,
                      eps_ratio, kappa, phi_out, phin_out);
}

JcpChannelValues jcp_far_cell_channels(const PrototypePanel &target,
                                       const Cell &source_cell,
                                       const PanelSourceMoments &moments,
                                       double eps_ratio, double kappa) {
  JcpChannelValues channels{};
  constexpr double inv_four_pi = 0.079577471545947667884;
  const double safe_eps_ratio = std::max(1.0e-12, eps_ratio);
  const double inv_eps_ratio = 1.0 / safe_eps_ratio;
  const double kk = std::max(0.0, kappa);
  const double ux = target.x - source_cell.qx;
  const double uy = target.y - source_cell.qy;
  const double uz = target.z - source_cell.qz;
  const double r2 = ux * ux + uy * uy + uz * uz;
  if (!(r2 > 1.0e-28)) {
    return channels;
  }
  const double r = std::sqrt(r2);
  const double inv_r = 1.0 / r;
  const double inv_r2 = inv_r * inv_r;
  const double inv_r3 = inv_r2 * inv_r;
  const double expkr = std::exp(-kk * r);
  const double one_plus_kr = 1.0 + kk * r;
  const double g0 = inv_four_pi * inv_r;
  const double gk = expkr * g0;
  const double dipole_dot =
      moments.phin_px * ux + moments.phin_py * uy + moments.phin_pz * uz;
  const double lap_dipole_scale = inv_four_pi * inv_r3;
  const double yuk_dipole_scale = expkr * inv_four_pi * (kk * inv_r2 + inv_r3);
  const double k1 = moments.q_phin * (g0 - gk) +
                    dipole_dot * (lap_dipole_scale - yuk_dipole_scale);

  const double source_normal_projection =
      moments.phi_nx * ux + moments.phi_ny * uy + moments.phi_nz * uz;
  const double source_d0 = inv_four_pi * source_normal_projection * inv_r3;
  const double source_dk = expkr * one_plus_kr * source_d0;
  double k2 = safe_eps_ratio * source_dk - source_d0;

  const double target_projection =
      target.nx * ux + target.ny * uy + target.nz * uz;
  const double target_d0 =
      -inv_four_pi * target_projection * inv_r3 * moments.q_phin;
  const double target_dk = expkr * one_plus_kr * target_d0;
  double k3 = target_d0 - inv_eps_ratio * target_dk;

  const double nxu = target_projection * inv_r;
  const double nyu = source_normal_projection * inv_r;
  const double nxny = target.nx * moments.phi_nx + target.ny * moments.phi_ny +
                      target.nz * moments.phi_nz;
  const double nxu_nyu = nxu * nyu;
  const double tangent_metric = nxny - nxu_nyu;
  const double mixed0 = inv_four_pi * inv_r3 * (nxny - 3.0 * nxu_nyu);
  const double fprime_k_over_r = -expkr * one_plus_kr * inv_four_pi * inv_r3;
  const double fsecond_k = expkr * inv_four_pi *
                           (kk * kk * inv_r + 2.0 * kk * inv_r2 + 2.0 * inv_r3);
  const double mixedk = -fsecond_k * nxu_nyu - fprime_k_over_r * tangent_metric;
  double k4 = mixedk - mixed0;

  // Degree one for K2, K3 and K4, as the dipole above gives K1.  With
  // u = x - q, a source at q + d and a radial f, the Taylor term of
  // f(|u - d|) is -d . grad f, and
  //   d_a d_b f = f'' u_a u_b + (f'/r)(delta_ab - u_a u_b)          (u unit)
  //   d_a d_b d_c f = A3 u_a u_b u_c + B3 (delta_ab u_c + delta_ac u_b
  //                   + delta_bc u_a),  A3 = f''' - 3 f''/r + 3 f'/r^2,
  //                   B3 = f''/r - f'/r^2.
  // K2 = n_y . (-grad H), H = eps Gk - G0: term  M : grad grad H.
  // K3 = n_x . grad P, P = G0 - Gk/eps: term  -n_x . grad grad P . p.
  // K4 = n_x . grad grad (G0 - Gk) . n_y: term
  //      -sum n_x,a M_bc d_a d_b d_c (G0 - Gk).
  // M = phi_n_p (sum A phi n_b d_c), p = (phin_px, phin_py, phin_pz).
  {
    const double hx = ux * inv_r, hy = uy * inv_r, hz = uz * inv_r;
    const std::array<double, 3> hat{hx, hy, hz};
    const std::array<double, 3> nx{target.nx, target.ny, target.nz};
    const auto &m = moments.phi_n_p;
    double umu = 0.0, nmu = 0.0, umn = 0.0;
    for (int b = 0; b < 3; ++b)
      for (int c = 0; c < 3; ++c) {
        umu += hat[b] * m[3 * b + c] * hat[c];
        nmu += nx[b] * m[3 * b + c] * hat[c];
        umn += hat[b] * m[3 * b + c] * nx[c];
      }
    const double trace = m[0] + m[4] + m[8];
    const double hess0 = inv_four_pi * inv_r3; // G0: f'' = 2 h, f'/r = -h
    k2 += safe_eps_ratio *
              (fsecond_k * umu + fprime_k_over_r * (trace - umu)) -
          hess0 * (3.0 * umu - trace);

    const double np = target.nx * moments.phin_px +
                      target.ny * moments.phin_py + target.nz * moments.phin_pz;
    const double pu = (moments.phin_px * ux + moments.phin_py * uy +
                       moments.phin_pz * uz) *
                      inv_r;
    const double hess_p0 = hess0 * (3.0 * nxu * pu - np);
    const double hess_pk =
        fsecond_k * nxu * pu + fprime_k_over_r * (np - nxu * pu);
    k3 -= hess_p0 - inv_eps_ratio * hess_pk;

    const double inv_r4 = inv_r2 * inv_r2;
    const double kr = kk * r;
    const double a3_0 = -15.0 * inv_four_pi * inv_r4;
    const double b3_0 = 3.0 * inv_four_pi * inv_r4;
    const double a3_k = -expkr * inv_four_pi * inv_r4 *
                        (((kr + 6.0) * kr + 15.0) * kr + 15.0);
    const double b3_k = expkr * inv_four_pi * inv_r4 * ((kr + 3.0) * kr + 3.0);
    const double t_term = nxu * umu;
    const double s_term = nmu + umn + nxu * trace;
    k4 += (a3_k - a3_0) * t_term + (b3_k - b3_0) * s_term;
  }

  channels[0] = k1;
  channels[1] = k2;
  channels[2] = k3;
  channels[3] = k4;
  return channels;
}

void add_jcp_far_cell(const PrototypePanel &target, const Cell &source_cell,
                      const PanelSourceMoments &moments, double eps_ratio,
                      double kappa, double &phi_out, double &phin_out) {
  const JcpChannelValues channels =
      jcp_far_cell_channels(target, source_cell, moments, eps_ratio, kappa);
  phi_out += channel_phi(channels);
  phin_out += channel_phin(channels);
}

} // namespace

namespace {

PrototypeConfig normalize_jcp_analytic_config(PrototypeConfig config) {
  config.max_leaf_size = std::max(1, config.max_leaf_size);
  config.threads = std::max(1, config.threads);
  return config;
}

bool same_double(double a, double b) noexcept {
  return std::abs(a - b) <= 1.0e-12 * std::max({1.0, std::abs(a), std::abs(b)});
}

void build_cartesian_node_moments_into(
    const Tree &tree, const std::vector<PrototypePanel> &panels,
    const std::vector<double> &phi, const std::vector<double> &phin, int order,
    TitanExecutor &executor, std::vector<CartesianJcpMoments> &moments) {
  moments.resize(tree.cells.size());
  const std::size_t workers = std::max<std::size_t>(1, executor.capacity());
  executor.parallel_for(
      tree.cells.size(), workers, 1,
      [&](std::size_t, std::size_t begin, std::size_t end) {
        for (std::size_t node_index = begin; node_index < end; ++node_index) {
          const Cell &cell = tree.cells[node_index];
          const std::span<const std::size_t> indices(
              tree.order.data() + cell.begin, cell.count());
          moments[node_index] = build_cartesian_jcp_moments(
              panels, phi, phin, indices, cell.qx, cell.qy, cell.qz, order);
        }
      });
}

std::vector<CartesianJcpMoments> build_cartesian_node_moments(
    const Tree &tree, const std::vector<PrototypePanel> &panels,
    const std::vector<double> &phi, const std::vector<double> &phin, int order,
    TitanExecutor &executor) {
  std::vector<CartesianJcpMoments> moments;
  build_cartesian_node_moments_into(tree, panels, phi, phin, order, executor,
                                    moments);
  return moments;
}

std::uint64_t rademacher_hash(std::size_t panel_id,
                              std::uint64_t seed) noexcept {
  std::uint64_t value =
      static_cast<std::uint64_t>(panel_id) + 0x9e3779b97f4a7c15ULL + seed;
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

// A packed near loop evaluates every listed non-self pair with the one-point
// centroid kernel.  For node-patch sources the near kernel integrates some
// pairs over the patch instead - the one-ring (7ba6ee23), and on the FMM route
// the curved near zone (geometric_buffer) - so those pairs need K_patch -
// K_centroid added back, or the accelerated operator is not the one the dense
// audit rows and the ILU blocks apply: production and the Cartesian full
// route once disagreed by 81% on barstar with zero far pairs
// (docs/TITAN_NODE_PATCH_ROOT_CAUSE_2026-09-16.md).  A near worklist is
// immutable across Krylov applies, so the deltas are built once with the
// packed plan and only contracted with the densities afterwards.
struct NodePatchNearCorrection {
  std::uint32_t target{0};
  std::uint32_t source{0};
  double dk1{0.0};
  double dk2{0.0};
  double dk3{0.0};
  double dk4{0.0};
};

// The curved values can take adaptive quadrature, so leaves are visited in
// parallel and their corrections joined in leaf order: the list, and so every
// apply, does not depend on the thread count.  Uncoupled runs carry only K2.
// The pairs that can need a correction are known from the geometry: the
// one-ring, and for a curved cell the near-zone targets stored with each
// source cell.  Visiting only those, instead of every pair of a near list
// (about 3,300 per row on a 21k-node protein), keeps the list and its order -
// span by span, ascending position - exactly what the full scan produced.
// Returns false when panels are not node patches in node order.
bool node_patch_correction_candidates(
    const std::vector<TitanPBE::Tabi::Panel> &np,
    std::vector<std::size_t> &offsets, std::vector<std::size_t> &sources) {
  if (np.empty() || np.front().node_patch_geometry == nullptr)
    return false;
  const auto &geometry = *np.front().node_patch_geometry;
  const std::size_t n = np.size();
  if (geometry.adjacent_offsets.size() != n + 1)
    return false;
  for (std::size_t k = 0; k < n; ++k)
    if (np[k].node_patch_index != k ||
        np[k].node_patch_geometry.get() != &geometry)
      return false;
  const auto *curved = geometry.curved.get();
  if (curved != nullptr && curved->cells.size() != n)
    return false;
  offsets.assign(n + 1, 0);
  // Symmetric one-ring, plus each curved source's near-zone targets.
  for (std::size_t t = 0; t < n; ++t)
    offsets[t + 1] += geometry.adjacent_offsets[t + 1] - geometry.adjacent_offsets[t];
  if (curved != nullptr)
    for (std::size_t j = 0; j < n; ++j)
      for (const std::size_t t : curved->cells[j].near_targets)
        if (t != j)
          ++offsets[t + 1];
  for (std::size_t t = 0; t < n; ++t)
    offsets[t + 1] += offsets[t];
  sources.assign(offsets[n], 0);
  std::vector<std::size_t> fill(offsets.begin(), offsets.end() - 1);
  for (std::size_t t = 0; t < n; ++t)
    for (std::size_t p = geometry.adjacent_offsets[t];
         p < geometry.adjacent_offsets[t + 1]; ++p)
      sources[fill[t]++] = geometry.adjacent_nodes[p];
  if (curved != nullptr)
    for (std::size_t j = 0; j < n; ++j)
      for (const std::size_t t : curved->cells[j].near_targets)
        if (t != j)
          sources[fill[t]++] = j;
  // A one-ring neighbour of a curved cell is also among its near targets.
  std::vector<std::size_t> unique_offsets(n + 1, 0), unique_sources;
  unique_sources.reserve(sources.size());
  for (std::size_t t = 0; t < n; ++t) {
    auto first = sources.begin() + static_cast<std::ptrdiff_t>(offsets[t]);
    auto last = sources.begin() + static_cast<std::ptrdiff_t>(offsets[t + 1]);
    std::sort(first, last);
    unique_sources.insert(unique_sources.end(), first, std::unique(first, last));
    unique_offsets[t + 1] = unique_sources.size();
  }
  offsets.swap(unique_offsets);
  sources.swap(unique_sources);
  return true;
}

// Clipped sphere-patch triangles: the near-pair dispatcher integrates the self
// pair and every pair sharing a vertex over the curved patches
// (sphere_coupled_near), and every other pair with the one-point rule.  The
// candidates are therefore the vertex-sharing pairs, as shared_vertex_count()
// counts them.
bool sphere_patch_correction_candidates(
    const std::vector<TitanPBE::Tabi::Panel> &np,
    std::vector<std::size_t> &offsets, std::vector<std::size_t> &sources) {
  constexpr std::size_t none = static_cast<std::size_t>(-1);
  if (np.empty() || np.front().sphere_patch_quadrature == nullptr)
    return false;
  const std::size_t n = np.size();
  std::size_t vertex_count = 0;
  for (const auto &p : np) {
    if (!p.has_triangle_geometry || p.sphere_patch_quadrature == nullptr)
      return false;
    for (const std::size_t v : {p.v0_index, p.v1_index, p.v2_index})
      if (v != none)
        vertex_count = std::max(vertex_count, v + 1);
  }
  std::vector<std::size_t> vertex_offsets(vertex_count + 1, 0);
  for (const auto &p : np)
    for (const std::size_t v : {p.v0_index, p.v1_index, p.v2_index})
      if (v != none)
        ++vertex_offsets[v + 1];
  for (std::size_t v = 0; v < vertex_count; ++v)
    vertex_offsets[v + 1] += vertex_offsets[v];
  std::vector<std::size_t> vertex_panels(vertex_offsets[vertex_count]);
  std::vector<std::size_t> fill(vertex_offsets.begin(), vertex_offsets.end() - 1);
  for (std::size_t k = 0; k < n; ++k)
    for (const std::size_t v : {np[k].v0_index, np[k].v1_index, np[k].v2_index})
      if (v != none)
        vertex_panels[fill[v]++] = k;
  offsets.assign(n + 1, 0);
  sources.clear();
  std::vector<std::size_t> row;
  for (std::size_t t = 0; t < n; ++t) {
    row.clear();
    for (const std::size_t v : {np[t].v0_index, np[t].v1_index, np[t].v2_index}) {
      if (v == none)
        continue;
      for (std::size_t q = vertex_offsets[v]; q < vertex_offsets[v + 1]; ++q)
        if (vertex_panels[q] != t)
          row.push_back(vertex_panels[q]);
    }
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
    sources.insert(sources.end(), row.begin(), row.end());
    offsets[t + 1] = sources.size();
  }
  return true;
}

// near_spans_too: the packed near loop sums the one-point kernel over the
// near lists too, so it needs the corrections of the near-list pairs as well
// as of the far-classified ones; a scalar near loop evaluates near-list pairs
// with their curved value and needs only the far-classified ones.
std::vector<NodePatchNearCorrection> build_node_patch_near_corrections(
    const std::vector<PackedFallbackLeafInput> &leaves,
    const std::vector<std::size_t> &source_order,
    const std::vector<TitanPBE::Tabi::Panel> &np, double eps_ratio,
    double kappa, bool coupled, TitanExecutor &executor,
    bool near_spans_too = true) {
  const double pair_kappa = coupled ? kappa : 0.0;
  std::vector<std::vector<NodePatchNearCorrection>> per_leaf(leaves.size());
  const auto correct = [&](std::vector<NodePatchNearCorrection> &out,
                           std::size_t target, std::size_t source) {
    const auto analytic = TitanPBE::Tabi::evaluate_jcp_near_pair_kernel(
        np[target], target, np[source], source, eps_ratio, pair_kappa, coupled,
        "analytic_singular", "off", nullptr);
    if (!analytic.valid || !(analytic.one_ring || analytic.geometric_buffer))
      return;
    const auto centroid = TitanPBE::Tabi::evaluate_jcp_centroid_monopole_kernel(
        np[target], np[source], eps_ratio, pair_kappa);
    if (!centroid.valid)
      throw std::runtime_error("packed node-patch near correction is invalid");
    out.push_back({static_cast<std::uint32_t>(target),
                   static_cast<std::uint32_t>(source),
                   coupled ? analytic.k1 - centroid.k1 : 0.0,
                   analytic.k2 - centroid.k2,
                   coupled ? analytic.k3 - centroid.k3 : 0.0,
                   coupled ? analytic.k4 - centroid.k4 : 0.0});
  };
  std::vector<std::size_t> candidate_offsets, candidate_sources;
  if (node_patch_correction_candidates(np, candidate_offsets,
                                       candidate_sources) ||
      sphere_patch_correction_candidates(np, candidate_offsets,
                                         candidate_sources)) {
    std::vector<std::size_t> position(source_order.size(), SIZE_MAX);
    for (std::size_t p = 0; p < source_order.size(); ++p)
      if (source_order[p] < position.size())
        position[source_order[p]] = p;
    // A candidate whose source the tree put in the far field still takes its
    // correction: the far field then sums the one-point kernel through the
    // expansions, and the correction (curved minus exact one-point) brings
    // the pair to its curved value up to the expansion error, as the direct
    // operator and the audit rows have it.  Such pairs follow the near-list
    // ones, by source index, so the list does not depend on the threads.
    std::atomic<std::size_t> far_classified{0};
    executor.parallel_for(
        leaves.size(), executor.capacity(), 1,
        [&](std::size_t, std::size_t begin, std::size_t end) {
          std::vector<std::array<std::size_t, 3>> ordered; // span, pos, source
          std::vector<std::size_t> far;
          for (std::size_t li = begin; li < end; ++li) {
            const PackedFallbackLeafInput &leaf = leaves[li];
            auto &out = per_leaf[li];
            for (const std::size_t target : leaf.target_panels) {
              ordered.clear();
              far.clear();
              for (std::size_t c = candidate_offsets[target];
                   c < candidate_offsets[target + 1]; ++c) {
                const std::size_t source = candidate_sources[c];
                const std::size_t pos = position[source];
                bool found = false;
                for (std::size_t s = 0; s < leaf.source_spans.size(); ++s) {
                  const auto &span = leaf.source_spans[s];
                  if (pos >= span.first_panel &&
                      pos < span.first_panel + span.panel_count) {
                    ordered.push_back({s, pos, source});
                    found = true;
                    break;
                  }
                }
                if (!found && source != target)
                  far.push_back(source);
              }
              std::sort(ordered.begin(), ordered.end());
              for (const auto &entry : ordered)
                if (near_spans_too)
                  correct(out, target, entry[2]);
              std::sort(far.begin(), far.end());
              const std::size_t before = out.size();
              for (const std::size_t source : far)
                correct(out, target, source);
              far_classified.fetch_add(out.size() - before,
                                       std::memory_order_relaxed);
            }
          }
        });
    if (far_classified.load() != 0)
      std::cerr << "TABI node-patch: " << far_classified.load()
                << " one-ring or curved near-zone pair(s) the tree puts in "
                   "the far field take their correction as well\n";
  } else if (near_spans_too)
  // No candidate list (the panels do not index one node-patch geometry in
  // order): only near-list pairs can be found here.  A scalar loop, which
  // asks for the far-classified pairs alone, evaluates those itself.
  executor.parallel_for(
      leaves.size(), executor.capacity(), 1,
      [&](std::size_t, std::size_t begin, std::size_t end) {
        for (std::size_t li = begin; li < end; ++li) {
          const PackedFallbackLeafInput &leaf = leaves[li];
          auto &out = per_leaf[li];
          for (const std::size_t target : leaf.target_panels) {
            for (const auto &span : leaf.source_spans) {
              for (std::size_t pos = span.first_panel;
                   pos < span.first_panel + span.panel_count; ++pos) {
                const std::size_t source = source_order[pos];
                if (source == target)
                  continue;
                const auto analytic =
                    TitanPBE::Tabi::evaluate_jcp_near_pair_kernel(
                        np[target], target, np[source], source, eps_ratio,
                        pair_kappa, coupled, "analytic_singular", "off",
                        nullptr);
                if (!analytic.valid ||
                    !(analytic.one_ring || analytic.geometric_buffer))
                  continue;
                const auto centroid =
                    TitanPBE::Tabi::evaluate_jcp_centroid_monopole_kernel(
                        np[target], np[source], eps_ratio, pair_kappa);
                if (!centroid.valid)
                  throw std::runtime_error(
                      "packed node-patch near correction is invalid");
                out.push_back({static_cast<std::uint32_t>(target),
                               static_cast<std::uint32_t>(source),
                               coupled ? analytic.k1 - centroid.k1 : 0.0,
                               analytic.k2 - centroid.k2,
                               coupled ? analytic.k3 - centroid.k3 : 0.0,
                               coupled ? analytic.k4 - centroid.k4 : 0.0});
              }
            }
          }
        }
      });
  std::vector<NodePatchNearCorrection> corrections;
  for (auto &leaf_corrections : per_leaf)
    corrections.insert(corrections.end(), leaf_corrections.begin(),
                       leaf_corrections.end());
  return corrections;
}

void apply_node_patch_near_corrections(
    const std::vector<NodePatchNearCorrection> &corrections,
    const std::vector<double> &phi_values,
    const std::vector<double> &phin_values, double *phi_out,
    double *phin_out) {
  for (const auto &c : corrections) {
    phi_out[c.target] +=
        c.dk2 * phi_values[c.source] + c.dk1 * phin_values[c.source];
    if (phin_out != nullptr)
      phin_out[c.target] +=
          c.dk3 * phin_values[c.source] + c.dk4 * phi_values[c.source];
  }
}

struct StaticLocalP4Plan {
  int local_order{4};
  double tolerance{2.0e-3};
  bool sampled_targets_only{false};
  std::uint64_t sampled_target_hash{0};
  std::vector<int> target_nodes;
  std::vector<std::vector<std::size_t>> target_panels;
  std::vector<TargetLeafWork> work;
  std::vector<std::vector<CartesianJcpM2lOperatorP4>> cached_operators;
  std::vector<double> compressed_cached_green_coefficients;
  std::vector<std::size_t> compressed_cache_offsets;
  std::size_t compressed_green_term_count{0};
  std::vector<std::vector<int>> refined_fallback_sources;
  std::unique_ptr<PackedJcpFallbackPlan> packed_fallback_plan;
  // Node-patch corrections for the packed fallback's pairs, built with it.
  std::vector<NodePatchNearCorrection> packed_node_patch_corrections;
  // The scalar fallback's corrections: far-classified pairs only.
  std::vector<NodePatchNearCorrection> scalar_far_node_patch_corrections;
  bool scalar_far_built{false};
  std::uint64_t hash{0};
  std::string refine_mode{"operator-tail-static"};
  std::size_t probe_count{4};
  std::string probe_mode{"rademacher"};
  bool solution_probe_used{false};
  double linf_max_abs_tail{0.0};
  double linf_max_relative_tail{0.0};
  std::size_t linf_outlier_panels{0};
  std::string geometry_guard{"off"};
  double geometry_theta{0.0};
  std::size_t geometry_guard_rejected_cells{0};
  std::size_t geometry_guard_direct_fallback_pairs{0};
  double geometry_gap_min{0.0};
  double geometry_ratio_max{0.0};
  std::size_t refined_cells{0};
  std::size_t split_children{0};
  std::size_t direct_fallback_panels{0};
  std::size_t direct_fallback_pairs{0};
  std::size_t far_pairs{0};
  std::size_t far_panel_pairs{0};
  std::size_t cache_bytes{0};
  std::size_t cache_operator_bytes{0};
  std::size_t cache_uncompressed_estimated_bytes{0};
  double cache_compression_ratio{1.0};
  std::string cache_layout{"off"};
  double plan_seconds{0.0};
  std::string plan_builder{"legacy-rescan"};
  double probe_moment_seconds{0.0};
  double pair_metric_seconds{0.0};
  double refinement_seconds{0.0};
  std::size_t pair_metric_evaluations{0};
  std::size_t pair_metric_cache_hits{0};
  std::size_t refinement_iterations{0};
  std::size_t metric_cache_peak_bytes{0};
  std::size_t parallel_target_leaves{0};
  double cache_seconds{0.0};
  std::size_t tiled_green_term_count{0};
  std::size_t tiled_operator_bytes{0};
  std::size_t tiled_peak_bytes{0};
  bool cache_ready{false};
};

void hash_mix(std::uint64_t &hash, std::uint64_t value) noexcept {
  hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
}

} // namespace

struct FmmM2lAnalyticContext::Impl {
  PrototypeConfig config;
  std::vector<PrototypePanel> panels;
  PointSoA source_points;
  PointSoA target_points;
  Tree source_tree;
  Tree target_tree;
  std::size_t near_pairs{0};
  std::size_t far_pairs{0};
  std::size_t work_items{0};
  double eps_ratio{1.0};
  double kappa{0.0};
  double setup_seconds{0.0};
  mutable std::unique_ptr<StaticLocalP4Plan> static_local_plan;
  mutable std::size_t static_plan_build_count{0};
  mutable std::size_t local_cache_build_count{0};
  mutable std::size_t local_cache_hits{0};
  // A prepared local operator is consumed serially by GMRES. Retaining these
  // arrays avoids repeated large allocations during Krylov matvecs while the
  // mutex makes accidental concurrent diagnostic calls fail safe.
  mutable std::mutex local_p4_apply_mutex;
  mutable std::vector<CartesianJcpMoments> local_p4_moment_workspace;
  mutable std::vector<CartesianJcpLocalP4> local_p4_local_workspace;
  mutable std::vector<std::vector<double>> local_p4_tiled_coulomb_workspace;
  mutable std::vector<std::vector<double>> local_p4_tiled_yukawa_workspace;
  // The degree-one operator uses the same immutable near worklist on every
  // Krylov apply. Build its packed representation lazily once production
  // triangle geometry is available, then only refresh density SoA values.
  mutable std::unique_ptr<PackedJcpFallbackPlan> degree1_packed_near_plan;
  // build_node_patch_near_corrections for the degree-one near plan.
  mutable std::vector<NodePatchNearCorrection>
      degree1_packed_one_ring_corrections;
  // The scalar near loop's corrections: far-classified pairs only.
  mutable bool degree1_scalar_far_built{false};
  mutable std::vector<NodePatchNearCorrection> degree1_scalar_far_corrections;
  mutable std::mutex degree1_apply_mutex;
  mutable std::vector<PanelSourceMoments> degree1_moment_workspace;
  bool local_p4_prepared{false};
  JcpLocalP4OperatorOptions local_p4_options;
  std::vector<TitanPBE::Tabi::Panel> local_p4_production_panels;
};

FmmM2lAnalyticContext::FmmM2lAnalyticContext(
    const PrototypeConfig &input_config, std::vector<PrototypePanel> panels,
    double eps_ratio, double kappa)
    : impl_(std::make_unique<Impl>()) {
  if (panels.empty()) {
    throw std::invalid_argument(
        "JCP analytic FMM/M2L context requires at least one panel");
  }
  if (!std::isfinite(kappa) || kappa < 0.0) {
    throw std::invalid_argument(
        "JCP analytic FMM/M2L context requires finite kappa >= 0");
  }
  const auto setup_begin = Clock::now();
  impl_->config = normalize_jcp_analytic_config(input_config);
  impl_->panels = std::move(panels);
  impl_->eps_ratio = eps_ratio;
  impl_->kappa = kappa;
  impl_->source_points = make_panel_source_points(impl_->panels);
  impl_->target_points = make_panel_geometry_points(impl_->panels);
  impl_->source_tree = build_tree(
      impl_->source_points,
      static_cast<std::size_t>(impl_->config.max_leaf_size));
  impl_->target_tree = build_tree(
      impl_->target_points,
      static_cast<std::size_t>(impl_->config.max_leaf_size));
  if (!impl_->source_tree.cells.empty() && !impl_->target_tree.cells.empty()) {
    build_work(impl_->target_tree, impl_->source_tree, 0, 0,
               impl_->config.separation_theta, true);
  }
  sort_work(impl_->target_tree);
  PrototypeMetrics counts;
  compute_counts_and_operators(impl_->target_tree, impl_->source_tree,
                               impl_->config, counts);
  impl_->near_pairs = counts.near_pairs;
  impl_->far_pairs = counts.far_pairs;
  for (int target_node : impl_->target_tree.target_leaf_nodes) {
    const TargetLeafWork &work =
        impl_->target_tree.work[static_cast<std::size_t>(target_node)];
    impl_->work_items += work.near_sources.size() + work.far_sources.size();
  }
  impl_->setup_seconds = elapsed_seconds(setup_begin, Clock::now());
}

FmmM2lAnalyticContext::~FmmM2lAnalyticContext() = default;

FmmM2lAnalyticContext::FmmM2lAnalyticContext(
    FmmM2lAnalyticContext &&) noexcept = default;

FmmM2lAnalyticContext &
FmmM2lAnalyticContext::operator=(FmmM2lAnalyticContext &&) noexcept = default;

bool FmmM2lAnalyticContext::matches(const PrototypeConfig &input_config,
                                    std::size_t panel_count, double eps_ratio,
                                    double kappa) const noexcept {
  if (impl_ == nullptr) {
    return false;
  }
  const PrototypeConfig config = normalize_jcp_analytic_config(input_config);
  return panel_count == impl_->panels.size() &&
         config.kernel == impl_->config.kernel &&
         config.max_leaf_size == impl_->config.max_leaf_size &&
         config.expansion_order == impl_->config.expansion_order &&
         same_double(config.separation_theta, impl_->config.separation_theta) &&
         same_double(eps_ratio, impl_->eps_ratio) &&
         same_double(kappa, impl_->kappa);
}

std::size_t FmmM2lAnalyticContext::panel_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->panels.size();
}

std::size_t FmmM2lAnalyticContext::context_nodes() const noexcept {
  return impl_ == nullptr ? 0
                          : impl_->source_tree.cells.size() +
                                impl_->target_tree.cells.size();
}

std::size_t FmmM2lAnalyticContext::context_work_items() const noexcept {
  return impl_ == nullptr ? 0 : impl_->work_items;
}

double FmmM2lAnalyticContext::setup_seconds() const noexcept {
  return impl_ == nullptr ? 0.0 : impl_->setup_seconds;
}

std::vector<std::vector<std::size_t>>
FmmM2lAnalyticContext::block_panel_ids(std::size_t max_block_size) const {
  if (impl_ == nullptr) {
    throw std::logic_error("JCP analytic FMM/M2L context is not initialized");
  }
  const std::size_t cap = std::max<std::size_t>(1, max_block_size);
  std::vector<std::vector<std::size_t>> blocks;
  for (const int node_index : impl_->target_tree.target_leaf_nodes) {
    if (node_index < 0 || static_cast<std::size_t>(node_index) >=
                              impl_->target_tree.cells.size()) {
      continue;
    }
    const Cell &cell =
        impl_->target_tree.cells[static_cast<std::size_t>(node_index)];
    const std::size_t end = std::min(cell.end, impl_->target_tree.order.size());
    for (std::size_t begin = cell.begin; begin < end; begin += cap) {
      const std::size_t block_end = std::min(end, begin + cap);
      std::vector<std::size_t> block;
      block.reserve(block_end - begin);
      for (std::size_t pos = begin; pos < block_end; ++pos) {
        const std::size_t panel = impl_->target_tree.order[pos];
        if (panel < impl_->panels.size()) {
          block.push_back(panel);
        }
      }
      if (!block.empty()) {
        blocks.push_back(std::move(block));
      }
    }
  }
  return blocks;
}

std::vector<FmmM2lOverlapPanelGroup>
FmmM2lAnalyticContext::overlap_panel_groups(
    std::size_t owned_block_size, std::size_t overlap_panel_cap) const {
  if (impl_ == nullptr) {
    throw std::logic_error("JCP analytic FMM/M2L context is not initialized");
  }
  const std::size_t owned_cap = std::max<std::size_t>(1, owned_block_size);
  const std::size_t halo_cap = overlap_panel_cap;
  std::vector<FmmM2lOverlapPanelGroup> groups;
  const std::size_t n = impl_->panels.size();

  for (const int node_index : impl_->target_tree.target_leaf_nodes) {
    if (node_index < 0 || static_cast<std::size_t>(node_index) >=
                              impl_->target_tree.cells.size()) {
      continue;
    }
    const Cell &cell =
        impl_->target_tree.cells[static_cast<std::size_t>(node_index)];
    const std::size_t end = std::min(cell.end, impl_->target_tree.order.size());
    const TargetLeafWork &work =
        impl_->target_tree.work[static_cast<std::size_t>(node_index)];

    for (std::size_t begin = cell.begin; begin < end; begin += owned_cap) {
      const std::size_t block_end = std::min(end, begin + owned_cap);
      FmmM2lOverlapPanelGroup group;
      group.owned_panels.reserve(block_end - begin);
      group.factor_panels.reserve(block_end - begin + halo_cap);
      std::vector<unsigned char> owned_mask(n, 0);
      std::vector<unsigned char> factor_mask(n, 0);

      double cx = 0.0;
      double cy = 0.0;
      double cz = 0.0;
      for (std::size_t pos = begin; pos < block_end; ++pos) {
        const std::size_t panel = impl_->target_tree.order[pos];
        if (panel >= n || owned_mask[panel])
          continue;
        owned_mask[panel] = 1;
        factor_mask[panel] = 1;
        group.owned_panels.push_back(panel);
        group.factor_panels.push_back(panel);
        cx += impl_->panels[panel].x;
        cy += impl_->panels[panel].y;
        cz += impl_->panels[panel].z;
      }
      if (group.owned_panels.empty())
        continue;
      const double inv_owned =
          1.0 / static_cast<double>(group.owned_panels.size());
      cx *= inv_owned;
      cy *= inv_owned;
      cz *= inv_owned;

      if (halo_cap > 0) {
        std::vector<std::pair<double, std::size_t>> candidates;
        for (const int source_node : work.near_sources) {
          if (source_node < 0 || static_cast<std::size_t>(source_node) >=
                                     impl_->source_tree.cells.size()) {
            continue;
          }
          const Cell &source_cell =
              impl_->source_tree.cells[static_cast<std::size_t>(source_node)];
          const std::size_t source_end =
              std::min(source_cell.end, impl_->source_tree.order.size());
          for (std::size_t pos = source_cell.begin; pos < source_end; ++pos) {
            const std::size_t panel = impl_->source_tree.order[pos];
            if (panel >= n || owned_mask[panel] || factor_mask[panel])
              continue;
            factor_mask[panel] = 1;
            const double dx = impl_->panels[panel].x - cx;
            const double dy = impl_->panels[panel].y - cy;
            const double dz = impl_->panels[panel].z - cz;
            candidates.push_back({dx * dx + dy * dy + dz * dz, panel});
          }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto &lhs, const auto &rhs) {
                    if (lhs.first != rhs.first)
                      return lhs.first < rhs.first;
                    return lhs.second < rhs.second;
                  });
        const std::size_t selected = std::min(halo_cap, candidates.size());
        group.halo_count = selected;
        for (std::size_t i = 0; i < selected; ++i)
          group.factor_panels.push_back(candidates[i].second);
      }
      groups.push_back(std::move(group));
    }
  }
  return groups;
}

std::vector<std::vector<std::size_t>>
FmmM2lAnalyticContext::near_panel_graph(std::size_t neighbor_cap,
                                        TitanExecutor *executor) const {
  if (impl_ == nullptr) {
    throw std::logic_error("JCP analytic FMM/M2L context is not initialized");
  }
  const std::size_t n = impl_->panels.size();
  const std::size_t cap = std::max<std::size_t>(1, neighbor_cap);
  std::vector<std::vector<std::size_t>> graph(n);
  // Every panel is a target of exactly one leaf and its row depends on that
  // leaf's near list only, so the leaves are processed in parallel, each
  // worker with its own visit stamps.
  const std::vector<int> &leaf_nodes = impl_->target_tree.target_leaf_nodes;
  const std::size_t workers =
      executor != nullptr
          ? std::max<std::size_t>(
                1U, std::min(executor->capacity(), leaf_nodes.size()))
          : 1U;
  std::vector<std::vector<std::uint32_t>> stamps(workers);
  std::vector<std::uint32_t> generations(workers, 0U);
  const auto build_rows = [&](std::size_t worker, std::size_t leaf_begin,
                              std::size_t leaf_end) {
  std::vector<std::uint32_t> &seen_generation = stamps[worker];
  if (seen_generation.size() != n)
    seen_generation.assign(n, 0U);
  std::uint32_t &generation = generations[worker];
  for (std::size_t leaf = leaf_begin; leaf < leaf_end; ++leaf) {
    const int node_index = leaf_nodes[leaf];
    if (node_index < 0 || static_cast<std::size_t>(node_index) >=
                              impl_->target_tree.cells.size()) {
      continue;
    }
    const Cell &target_cell =
        impl_->target_tree.cells[static_cast<std::size_t>(node_index)];
    const std::size_t target_end =
        std::min(target_cell.end, impl_->target_tree.order.size());
    const TargetLeafWork &work =
        impl_->target_tree.work[static_cast<std::size_t>(node_index)];

    for (std::size_t tpos = target_cell.begin; tpos < target_end; ++tpos) {
      const std::size_t target_panel = impl_->target_tree.order[tpos];
      if (target_panel >= n)
        continue;
      if (++generation == 0U) {
        std::fill(seen_generation.begin(), seen_generation.end(), 0U);
        generation = 1U;
      }
      std::vector<std::pair<double, std::size_t>> candidates;
      auto add_candidate = [&](std::size_t source_panel) {
        if (source_panel >= n || seen_generation[source_panel] == generation)
          return;
        seen_generation[source_panel] = generation;
        const double dx =
            impl_->panels[source_panel].x - impl_->panels[target_panel].x;
        const double dy =
            impl_->panels[source_panel].y - impl_->panels[target_panel].y;
        const double dz =
            impl_->panels[source_panel].z - impl_->panels[target_panel].z;
        candidates.push_back({dx * dx + dy * dy + dz * dz, source_panel});
      };
      add_candidate(target_panel);
      for (const int source_node : work.near_sources) {
        if (source_node < 0 || static_cast<std::size_t>(source_node) >=
                                   impl_->source_tree.cells.size()) {
          continue;
        }
        const Cell &source_cell =
            impl_->source_tree.cells[static_cast<std::size_t>(source_node)];
        const std::size_t source_end =
            std::min(source_cell.end, impl_->source_tree.order.size());
        for (std::size_t spos = source_cell.begin; spos < source_end; ++spos)
          add_candidate(impl_->source_tree.order[spos]);
      }
      std::vector<std::size_t> row;
      const std::size_t selected = std::min(cap, candidates.size());
      const auto distance_then_id = [](const auto &lhs, const auto &rhs) {
        if (lhs.first != rhs.first)
          return lhs.first < rhs.first;
        return lhs.second < rhs.second;
      };
      if (selected < candidates.size()) {
        std::partial_sort(candidates.begin(),
                          candidates.begin() +
                              static_cast<std::ptrdiff_t>(selected),
                          candidates.end(), distance_then_id);
      } else {
        std::sort(candidates.begin(), candidates.end(), distance_then_id);
      }
      row.reserve(selected);
      for (std::size_t i = 0; i < selected; ++i)
        row.push_back(candidates[i].second);
      std::sort(row.begin(), row.end());
      row.erase(std::unique(row.begin(), row.end()), row.end());
      graph[target_panel] = std::move(row);
    }
  }
  };
  if (workers > 1U)
    executor->parallel_for(leaf_nodes.size(), workers, 1U, build_rows);
  else
    build_rows(0U, 0U, leaf_nodes.size());
  for (std::size_t panel = 0; panel < n; ++panel) {
    if (graph[panel].empty())
      graph[panel].push_back(panel);
  }
  return graph;
}

std::vector<std::vector<std::size_t>>
FmmM2lAnalyticContext::distance_knn_panel_graph(
    std::size_t neighbor_cap) const {
  if (impl_ == nullptr) {
    throw std::logic_error("JCP analytic FMM/M2L context is not initialized");
  }
  const std::size_t n = impl_->panels.size();
  const std::size_t cap = std::max<std::size_t>(1, neighbor_cap);
  std::vector<std::vector<std::size_t>> graph(n);
  if (n == 0 || impl_->source_tree.cells.empty())
    return graph;

  using DistanceId = std::pair<double, std::size_t>;
  using DistanceNode = std::pair<double, int>;
  const auto node_lower_distance_sq = [&](const Cell &cell,
                                          const PrototypePanel &target) {
    const double dx =
        target.x < cell.min_x
            ? cell.min_x - target.x
            : (target.x > cell.max_x ? target.x - cell.max_x : 0.0);
    const double dy =
        target.y < cell.min_y
            ? cell.min_y - target.y
            : (target.y > cell.max_y ? target.y - cell.max_y : 0.0);
    const double dz =
        target.z < cell.min_z
            ? cell.min_z - target.z
            : (target.z > cell.max_z ? target.z - cell.max_z : 0.0);
    return dx * dx + dy * dy + dz * dz;
  };

  for (std::size_t target_id = 0; target_id < n; ++target_id) {
    const PrototypePanel &target = impl_->panels[target_id];
    std::priority_queue<DistanceNode, std::vector<DistanceNode>,
                        std::greater<DistanceNode>>
        pending;
    std::priority_queue<DistanceId> nearest;
    pending.push(
        {node_lower_distance_sq(impl_->source_tree.cells[0], target), 0});
    while (!pending.empty()) {
      const auto [lower_sq, node_index] = pending.top();
      pending.pop();
      if (nearest.size() == cap && lower_sq > nearest.top().first)
        break;
      if (node_index < 0 || static_cast<std::size_t>(node_index) >=
                                impl_->source_tree.cells.size())
        continue;
      const Cell &cell =
          impl_->source_tree.cells[static_cast<std::size_t>(node_index)];
      if (cell.is_leaf()) {
        const std::size_t end =
            std::min(cell.end, impl_->source_tree.order.size());
        for (std::size_t pos = cell.begin; pos < end; ++pos) {
          const std::size_t source_id = impl_->source_tree.order[pos];
          if (source_id >= n)
            continue;
          const PrototypePanel &source = impl_->panels[source_id];
          const double dx = source.x - target.x;
          const double dy = source.y - target.y;
          const double dz = source.z - target.z;
          const DistanceId candidate{dx * dx + dy * dy + dz * dz, source_id};
          if (nearest.size() < cap) {
            nearest.push(candidate);
          } else if (candidate < nearest.top()) {
            nearest.pop();
            nearest.push(candidate);
          }
        }
        continue;
      }
      for (const int child : {cell.left, cell.right}) {
        if (child < 0 ||
            static_cast<std::size_t>(child) >= impl_->source_tree.cells.size())
          continue;
        const double child_lower = node_lower_distance_sq(
            impl_->source_tree.cells[static_cast<std::size_t>(child)], target);
        if (nearest.size() < cap || child_lower <= nearest.top().first)
          pending.push({child_lower, child});
      }
    }
    auto &row = graph[target_id];
    row.reserve(nearest.size());
    while (!nearest.empty()) {
      row.push_back(nearest.top().second);
      nearest.pop();
    }
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
    if (row.empty())
      row.push_back(target_id);
  }
  return graph;
}

std::vector<FmmM2lOverlapPanelGroup>
FmmM2lAnalyticContext::distance_knn_overlap_panel_groups(
    std::size_t owned_block_size, std::size_t overlap_panel_cap) const {
  if (impl_ == nullptr) {
    throw std::logic_error("JCP analytic FMM/M2L context is not initialized");
  }
  const auto owned_blocks = block_panel_ids(owned_block_size);
  const auto graph =
      distance_knn_panel_graph(std::max<std::size_t>(1, overlap_panel_cap));
  using DistanceId = std::pair<double, std::size_t>;
  const std::size_t n = impl_->panels.size();
  std::vector<std::uint32_t> owned_generation(n, 0U);
  std::vector<std::uint32_t> candidate_generation(n, 0U);
  std::uint32_t generation = 0U;
  std::vector<FmmM2lOverlapPanelGroup> groups;
  groups.reserve(owned_blocks.size());
  for (const auto &owned : owned_blocks) {
    if (++generation == 0U) {
      std::fill(owned_generation.begin(), owned_generation.end(), 0U);
      std::fill(candidate_generation.begin(), candidate_generation.end(), 0U);
      generation = 1U;
    }
    FmmM2lOverlapPanelGroup group;
    double cx = 0.0;
    double cy = 0.0;
    double cz = 0.0;
    for (const std::size_t panel : owned) {
      if (panel >= n || owned_generation[panel] == generation)
        continue;
      owned_generation[panel] = generation;
      candidate_generation[panel] = generation;
      group.owned_panels.push_back(panel);
      group.factor_panels.push_back(panel);
      cx += impl_->panels[panel].x;
      cy += impl_->panels[panel].y;
      cz += impl_->panels[panel].z;
    }
    if (group.owned_panels.empty())
      continue;
    const double inv_owned =
        1.0 / static_cast<double>(group.owned_panels.size());
    cx *= inv_owned;
    cy *= inv_owned;
    cz *= inv_owned;
    std::vector<DistanceId> candidates;
    if (overlap_panel_cap > 0) {
      for (const std::size_t panel : group.owned_panels) {
        for (const std::size_t source : graph[panel]) {
          if (source >= n || candidate_generation[source] == generation)
            continue;
          candidate_generation[source] = generation;
          const double dx = impl_->panels[source].x - cx;
          const double dy = impl_->panels[source].y - cy;
          const double dz = impl_->panels[source].z - cz;
          candidates.push_back({dx * dx + dy * dy + dz * dz, source});
        }
      }
      std::sort(candidates.begin(), candidates.end());
      const std::size_t selected =
          std::min(overlap_panel_cap, candidates.size());
      group.halo_count = selected;
      for (std::size_t i = 0; i < selected; ++i)
        group.factor_panels.push_back(candidates[i].second);
    }
    groups.push_back(std::move(group));
  }
  return groups;
}

JcpAnalyticOperatorResult FmmM2lAnalyticContext::apply(
    const std::vector<double> &phi_values,
    const std::vector<double> &phin_values, TitanExecutor &executor,
    bool coupled, const std::string &near_integration,
    const std::string &quad_buffer,
    const std::vector<TitanPBE::Tabi::Panel> *production_panels) const {
  std::vector<double> phi_out;
  std::vector<double> phin_out;
  JcpAnalyticOperatorResult result = apply_into(
      phi_values, phin_values, executor, phi_out, phin_out, coupled,
      near_integration, quad_buffer, production_panels);
  result.phi = std::move(phi_out);
  result.phin = std::move(phin_out);
  return result;
}

JcpAnalyticOperatorResult FmmM2lAnalyticContext::apply_into(
    const std::vector<double> &phi_values,
    const std::vector<double> &phin_values, TitanExecutor &executor,
    std::vector<double> &phi_out, std::vector<double> &phin_out, bool coupled,
    const std::string &near_integration, const std::string &quad_buffer,
    const std::vector<TitanPBE::Tabi::Panel> *production_panels) const {
  if (impl_ == nullptr) {
    throw std::logic_error("JCP analytic FMM/M2L context is not initialized");
  }
  if (phi_values.size() != impl_->panels.size() ||
      phin_values.size() != impl_->panels.size()) {
    throw std::invalid_argument(
        "JCP analytic FMM/M2L context density vectors must match panels");
  }
  const bool use_production_near_kernel =
      near_integration != "centroid" && near_integration != "off";
  if (use_production_near_kernel &&
      (production_panels == nullptr ||
       production_panels->size() != impl_->panels.size())) {
    throw std::invalid_argument(
        "JCP analytic FMM/M2L analytic near integration requires production "
        "panel geometry");
  }

  JcpAnalyticOperatorResult result;
  result.panels = impl_->panels.size();
  result.near_pairs = impl_->near_pairs;
  result.far_pairs = impl_->far_pairs;
  result.setup_seconds = impl_->setup_seconds;
  result.context_nodes = context_nodes();
  result.context_work_items = context_work_items();
  phi_out.assign(impl_->panels.size(), 0.0);
  phin_out.assign(impl_->panels.size(), 0.0);

  std::unique_lock<std::mutex> degree1_apply_lock(
      impl_->degree1_apply_mutex);
  const bool node_patch_sources =
      production_panels != nullptr && !production_panels->empty() &&
      production_panels->front().node_patch_geometry != nullptr;
  const bool use_packed_near =
      use_production_near_kernel && near_integration == "analytic_singular" &&
      quad_buffer == "off" && packed_jcp_fallback_avx2_available();
  // Target leaves with their near source spans, for the correction lists.
  const auto degree1_leaves = [&] {
    std::vector<PackedFallbackLeafInput> leaves;
    leaves.reserve(impl_->target_tree.target_leaf_nodes.size());
    for (const int target_node : impl_->target_tree.target_leaf_nodes) {
      const Cell &target_cell =
          impl_->target_tree.cells[static_cast<std::size_t>(target_node)];
      const TargetLeafWork &work =
          impl_->target_tree.work[static_cast<std::size_t>(target_node)];
      PackedFallbackLeafInput leaf;
      leaf.target_panels.reserve(target_cell.count());
      for (std::size_t pos = target_cell.begin; pos < target_cell.end; ++pos)
        leaf.target_panels.push_back(impl_->target_tree.order[pos]);
      leaf.source_spans.reserve(work.near_sources.size());
      for (const int source_node : work.near_sources) {
        const Cell &source_cell =
            impl_->source_tree.cells[static_cast<std::size_t>(source_node)];
        leaf.source_spans.push_back(
            {source_cell.begin, source_cell.count(), false});
      }
      leaves.push_back(std::move(leaf));
    }
    return leaves;
  };
  // The scalar near loop takes every near-list pair with its own (curved)
  // value; a one-ring or curved near-zone pair the tree puts in the far
  // field gets its correction as on the packed route.
  const bool sphere_patch_sources =
      production_panels != nullptr && !production_panels->empty() &&
      production_panels->front().sphere_patch_quadrature != nullptr;
  if (!use_packed_near && use_production_near_kernel &&
      (node_patch_sources || sphere_patch_sources) &&
      !impl_->degree1_scalar_far_built) {
    impl_->degree1_scalar_far_corrections = build_node_patch_near_corrections(
        degree1_leaves(), impl_->source_tree.order, *production_panels,
        impl_->eps_ratio, impl_->kappa, coupled, executor, false);
    impl_->degree1_scalar_far_built = true;
  }
  if (use_packed_near) {
    if (impl_->degree1_packed_near_plan == nullptr) {
      const std::vector<PackedFallbackLeafInput> leaves = degree1_leaves();
      impl_->degree1_packed_near_plan = std::make_unique<PackedJcpFallbackPlan>(
          build_packed_jcp_fallback_plan(
              *production_panels, impl_->source_tree.order, leaves, 4, 32));
      // Built from the same leaves the plan is built from, so the corrected
      // pairs are exactly the pairs the packed kernel evaluated; self pairs
      // already take the scalar path inside the packed loop.
      impl_->degree1_packed_one_ring_corrections.clear();
      // Clipped sphere patches take their one-ring pairs from the curved
      // dispatcher too (sphere_coupled_near), as the direct operator does.
      if (node_patch_sources ||
          production_panels->front().sphere_patch_quadrature != nullptr)
        impl_->degree1_packed_one_ring_corrections =
            build_node_patch_near_corrections(
                leaves, impl_->source_tree.order, *production_panels,
                impl_->eps_ratio, impl_->kappa, coupled, executor);
    }
    const PackedFallbackApplyMetrics packed = accumulate_packed_jcp_fallback(
        *impl_->degree1_packed_near_plan, *production_panels, phi_values,
        phin_values, impl_->eps_ratio, impl_->kappa, coupled, near_integration,
        quad_buffer, "packed-avx2", executor, phi_out, phin_out);
    apply_node_patch_near_corrections(
        impl_->degree1_packed_one_ring_corrections, phi_values, phin_values,
        phi_out.data(), phin_out.data());
    result.near_one_ring_pairs +=
        impl_->degree1_packed_one_ring_corrections.size();
    result.near_self_pairs = packed.scalar_self_pairs;
    if (node_patch_sources) {
      result.near_duffy_pairs = packed.scalar_self_pairs;
      // Self, one-ring and (curved) near-zone pairs are replaced in both
      // cases; the name says which cell they are integrated over.
      result.near_kernel_policy =
          TitanPBE::Tabi::curved_dual_cell(production_panels->front())
              ? "packed-avx2-node-patch-curved"
              : "packed-avx2-node-patch-flat";
    } else {
      result.near_kernel_policy = "packed-avx2-self-asi";
    }
    const std::size_t corrected = packed.scalar_self_pairs;
    result.near_centroid_pairs =
        impl_->near_pairs >= corrected ? impl_->near_pairs - corrected : 0;
  }

  compute_panel_source_moments_into(
      impl_->source_tree, impl_->panels, phi_values, phin_values,
      impl_->degree1_moment_workspace);
  const std::vector<PanelSourceMoments> &source_moments =
      impl_->degree1_moment_workspace;

  const auto apply_begin = Clock::now();
  const std::size_t workers = std::max<std::size_t>(1, executor.capacity());
  std::vector<TitanPBE::Tabi::TreecodeStats> local_near_stats(
      use_production_near_kernel ? workers : 0);
  executor.parallel_for(
      impl_->target_tree.target_leaf_nodes.size(), workers, 1,
      [&](std::size_t worker, std::size_t begin, std::size_t end) {
        TitanPBE::Tabi::TreecodeStats *worker_near_stats =
            use_production_near_kernel ? &local_near_stats[worker] : nullptr;
        for (std::size_t leaf_pos = begin; leaf_pos < end; ++leaf_pos) {
          const int target_node =
              impl_->target_tree.target_leaf_nodes[leaf_pos];
          const Cell &target_cell =
              impl_->target_tree.cells[static_cast<std::size_t>(target_node)];
          const TargetLeafWork &work =
              impl_->target_tree.work[static_cast<std::size_t>(target_node)];
          for (std::size_t tpos = target_cell.begin; tpos < target_cell.end;
               ++tpos) {
            const std::size_t ti = impl_->target_tree.order[tpos];
            double phi_sum = use_packed_near ? phi_out[ti] : 0.0;
            double phin_sum = use_packed_near ? phin_out[ti] : 0.0;
            const PrototypePanel &target = impl_->panels[ti];
            if (!use_packed_near)
              for (int source_node : work.near_sources) {
                const Cell &source_cell =
                    impl_->source_tree
                        .cells[static_cast<std::size_t>(source_node)];
                for (std::size_t spos = source_cell.begin;
                     spos < source_cell.end; ++spos) {
                  const std::size_t si = impl_->source_tree.order[spos];
                  if (use_production_near_kernel) {
                    add_jcp_near_kernel_pair(
                        (*production_panels)[ti], ti, (*production_panels)[si],
                        si, phi_values[si], phin_values[si], impl_->eps_ratio,
                        impl_->kappa, coupled, near_integration, quad_buffer,
                        phi_sum, phin_sum, worker_near_stats);
                  } else {
                    if (ti == si) {
                      continue;
                    }
                    add_jcp_direct_pair(target, impl_->panels[si],
                                        phi_values[si], phin_values[si],
                                        impl_->eps_ratio, impl_->kappa, phi_sum,
                                        phin_sum);
                  }
                }
              }
            for (int source_node : work.far_sources) {
              const std::size_t source_index =
                  static_cast<std::size_t>(source_node);
              add_jcp_far_cell(target, impl_->source_tree.cells[source_index],
                               source_moments[source_index], impl_->eps_ratio,
                               impl_->kappa, phi_sum, phin_sum);
            }
            phi_out[ti] = phi_sum;
            phin_out[ti] = phin_sum;
          }
        }
      });
  if (!use_packed_near && use_production_near_kernel && node_patch_sources) {
    apply_node_patch_near_corrections(impl_->degree1_scalar_far_corrections,
                                      phi_values, phin_values, phi_out.data(),
                                      phin_out.data());
    result.near_one_ring_pairs += impl_->degree1_scalar_far_corrections.size();
  }
  result.apply_seconds = elapsed_seconds(apply_begin, Clock::now());

  if (use_production_near_kernel && !use_packed_near) {
    for (const TitanPBE::Tabi::TreecodeStats &worker_stats : local_near_stats) {
      const std::size_t previous_duffy = result.near_duffy_pairs;
      result.near_self_pairs += worker_stats.near_self_pairs;
      result.near_one_ring_pairs += worker_stats.near_one_ring_pairs;
      result.near_duffy_pairs += worker_stats.near_duffy_pairs;
      result.near_centroid_pairs += worker_stats.near_centroid_pairs;
      result.near_fallbacks += worker_stats.near_fallbacks;
      result.duffy_subdivisions_max = std::max(
          result.duffy_subdivisions_max, worker_stats.duffy_subdivisions_max);
      if (worker_stats.near_duffy_pairs > 0) {
        const double old_count = static_cast<double>(previous_duffy);
        const double new_count = static_cast<double>(result.near_duffy_pairs);
        result.duffy_subdivisions_mean =
            new_count <= 0.0
                ? 0.0
                : (result.duffy_subdivisions_mean * old_count +
                   worker_stats.duffy_subdivisions_mean *
                       static_cast<double>(worker_stats.near_duffy_pairs)) /
                      new_count;
      }
      if (worker_stats.near_self_pairs + worker_stats.near_one_ring_pairs +
              worker_stats.near_duffy_pairs + worker_stats.near_centroid_pairs +
              worker_stats.near_geometric_buffer_pairs >
          0) {
        result.near_kernel_policy = worker_stats.near_kernel_policy;
      }
    }
  } else {
    result.near_centroid_pairs = result.near_pairs;
  }

  result.finite_outputs =
      std::all_of(phi_out.begin(), phi_out.end(),
                  [](double value) { return std::isfinite(value); }) &&
      std::all_of(phin_out.begin(), phin_out.end(),
                  [](double value) { return std::isfinite(value); });
  return result;
}

JcpOperatorAttributionResult FmmM2lAnalyticContext::attribute(
    const std::vector<double> &phi_values,
    const std::vector<double> &phin_values, std::size_t sample_count,
    const std::string &probe, TitanExecutor &executor, bool coupled,
    const std::string &near_integration, const std::string &quad_buffer,
    const std::vector<TitanPBE::Tabi::Panel> *production_panels) const {
  return attribute_impl(phi_values, phin_values, sample_count, probe, executor,
                        coupled, near_integration, quad_buffer,
                        production_panels, 0, "off", 4.0e-3);
}

JcpLocalCartesianCandidateResult
FmmM2lAnalyticContext::apply_local_cartesian_candidate(
    const std::vector<double> &phi_values,
    const std::vector<double> &phin_values, TitanExecutor &executor,
    bool coupled, const std::string &near_integration,
    const std::string &quad_buffer,
    const std::vector<TitanPBE::Tabi::Panel> *production_panels,
    const std::string &refine_mode, double refine_tolerance,
    const std::string &cache_mode, std::size_t cache_budget_mb,
    const std::string &fallback_mode, const std::string &plan_builder,
    std::size_t refine_probe_count, const std::string &refine_probe_mode,
    const std::string &geometry_guard, double geometry_theta,
    const std::vector<std::size_t> *sampled_target_indices) const {
  if (impl_ == nullptr)
    throw std::logic_error("JCP analytic FMM/M2L context is not initialized");
  const std::size_t n = impl_->panels.size();
  if (phi_values.size() != n || phin_values.size() != n)
    throw std::invalid_argument(
        "JCP local Cartesian densities must match panels");
  if (production_panels == nullptr || production_panels->size() != n)
    throw std::invalid_argument(
        "JCP local Cartesian candidate requires production panel geometry");
  const int local_order = impl_->config.expansion_order;
  if (local_order < 2 || local_order > 5)
    throw std::invalid_argument(
        "JCP local Cartesian candidate requires Taylor order 2, 3, 4, or 5");
  if (refine_mode != "operator-tail-static" &&
      refine_mode != "operator-tail-linf")
    throw std::invalid_argument(
        "JCP local Cartesian candidate requires operator-tail-static or "
        "operator-tail-linf refinement");
  if (!std::isfinite(refine_tolerance) || refine_tolerance < 1.0e-4 ||
      refine_tolerance > 5.0e-2)
    throw std::invalid_argument(
        "JCP local p4 static tolerance must be in [1e-4, 5e-2]");
  if (cache_mode != "off" && cache_mode != "kernel-jet" &&
      cache_mode != "green-jet-compressed" && cache_mode != "green-jet-tiled")
    throw std::invalid_argument("JCP local p4 cache must be off, kernel-jet, "
                                "green-jet-compressed, or green-jet-tiled");
  if (fallback_mode != "scalar" && fallback_mode != "packed-avx2")
    throw std::invalid_argument(
        "JCP local p4 fallback must be scalar or packed-avx2");
  if (plan_builder != "legacy-rescan" && plan_builder != "metric-cache")
    throw std::invalid_argument(
        "JCP local p4 plan builder must be legacy-rescan or metric-cache");
  if (refine_probe_count != 4 && refine_probe_count != 8 &&
      refine_probe_count != 16)
    throw std::invalid_argument(
        "JCP local p4 refine probe count must be 4, 8, or 16");
  if (refine_probe_mode != "rademacher" && refine_probe_mode != "solution" &&
      refine_probe_mode != "mixed")
    throw std::invalid_argument("JCP local p4 refine probe mode must be "
                                "rademacher, solution, or mixed");
  if (geometry_guard != "off" && geometry_guard != "strict-geometry-mac")
    throw std::invalid_argument(
        "JCP local p4 geometry guard must be off or strict-geometry-mac");
  if (geometry_guard == "strict-geometry-mac" &&
      (!(geometry_theta > 0.0) || !std::isfinite(geometry_theta)))
    throw std::invalid_argument(
        "JCP local p4 strict geometry theta must be positive and finite");

  const bool sampled_targets_only = sampled_target_indices != nullptr;
  std::vector<std::size_t> sampled_targets;
  std::uint64_t sampled_target_hash = 0;
  if (sampled_targets_only) {
    sampled_targets = *sampled_target_indices;
    if (sampled_targets.empty())
      throw std::invalid_argument(
          "JCP local Cartesian sampled candidate requires target rows");
    std::sort(sampled_targets.begin(), sampled_targets.end());
    if (sampled_targets.back() >= n ||
        std::adjacent_find(sampled_targets.begin(), sampled_targets.end()) !=
            sampled_targets.end())
      throw std::invalid_argument("JCP local Cartesian sampled candidate "
                                  "target rows must be unique and in range");
    sampled_target_hash = 0x91e10da5c79f6d5bULL;
    hash_mix(sampled_target_hash,
             static_cast<std::uint64_t>(sampled_targets.size()));
    for (const std::size_t target : sampled_targets)
      hash_mix(sampled_target_hash, static_cast<std::uint64_t>(target));
  }
  std::lock_guard<std::mutex> apply_lock(impl_->local_p4_apply_mutex);

  const auto build_plan = [&]() {
    const auto plan_begin = Clock::now();
    auto plan = std::make_unique<StaticLocalP4Plan>();
    plan->local_order = local_order;
    plan->tolerance = refine_tolerance;
    plan->refine_mode = refine_mode;
    plan->plan_builder = plan_builder;
    plan->probe_count = refine_probe_count;
    plan->probe_mode = refine_probe_mode;
    plan->solution_probe_used = refine_probe_mode != "rademacher";
    plan->sampled_targets_only = sampled_targets_only;
    plan->sampled_target_hash = sampled_target_hash;
    plan->geometry_guard = geometry_guard;
    plan->geometry_theta =
        geometry_guard == "strict-geometry-mac" ? geometry_theta : 0.0;
    plan->geometry_gap_min = std::numeric_limits<double>::infinity();
    if (sampled_targets_only) {
      std::vector<int> target_leaf_by_panel(n, -1);
      for (const int target_node : impl_->target_tree.target_leaf_nodes) {
        const Cell &target_cell =
            impl_->target_tree.cells[static_cast<std::size_t>(target_node)];
        for (std::size_t pos = target_cell.begin; pos < target_cell.end;
             ++pos) {
          const std::size_t panel = impl_->target_tree.order[pos];
          if (panel < n)
            target_leaf_by_panel[panel] = target_node;
        }
      }
      std::vector<std::pair<int, std::size_t>> by_node;
      by_node.reserve(sampled_targets.size());
      for (const std::size_t target : sampled_targets) {
        const int node = target_leaf_by_panel[target];
        if (node < 0)
          throw std::logic_error(
              "JCP local p4 sampled target is not in a target leaf");
        by_node.push_back({node, target});
      }
      std::sort(by_node.begin(), by_node.end());
      for (const auto &entry : by_node) {
        if (plan->target_nodes.empty() ||
            plan->target_nodes.back() != entry.first) {
          plan->target_nodes.push_back(entry.first);
          plan->target_panels.emplace_back();
        }
        plan->target_panels.back().push_back(entry.second);
      }
    } else {
      plan->target_nodes = impl_->target_tree.target_leaf_nodes;
      plan->target_panels.reserve(plan->target_nodes.size());
      for (const int target_node : plan->target_nodes) {
        const Cell &target_cell =
            impl_->target_tree.cells[static_cast<std::size_t>(target_node)];
        std::vector<std::size_t> leaf_panels;
        leaf_panels.reserve(target_cell.count());
        for (std::size_t pos = target_cell.begin; pos < target_cell.end; ++pos)
          leaf_panels.push_back(impl_->target_tree.order[pos]);
        plan->target_panels.push_back(std::move(leaf_panels));
      }
    }
    plan->work.reserve(plan->target_nodes.size());
    plan->refined_fallback_sources.resize(plan->target_nodes.size());
    for (const int target_node : plan->target_nodes)
      plan->work.push_back(
          impl_->target_tree.work[static_cast<std::size_t>(target_node)]);

    const auto probe_moment_begin = Clock::now();
    std::vector<std::vector<CartesianJcpMoments>> probe_moments(
        refine_probe_count);
    const std::array<std::uint64_t, 16> seeds{17,  29,  43,  71,  83,  97,
                                              113, 131, 149, 167, 181, 197,
                                              211, 229, 241, 263};
    const double normalization = 1.0 / std::sqrt(static_cast<double>(n));
    const auto normalized_density_probe = [&](std::size_t probe,
                                              std::vector<double> &probe_phi,
                                              std::vector<double> &probe_phin) {
      double phi_norm_sq = 0.0;
      double phin_norm_sq = 0.0;
      for (std::size_t panel = 0; panel < n; ++panel) {
        phi_norm_sq += phi_values[panel] * phi_values[panel];
        phin_norm_sq += phin_values[panel] * phin_values[panel];
      }
      const double phi_norm = std::sqrt(std::max(0.0, phi_norm_sq));
      const double phin_norm = std::sqrt(std::max(0.0, phin_norm_sq));
      const double combined =
          std::sqrt(std::max(0.0, phi_norm_sq + phin_norm_sq));
      if (probe == 0 && combined > 0.0) {
        for (std::size_t panel = 0; panel < n; ++panel) {
          probe_phi[panel] = phi_values[panel] / combined;
          probe_phin[panel] = phin_values[panel] / combined;
        }
      } else if (probe == 1 && phi_norm > 0.0) {
        for (std::size_t panel = 0; panel < n; ++panel)
          probe_phi[panel] = phi_values[panel] / phi_norm;
      } else if (probe == 2 && phin_norm > 0.0) {
        for (std::size_t panel = 0; panel < n; ++panel)
          probe_phin[panel] = phin_values[panel] / phin_norm;
      }
    };
    for (std::size_t probe = 0; probe < probe_moments.size(); ++probe) {
      std::vector<double> probe_phi(n, 0.0);
      std::vector<double> probe_phin(n, 0.0);
      const bool solution_slot = refine_probe_mode == "solution" ||
                                 (refine_probe_mode == "mixed" && probe < 3);
      if (solution_slot) {
        normalized_density_probe(probe, probe_phi, probe_phin);
      }
      if (!solution_slot ||
          (std::all_of(probe_phi.begin(), probe_phi.end(),
                       [](double value) { return value == 0.0; }) &&
           std::all_of(probe_phin.begin(), probe_phin.end(),
                       [](double value) { return value == 0.0; }))) {
        for (std::size_t panel = 0; panel < n; ++panel) {
          const std::size_t panel_id = impl_->source_points.panel_id[panel];
          const double sign =
              (rademacher_hash(panel_id, seeds[probe]) & 1ULL) != 0
                  ? normalization
                  : -normalization;
          if (probe < refine_probe_count / 2)
            probe_phi[panel] = sign;
          else
            probe_phin[panel] = sign;
        }
      }
      probe_moments[probe] = build_cartesian_node_moments(
          impl_->source_tree, impl_->panels, probe_phi, probe_phin, local_order,
          executor);
    }
    plan->probe_moment_seconds =
        elapsed_seconds(probe_moment_begin, Clock::now());

    struct PairMetric {
      int source_node{-1};
      std::vector<double> tail_sq_by_probe;
      std::vector<double> signal_sq_by_probe;
      double tail_sq{0.0};
      double max_abs_tail{0.0};
      double max_relative_tail{0.0};
      std::size_t outlier_panels{0};
    };

    struct LeafPlanResult {
      TargetLeafWork work;
      std::vector<int> refined_fallback_sources;
      std::size_t refined_cells{0};
      std::size_t split_children{0};
      std::size_t direct_fallback_panels{0};
      std::size_t direct_fallback_pairs{0};
      std::size_t far_pairs{0};
      std::size_t far_panel_pairs{0};
      std::size_t pair_metric_evaluations{0};
      std::size_t pair_metric_cache_hits{0};
      std::size_t refinement_iterations{0};
      std::size_t geometry_guard_rejected_cells{0};
      std::size_t geometry_guard_direct_fallback_pairs{0};
      double geometry_gap_min{std::numeric_limits<double>::infinity()};
      double geometry_ratio_max{0.0};
      double linf_max_abs_tail{0.0};
      double linf_max_relative_tail{0.0};
      std::size_t linf_outlier_panels{0};
      double pair_metric_seconds{0.0};
    };

    std::vector<LeafPlanResult> leaf_results(plan->target_nodes.size());
    for (std::size_t target_index = 0; target_index < plan->target_nodes.size();
         ++target_index)
      leaf_results[target_index].work = plan->work[target_index];

    std::atomic<std::size_t> current_metric_cache_bytes{0};
    std::atomic<std::size_t> peak_metric_cache_bytes{0};
    auto record_metric_cache_bytes = [&](std::size_t bytes) {
      const std::size_t current =
          current_metric_cache_bytes.fetch_add(bytes) + bytes;
      std::size_t peak = peak_metric_cache_bytes.load();
      while (peak < current &&
             !peak_metric_cache_bytes.compare_exchange_weak(peak, current)) {
      }
    };

    const auto process_target_leaf = [&](std::size_t target_index) {
      LeafPlanResult &leaf = leaf_results[target_index];
      const int target_node = plan->target_nodes[target_index];
      const Cell &target_cell =
          impl_->target_tree.cells[static_cast<std::size_t>(target_node)];
      const std::vector<std::size_t> &active_target_panels =
          plan->target_panels[target_index];
      TargetLeafWork &work = leaf.work;
      std::vector<PairMetric> metric_cache;
      std::size_t metric_cache_bytes = 0;
      if (plan_builder == "metric-cache" && !work.far_sources.empty()) {
        metric_cache.resize(impl_->source_tree.cells.size());
        metric_cache_bytes = metric_cache.capacity() * sizeof(PairMetric);
        record_metric_cache_bytes(metric_cache_bytes);
      }
      auto sort_leaf_work = [&]() {
        std::sort(work.near_sources.begin(), work.near_sources.end());
        work.near_sources.erase(
            std::unique(work.near_sources.begin(), work.near_sources.end()),
            work.near_sources.end());
        std::sort(work.far_sources.begin(), work.far_sources.end());
        work.far_sources.erase(
            std::unique(work.far_sources.begin(), work.far_sources.end()),
            work.far_sources.end());
      };
      std::function<void(int)> append_source = [&](int source_node) {
        const Cell &source =
            impl_->source_tree.cells[static_cast<std::size_t>(source_node)];
        if (accept_far(impl_->target_tree, impl_->source_tree, target_node,
                       source_node, impl_->config.separation_theta, true)) {
          work.far_sources.push_back(source_node);
        } else if (source.is_leaf()) {
          work.near_sources.push_back(source_node);
        } else {
          append_source(source.left);
          append_source(source.right);
        }
      };

      auto force_direct_source = [&](int source_node) {
        const Cell &source =
            impl_->source_tree.cells[static_cast<std::size_t>(source_node)];
        work.near_sources.push_back(source_node);
        leaf.refined_fallback_sources.push_back(source_node);
        leaf.direct_fallback_panels += source.count();
        leaf.direct_fallback_pairs +=
            active_target_panels.size() * source.count();
      };

      if (geometry_guard == "strict-geometry-mac") {
        bool changed = true;
        while (changed) {
          changed = false;
          std::vector<int> far_snapshot = work.far_sources;
          for (const int source_node : far_snapshot) {
            const Cell &source =
                impl_->source_tree.cells[static_cast<std::size_t>(source_node)];
            const double gap = aabb_gap(target_cell, source);
            const double ratio = strict_geometry_ratio(target_cell, source);
            leaf.geometry_gap_min = std::min(leaf.geometry_gap_min, gap);
            leaf.geometry_ratio_max = std::max(leaf.geometry_ratio_max, ratio);
            if (!(ratio > geometry_theta))
              continue;
            work.far_sources.erase(std::remove(work.far_sources.begin(),
                                               work.far_sources.end(),
                                               source_node),
                                   work.far_sources.end());
            ++leaf.geometry_guard_rejected_cells;
            changed = true;
            if (source.is_leaf()) {
              force_direct_source(source_node);
              leaf.geometry_guard_direct_fallback_pairs +=
                  active_target_panels.size() * source.count();
            } else {
              append_source(source.left);
              append_source(source.right);
            }
            sort_leaf_work();
            break;
          }
        }
      }

      const auto compute_metric = [&](int source_node) {
        const std::size_t source_index = static_cast<std::size_t>(source_node);
        if (plan_builder == "metric-cache" &&
            metric_cache[source_index].source_node == source_node) {
          ++leaf.pair_metric_cache_hits;
          return metric_cache[source_index];
        }
        const auto metric_begin = Clock::now();
        const Cell &source_cell = impl_->source_tree.cells[source_index];
        PairMetric metric;
        metric.source_node = source_node;
        metric.tail_sq_by_probe.assign(probe_moments.size(), 0.0);
        metric.signal_sq_by_probe.assign(probe_moments.size(), 0.0);
        // The translation depends on the two centres only: one build serves
        // every probe.
        const CartesianJcpM2lOperatorP4 m2l =
            build_cartesian_jcp_m2l_operator_p4(
                source_cell.qx, source_cell.qy, source_cell.qz,
                target_cell.qx, target_cell.qy, target_cell.qz,
                impl_->eps_ratio, impl_->kappa);
        for (std::size_t probe = 0; probe < probe_moments.size(); ++probe) {
          const CartesianJcpLocalP4 translated =
              apply_cartesian_jcp_m2l_operator_p4_order(
                  probe_moments[probe][source_index], m2l, local_order);
          double pair_tail_sq = 0.0;
          double pair_signal_sq = 0.0;
          const int tail_low_order = std::max(1, local_order - 1);
          for (const std::size_t target_panel : active_target_panels) {
            const PrototypePanel &target = impl_->panels[target_panel];
            const auto both = evaluate_cartesian_jcp_l2p_two_orders(
                target, target_cell.qx, target_cell.qy, target_cell.qz,
                translated, local_order, tail_low_order);
            const auto &p_high = both[0];
            const auto &p_low = both[1];
            for (const std::pair<int, int> range :
                 {std::pair<int, int>{0, 2}, {2, 4}}) {
              const double high_row =
                  p_high[range.first] + p_high[range.first + 1];
              const double low_row =
                  p_low[range.first] + p_low[range.first + 1];
              const double tail = high_row - low_row;
              const double abs_tail = std::abs(tail);
              const double relative_tail =
                  abs_tail / std::max(1.0e-300, std::abs(high_row));
              pair_tail_sq += tail * tail;
              pair_signal_sq += high_row * high_row;
              metric.max_abs_tail = std::max(metric.max_abs_tail, abs_tail);
              metric.max_relative_tail =
                  std::max(metric.max_relative_tail, relative_tail);
              if (relative_tail > refine_tolerance)
                ++metric.outlier_panels;
            }
          }
          metric.tail_sq_by_probe[probe] = pair_tail_sq;
          metric.signal_sq_by_probe[probe] = pair_signal_sq;
          metric.tail_sq = std::max(metric.tail_sq, pair_tail_sq);
        }
        ++leaf.pair_metric_evaluations;
        leaf.pair_metric_seconds += elapsed_seconds(metric_begin, Clock::now());
        if (plan_builder == "metric-cache")
          metric_cache[source_index] = metric;
        return metric;
      };

      while (true) {
        ++leaf.refinement_iterations;
        std::vector<PairMetric> metrics;
        metrics.reserve(work.far_sources.size());
        std::vector<double> total_tail_sq(probe_moments.size(), 0.0);
        std::vector<double> total_signal_sq(probe_moments.size(), 0.0);
        for (const int source_node : work.far_sources) {
          const PairMetric metric = compute_metric(source_node);
          for (std::size_t probe = 0; probe < probe_moments.size(); ++probe) {
            total_tail_sq[probe] += metric.tail_sq_by_probe[probe];
            total_signal_sq[probe] += metric.signal_sq_by_probe[probe];
          }
          leaf.linf_max_abs_tail =
              std::max(leaf.linf_max_abs_tail, metric.max_abs_tail);
          leaf.linf_max_relative_tail =
              std::max(leaf.linf_max_relative_tail, metric.max_relative_tail);
          leaf.linf_outlier_panels += metric.outlier_panels;
          metrics.push_back(metric);
        }
        double tail_ratio = 0.0;
        if (refine_mode == "operator-tail-linf") {
          for (const PairMetric &metric : metrics)
            tail_ratio = std::max(tail_ratio, metric.max_relative_tail);
        } else {
          for (std::size_t probe = 0; probe < total_tail_sq.size(); ++probe) {
            tail_ratio = std::max(
                tail_ratio,
                std::sqrt(std::max(0.0, total_tail_sq[probe])) /
                    std::max(1.0e-300,
                             std::sqrt(std::max(0.0, total_signal_sq[probe]))));
          }
        }
        if (tail_ratio <= refine_tolerance || metrics.empty())
          break;
        std::sort(metrics.begin(), metrics.end(),
                  [&](const PairMetric &lhs, const PairMetric &rhs) {
                    if (refine_mode == "operator-tail-linf") {
                      if (lhs.max_relative_tail != rhs.max_relative_tail)
                        return lhs.max_relative_tail > rhs.max_relative_tail;
                      if (lhs.max_abs_tail != rhs.max_abs_tail)
                        return lhs.max_abs_tail > rhs.max_abs_tail;
                    }
                    if (lhs.tail_sq != rhs.tail_sq)
                      return lhs.tail_sq > rhs.tail_sq;
                    const int lhs_depth =
                        impl_->source_tree
                            .cells[static_cast<std::size_t>(lhs.source_node)]
                            .depth;
                    const int rhs_depth =
                        impl_->source_tree
                            .cells[static_cast<std::size_t>(rhs.source_node)]
                            .depth;
                    if (lhs_depth != rhs_depth)
                      return lhs_depth < rhs_depth;
                    return lhs.source_node < rhs.source_node;
                  });
        const int selected = metrics.front().source_node;
        work.far_sources.erase(std::remove(work.far_sources.begin(),
                                           work.far_sources.end(), selected),
                               work.far_sources.end());
        const Cell &selected_cell =
            impl_->source_tree.cells[static_cast<std::size_t>(selected)];
        ++leaf.refined_cells;
        if (selected_cell.is_leaf()) {
          force_direct_source(selected);
        } else {
          append_source(selected_cell.left);
          append_source(selected_cell.right);
          leaf.split_children += 2;
        }
        sort_leaf_work();
      }
      leaf.far_pairs = work.far_sources.size();
      for (const int source_node : work.far_sources) {
        const Cell &source =
            impl_->source_tree.cells[static_cast<std::size_t>(source_node)];
        leaf.far_panel_pairs += active_target_panels.size() * source.count();
      }
      if (metric_cache_bytes > 0) {
        std::vector<PairMetric>().swap(metric_cache);
        current_metric_cache_bytes.fetch_sub(metric_cache_bytes);
      }
    };

    const auto refinement_begin = Clock::now();
    if (plan_builder == "metric-cache") {
      const std::size_t workers = std::max<std::size_t>(1, executor.capacity());
      executor.parallel_for(
          plan->target_nodes.size(), workers, 1,
          [&](std::size_t, std::size_t begin, std::size_t end) {
            for (std::size_t target_index = begin; target_index < end;
                 ++target_index)
              process_target_leaf(target_index);
          });
      plan->parallel_target_leaves = plan->target_nodes.size();
    } else {
      for (std::size_t target_index = 0;
           target_index < plan->target_nodes.size(); ++target_index)
        process_target_leaf(target_index);
    }
    plan->refinement_seconds = elapsed_seconds(refinement_begin, Clock::now());
    plan->metric_cache_peak_bytes = peak_metric_cache_bytes.load();
    for (std::size_t target_index = 0; target_index < leaf_results.size();
         ++target_index) {
      LeafPlanResult &leaf = leaf_results[target_index];
      plan->work[target_index] = std::move(leaf.work);
      plan->refined_fallback_sources[target_index] =
          std::move(leaf.refined_fallback_sources);
      plan->refined_cells += leaf.refined_cells;
      plan->split_children += leaf.split_children;
      plan->direct_fallback_panels += leaf.direct_fallback_panels;
      plan->direct_fallback_pairs += leaf.direct_fallback_pairs;
      plan->far_pairs += leaf.far_pairs;
      plan->far_panel_pairs += leaf.far_panel_pairs;
      plan->pair_metric_evaluations += leaf.pair_metric_evaluations;
      plan->pair_metric_cache_hits += leaf.pair_metric_cache_hits;
      plan->refinement_iterations += leaf.refinement_iterations;
      plan->linf_max_abs_tail =
          std::max(plan->linf_max_abs_tail, leaf.linf_max_abs_tail);
      plan->linf_max_relative_tail =
          std::max(plan->linf_max_relative_tail, leaf.linf_max_relative_tail);
      plan->linf_outlier_panels += leaf.linf_outlier_panels;
      plan->geometry_guard_rejected_cells += leaf.geometry_guard_rejected_cells;
      plan->geometry_guard_direct_fallback_pairs +=
          leaf.geometry_guard_direct_fallback_pairs;
      plan->geometry_gap_min =
          std::min(plan->geometry_gap_min, leaf.geometry_gap_min);
      plan->geometry_ratio_max =
          std::max(plan->geometry_ratio_max, leaf.geometry_ratio_max);
      plan->pair_metric_seconds += leaf.pair_metric_seconds;
    }
    if (!std::isfinite(plan->geometry_gap_min))
      plan->geometry_gap_min = 0.0;

    std::uint64_t hash = 0xcbf29ce484222325ULL;
    std::uint64_t tolerance_bits = 0;
    static_assert(sizeof(tolerance_bits) == sizeof(refine_tolerance));
    std::memcpy(&tolerance_bits, &refine_tolerance, sizeof(tolerance_bits));
    hash_mix(hash, static_cast<std::uint64_t>(local_order));
    hash_mix(hash, tolerance_bits);
    hash_mix(hash, sampled_targets_only ? 1ULL : 0ULL);
    hash_mix(hash, sampled_target_hash);
    hash_mix(hash, static_cast<std::uint64_t>(refine_probe_count));
    for (const char ch : refine_probe_mode)
      hash_mix(hash,
               static_cast<std::uint64_t>(static_cast<unsigned char>(ch)));
    for (const char ch : refine_mode)
      hash_mix(hash,
               static_cast<std::uint64_t>(static_cast<unsigned char>(ch)));
    for (const char ch : geometry_guard)
      hash_mix(hash,
               static_cast<std::uint64_t>(static_cast<unsigned char>(ch)));
    std::uint64_t geometry_bits = 0;
    static_assert(sizeof(geometry_bits) == sizeof(geometry_theta));
    std::memcpy(&geometry_bits, &geometry_theta, sizeof(geometry_bits));
    hash_mix(hash, geometry_bits);
    for (std::size_t i = 0; i < plan->target_nodes.size(); ++i) {
      hash_mix(hash, static_cast<std::uint64_t>(plan->target_nodes[i]));
      for (const std::size_t panel : plan->target_panels[i])
        hash_mix(hash, static_cast<std::uint64_t>(panel));
      for (const int node : plan->work[i].near_sources)
        hash_mix(hash, (static_cast<std::uint64_t>(node) << 1) | 1ULL);
      for (const int node : plan->work[i].far_sources)
        hash_mix(hash, static_cast<std::uint64_t>(node) << 1);
    }
    plan->hash = hash;
    plan->plan_seconds = elapsed_seconds(plan_begin, Clock::now());
    impl_->static_local_plan = std::move(plan);
    ++impl_->static_plan_build_count;
  };

  if (!impl_->static_local_plan ||
      impl_->static_local_plan->local_order != local_order ||
      !same_double(impl_->static_local_plan->tolerance, refine_tolerance) ||
      impl_->static_local_plan->refine_mode != refine_mode ||
      impl_->static_local_plan->plan_builder != plan_builder ||
      impl_->static_local_plan->probe_count != refine_probe_count ||
      impl_->static_local_plan->probe_mode != refine_probe_mode ||
      impl_->static_local_plan->sampled_targets_only != sampled_targets_only ||
      impl_->static_local_plan->sampled_target_hash != sampled_target_hash ||
      impl_->static_local_plan->geometry_guard != geometry_guard ||
      !same_double(impl_->static_local_plan->geometry_theta,
                   geometry_guard == "strict-geometry-mac" ? geometry_theta
                                                           : 0.0))
    build_plan();
  StaticLocalP4Plan &plan = *impl_->static_local_plan;

  const bool packed_fallback = fallback_mode == "packed-avx2";
  if ((packed_fallback && !plan.packed_fallback_plan) ||
      (fallback_mode == "scalar" && !plan.scalar_far_built)) {
    std::vector<PackedFallbackLeafInput> leaves;
    leaves.reserve(plan.target_nodes.size());
    for (std::size_t target_index = 0; target_index < plan.target_nodes.size();
         ++target_index) {
      PackedFallbackLeafInput input;
      input.target_panels = plan.target_panels[target_index];
      for (const int source_node : plan.work[target_index].near_sources) {
        const Cell &source_cell =
            impl_->source_tree.cells[static_cast<std::size_t>(source_node)];
        const auto &refined = plan.refined_fallback_sources[target_index];
        input.source_spans.push_back({source_cell.begin, source_cell.count(),
                                      std::find(refined.begin(), refined.end(),
                                                source_node) != refined.end()});
      }
      leaves.push_back(std::move(input));
    }
    const bool node_patch =
        !production_panels->empty() &&
        production_panels->front().node_patch_geometry != nullptr;
    // Each fallback builds the list it applies: the packed loop sums the
    // one-point kernel, so every one-ring and curved near-zone pair needs
    // its correction; the scalar loop evaluates near-list pairs with their
    // curved value, so only the pairs the local expansions take need one.
    if (packed_fallback) {
      plan.packed_fallback_plan =
          std::make_unique<PackedJcpFallbackPlan>(build_packed_jcp_fallback_plan(
              *production_panels, impl_->source_tree.order, leaves, 4, 32));
      plan.packed_node_patch_corrections.clear();
      if (node_patch ||
          production_panels->front().sphere_patch_quadrature != nullptr)
        plan.packed_node_patch_corrections = build_node_patch_near_corrections(
            leaves, impl_->source_tree.order, *production_panels,
            impl_->eps_ratio, impl_->kappa, coupled, executor);
    } else {
      plan.scalar_far_built = true;
      plan.scalar_far_node_patch_corrections.clear();
      if (node_patch ||
          production_panels->front().sphere_patch_quadrature != nullptr)
        plan.scalar_far_node_patch_corrections =
            build_node_patch_near_corrections(
                leaves, impl_->source_tree.order, *production_panels,
                impl_->eps_ratio, impl_->kappa, coupled, executor, false);
    }
  }

  const std::size_t budget_bytes =
      cache_budget_mb * std::size_t{1024} * std::size_t{1024};
  const bool tiled_cache = cache_mode == "green-jet-tiled";
  const bool global_cache_requested =
      cache_mode == "kernel-jet" || cache_mode == "green-jet-compressed";
  const bool cache_requested = global_cache_requested || tiled_cache;
  if (global_cache_requested && plan.cache_ready &&
      plan.cache_layout != cache_mode) {
    plan.cached_operators.clear();
    plan.compressed_cached_green_coefficients.clear();
    plan.compressed_cache_offsets.clear();
    plan.compressed_green_term_count = 0;
    plan.cache_ready = false;
    plan.cache_bytes = 0;
    plan.cache_operator_bytes = 0;
    plan.cache_uncompressed_estimated_bytes = 0;
    plan.cache_compression_ratio = 1.0;
    plan.cache_layout = "off";
  }
  if (global_cache_requested && !plan.cache_ready) {
    const std::size_t operator_bytes =
        cache_mode == "green-jet-compressed"
            ? cartesian_compressed_green_operator_bytes_order(local_order)
            : sizeof(CartesianJcpM2lOperatorP4);
    const std::size_t offset_bytes =
        cache_mode == "green-jet-compressed" && plan.far_pairs > 0
            ? (plan.work.size() + 1) * sizeof(std::size_t)
            : 0;
    const std::size_t required_bytes =
        plan.far_pairs * operator_bytes + offset_bytes;
    if (required_bytes > budget_bytes)
      throw std::runtime_error(
          "cache_budget_exceeded: required=" + std::to_string(required_bytes) +
          " budget=" + std::to_string(budget_bytes));
    const auto cache_begin = Clock::now();
    if (cache_mode == "green-jet-compressed") {
      const std::size_t green_terms =
          cartesian_compressed_green_jet_term_count_order(local_order);
      plan.compressed_green_term_count = green_terms;
      if (plan.far_pairs > 0) {
        plan.compressed_cache_offsets.assign(plan.work.size() + 1, 0);
        for (std::size_t target_index = 0; target_index < plan.work.size();
             ++target_index)
          plan.compressed_cache_offsets[target_index + 1] =
              plan.compressed_cache_offsets[target_index] +
              plan.work[target_index].far_sources.size();
        if (plan.compressed_cache_offsets.back() != plan.far_pairs)
          throw std::logic_error("compressed cache: far-pair count mismatch");
        plan.compressed_cached_green_coefficients.assign(
            plan.far_pairs * 2 * green_terms, 0.0);
        // Every target leaf fills its own slice, so the leaves are built in
        // parallel; each operator is the one the serial loop built.
        executor.parallel_for(
            plan.work.size(), std::max<std::size_t>(1, executor.capacity()), 1,
            [&](std::size_t, std::size_t begin, std::size_t end) {
              for (std::size_t target_index = begin; target_index < end;
                   ++target_index) {
                const Cell &target_cell =
                    impl_->target_tree.cells[static_cast<std::size_t>(
                        plan.target_nodes[target_index])];
                std::size_t far_pair_index =
                    plan.compressed_cache_offsets[target_index];
                for (const int source_node :
                     plan.work[target_index].far_sources) {
                  const Cell &source_cell =
                      impl_->source_tree.cells[static_cast<std::size_t>(
                          source_node)];
                  const std::size_t base = far_pair_index * 2 * green_terms;
                  build_cartesian_jcp_compressed_green_operator_p4_order(
                      source_cell.qx, source_cell.qy, source_cell.qz,
                      target_cell.qx, target_cell.qy, target_cell.qz,
                      impl_->kappa, local_order,
                      std::span<double>(
                          plan.compressed_cached_green_coefficients.data() +
                              base,
                          green_terms),
                      std::span<double>(
                          plan.compressed_cached_green_coefficients.data() +
                              base + green_terms,
                          green_terms));
                  ++far_pair_index;
                }
              }
            });
      }
    } else {
      plan.cached_operators.resize(plan.work.size());
      executor.parallel_for(
          plan.work.size(), std::max<std::size_t>(1, executor.capacity()), 1,
          [&](std::size_t, std::size_t begin, std::size_t end) {
            for (std::size_t target_index = begin; target_index < end;
                 ++target_index) {
              const Cell &target_cell =
                  impl_->target_tree.cells[static_cast<std::size_t>(
                      plan.target_nodes[target_index])];
              auto &operators = plan.cached_operators[target_index];
              operators.reserve(plan.work[target_index].far_sources.size());
              for (const int source_node :
                   plan.work[target_index].far_sources) {
                const Cell &source_cell =
                    impl_->source_tree.cells[static_cast<std::size_t>(
                        source_node)];
                operators.push_back(build_cartesian_jcp_m2l_operator_p4(
                    source_cell.qx, source_cell.qy, source_cell.qz,
                    target_cell.qx, target_cell.qy, target_cell.qz,
                    impl_->eps_ratio, impl_->kappa));
              }
            }
          });
    }
    plan.cache_bytes = required_bytes;
    plan.cache_operator_bytes = operator_bytes;
    plan.cache_uncompressed_estimated_bytes =
        plan.far_pairs * sizeof(CartesianJcpM2lOperatorP4);
    plan.cache_compression_ratio =
        required_bytes > 0
            ? static_cast<double>(plan.cache_uncompressed_estimated_bytes) /
                  static_cast<double>(required_bytes)
            : 1.0;
    plan.cache_layout = cache_mode;
    plan.cache_ready = true;
    plan.cache_seconds = elapsed_seconds(cache_begin, Clock::now());
    ++impl_->local_cache_build_count;
  }
  if (global_cache_requested && plan.cache_layout == cache_mode &&
      plan.cache_bytes > budget_bytes)
    throw std::runtime_error(
        "cache_budget_exceeded: existing cache exceeds budget");

  const std::size_t workers = std::max<std::size_t>(1, executor.capacity());
  const std::size_t tiled_operator_bytes =
      tiled_cache ? cartesian_compressed_green_operator_bytes_order(local_order)
                  : 0;
  const std::size_t tiled_cache_bytes =
      tiled_cache ? workers * tiled_operator_bytes : 0;
  if (tiled_cache && tiled_cache_bytes > budget_bytes)
    throw std::runtime_error(
        "cache_budget_exceeded: required=" + std::to_string(tiled_cache_bytes) +
        " budget=" + std::to_string(budget_bytes));
  if (tiled_cache) {
    plan.tiled_green_term_count =
        cartesian_compressed_green_jet_term_count_order(local_order);
    plan.tiled_operator_bytes = tiled_operator_bytes;
    plan.tiled_peak_bytes = tiled_cache_bytes;
  }

  JcpLocalP4CandidateResult result;
  result.refine_mode = refine_mode;
  result.local_source_order = local_order;
  result.local_target_order = local_order;
  result.local_kernel_derivative_order = 2 * local_order;
  result.refine_tolerance = refine_tolerance;
  result.cache_mode = cache_mode;
  result.cache_layout =
      tiled_cache ? "green-jet-tiled"
                  : (global_cache_requested ? plan.cache_layout : "off");
  result.plan_hash = plan.hash;
  result.plan_build_count = impl_->static_plan_build_count;
  result.static_probe_count = plan.probe_count;
  result.refine_probe_mode = plan.probe_mode;
  result.solution_probe_used = plan.solution_probe_used;
  result.linf_max_abs_tail = plan.linf_max_abs_tail;
  result.linf_max_relative_tail = plan.linf_max_relative_tail;
  result.linf_outlier_panels = plan.linf_outlier_panels;
  result.geometry_guard = plan.geometry_guard;
  result.geometry_theta = plan.geometry_theta;
  result.geometry_guard_rejected_cells = plan.geometry_guard_rejected_cells;
  result.geometry_guard_direct_fallback_pairs =
      plan.geometry_guard_direct_fallback_pairs;
  result.geometry_gap_min = plan.geometry_gap_min;
  result.geometry_ratio_max = plan.geometry_ratio_max;
  result.refined_cells = plan.refined_cells;
  result.split_children = plan.split_children;
  result.direct_fallback_panels = plan.direct_fallback_panels;
  result.direct_fallback_pairs = plan.direct_fallback_pairs;
  result.far_pairs = plan.far_pairs;
  result.far_panel_pairs = plan.far_panel_pairs;
  result.cache_operator_count =
      tiled_cache
          ? workers
          : (global_cache_requested && plan.cache_ready ? plan.far_pairs : 0);
  result.cache_bytes =
      tiled_cache
          ? tiled_cache_bytes
          : (global_cache_requested && plan.cache_ready ? plan.cache_bytes : 0);
  result.cache_operator_bytes =
      tiled_cache ? tiled_operator_bytes
                  : (global_cache_requested ? plan.cache_operator_bytes : 0);
  result.cache_uncompressed_estimated_bytes =
      cache_requested ? plan.far_pairs * sizeof(CartesianJcpM2lOperatorP4) : 0;
  result.cache_compression_ratio =
      result.cache_bytes > 0
          ? static_cast<double>(result.cache_uncompressed_estimated_bytes) /
                static_cast<double>(result.cache_bytes)
          : 1.0;
  result.cache_far_pair_offsets = cache_mode == "green-jet-compressed"
                                      ? plan.compressed_cache_offsets.size()
                                      : 0;
  // Tiled mode retains only per-worker scratch jets. Every far interaction
  // rebuilds its coefficients during this apply, so it is not a reusable
  // geometry cache and must not claim a cache build or cache hit.
  result.cache_build_count =
      global_cache_requested ? impl_->local_cache_build_count : 0;
  result.cache_tile_bytes = tiled_cache ? tiled_operator_bytes : 0;
  result.cache_tile_count = tiled_cache ? workers : 0;
  result.cache_tile_builds = tiled_cache ? plan.far_pairs : 0;
  result.fallback_requested = fallback_mode;
  result.fallback_effective = fallback_mode;
  if (plan.packed_fallback_plan) {
    const PackedFallbackPlanMetrics metrics =
        plan.packed_fallback_plan->metrics();
    result.fallback_worklist_hash = metrics.worklist_hash;
    result.fallback_worklist_build_count = metrics.build_count;
    result.fallback_tasks = metrics.task_count;
    result.fallback_metadata_bytes = metrics.metadata_bytes;
    result.fallback_base_near_pairs = metrics.base_near_pairs;
    result.fallback_refined_pairs = metrics.refined_fallback_pairs;
  }
  result.plan_seconds = plan.plan_seconds;
  result.plan_builder = plan.plan_builder;
  result.plan_probe_moment_seconds = plan.probe_moment_seconds;
  result.plan_pair_metric_seconds = plan.pair_metric_seconds;
  result.plan_refinement_seconds = plan.refinement_seconds;
  result.plan_pair_metric_evaluations = plan.pair_metric_evaluations;
  result.plan_pair_metric_cache_hits = plan.pair_metric_cache_hits;
  result.plan_refinement_iterations = plan.refinement_iterations;
  result.plan_metric_cache_peak_bytes = plan.metric_cache_peak_bytes;
  result.plan_parallel_target_leaves = plan.parallel_target_leaves;
  result.cache_seconds = tiled_cache ? 0.0 : plan.cache_seconds;
  result.phi.assign(n, 0.0);
  result.phin.assign(n, 0.0);

  const auto apply_begin = Clock::now();
  const auto moment_begin = Clock::now();
  std::vector<CartesianJcpMoments> &moments = impl_->local_p4_moment_workspace;
  build_cartesian_node_moments_into(impl_->source_tree, impl_->panels,
                                    phi_values, phin_values, local_order,
                                    executor, moments);
  result.moment_seconds = elapsed_seconds(moment_begin, Clock::now());

  std::vector<CartesianJcpLocalP4> &locals = impl_->local_p4_local_workspace;
  locals.assign(plan.work.size(), CartesianJcpLocalP4{});
  const auto m2l_begin = Clock::now();
  const std::size_t tiled_green_terms =
      tiled_cache ? plan.tiled_green_term_count : 0;
  std::vector<std::vector<double>> &worker_coulomb =
      impl_->local_p4_tiled_coulomb_workspace;
  std::vector<std::vector<double>> &worker_yukawa =
      impl_->local_p4_tiled_yukawa_workspace;
  if (tiled_cache) {
    worker_coulomb.resize(workers);
    worker_yukawa.resize(workers);
    for (std::size_t worker = 0; worker < workers; ++worker) {
      worker_coulomb[worker].assign(tiled_green_terms, 0.0);
      worker_yukawa[worker].assign(tiled_green_terms, 0.0);
    }
  } else {
    worker_coulomb.clear();
    worker_yukawa.clear();
  }
  executor.parallel_for(
      plan.work.size(), workers, 1,
      [&](std::size_t worker_id, std::size_t begin, std::size_t end) {
        std::vector<double> *tiled_coulomb =
            tiled_cache && !worker_coulomb.empty()
                ? &worker_coulomb[worker_id % worker_coulomb.size()]
                : nullptr;
        std::vector<double> *tiled_yukawa =
            tiled_cache && !worker_yukawa.empty()
                ? &worker_yukawa[worker_id % worker_yukawa.size()]
                : nullptr;
        for (std::size_t target_index = begin; target_index < end;
             ++target_index) {
          const Cell &target_cell =
              impl_->target_tree.cells[static_cast<std::size_t>(
                  plan.target_nodes[target_index])];
          CartesianJcpLocalP4 accumulated;
          for (std::size_t far_index = 0;
               far_index < plan.work[target_index].far_sources.size();
               ++far_index) {
            const int source_node =
                plan.work[target_index].far_sources[far_index];
            CartesianJcpLocalP4 translated;
            if (cache_mode == "kernel-jet") {
              translated = apply_cartesian_jcp_m2l_operator_p4_order(
                  moments[static_cast<std::size_t>(source_node)],
                  plan.cached_operators[target_index][far_index], local_order);
            } else if (cache_mode == "green-jet-compressed") {
              const std::size_t far_pair_offset =
                  plan.compressed_cache_offsets[target_index] + far_index;
              const std::size_t green_terms = plan.compressed_green_term_count;
              const std::size_t offset = far_pair_offset * 2 * green_terms;
              translated =
                  apply_cartesian_jcp_compressed_green_operator_p4_order(
                      moments[static_cast<std::size_t>(source_node)],
                      std::span<const double>(
                          plan.compressed_cached_green_coefficients.data() +
                              offset,
                          green_terms),
                      std::span<const double>(
                          plan.compressed_cached_green_coefficients.data() +
                              offset + green_terms,
                          green_terms),
                      impl_->eps_ratio, local_order);
            } else if (cache_mode == "green-jet-tiled") {
              const Cell &source_cell =
                  impl_->source_tree
                      .cells[static_cast<std::size_t>(source_node)];
              build_cartesian_jcp_compressed_green_operator_p4_order(
                  source_cell.qx, source_cell.qy, source_cell.qz,
                  target_cell.qx, target_cell.qy, target_cell.qz, impl_->kappa,
                  local_order,
                  std::span<double>(tiled_coulomb->data(),
                                    tiled_coulomb->size()),
                  std::span<double>(tiled_yukawa->data(),
                                    tiled_yukawa->size()));
              translated =
                  apply_cartesian_jcp_compressed_green_operator_p4_order(
                      moments[static_cast<std::size_t>(source_node)],
                      std::span<const double>(tiled_coulomb->data(),
                                              tiled_coulomb->size()),
                      std::span<const double>(tiled_yukawa->data(),
                                              tiled_yukawa->size()),
                      impl_->eps_ratio, local_order);
            } else {
              const Cell &source_cell =
                  impl_->source_tree
                      .cells[static_cast<std::size_t>(source_node)];
              translated = apply_cartesian_jcp_m2l_operator_p4_order(
                  moments[static_cast<std::size_t>(source_node)],
                  build_cartesian_jcp_m2l_operator_p4(
                      source_cell.qx, source_cell.qy, source_cell.qz,
                      target_cell.qx, target_cell.qy, target_cell.qz,
                      impl_->eps_ratio, impl_->kappa),
                  local_order);
            }
            accumulate_cartesian_jcp_local(accumulated, translated);
          }
          locals[target_index] = accumulated;
        }
      });
  result.m2l_seconds = elapsed_seconds(m2l_begin, Clock::now());
  if (global_cache_requested)
    impl_->local_cache_hits += plan.far_pairs;
  result.cache_hits = global_cache_requested ? impl_->local_cache_hits : 0;
  if (tiled_cache)
    result.cache_tile_seconds = result.m2l_seconds;

  const auto l2p_begin = Clock::now();
  executor.parallel_for(
      plan.work.size(), workers, 1,
      [&](std::size_t, std::size_t begin, std::size_t end) {
        for (std::size_t target_index = begin; target_index < end;
             ++target_index) {
          const Cell &target_cell =
              impl_->target_tree.cells[static_cast<std::size_t>(
                  plan.target_nodes[target_index])];
          for (const std::size_t target_panel :
               plan.target_panels[target_index]) {
            const auto channels = evaluate_cartesian_jcp_l2p_order(
                impl_->panels[target_panel], target_cell.qx, target_cell.qy,
                target_cell.qz, locals[target_index], local_order);
            result.phi[target_panel] = channel_phi(channels);
            result.phin[target_panel] = coupled ? channel_phin(channels) : 0.0;
          }
        }
      });
  result.l2p_seconds = elapsed_seconds(l2p_begin, Clock::now());

  if (fallback_mode == "packed-avx2") {
    PackedFallbackApplyResult fallback = apply_packed_jcp_fallback(
        *plan.packed_fallback_plan, *production_panels, phi_values, phin_values,
        impl_->eps_ratio, impl_->kappa, coupled, near_integration, quad_buffer,
        fallback_mode, executor);
    for (std::size_t panel = 0; panel < n; ++panel) {
      result.phi[panel] += fallback.phi[panel];
      if (coupled)
        result.phin[panel] += fallback.phin[panel];
    }
    apply_node_patch_near_corrections(
        plan.packed_node_patch_corrections, phi_values, phin_values,
        result.phi.data(), coupled ? result.phin.data() : nullptr);
    result.direct_fallback_seconds = fallback.apply_seconds;
    result.fallback_effective = fallback.effective_kernel;
    result.fallback_vectorized_sources = fallback.vectorized_sources;
    result.fallback_vectorized_lanes = fallback.vectorized_lanes;
    result.fallback_scalar_self_pairs = fallback.scalar_self_pairs;
    result.finite_outputs = result.finite_outputs && fallback.finite_outputs;
  } else {
    const auto direct_begin = Clock::now();
    executor.parallel_for(
        plan.work.size(), workers, 1,
        [&](std::size_t, std::size_t begin, std::size_t end) {
          for (std::size_t target_index = begin; target_index < end;
               ++target_index) {
            for (const std::size_t target_panel :
                 plan.target_panels[target_index]) {
              JcpChannelValues direct{};
              for (const int source_node :
                   plan.work[target_index].near_sources) {
                const Cell &source_cell =
                    impl_->source_tree
                        .cells[static_cast<std::size_t>(source_node)];
                for (std::size_t source_pos = source_cell.begin;
                     source_pos < source_cell.end; ++source_pos) {
                  const std::size_t source_panel =
                      impl_->source_tree.order[source_pos];
                  add_channels(
                      direct,
                      jcp_near_kernel_pair_channels(
                          (*production_panels)[target_panel], target_panel,
                          (*production_panels)[source_panel], source_panel,
                          phi_values[source_panel], phin_values[source_panel],
                          impl_->eps_ratio, impl_->kappa, coupled,
                          near_integration, quad_buffer, nullptr));
                }
              }
              result.phi[target_panel] += channel_phi(direct);
              if (coupled)
                result.phin[target_panel] += channel_phin(direct);
            }
          }
        });
    apply_node_patch_near_corrections(
        plan.scalar_far_node_patch_corrections, phi_values, phin_values,
        result.phi.data(), coupled ? result.phin.data() : nullptr);
    result.direct_fallback_seconds =
        elapsed_seconds(direct_begin, Clock::now());
  }
  result.apply_seconds = elapsed_seconds(apply_begin, Clock::now());
  result.finite_outputs =
      result.finite_outputs &&
      std::all_of(result.phi.begin(), result.phi.end(),
                  [](double value) { return std::isfinite(value); }) &&
      std::all_of(result.phin.begin(), result.phin.end(),
                  [](double value) { return std::isfinite(value); });
  if (!result.finite_outputs)
    result.status = "non_finite_output";
  return result;
}

JcpLocalP4CandidateResult FmmM2lAnalyticContext::apply_local_p4_candidate(
    const std::vector<double> &phi_values,
    const std::vector<double> &phin_values, TitanExecutor &executor,
    bool coupled, const std::string &near_integration,
    const std::string &quad_buffer,
    const std::vector<TitanPBE::Tabi::Panel> *production_panels,
    const std::string &refine_mode, double refine_tolerance,
    const std::string &cache_mode, std::size_t cache_budget_mb,
    const std::string &fallback_mode, const std::string &plan_builder,
    std::size_t refine_probe_count, const std::string &refine_probe_mode,
    const std::string &geometry_guard, double geometry_theta,
    const std::vector<std::size_t> *sampled_target_indices) const {
  return apply_local_cartesian_candidate(
      phi_values, phin_values, executor, coupled, near_integration, quad_buffer,
      production_panels, refine_mode, refine_tolerance, cache_mode,
      cache_budget_mb, fallback_mode, plan_builder, refine_probe_count,
      refine_probe_mode, geometry_guard, geometry_theta,
      sampled_target_indices);
}

JcpLocalP4CandidateResult FmmM2lAnalyticContext::prepare_local_p4_operator(
    const std::vector<TitanPBE::Tabi::Panel> &production_panels,
    TitanExecutor &executor, const JcpLocalP4OperatorOptions &options) {
  if (impl_ == nullptr)
    throw std::logic_error("JCP analytic FMM/M2L context is not initialized");
  if (production_panels.size() != impl_->panels.size())
    throw std::invalid_argument(
        "JCP local p4 prepared geometry must match panels");

  impl_->local_p4_prepared = false;
  impl_->local_p4_options = options;
  impl_->local_p4_production_panels = production_panels;
  const std::vector<double> zero(impl_->panels.size(), 0.0);
  JcpLocalP4CandidateResult result = apply_local_cartesian_candidate(
      zero, zero, executor, true, options.near_integration, options.quad_buffer,
      &impl_->local_p4_production_panels, options.refine_mode,
      options.refine_tolerance, options.cache_mode, options.cache_budget_mb,
      options.fallback_mode, options.plan_builder, options.refine_probe_count,
      options.refine_probe_mode, options.geometry_guard,
      options.geometry_theta);
  impl_->local_p4_prepared = true;
  return result;
}

JcpLocalP4CandidateResult
FmmM2lAnalyticContext::apply_prepared_local_p4_operator(
    const std::vector<double> &phi, const std::vector<double> &phin,
    TitanExecutor &executor, bool coupled) const {
  if (impl_ == nullptr || !impl_->local_p4_prepared)
    throw std::logic_error("local_p4_operator_not_prepared");
  const JcpLocalP4OperatorOptions &options = impl_->local_p4_options;
  return apply_local_cartesian_candidate(
      phi, phin, executor, coupled, options.near_integration,
      options.quad_buffer, &impl_->local_p4_production_panels,
      options.refine_mode, options.refine_tolerance, options.cache_mode,
      options.cache_budget_mb, options.fallback_mode, options.plan_builder,
      options.refine_probe_count, options.refine_probe_mode,
      options.geometry_guard, options.geometry_theta);
}

void assemble_jcp_local_p4_system_matvec(
    const JcpLocalP4CandidateResult &interaction,
    const std::vector<double> &unknowns, double lhs_phi, double lhs_phin,
    bool coupled, std::vector<double> &out) {
  const std::size_t n = interaction.phi.size();
  if (interaction.phin.size() != n || unknowns.size() < (coupled ? 2 * n : n))
    throw std::invalid_argument("gmres_p4_incomplete_unknown_vector");
  out.assign(coupled ? 2 * n : n, 0.0);
  for (std::size_t panel = 0; panel < n; ++panel) {
    out[panel] = lhs_phi * unknowns[panel] - interaction.phi[panel];
    if (coupled)
      out[n + panel] = lhs_phin * unknowns[n + panel] - interaction.phin[panel];
  }
}

JcpOperatorAttributionResult
FmmM2lAnalyticContext::attribute_local_p4_candidate(
    const std::vector<double> &phi_values,
    const std::vector<double> &phin_values, std::size_t sample_count,
    const std::string &probe, TitanExecutor &executor, bool coupled,
    const std::string &near_integration, const std::string &quad_buffer,
    const std::vector<TitanPBE::Tabi::Panel> *production_panels,
    const std::string &refine_mode, double refine_tolerance,
    const std::string &cache_mode, std::size_t cache_budget_mb,
    std::size_t repeat_count, const std::string &fallback_mode,
    const std::string &plan_builder, std::size_t refine_probe_count,
    const std::string &refine_probe_mode, const std::string &geometry_guard,
    double geometry_theta) const {
  repeat_count = std::max<std::size_t>(1, repeat_count);
  JcpLocalP4CandidateResult candidate;
  std::vector<double> apply_times;
  apply_times.reserve(repeat_count);
  for (std::size_t repeat = 0; repeat < repeat_count; ++repeat) {
    candidate = apply_local_p4_candidate(
        phi_values, phin_values, executor, coupled, near_integration,
        quad_buffer, production_panels, refine_mode, refine_tolerance,
        cache_mode, cache_budget_mb, fallback_mode, plan_builder,
        refine_probe_count, refine_probe_mode, geometry_guard, geometry_theta);
    apply_times.push_back(candidate.apply_seconds);
  }
  std::sort(apply_times.begin(), apply_times.end());
  const double median_apply = apply_times[apply_times.size() / 2];
  JcpOperatorAttributionResult result;
  result.approximation_mode =
      cache_mode == "green-jet-compressed"
          ? "local_m2l_p4_static_green_cached"
          : (cache_mode == "kernel-jet" ? "local_m2l_p4_static_cached"
                                        : "local_m2l_p4_static");
  result.probe = probe;
  result.configured_expansion_order = impl_->config.expansion_order;
  result.implemented_far_moment_degree = candidate.local_source_order;
  result.local_source_order = candidate.local_source_order;
  result.local_target_order = candidate.local_target_order;
  result.local_kernel_derivative_order =
      candidate.local_kernel_derivative_order;
  result.local_refine_mode = candidate.refine_mode;
  result.local_refine_tolerance = candidate.refine_tolerance;
  result.local_refined_cells = candidate.refined_cells;
  result.local_split_children = candidate.split_children;
  result.local_direct_fallback_panels = candidate.direct_fallback_panels;
  result.local_direct_fallback_pairs = candidate.direct_fallback_pairs;
  result.local_far_pairs_after = candidate.far_pairs;
  result.local_plan_hash = candidate.plan_hash;
  result.local_plan_build_count = candidate.plan_build_count;
  result.local_static_probe_count = candidate.static_probe_count;
  result.local_refine_probe_mode = candidate.refine_probe_mode;
  result.local_solution_probe_used = candidate.solution_probe_used;
  result.local_linf_max_abs_tail = candidate.linf_max_abs_tail;
  result.local_linf_max_relative_tail = candidate.linf_max_relative_tail;
  result.local_linf_outlier_panels = candidate.linf_outlier_panels;
  result.local_geometry_guard = candidate.geometry_guard;
  result.local_geometry_theta = candidate.geometry_theta;
  result.local_geometry_guard_rejected_cells =
      candidate.geometry_guard_rejected_cells;
  result.local_geometry_guard_direct_fallback_pairs =
      candidate.geometry_guard_direct_fallback_pairs;
  result.local_geometry_gap_min = candidate.geometry_gap_min;
  result.local_geometry_ratio_max = candidate.geometry_ratio_max;
  result.local_cache_mode = candidate.cache_mode;
  result.local_cache_layout = candidate.cache_layout;
  result.local_cache_operators = candidate.cache_operator_count;
  result.local_cache_bytes = candidate.cache_bytes;
  result.local_cache_operator_bytes = candidate.cache_operator_bytes;
  result.local_cache_uncompressed_estimated_bytes =
      candidate.cache_uncompressed_estimated_bytes;
  result.local_cache_compression_ratio = candidate.cache_compression_ratio;
  result.local_cache_far_pair_offsets = candidate.cache_far_pair_offsets;
  result.local_cache_build_count = candidate.cache_build_count;
  result.local_cache_hits = candidate.cache_hits;
  result.local_cache_tile_bytes = candidate.cache_tile_bytes;
  result.local_cache_tile_count = candidate.cache_tile_count;
  result.local_cache_tile_builds = candidate.cache_tile_builds;
  result.local_cache_tile_seconds = candidate.cache_tile_seconds;
  result.local_plan_seconds = candidate.plan_seconds;
  result.local_plan_builder = candidate.plan_builder;
  result.local_plan_probe_moment_seconds = candidate.plan_probe_moment_seconds;
  result.local_plan_pair_metric_seconds = candidate.plan_pair_metric_seconds;
  result.local_plan_refinement_seconds = candidate.plan_refinement_seconds;
  result.local_plan_pair_metric_evaluations =
      candidate.plan_pair_metric_evaluations;
  result.local_plan_pair_metric_cache_hits =
      candidate.plan_pair_metric_cache_hits;
  result.local_plan_refinement_iterations =
      candidate.plan_refinement_iterations;
  result.local_plan_metric_cache_peak_bytes =
      candidate.plan_metric_cache_peak_bytes;
  result.local_plan_parallel_target_leaves =
      candidate.plan_parallel_target_leaves;
  result.local_cache_seconds = candidate.cache_seconds;
  result.moment_build_seconds = candidate.moment_seconds;
  result.m2l_seconds = candidate.m2l_seconds;
  result.l2p_seconds = candidate.l2p_seconds;
  result.local_candidate_apply_seconds = candidate.apply_seconds;
  result.local_candidate_repeat_count = repeat_count;
  result.local_candidate_median_apply_seconds = median_apply;
  result.local_fallback_requested = candidate.fallback_requested;
  result.local_fallback_effective = candidate.fallback_effective;
  result.local_fallback_worklist_hash = candidate.fallback_worklist_hash;
  result.local_fallback_worklist_build_count =
      candidate.fallback_worklist_build_count;
  result.local_fallback_tasks = candidate.fallback_tasks;
  result.local_fallback_metadata_bytes = candidate.fallback_metadata_bytes;
  result.local_fallback_base_near_pairs = candidate.fallback_base_near_pairs;
  result.local_fallback_refined_pairs = candidate.fallback_refined_pairs;
  result.local_fallback_vectorized_sources =
      candidate.fallback_vectorized_sources;
  result.local_fallback_vectorized_lanes = candidate.fallback_vectorized_lanes;
  result.local_fallback_scalar_self_pairs =
      candidate.fallback_scalar_self_pairs;
  result.local_fallback_seconds = candidate.direct_fallback_seconds;
  result.finite_outputs = candidate.finite_outputs;

  const std::size_t n = impl_->panels.size();
  const std::size_t target_count =
      std::min(n, std::max<std::size_t>(1, sample_count));
  std::size_t previous = std::numeric_limits<std::size_t>::max();
  for (std::size_t sample = 0; sample < target_count; ++sample) {
    std::size_t target =
        ((2 * sample + 1) * n) / std::max<std::size_t>(1, 2 * target_count);
    target = std::min(target, n - 1);
    if (target != previous) {
      result.sampled_targets.push_back(target);
      previous = target;
    }
  }
  const auto direct_begin = Clock::now();
  for (const std::size_t target_panel : result.sampled_targets) {
    JcpOperatorAttributionRecord record;
    record.probe = probe;
    record.target_panel = target_panel;
    record.region = "far";
    record.source_node = -3;
    record.source_depth = -1;
    record.source_panels = n;
    record.approximate = {candidate.phi[target_panel], 0.0,
                          candidate.phin[target_panel], 0.0};
    for (std::size_t source_panel = 0; source_panel < n; ++source_panel) {
      add_channels(record.direct,
                   jcp_near_kernel_pair_channels(
                       (*production_panels)[target_panel], target_panel,
                       (*production_panels)[source_panel], source_panel,
                       phi_values[source_panel], phin_values[source_panel],
                       impl_->eps_ratio, impl_->kappa, coupled,
                       near_integration, quad_buffer, nullptr));
    }
    result.records.push_back(record);
    ++result.far_records;
  }
  result.local_candidate_direct_reference_seconds =
      elapsed_seconds(direct_begin, Clock::now());
  result.recomposition_rel_error = 0.0;
  return result;
}

JcpOperatorAttributionResult FmmM2lAnalyticContext::attribute_cartesian(
    const std::vector<double> &phi_values,
    const std::vector<double> &phin_values, std::size_t sample_count,
    const std::string &probe, TitanExecutor &executor, bool coupled,
    const std::string &near_integration, const std::string &quad_buffer,
    const std::vector<TitanPBE::Tabi::Panel> *production_panels) const {
  return attribute_impl(phi_values, phin_values, sample_count, probe, executor,
                        coupled, near_integration, quad_buffer,
                        production_panels, 1, "off", 4.0e-3);
}

JcpOperatorAttributionResult FmmM2lAnalyticContext::attribute_local_p4(
    const std::vector<double> &phi_values,
    const std::vector<double> &phin_values, std::size_t sample_count,
    const std::string &probe, TitanExecutor &executor, bool coupled,
    const std::string &near_integration, const std::string &quad_buffer,
    const std::vector<TitanPBE::Tabi::Panel> *production_panels,
    const std::string &refine_mode, double refine_tolerance) const {
  return attribute_impl(phi_values, phin_values, sample_count, probe, executor,
                        coupled, near_integration, quad_buffer,
                        production_panels, 2, refine_mode, refine_tolerance);
}

JcpOperatorAttributionResult FmmM2lAnalyticContext::attribute_impl(
    const std::vector<double> &phi_values,
    const std::vector<double> &phin_values, std::size_t sample_count,
    const std::string &probe, TitanExecutor &executor, bool coupled,
    const std::string &near_integration, const std::string &quad_buffer,
    const std::vector<TitanPBE::Tabi::Panel> *production_panels,
    int approximation_mode, const std::string &refine_mode,
    double refine_tolerance) const {
  if (impl_ == nullptr) {
    throw std::logic_error("JCP analytic FMM/M2L context is not initialized");
  }
  const std::size_t n = impl_->panels.size();
  if (phi_values.size() != n || phin_values.size() != n) {
    throw std::invalid_argument(
        "JCP operator attribution density vectors must match panels");
  }
  if (production_panels == nullptr || production_panels->size() != n) {
    throw std::invalid_argument(
        "JCP operator attribution requires production panel geometry");
  }

  const bool cartesian = approximation_mode != 0;
  const bool local_mode = approximation_mode == 2;
  const bool refine_local = local_mode && refine_mode == "p3-p4-tail";
  if (local_mode && refine_mode != "off" && !refine_local) {
    throw std::invalid_argument(
        "JCP local M2L refinement must be off or p3-p4-tail");
  }
  if (refine_local &&
      (!std::isfinite(refine_tolerance) || refine_tolerance < 1.0e-4 ||
       refine_tolerance > 5.0e-2)) {
    throw std::invalid_argument(
        "JCP local M2L refinement tolerance must be in [1e-4, 5e-2]");
  }
  JcpOperatorAttributionResult result;
  result.approximation_mode =
      local_mode
          ? "local_m2l_p4"
          : (cartesian
                 ? "cartesian_p" + std::to_string(impl_->config.expansion_order)
                 : "legacy_degree1");
  result.probe = probe;
  result.local_refine_mode = local_mode ? refine_mode : "off";
  result.local_refine_tolerance = refine_tolerance;
  result.configured_expansion_order = impl_->config.expansion_order;
  result.implemented_far_moment_degree =
      cartesian ? impl_->config.expansion_order : 1;
  const std::size_t target_count =
      std::min(n, std::max<std::size_t>(1, sample_count));
  result.sampled_targets.reserve(target_count);
  std::size_t previous = std::numeric_limits<std::size_t>::max();
  for (std::size_t sample = 0; sample < target_count; ++sample) {
    std::size_t target =
        ((2 * sample + 1) * n) / std::max<std::size_t>(1, 2 * target_count);
    target = std::min(target, n - 1);
    if (target != previous) {
      result.sampled_targets.push_back(target);
      previous = target;
    }
  }

  std::vector<int> target_leaf_by_panel(n, -1);
  for (const int target_node : impl_->target_tree.target_leaf_nodes) {
    const Cell &target_cell =
        impl_->target_tree.cells[static_cast<std::size_t>(target_node)];
    for (std::size_t pos = target_cell.begin; pos < target_cell.end; ++pos) {
      const std::size_t panel = impl_->target_tree.order[pos];
      if (panel < n)
        target_leaf_by_panel[panel] = target_node;
    }
  }

  std::vector<PanelSourceMoments> source_moments;
  std::vector<CartesianJcpMoments> cartesian_moments;
  const auto moment_begin = Clock::now();
  if (cartesian) {
    cartesian_moments.resize(impl_->source_tree.cells.size());
    const std::size_t workers = std::max<std::size_t>(1, executor.capacity());
    executor.parallel_for(
        impl_->source_tree.cells.size(), workers, 1,
        [&](std::size_t, std::size_t begin, std::size_t end) {
          for (std::size_t source_index = begin; source_index < end;
               ++source_index) {
            const Cell &source_cell = impl_->source_tree.cells[source_index];
            const std::span<const std::size_t> source_indices(
                impl_->source_tree.order.data() + source_cell.begin,
                source_cell.count());
            cartesian_moments[source_index] = build_cartesian_jcp_moments(
                impl_->panels, phi_values, phin_values, source_indices,
                source_cell.qx, source_cell.qy, source_cell.qz,
                impl_->config.expansion_order);
          }
        });
  } else {
    source_moments = compute_panel_source_moments(
        impl_->source_tree, impl_->panels, phi_values, phin_values);
  }
  result.moment_build_seconds = elapsed_seconds(moment_begin, Clock::now());

  std::vector<int> local_target_nodes;
  std::vector<int> local_index_by_target_node(impl_->target_tree.cells.size(),
                                              -1);
  std::vector<CartesianJcpLocalP4> local_expansions;
  std::vector<TargetLeafWork> local_work;
  if (local_mode) {
    for (const std::size_t target_panel : result.sampled_targets) {
      const int target_node = target_leaf_by_panel[target_panel];
      if (target_node >= 0)
        local_target_nodes.push_back(target_node);
    }
    std::sort(local_target_nodes.begin(), local_target_nodes.end());
    local_target_nodes.erase(
        std::unique(local_target_nodes.begin(), local_target_nodes.end()),
        local_target_nodes.end());
    local_expansions.resize(local_target_nodes.size());
    local_work.resize(local_target_nodes.size());
    for (std::size_t i = 0; i < local_target_nodes.size(); ++i)
      local_index_by_target_node[static_cast<std::size_t>(
          local_target_nodes[i])] = static_cast<int>(i);
    for (std::size_t i = 0; i < local_target_nodes.size(); ++i) {
      local_work[i] =
          impl_->target_tree
              .work[static_cast<std::size_t>(local_target_nodes[i])];
      result.local_far_pairs_before += local_work[i].far_sources.size();
    }

    if (refine_local) {
      const auto refine_begin = Clock::now();
      struct PairTailMetric {
        int source_node{-1};
        double tail_sq{0.0};
        double signal_sq{0.0};
      };
      for (std::size_t local_index = 0; local_index < local_target_nodes.size();
           ++local_index) {
        const int target_node = local_target_nodes[local_index];
        const Cell &target_cell =
            impl_->target_tree.cells[static_cast<std::size_t>(target_node)];
        TargetLeafWork &work = local_work[local_index];
        double first_tail_ratio = 0.0;
        double final_tail_ratio = 0.0;
        bool first_iteration = true;

        auto sort_leaf_work = [&]() {
          std::sort(work.near_sources.begin(), work.near_sources.end());
          work.near_sources.erase(
              std::unique(work.near_sources.begin(), work.near_sources.end()),
              work.near_sources.end());
          std::sort(work.far_sources.begin(), work.far_sources.end());
          work.far_sources.erase(
              std::unique(work.far_sources.begin(), work.far_sources.end()),
              work.far_sources.end());
        };
        std::function<void(int)> append_source = [&](int source_node) {
          const Cell &source_cell =
              impl_->source_tree.cells[static_cast<std::size_t>(source_node)];
          if (accept_far(impl_->target_tree, impl_->source_tree, target_node,
                         source_node, impl_->config.separation_theta, true)) {
            work.far_sources.push_back(source_node);
            return;
          }
          if (source_cell.is_leaf()) {
            work.near_sources.push_back(source_node);
            return;
          }
          append_source(source_cell.left);
          append_source(source_cell.right);
        };

        while (true) {
          std::vector<PairTailMetric> metrics;
          metrics.reserve(work.far_sources.size());
          double tail_sq = 0.0;
          double signal_sq = 0.0;
          for (const int source_node : work.far_sources) {
            const std::size_t source_index =
                static_cast<std::size_t>(source_node);
            const Cell &source_cell = impl_->source_tree.cells[source_index];
            const CartesianJcpLocalP4 translated =
                translate_cartesian_jcp_m2l_p4(cartesian_moments[source_index],
                                               source_cell.qx, source_cell.qy,
                                               source_cell.qz, target_cell.qx,
                                               target_cell.qy, target_cell.qz,
                                               impl_->eps_ratio, impl_->kappa);
            PairTailMetric metric;
            metric.source_node = source_node;
            for (std::size_t pos = target_cell.begin; pos < target_cell.end;
                 ++pos) {
              const std::size_t target_panel = impl_->target_tree.order[pos];
              const PrototypePanel &target = impl_->panels[target_panel];
              const auto p4 = evaluate_cartesian_jcp_l2p_order(
                  target, target_cell.qx, target_cell.qy, target_cell.qz,
                  translated, 4);
              const auto p3 = evaluate_cartesian_jcp_l2p_order(
                  target, target_cell.qx, target_cell.qy, target_cell.qz,
                  translated, 3);
              const double p4_phi = p4[0] + p4[1];
              const double p3_phi = p3[0] + p3[1];
              const double phi_tail = p4_phi - p3_phi;
              metric.tail_sq += phi_tail * phi_tail;
              metric.signal_sq += p4_phi * p4_phi;
              if (coupled) {
                const double p4_phin = p4[2] + p4[3];
                const double p3_phin = p3[2] + p3[3];
                const double phin_tail = p4_phin - p3_phin;
                metric.tail_sq += phin_tail * phin_tail;
                metric.signal_sq += p4_phin * p4_phin;
              }
            }
            tail_sq += metric.tail_sq;
            signal_sq += metric.signal_sq;
            metrics.push_back(metric);
          }
          final_tail_ratio =
              std::sqrt(std::max(0.0, tail_sq)) /
              std::max(1.0e-300, std::sqrt(std::max(0.0, signal_sq)));
          if (first_iteration) {
            first_tail_ratio = final_tail_ratio;
            first_iteration = false;
          }
          if (final_tail_ratio <= refine_tolerance || metrics.empty())
            break;

          std::sort(metrics.begin(), metrics.end(),
                    [&](const PairTailMetric &lhs, const PairTailMetric &rhs) {
                      if (lhs.tail_sq != rhs.tail_sq)
                        return lhs.tail_sq > rhs.tail_sq;
                      const int lhs_depth =
                          impl_->source_tree
                              .cells[static_cast<std::size_t>(lhs.source_node)]
                              .depth;
                      const int rhs_depth =
                          impl_->source_tree
                              .cells[static_cast<std::size_t>(rhs.source_node)]
                              .depth;
                      if (lhs_depth != rhs_depth)
                        return lhs_depth < rhs_depth;
                      return lhs.source_node < rhs.source_node;
                    });
          const int selected = metrics.front().source_node;
          work.far_sources.erase(std::remove(work.far_sources.begin(),
                                             work.far_sources.end(), selected),
                                 work.far_sources.end());
          const Cell &selected_cell =
              impl_->source_tree.cells[static_cast<std::size_t>(selected)];
          ++result.local_refined_cells;
          if (selected_cell.is_leaf()) {
            work.near_sources.push_back(selected);
            result.local_direct_fallback_panels += selected_cell.count();
            result.local_direct_fallback_pairs +=
                target_cell.count() * selected_cell.count();
          } else {
            append_source(selected_cell.left);
            append_source(selected_cell.right);
            result.local_split_children += 2;
          }
          sort_leaf_work();
        }
        result.local_tail_ratio_before =
            std::max(result.local_tail_ratio_before, first_tail_ratio);
        result.local_tail_ratio_after =
            std::max(result.local_tail_ratio_after, final_tail_ratio);
      }
      result.local_refinement_seconds =
          elapsed_seconds(refine_begin, Clock::now());
    }
    for (const TargetLeafWork &work : local_work)
      result.local_far_pairs_after += work.far_sources.size();

    const auto m2l_begin = Clock::now();
    const std::size_t workers = std::max<std::size_t>(1, executor.capacity());
    executor.parallel_for(
        local_target_nodes.size(), workers, 1,
        [&](std::size_t, std::size_t begin, std::size_t end) {
          for (std::size_t local_index = begin; local_index < end;
               ++local_index) {
            const int target_node = local_target_nodes[local_index];
            const Cell &target_cell =
                impl_->target_tree.cells[static_cast<std::size_t>(target_node)];
            const TargetLeafWork &work = local_work[local_index];
            CartesianJcpLocalP4 accumulated;
            for (const int source_node : work.far_sources) {
              const std::size_t source_index =
                  static_cast<std::size_t>(source_node);
              const Cell &source_cell = impl_->source_tree.cells[source_index];
              const CartesianJcpLocalP4 translated =
                  translate_cartesian_jcp_m2l_p4(
                      cartesian_moments[source_index], source_cell.qx,
                      source_cell.qy, source_cell.qz, target_cell.qx,
                      target_cell.qy, target_cell.qz, impl_->eps_ratio,
                      impl_->kappa);
              accumulate_cartesian_jcp_local(accumulated, translated);
            }
            local_expansions[local_index] = accumulated;
          }
        });
    result.m2l_seconds = elapsed_seconds(m2l_begin, Clock::now());
    result.local_target_leaves = local_target_nodes.size();
    result.local_m2l_pairs = result.local_far_pairs_after;
  }
  JcpAnalyticOperatorResult applied;
  if (!cartesian) {
    applied = apply(phi_values, phin_values, executor, coupled,
                    near_integration, quad_buffer, production_panels);
  }

  double recompose_diff_sq = 0.0;
  double applied_sq = 0.0;
  for (const std::size_t target_panel : result.sampled_targets) {
    const int target_node = target_leaf_by_panel[target_panel];
    if (target_node < 0) {
      result.finite_outputs = false;
      continue;
    }
    const Cell &target_cell =
        impl_->target_tree.cells[static_cast<std::size_t>(target_node)];
    const int local_index =
        local_mode
            ? local_index_by_target_node[static_cast<std::size_t>(target_node)]
            : -1;
    const TargetLeafWork &work =
        local_mode && local_index >= 0
            ? local_work[static_cast<std::size_t>(local_index)]
            : impl_->target_tree.work[static_cast<std::size_t>(target_node)];
    const PrototypePanel &target = impl_->panels[target_panel];

    JcpChannelValues reconstructed{};
    JcpOperatorAttributionRecord near_record;
    near_record.probe = probe;
    near_record.target_panel = target_panel;
    near_record.region = "near";
    near_record.source_depth = -1;
    for (const int source_node : work.near_sources) {
      const Cell &source_cell =
          impl_->source_tree.cells[static_cast<std::size_t>(source_node)];
      near_record.source_panels += source_cell.count();
      for (std::size_t pos = source_cell.begin; pos < source_cell.end; ++pos) {
        const std::size_t source_panel = impl_->source_tree.order[pos];
        const JcpChannelValues channels = jcp_near_kernel_pair_channels(
            (*production_panels)[target_panel], target_panel,
            (*production_panels)[source_panel], source_panel,
            phi_values[source_panel], phin_values[source_panel],
            impl_->eps_ratio, impl_->kappa, coupled, near_integration,
            quad_buffer, nullptr);
        add_channels(near_record.approximate, channels);
        add_channels(near_record.direct, channels);
      }
    }
    add_channels(reconstructed, near_record.approximate);
    result.records.push_back(near_record);
    ++result.near_records;

    if (local_mode) {
      JcpOperatorAttributionRecord record;
      record.probe = probe;
      record.target_panel = target_panel;
      record.region = "far";
      record.source_node = -2;
      record.source_depth = -1;
      record.distance = 0.0;
      record.source_radius = 0.0;
      record.mac_ratio = 0.0;
      if (local_index < 0) {
        result.finite_outputs = false;
      } else {
        const auto l2p_begin = Clock::now();
        record.approximate = evaluate_cartesian_jcp_l2p(
            target, target_cell.qx, target_cell.qy, target_cell.qz,
            local_expansions[static_cast<std::size_t>(local_index)]);
        result.l2p_seconds += elapsed_seconds(l2p_begin, Clock::now());
      }
      for (const int source_node : work.far_sources) {
        const Cell &source_cell =
            impl_->source_tree.cells[static_cast<std::size_t>(source_node)];
        record.source_panels += source_cell.count();
        for (std::size_t pos = source_cell.begin; pos < source_cell.end;
             ++pos) {
          const std::size_t source_panel = impl_->source_tree.order[pos];
          const JcpChannelValues channels = jcp_near_kernel_pair_channels(
              (*production_panels)[target_panel], target_panel,
              (*production_panels)[source_panel], source_panel,
              phi_values[source_panel], phin_values[source_panel],
              impl_->eps_ratio, impl_->kappa, coupled, near_integration,
              quad_buffer, nullptr);
          add_channels(record.direct, channels);
        }
      }
      add_channels(reconstructed, record.approximate);
      result.records.push_back(record);
      ++result.far_records;
    } else
      for (const int source_node : work.far_sources) {
        const std::size_t source_index = static_cast<std::size_t>(source_node);
        const Cell &source_cell = impl_->source_tree.cells[source_index];
        JcpOperatorAttributionRecord record;
        record.probe = probe;
        record.target_panel = target_panel;
        record.region = "far";
        record.source_node = source_node;
        record.source_depth = source_cell.depth;
        record.source_panels = source_cell.count();
        // Match the exact cell-center metric used by accept_far/build_work so
        // depth/MAC attribution bins describe the interaction list that ran.
        const double dx = target_cell.cx - source_cell.cx;
        const double dy = target_cell.cy - source_cell.cy;
        const double dz = target_cell.cz - source_cell.cz;
        record.distance = std::sqrt(dx * dx + dy * dy + dz * dz);
        record.source_radius = source_cell.radius;
        record.mac_ratio =
            record.distance > 1.0e-12
                ? (target_cell.radius + source_cell.radius) / record.distance
                : std::numeric_limits<double>::infinity();
        const double eqx = target_cell.qx - source_cell.qx;
        const double eqy = target_cell.qy - source_cell.qy;
        const double eqz = target_cell.qz - source_cell.qz;
        record.expansion_distance =
            std::sqrt(eqx * eqx + eqy * eqy + eqz * eqz);
        record.target_expansion_radius = target_cell.expansion_radius;
        record.source_expansion_radius = source_cell.expansion_radius;
        record.target_center_offset = target_cell.center_offset;
        record.source_center_offset = source_cell.center_offset;
        record.expansion_mac_ratio =
            record.expansion_distance > 1.0e-12
                ? (target_cell.expansion_radius +
                   source_cell.expansion_radius) /
                      record.expansion_distance
                : std::numeric_limits<double>::infinity();
        const auto far_begin = Clock::now();
        record.approximate =
            cartesian
                ? evaluate_cartesian_jcp_m2p(target, source_cell.qx,
                                             source_cell.qy, source_cell.qz,
                                             cartesian_moments[source_index],
                                             impl_->eps_ratio, impl_->kappa)
                : jcp_far_cell_channels(target, source_cell,
                                        source_moments[source_index],
                                        impl_->eps_ratio, impl_->kappa);
        result.far_evaluation_seconds +=
            elapsed_seconds(far_begin, Clock::now());
        for (std::size_t pos = source_cell.begin; pos < source_cell.end;
             ++pos) {
          const std::size_t source_panel = impl_->source_tree.order[pos];
          const JcpChannelValues channels = jcp_near_kernel_pair_channels(
              (*production_panels)[target_panel], target_panel,
              (*production_panels)[source_panel], source_panel,
              phi_values[source_panel], phin_values[source_panel],
              impl_->eps_ratio, impl_->kappa, coupled, near_integration,
              quad_buffer, nullptr);
          add_channels(record.direct, channels);
        }
        add_channels(reconstructed, record.approximate);
        result.records.push_back(record);
        ++result.far_records;
      }

    const double reconstructed_phi = channel_phi(reconstructed);
    const double reconstructed_phin = channel_phin(reconstructed);
    const double applied_phi =
        cartesian ? reconstructed_phi : applied.phi[target_panel];
    const double applied_phin =
        cartesian ? reconstructed_phin : applied.phin[target_panel];
    const double phi_delta = reconstructed_phi - applied_phi;
    recompose_diff_sq += phi_delta * phi_delta;
    applied_sq += applied_phi * applied_phi;
    if (coupled) {
      const double phin_delta = reconstructed_phin - applied_phin;
      recompose_diff_sq += phin_delta * phin_delta;
      applied_sq += applied_phin * applied_phin;
    }
  }
  result.recomposition_rel_error =
      std::sqrt(std::max(0.0, recompose_diff_sq)) /
      std::max(1.0e-300, std::sqrt(std::max(0.0, applied_sq)));
  if (local_mode)
    result.far_evaluation_seconds = result.m2l_seconds + result.l2p_seconds;
  for (const JcpOperatorAttributionRecord &record : result.records) {
    result.finite_outputs = result.finite_outputs &&
                            std::isfinite(record.distance) &&
                            std::isfinite(record.source_radius) &&
                            std::isfinite(record.mac_ratio) &&
                            std::isfinite(record.expansion_distance) &&
                            std::isfinite(record.target_expansion_radius) &&
                            std::isfinite(record.source_expansion_radius) &&
                            std::isfinite(record.target_center_offset) &&
                            std::isfinite(record.source_center_offset) &&
                            std::isfinite(record.expansion_mac_ratio);
    for (std::size_t channel = 0; channel < 4; ++channel) {
      result.finite_outputs = result.finite_outputs &&
                              std::isfinite(record.approximate[channel]) &&
                              std::isfinite(record.direct[channel]);
    }
  }
  result.finite_outputs = result.finite_outputs &&
                          (cartesian || applied.finite_outputs) &&
                          std::isfinite(result.recomposition_rel_error);
  return result;
}

const char *kernel_kind_name(KernelKind kernel) noexcept {
  switch (kernel) {
  case KernelKind::Laplace:
    return "laplace";
  case KernelKind::Yukawa:
    return "yukawa";
  }
  return "unknown";
}

PrototypeEvaluation
evaluate_source_target_cloud(const PrototypeConfig &input_config,
                             const PrototypePointCloud &source_points,
                             const PrototypePointCloud &target_points,
                             bool exclude_same_index = false,
                             bool compute_direct_reference = true) {
  const PointSoA sources = from_public_point_cloud(source_points);
  const PointSoA targets = from_public_point_cloud(target_points);
  PrototypeConfig config = input_config;
  config.max_leaf_size = std::max(1, config.max_leaf_size);
  config.threads = std::max(1, config.threads);
  if (config.kernel == KernelKind::Yukawa && !(config.kappa > 0.0)) {
    config.kappa = 0.125;
  }

  PrototypeEvaluation evaluation;
  PrototypeMetrics &metrics = evaluation.metrics;
  metrics.sources = sources.x.size();
  metrics.targets = targets.x.size();

  const auto setup_begin = Clock::now();
  Tree source_tree =
      build_tree(sources, static_cast<std::size_t>(config.max_leaf_size));
  Tree target_tree =
      build_tree(targets, static_cast<std::size_t>(config.max_leaf_size));
  if (!source_tree.cells.empty() && !target_tree.cells.empty()) {
    build_work(target_tree, source_tree, 0, 0, config.separation_theta,
               exclude_same_index);
  }
  sort_work(target_tree);
  metrics.cells = source_tree.cells.size() + target_tree.cells.size();
  compute_counts_and_operators(target_tree, source_tree, config, metrics);
  const auto setup_end = Clock::now();
  metrics.setup_seconds = elapsed_seconds(setup_begin, setup_end);

  if (compute_direct_reference) {
    const auto direct_begin = Clock::now();
    apply_direct_reference(sources, targets, config, evaluation.reference,
                           exclude_same_index);
    const auto direct_end = Clock::now();
    metrics.direct_seconds = elapsed_seconds(direct_begin, direct_end);
  }

  const auto apply_begin = Clock::now();
  apply_prototype(sources, source_tree, targets, target_tree, config,
                  evaluation.approx, exclude_same_index);
  const auto apply_end = Clock::now();
  metrics.apply_seconds = elapsed_seconds(apply_begin, apply_end);

  if (compute_direct_reference) {
    compute_error(evaluation.reference, evaluation.approx, metrics);
  } else {
    metrics.finite_outputs = finite_vector(evaluation.approx);
  }
  return evaluation;
}

PrototypeEvaluation
evaluate_point_cloud(const PrototypeConfig &input_config,
                     const PrototypePointCloud &input_points) {
  return evaluate_source_target_cloud(input_config, input_points, input_points);
}

PrototypeMetrics
run_point_cloud_probe(const PrototypeConfig &input_config,
                      const PrototypePointCloud &input_points) {
  return evaluate_point_cloud(input_config, input_points).metrics;
}

PrototypeMetrics run_synthetic_probe(const PrototypeConfig &config,
                                     std::size_t panel_count, unsigned seed) {
  if (panel_count == 0) {
    throw std::invalid_argument(
        "FMM/M2L prototype requires at least one panel");
  }
  const PointSoA generated = make_synthetic_points(panel_count, seed);
  PrototypePointCloud public_points;
  public_points.x.assign(generated.x.begin(), generated.x.end());
  public_points.y.assign(generated.y.begin(), generated.y.end());
  public_points.z.assign(generated.z.begin(), generated.z.end());
  public_points.q.assign(generated.q.begin(), generated.q.end());
  return run_point_cloud_probe(config, public_points);
}

PrototypePointCloud make_mini_protein_like_point_cloud(std::size_t panel_count,
                                                       unsigned seed) {
  if (panel_count == 0) {
    throw std::invalid_argument(
        "mini-protein-like fixture requires at least one point");
  }
  PrototypePointCloud points;
  points.x.resize(panel_count);
  points.y.resize(panel_count);
  points.z.resize(panel_count);
  points.q.resize(panel_count);

  std::uint64_t state =
      (static_cast<std::uint64_t>(seed) << 32) ^ 0xd1b54a32d192ed03ull;
  auto next01 = [&]() {
    state = state * 2862933555777941757ull + 3037000493ull;
    const std::uint64_t bits = (state >> 11) | 0x3ff0000000000000ull;
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value - 1.0;
  };

  for (std::size_t i = 0; i < panel_count; ++i) {
    const double t = 12.5663706143591729539 * static_cast<double>(i) /
                     static_cast<double>(panel_count);
    const double strand = static_cast<double>(i % 5) - 2.0;
    const double shell = 0.55 + 0.25 * next01();
    const double wobble = 0.15 * (next01() - 0.5);
    points.x[i] = 2.8 * std::cos(0.5 * t) + shell * std::cos(t) + 0.18 * strand;
    points.y[i] = 1.4 * std::sin(0.5 * t) + shell * std::sin(t) + wobble;
    points.z[i] = 0.22 * static_cast<double>(i % 31) + 0.45 * std::sin(1.5 * t);
    points.q[i] =
        ((i % 7) == 0 ? 1.0 : ((i % 11) == 0 ? -1.0 : 0.12 * (next01() - 0.5)));
  }
  const double z_mean = std::accumulate(points.z.begin(), points.z.end(), 0.0) /
                        static_cast<double>(panel_count);
  for (double &z : points.z) {
    z -= z_mean;
  }
  return points;
}

std::vector<PrototypePanel>
make_tabi_panel_like_fixture(std::size_t panel_count, unsigned seed) {
  if (panel_count == 0) {
    throw std::invalid_argument(
        "TABI-panel-like fixture requires at least one panel");
  }
  std::vector<PrototypePanel> panels(panel_count);
  std::uint64_t state =
      (static_cast<std::uint64_t>(seed) << 32) ^ 0x94d049bb133111ebull;
  auto next01 = [&]() {
    state = state * 3202034522624059733ull + 1ull;
    const std::uint64_t bits = (state >> 11) | 0x3ff0000000000000ull;
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value - 1.0;
  };

  const double golden = 2.3999632297286533222;
  for (std::size_t i = 0; i < panel_count; ++i) {
    const double u = -1.0 + 2.0 * (static_cast<double>(i) + 0.5) /
                                static_cast<double>(panel_count);
    const double r = std::sqrt(std::max(0.0, 1.0 - u * u));
    const double theta = golden * static_cast<double>(i);
    const double ripple =
        1.0 + 0.08 * std::sin(5.0 * theta) + 0.03 * (next01() - 0.5);
    PrototypePanel panel;
    panel.x = 2.2 * ripple * r * std::cos(theta);
    panel.y = 1.4 * ripple * r * std::sin(theta);
    panel.z = 3.0 * ripple * u;
    double nx = panel.x / (2.2 * 2.2);
    double ny = panel.y / (1.4 * 1.4);
    double nz = panel.z / (3.0 * 3.0);
    const double norm = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (norm > 0.0) {
      nx /= norm;
      ny /= norm;
      nz /= norm;
    }
    panel.nx = nx;
    panel.ny = ny;
    panel.nz = nz;
    panel.area = 4.0 * 3.14159265358979323846 * 2.2 * 1.4 /
                 static_cast<double>(panel_count) * (0.8 + 0.4 * std::abs(nz));
    panel.scalar_density =
        ((i % 9) == 0 ? 1.0 : ((i % 13) == 0 ? -0.85 : 0.08 * std::sin(theta)));
    panels[i] = panel;
  }
  return panels;
}

PrototypePointCloud
point_cloud_from_panels(const std::vector<PrototypePanel> &panels) {
  if (panels.empty()) {
    throw std::invalid_argument(
        "FMM/M2L panel adapter requires at least one panel");
  }
  PrototypePointCloud points;
  points.x.reserve(panels.size());
  points.y.reserve(panels.size());
  points.z.reserve(panels.size());
  points.q.reserve(panels.size());
  for (const PrototypePanel &panel : panels) {
    if (!(panel.area > 0.0)) {
      throw std::invalid_argument(
          "FMM/M2L panel adapter requires positive panel area");
    }
    points.x.push_back(panel.x);
    points.y.push_back(panel.y);
    points.z.push_back(panel.z);
    points.q.push_back(panel.area * panel.scalar_density);
  }
  return points;
}

std::array<PrototypePointCloud, 4>
make_jcp_channel_point_clouds(const std::vector<PrototypePanel> &panels) {
  if (panels.empty()) {
    throw std::invalid_argument(
        "JCP-channel prototype requires at least one panel");
  }
  std::array<PrototypePointCloud, 4> channels;
  for (auto &cloud : channels) {
    cloud.x.reserve(panels.size());
    cloud.y.reserve(panels.size());
    cloud.z.reserve(panels.size());
    cloud.q.reserve(panels.size());
  }

  for (std::size_t i = 0; i < panels.size(); ++i) {
    const PrototypePanel &panel = panels[i];
    if (!(panel.area > 0.0)) {
      throw std::invalid_argument(
          "JCP-channel prototype requires positive panel area");
    }
    const double t = static_cast<double>(i + 1);
    const double phi = panel.scalar_density;
    const double phin = 0.65 * std::sin(0.017 * t) + 0.35 * panel.nz;
    const double normal_mix =
        0.45 * panel.nx - 0.25 * panel.ny + 0.15 * panel.nz;
    const std::array<double, 4> q{
        panel.area * phin,
        panel.area * phi * normal_mix,
        panel.area * phin * (1.0 + 0.2 * panel.nz),
        panel.area * phi * (0.5 + 0.5 * std::abs(panel.nz)),
    };
    for (std::size_t c = 0; c < channels.size(); ++c) {
      channels[c].x.push_back(panel.x);
      channels[c].y.push_back(panel.y);
      channels[c].z.push_back(panel.z);
      channels[c].q.push_back(q[c]);
    }
  }
  return channels;
}

PrototypePointCloud
make_jcp_k1_source_cloud(const std::vector<PrototypePanel> &panels) {
  constexpr double inv_four_pi = 0.079577471545947667884;
  PrototypePointCloud cloud;
  cloud.x.reserve(panels.size());
  cloud.y.reserve(panels.size());
  cloud.z.reserve(panels.size());
  cloud.q.reserve(panels.size());
  for (std::size_t i = 0; i < panels.size(); ++i) {
    const PrototypePanel &panel = panels[i];
    cloud.x.push_back(panel.x);
    cloud.y.push_back(panel.y);
    cloud.z.push_back(panel.z);
    cloud.q.push_back(panel.area * jcp_phin_for_panel(panel, i) * inv_four_pi);
  }
  return cloud;
}

PrototypePointCloud
make_panel_target_cloud(const std::vector<PrototypePanel> &panels,
                        double normal_shift) {
  PrototypePointCloud cloud;
  cloud.x.reserve(panels.size());
  cloud.y.reserve(panels.size());
  cloud.z.reserve(panels.size());
  cloud.q.assign(panels.size(), 0.0);
  cloud.panel_id.reserve(panels.size());
  for (std::size_t i = 0; i < panels.size(); ++i) {
    const PrototypePanel &panel = panels[i];
    cloud.x.push_back(panel.x + normal_shift * panel.nx);
    cloud.y.push_back(panel.y + normal_shift * panel.ny);
    cloud.z.push_back(panel.z + normal_shift * panel.nz);
    cloud.panel_id.push_back(i);
  }
  return cloud;
}

PrototypePointCloud
make_source_normal_dipole_cloud(const std::vector<PrototypePanel> &panels,
                                bool use_phi, double h) {
  constexpr double inv_four_pi = 0.079577471545947667884;
  PrototypePointCloud cloud;
  cloud.x.reserve(panels.size() * 2);
  cloud.y.reserve(panels.size() * 2);
  cloud.z.reserve(panels.size() * 2);
  cloud.q.reserve(panels.size() * 2);
  cloud.panel_id.reserve(panels.size() * 2);
  for (std::size_t i = 0; i < panels.size(); ++i) {
    const PrototypePanel &panel = panels[i];
    const double value =
        use_phi ? panel.scalar_density : jcp_phin_for_panel(panel, i);
    const double q = panel.area * value * inv_four_pi / (2.0 * h);
    cloud.x.push_back(panel.x + h * panel.nx);
    cloud.y.push_back(panel.y + h * panel.ny);
    cloud.z.push_back(panel.z + h * panel.nz);
    cloud.q.push_back(q);
    cloud.panel_id.push_back(i);
    cloud.x.push_back(panel.x - h * panel.nx);
    cloud.y.push_back(panel.y - h * panel.ny);
    cloud.z.push_back(panel.z - h * panel.nz);
    cloud.q.push_back(-q);
    cloud.panel_id.push_back(i);
  }
  return cloud;
}

AlignedDoubleVector scalar_diff(const AlignedDoubleVector &lhs,
                                const AlignedDoubleVector &rhs, double scale) {
  AlignedDoubleVector out(lhs.size(), 0.0);
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    out[i] = (lhs[i] - rhs[i]) * scale;
  }
  return out;
}

struct DerivativeReference {
  AlignedDoubleVector k2;
  AlignedDoubleVector k3;
  AlignedDoubleVector k4;
};

DerivativeReference
compute_direct_derivative_reference(const std::vector<PrototypePanel> &panels,
                                    double eps_ratio, double kappa) {
  constexpr double inv_four_pi = 0.079577471545947667884;
  const double inv_eps_ratio = 1.0 / std::max(1.0e-12, eps_ratio);
  const double kk = std::max(0.0, kappa);
  DerivativeReference ref;
  ref.k2.assign(panels.size(), 0.0);
  ref.k3.assign(panels.size(), 0.0);
  ref.k4.assign(panels.size(), 0.0);
  for (std::size_t i = 0; i < panels.size(); ++i) {
    const PrototypePanel &target = panels[i];
    for (std::size_t j = 0; j < panels.size(); ++j) {
      if (i == j) {
        continue;
      }
      const PrototypePanel &source = panels[j];
      const double ux = target.x - source.x;
      const double uy = target.y - source.y;
      const double uz = target.z - source.z;
      const double r2 = ux * ux + uy * uy + uz * uz;
      if (!(r2 > 1.0e-28)) {
        continue;
      }
      const double r = std::sqrt(r2);
      const double inv_r = 1.0 / r;
      const double inv_r2 = inv_r * inv_r;
      const double inv_r3 = inv_r2 * inv_r;
      const double source_projection =
          source.nx * ux + source.ny * uy + source.nz * uz;
      const double target_projection =
          target.nx * ux + target.ny * uy + target.nz * uz;
      const double source_d0 = inv_four_pi * source_projection * inv_r3;
      const double expkr = std::exp(-kk * r);
      const double one_plus_kr = 1.0 + kk * r;
      const double source_dk = expkr * one_plus_kr * source_d0;
      const double target_d0 = -inv_four_pi * target_projection * inv_r3;
      const double target_dk = expkr * one_plus_kr * target_d0;
      const double nxu = target_projection * inv_r;
      const double nyu = source_projection * inv_r;
      const double nxny =
          target.nx * source.nx + target.ny * source.ny + target.nz * source.nz;
      const double nxu_nyu = nxu * nyu;
      const double tangent_metric = nxny - nxu_nyu;
      const double mixed0 = inv_four_pi * inv_r3 * (nxny - 3.0 * nxu_nyu);
      const double fprime_k_over_r =
          -expkr * one_plus_kr * inv_four_pi * inv_r3;
      const double fsecond_k =
          expkr * inv_four_pi *
          (kk * kk * inv_r + 2.0 * kk * inv_r2 + 2.0 * inv_r3);
      const double mixedk =
          -fsecond_k * nxu_nyu - fprime_k_over_r * tangent_metric;
      const double phi = source.scalar_density;
      const double phin = jcp_phin_for_panel(source, j);
      ref.k2[i] += source.area * (eps_ratio * source_dk - source_d0) * phi;
      ref.k3[i] += source.area * (target_d0 - inv_eps_ratio * target_dk) * phin;
      ref.k4[i] += source.area * (mixedk - mixed0) * phi;
    }
  }
  return ref;
}

AlignedDoubleVector
compute_direct_jcp_k1_reference(const std::vector<PrototypePanel> &panels,
                                double kappa) {
  constexpr double inv_four_pi = 0.079577471545947667884;
  AlignedDoubleVector out(panels.size(), 0.0);
  const double kk = std::max(0.0, kappa);
  for (std::size_t i = 0; i < panels.size(); ++i) {
    double sum = 0.0;
    const PrototypePanel &target = panels[i];
    for (std::size_t j = 0; j < panels.size(); ++j) {
      if (i == j) {
        continue;
      }
      const PrototypePanel &source = panels[j];
      const double dx = target.x - source.x;
      const double dy = target.y - source.y;
      const double dz = target.z - source.z;
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (!(r2 > 1.0e-28)) {
        continue;
      }
      const double r = std::sqrt(r2);
      const double phin = jcp_phin_for_panel(source, j);
      const double g0 = inv_four_pi / r;
      const double gk = std::exp(-kk * r) * g0;
      sum += source.area * (g0 - gk) * phin;
    }
    out[i] = sum;
  }
  return out;
}

JcpDirectParityMetrics
compare_jcp_direct_reference(const std::vector<PrototypePanel> &panels,
                             const std::vector<double> &production_phi,
                             const std::vector<double> &production_phin,
                             double eps_ratio, double kappa) {
  if (panels.empty()) {
    throw std::invalid_argument(
        "JCP direct parity requires at least one panel");
  }
  if (production_phi.size() != panels.size() ||
      production_phin.size() != panels.size()) {
    throw std::invalid_argument(
        "JCP direct parity production vectors do not match panel count");
  }

  const AlignedDoubleVector k1 = compute_direct_jcp_k1_reference(panels, kappa);
  const DerivativeReference derivative =
      compute_direct_derivative_reference(panels, eps_ratio, kappa);
  AlignedDoubleVector analytic_phi(panels.size(), 0.0);
  AlignedDoubleVector analytic_phin(panels.size(), 0.0);
  AlignedDoubleVector production_phi_aligned(production_phi.begin(),
                                             production_phi.end());
  AlignedDoubleVector production_phin_aligned(production_phin.begin(),
                                              production_phin.end());
  for (std::size_t i = 0; i < panels.size(); ++i) {
    analytic_phi[i] = k1[i] + derivative.k2[i];
    analytic_phin[i] = derivative.k3[i] + derivative.k4[i];
  }

  PrototypeMetrics phi_error;
  PrototypeMetrics phin_error;
  compute_error(production_phi_aligned, analytic_phi, phi_error);
  compute_error(production_phin_aligned, analytic_phin, phin_error);

  double err2 = 0.0;
  double ref2 = 0.0;
  for (std::size_t i = 0; i < panels.size(); ++i) {
    const double phi_diff = analytic_phi[i] - production_phi_aligned[i];
    const double phin_diff = analytic_phin[i] - production_phin_aligned[i];
    err2 += phi_diff * phi_diff + phin_diff * phin_diff;
    ref2 += production_phi_aligned[i] * production_phi_aligned[i] +
            production_phin_aligned[i] * production_phin_aligned[i];
  }

  JcpDirectParityMetrics metrics;
  metrics.panels = panels.size();
  metrics.phi_rel_rms_error = phi_error.rel_rms_error;
  metrics.phin_rel_rms_error = phin_error.rel_rms_error;
  metrics.aggregate_rel_rms_error = std::sqrt(err2 / std::max(ref2, 1.0e-30));
  metrics.max_abs_error =
      std::max(phi_error.max_abs_error, phin_error.max_abs_error);
  metrics.finite_outputs = phi_error.finite_outputs &&
                           phin_error.finite_outputs &&
                           std::isfinite(metrics.aggregate_rel_rms_error);
  return metrics;
}

JcpK1BridgeMetrics
run_jcp_k1_bridge_probe(const PrototypeConfig &input_config,
                        const std::vector<PrototypePanel> &panels) {
  if (panels.empty()) {
    throw std::invalid_argument("JCP K1 bridge requires at least one panel");
  }
  PrototypeConfig laplace = input_config;
  laplace.kernel = KernelKind::Laplace;
  PrototypeConfig yukawa = input_config;
  yukawa.kernel = KernelKind::Yukawa;
  if (!(yukawa.kappa > 0.0)) {
    yukawa.kappa = 0.2;
  }

  const PrototypePointCloud source_cloud = make_jcp_k1_source_cloud(panels);
  JcpK1BridgeMetrics metrics;
  metrics.panels = panels.size();

  const auto direct_begin = Clock::now();
  const AlignedDoubleVector direct =
      compute_direct_jcp_k1_reference(panels, yukawa.kappa);
  const auto direct_end = Clock::now();
  metrics.direct_seconds = elapsed_seconds(direct_begin, direct_end);

  PrototypeEvaluation lap_eval = evaluate_point_cloud(laplace, source_cloud);
  PrototypeEvaluation yuk_eval = evaluate_point_cloud(yukawa, source_cloud);
  metrics.setup_seconds =
      lap_eval.metrics.setup_seconds + yuk_eval.metrics.setup_seconds;
  metrics.apply_seconds =
      lap_eval.metrics.apply_seconds + yuk_eval.metrics.apply_seconds;
  metrics.near_pairs =
      lap_eval.metrics.near_pairs + yuk_eval.metrics.near_pairs;
  metrics.far_pairs = lap_eval.metrics.far_pairs + yuk_eval.metrics.far_pairs;
  metrics.finite_outputs =
      lap_eval.metrics.finite_outputs && yuk_eval.metrics.finite_outputs;

  AlignedDoubleVector approx(direct.size(), 0.0);
  for (std::size_t i = 0; i < direct.size(); ++i) {
    approx[i] = lap_eval.approx[i] - yuk_eval.approx[i];
  }
  PrototypeMetrics err_metrics;
  compute_error(direct, approx, err_metrics);
  metrics.rel_rms_error = err_metrics.rel_rms_error;
  metrics.rms_abs_error = err_metrics.rms_abs_error;
  metrics.max_abs_error = err_metrics.max_abs_error;
  metrics.finite_outputs = metrics.finite_outputs && err_metrics.finite_outputs;
  return metrics;
}

std::string classify_scope_e_decision(const JcpK1BridgeMetrics &metrics) {
  if (!metrics.finite_outputs || !std::isfinite(metrics.rel_rms_error)) {
    return "accuracy_fail";
  }
  if (metrics.rel_rms_error > 5.0e-2) {
    return "accuracy_fail";
  }
  if (metrics.far_pairs == 0) {
    return "no_far_work";
  }
  if (metrics.setup_seconds > metrics.direct_seconds) {
    return "setup_limited";
  }
  if (!(metrics.apply_seconds < metrics.direct_seconds)) {
    return "no_speedup";
  }
  return "pass";
}

JcpDerivativeBridgeMetrics
run_jcp_derivative_bridge_probe(const PrototypeConfig &input_config,
                                const std::vector<PrototypePanel> &panels) {
  if (panels.empty()) {
    throw std::invalid_argument(
        "JCP derivative bridge requires at least one panel");
  }
  PrototypeConfig laplace = input_config;
  laplace.kernel = KernelKind::Laplace;
  PrototypeConfig yukawa = input_config;
  yukawa.kernel = KernelKind::Yukawa;
  if (!(yukawa.kappa > 0.0)) {
    yukawa.kappa = 0.2;
  }
  constexpr double eps_ratio = 80.0;
  constexpr double inv_eps_ratio = 1.0 / eps_ratio;
  constexpr double h = 1.0e-3;

  JcpDerivativeBridgeMetrics metrics;
  metrics.panels = panels.size();
  metrics.channels = 3;

  const auto direct_begin = Clock::now();
  const DerivativeReference direct =
      compute_direct_derivative_reference(panels, eps_ratio, yukawa.kappa);
  const auto direct_end = Clock::now();
  metrics.direct_seconds = elapsed_seconds(direct_begin, direct_end);

  const PrototypePointCloud target_base = make_panel_target_cloud(panels, 0.0);
  const PrototypePointCloud target_plus = make_panel_target_cloud(panels, h);
  const PrototypePointCloud target_minus = make_panel_target_cloud(panels, -h);
  const PrototypePointCloud phi_source_dipole =
      make_source_normal_dipole_cloud(panels, true, h);
  const PrototypePointCloud phin_source = make_jcp_k1_source_cloud(panels);

  auto add_eval_metrics = [&](const PrototypeEvaluation &eval) {
    metrics.setup_seconds += eval.metrics.setup_seconds;
    metrics.apply_seconds += eval.metrics.apply_seconds;
    metrics.near_pairs += eval.metrics.near_pairs;
    metrics.far_pairs += eval.metrics.far_pairs;
    metrics.finite_outputs =
        metrics.finite_outputs && eval.metrics.finite_outputs;
  };

  const PrototypeEvaluation k2_lap = evaluate_source_target_cloud(
      laplace, phi_source_dipole, target_base, true);
  const PrototypeEvaluation k2_yuk = evaluate_source_target_cloud(
      yukawa, phi_source_dipole, target_base, true);
  add_eval_metrics(k2_lap);
  add_eval_metrics(k2_yuk);

  const PrototypeEvaluation k3_lap_plus =
      evaluate_source_target_cloud(laplace, phin_source, target_plus, true);
  const PrototypeEvaluation k3_lap_minus =
      evaluate_source_target_cloud(laplace, phin_source, target_minus, true);
  const PrototypeEvaluation k3_yuk_plus =
      evaluate_source_target_cloud(yukawa, phin_source, target_plus, true);
  const PrototypeEvaluation k3_yuk_minus =
      evaluate_source_target_cloud(yukawa, phin_source, target_minus, true);
  add_eval_metrics(k3_lap_plus);
  add_eval_metrics(k3_lap_minus);
  add_eval_metrics(k3_yuk_plus);
  add_eval_metrics(k3_yuk_minus);

  const PrototypeEvaluation k4_lap_plus = evaluate_source_target_cloud(
      laplace, phi_source_dipole, target_plus, true);
  const PrototypeEvaluation k4_lap_minus = evaluate_source_target_cloud(
      laplace, phi_source_dipole, target_minus, true);
  const PrototypeEvaluation k4_yuk_plus = evaluate_source_target_cloud(
      yukawa, phi_source_dipole, target_plus, true);
  const PrototypeEvaluation k4_yuk_minus = evaluate_source_target_cloud(
      yukawa, phi_source_dipole, target_minus, true);
  add_eval_metrics(k4_lap_plus);
  add_eval_metrics(k4_lap_minus);
  add_eval_metrics(k4_yuk_plus);
  add_eval_metrics(k4_yuk_minus);

  AlignedDoubleVector approx_k2(panels.size(), 0.0);
  AlignedDoubleVector approx_k3(panels.size(), 0.0);
  AlignedDoubleVector approx_k4(panels.size(), 0.0);
  for (std::size_t i = 0; i < panels.size(); ++i) {
    approx_k2[i] = eps_ratio * k2_yuk.approx[i] - k2_lap.approx[i];
    const double lap_target_deriv =
        (k3_lap_plus.approx[i] - k3_lap_minus.approx[i]) / (2.0 * h);
    const double yuk_target_deriv =
        (k3_yuk_plus.approx[i] - k3_yuk_minus.approx[i]) / (2.0 * h);
    approx_k3[i] = lap_target_deriv - inv_eps_ratio * yuk_target_deriv;
    const double lap_mixed =
        (k4_lap_plus.approx[i] - k4_lap_minus.approx[i]) / (2.0 * h);
    const double yuk_mixed =
        (k4_yuk_plus.approx[i] - k4_yuk_minus.approx[i]) / (2.0 * h);
    approx_k4[i] = yuk_mixed - lap_mixed;
  }

  PrototypeMetrics e2, e3, e4;
  compute_error(direct.k2, approx_k2, e2);
  compute_error(direct.k3, approx_k3, e3);
  compute_error(direct.k4, approx_k4, e4);
  metrics.k2_rel_rms_error = e2.rel_rms_error;
  metrics.k3_rel_rms_error = e3.rel_rms_error;
  metrics.k4_rel_rms_error = e4.rel_rms_error;
  auto vector_rms = [](const AlignedDoubleVector &values) {
    double sum2 = 0.0;
    for (double value : values) {
      sum2 += value * value;
    }
    return values.empty()
               ? 0.0
               : std::sqrt(sum2 / static_cast<double>(values.size()));
  };
  metrics.k2_reference_rms = vector_rms(direct.k2);
  metrics.k3_reference_rms = vector_rms(direct.k3);
  metrics.k4_reference_rms = vector_rms(direct.k4);
  metrics.max_channel_rel_rms_error =
      std::max({e2.rel_rms_error, e3.rel_rms_error, e4.rel_rms_error});
  metrics.max_abs_error =
      std::max({e2.max_abs_error, e3.max_abs_error, e4.max_abs_error});
  metrics.finite_outputs = metrics.finite_outputs && e2.finite_outputs &&
                           e3.finite_outputs && e4.finite_outputs;
  const double err2 = e2.rms_abs_error * e2.rms_abs_error * panels.size() +
                      e3.rms_abs_error * e3.rms_abs_error * panels.size() +
                      e4.rms_abs_error * e4.rms_abs_error * panels.size();
  const double ref2 =
      (e2.rms_abs_error * e2.rms_abs_error * panels.size()) /
          std::max(e2.rel_rms_error * e2.rel_rms_error, 1.0e-30) +
      (e3.rms_abs_error * e3.rms_abs_error * panels.size()) /
          std::max(e3.rel_rms_error * e3.rel_rms_error, 1.0e-30) +
      (e4.rms_abs_error * e4.rms_abs_error * panels.size()) /
          std::max(e4.rel_rms_error * e4.rel_rms_error, 1.0e-30);
  metrics.aggregate_rel_rms_error = std::sqrt(err2 / std::max(ref2, 1.0e-30));
  return metrics;
}

std::string
classify_scope_f_decision(const JcpDerivativeBridgeMetrics &metrics) {
  if (!metrics.finite_outputs ||
      !std::isfinite(metrics.aggregate_rel_rms_error)) {
    return "accuracy_fail";
  }
  if (metrics.aggregate_rel_rms_error > 1.0e-1 ||
      metrics.max_channel_rel_rms_error > 2.0e-1) {
    return "accuracy_fail";
  }
  if (metrics.far_pairs == 0) {
    return "no_far_work";
  }
  if (metrics.setup_seconds > metrics.direct_seconds) {
    return "setup_limited";
  }
  if (!(metrics.apply_seconds < metrics.direct_seconds)) {
    return "no_speedup";
  }
  return "pass";
}

JcpOfflineOperatorResult
evaluate_jcp_offline_operator(const PrototypeConfig &input_config,
                              const std::vector<PrototypePanel> &panels,
                              double eps_ratio, double kappa) {
  if (panels.empty()) {
    throw std::invalid_argument(
        "JCP offline operator requires at least one panel");
  }
  PrototypeConfig laplace = input_config;
  laplace.kernel = KernelKind::Laplace;
  PrototypeConfig yukawa = input_config;
  yukawa.kernel = KernelKind::Yukawa;
  yukawa.kappa = kappa > 0.0 ? kappa : input_config.kappa;
  if (!(yukawa.kappa > 0.0)) {
    yukawa.kappa = 0.2;
  }
  constexpr double h = 1.0e-3;
  const double safe_eps_ratio = std::max(1.0e-12, eps_ratio);
  const double inv_eps_ratio = 1.0 / safe_eps_ratio;

  JcpOfflineOperatorResult result;
  result.panels = panels.size();
  result.phi.assign(panels.size(), 0.0);
  result.phin.assign(panels.size(), 0.0);

  const PrototypePointCloud target_base = make_panel_target_cloud(panels, 0.0);
  const PrototypePointCloud target_plus = make_panel_target_cloud(panels, h);
  const PrototypePointCloud target_minus = make_panel_target_cloud(panels, -h);
  const PrototypePointCloud phin_source = make_jcp_k1_source_cloud(panels);
  const PrototypePointCloud phi_source_dipole =
      make_source_normal_dipole_cloud(panels, true, h);

  auto add_eval_metrics = [&](const PrototypeEvaluation &eval) {
    result.setup_seconds += eval.metrics.setup_seconds;
    result.apply_seconds += eval.metrics.apply_seconds;
    result.near_pairs += eval.metrics.near_pairs;
    result.far_pairs += eval.metrics.far_pairs;
    result.finite_outputs =
        result.finite_outputs && eval.metrics.finite_outputs;
  };

  const PrototypeEvaluation k1_lap = evaluate_source_target_cloud(
      laplace, phin_source, phin_source, false, false);
  const PrototypeEvaluation k1_yuk = evaluate_source_target_cloud(
      yukawa, phin_source, phin_source, false, false);
  add_eval_metrics(k1_lap);
  add_eval_metrics(k1_yuk);

  const PrototypeEvaluation k2_lap = evaluate_source_target_cloud(
      laplace, phi_source_dipole, target_base, true, false);
  const PrototypeEvaluation k2_yuk = evaluate_source_target_cloud(
      yukawa, phi_source_dipole, target_base, true, false);
  add_eval_metrics(k2_lap);
  add_eval_metrics(k2_yuk);

  const PrototypeEvaluation k3_lap_plus = evaluate_source_target_cloud(
      laplace, phin_source, target_plus, true, false);
  const PrototypeEvaluation k3_lap_minus = evaluate_source_target_cloud(
      laplace, phin_source, target_minus, true, false);
  const PrototypeEvaluation k3_yuk_plus = evaluate_source_target_cloud(
      yukawa, phin_source, target_plus, true, false);
  const PrototypeEvaluation k3_yuk_minus = evaluate_source_target_cloud(
      yukawa, phin_source, target_minus, true, false);
  add_eval_metrics(k3_lap_plus);
  add_eval_metrics(k3_lap_minus);
  add_eval_metrics(k3_yuk_plus);
  add_eval_metrics(k3_yuk_minus);

  const PrototypeEvaluation k4_lap_plus = evaluate_source_target_cloud(
      laplace, phi_source_dipole, target_plus, true, false);
  const PrototypeEvaluation k4_lap_minus = evaluate_source_target_cloud(
      laplace, phi_source_dipole, target_minus, true, false);
  const PrototypeEvaluation k4_yuk_plus = evaluate_source_target_cloud(
      yukawa, phi_source_dipole, target_plus, true, false);
  const PrototypeEvaluation k4_yuk_minus = evaluate_source_target_cloud(
      yukawa, phi_source_dipole, target_minus, true, false);
  add_eval_metrics(k4_lap_plus);
  add_eval_metrics(k4_lap_minus);
  add_eval_metrics(k4_yuk_plus);
  add_eval_metrics(k4_yuk_minus);

  for (std::size_t i = 0; i < panels.size(); ++i) {
    const double k1 = k1_lap.approx[i] - k1_yuk.approx[i];
    const double k2 = safe_eps_ratio * k2_yuk.approx[i] - k2_lap.approx[i];
    const double lap_target_deriv =
        (k3_lap_plus.approx[i] - k3_lap_minus.approx[i]) / (2.0 * h);
    const double yuk_target_deriv =
        (k3_yuk_plus.approx[i] - k3_yuk_minus.approx[i]) / (2.0 * h);
    const double k3 = lap_target_deriv - inv_eps_ratio * yuk_target_deriv;
    const double lap_mixed =
        (k4_lap_plus.approx[i] - k4_lap_minus.approx[i]) / (2.0 * h);
    const double yuk_mixed =
        (k4_yuk_plus.approx[i] - k4_yuk_minus.approx[i]) / (2.0 * h);
    const double k4 = yuk_mixed - lap_mixed;
    result.phi[i] = k1 + k2;
    result.phin[i] = k3 + k4;
    result.finite_outputs = result.finite_outputs &&
                            std::isfinite(result.phi[i]) &&
                            std::isfinite(result.phin[i]);
  }
  return result;
}

JcpAnalyticOperatorResult evaluate_jcp_offline_operator_analytic(
    const PrototypeConfig &input_config,
    const std::vector<PrototypePanel> &panels, double eps_ratio, double kappa) {
  std::vector<double> phi_values(panels.size(), 0.0);
  std::vector<double> phin_values(panels.size(), 0.0);
  for (std::size_t i = 0; i < panels.size(); ++i) {
    phi_values[i] = panels[i].scalar_density;
    phin_values[i] = jcp_phin_for_panel(panels[i], i);
  }
  return evaluate_jcp_offline_operator_analytic(
      input_config, panels, phi_values, phin_values, eps_ratio, kappa);
}

JcpAnalyticOperatorResult evaluate_jcp_offline_operator_analytic(
    const PrototypeConfig &input_config,
    const std::vector<PrototypePanel> &panels,
    const std::vector<double> &phi_values,
    const std::vector<double> &phin_values, double eps_ratio, double kappa) {
  PrototypeConfig config = normalize_jcp_analytic_config(input_config);
  TitanExecutor executor(static_cast<std::size_t>(std::max(1, config.threads)));
  return evaluate_jcp_offline_operator_analytic(
      config, panels, phi_values, phin_values, eps_ratio, kappa, executor);
}

JcpAnalyticOperatorResult evaluate_jcp_offline_operator_analytic(
    const PrototypeConfig &input_config,
    const std::vector<PrototypePanel> &panels,
    const std::vector<double> &phi_values,
    const std::vector<double> &phin_values, double eps_ratio, double kappa,
    TitanExecutor &executor) {
  if (panels.empty()) {
    throw std::invalid_argument(
        "JCP analytic offline operator requires at least one panel");
  }
  if (phi_values.size() != panels.size() ||
      phin_values.size() != panels.size()) {
    throw std::invalid_argument(
        "JCP analytic offline operator density vectors must match panels");
  }
  FmmM2lAnalyticContext context(input_config, panels, eps_ratio, kappa);
  return context.apply(phi_values, phin_values, executor);
}

namespace {

constexpr double kScopeKGmresLhsPhi = 10.0;
constexpr double kScopeKGmresLhsPhin = 10.0;

double l2_norm(const std::vector<double> &values) {
  double sum = 0.0;
  for (double value : values) {
    sum += value * value;
  }
  return std::sqrt(sum);
}

std::vector<double> subtract_vectors(const std::vector<double> &a,
                                     const std::vector<double> &b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("Scope K vector size mismatch");
  }
  std::vector<double> out(a.size(), 0.0);
  for (std::size_t i = 0; i < a.size(); ++i) {
    out[i] = a[i] - b[i];
  }
  return out;
}

double relative_norm(const std::vector<double> &residual,
                     const std::vector<double> &rhs) {
  return l2_norm(residual) / std::max(l2_norm(rhs), 1.0e-30);
}

bool finite_vector(const std::vector<double> &values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

std::vector<double> make_scope_k_truth_vector(std::size_t panels) {
  std::vector<double> x(2 * panels, 0.0);
  for (std::size_t i = 0; i < panels; ++i) {
    const double t = static_cast<double>(i + 1);
    x[i] = 0.7 * std::cos(0.013 * t) + 0.3 * std::sin(0.029 * t);
    x[panels + i] = 0.65 * std::sin(0.017 * t) + 0.2 * std::cos(0.011 * t);
  }
  return x;
}

std::vector<double>
direct_jcp_scope_k_matvec(const std::vector<PrototypePanel> &panels,
                          const std::vector<double> &x, double eps_ratio,
                          double kappa) {
  const std::size_t n = panels.size();
  if (x.size() != 2 * n) {
    throw std::invalid_argument("Scope K direct matvec requires a 2N vector");
  }
  std::vector<double> out(2 * n, 0.0);
  for (std::size_t ti = 0; ti < n; ++ti) {
    double phi_sum = 0.0;
    double phin_sum = 0.0;
    for (std::size_t si = 0; si < n; ++si) {
      if (ti == si) {
        continue;
      }
      PrototypePanel source = panels[si];
      source.scalar_density = x[si];
      add_jcp_direct_pair(panels[ti], source, x[n + si], eps_ratio, kappa,
                          phi_sum, phin_sum);
    }
    out[ti] = kScopeKGmresLhsPhi * x[ti] - phi_sum;
    out[n + ti] = kScopeKGmresLhsPhin * x[n + ti] - phin_sum;
  }
  return out;
}

struct AdapterMatvecOutcome {
  std::vector<double> out;
  std::size_t near_pairs = 0;
  std::size_t far_pairs = 0;
  double setup_ms = 0.0;
  double apply_ms = 0.0;
  bool finite_outputs = true;
};

AdapterMatvecOutcome adapter_jcp_scope_k_matvec(
    const PrototypeConfig &config, const std::vector<PrototypePanel> &panels,
    const std::vector<double> &x, double eps_ratio, double kappa) {
  const std::size_t n = panels.size();
  if (x.size() != 2 * n) {
    throw std::invalid_argument("Scope K adapter matvec requires a 2N vector");
  }
  std::vector<double> phi(x.begin(),
                          x.begin() + static_cast<std::ptrdiff_t>(n));
  std::vector<double> phin(x.begin() + static_cast<std::ptrdiff_t>(n), x.end());
  const JcpAnalyticOperatorResult result =
      evaluate_jcp_offline_operator_analytic(config, panels, phi, phin,
                                             eps_ratio, kappa);
  AdapterMatvecOutcome out;
  out.out.assign(2 * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    out.out[i] = kScopeKGmresLhsPhi * x[i] - result.phi[i];
    out.out[n + i] = kScopeKGmresLhsPhin * x[n + i] - result.phin[i];
  }
  out.near_pairs = result.near_pairs;
  out.far_pairs = result.far_pairs;
  out.setup_ms = result.setup_seconds * 1000.0;
  out.apply_ms = result.apply_seconds * 1000.0;
  out.finite_outputs = result.finite_outputs && finite_vector(out.out);
  return out;
}

std::vector<double> solve_dense_system(std::vector<std::vector<double>> a,
                                       std::vector<double> b) {
  const std::size_t n = b.size();
  for (std::size_t col = 0; col < n; ++col) {
    std::size_t pivot = col;
    double best = std::abs(a[col][col]);
    for (std::size_t row = col + 1; row < n; ++row) {
      const double value = std::abs(a[row][col]);
      if (value > best) {
        best = value;
        pivot = row;
      }
    }
    if (!(best > 1.0e-30)) {
      a[col][col] += 1.0e-12;
      best = std::abs(a[col][col]);
    }
    if (pivot != col) {
      std::swap(a[pivot], a[col]);
      std::swap(b[pivot], b[col]);
    }
    const double diag = a[col][col];
    for (std::size_t row = col + 1; row < n; ++row) {
      const double factor = a[row][col] / diag;
      if (factor == 0.0) {
        continue;
      }
      for (std::size_t k = col; k < n; ++k) {
        a[row][k] -= factor * a[col][k];
      }
      b[row] -= factor * b[col];
    }
  }
  std::vector<double> x(n, 0.0);
  for (std::size_t ii = 0; ii < n; ++ii) {
    const std::size_t i = n - 1 - ii;
    double rhs = b[i];
    for (std::size_t j = i + 1; j < n; ++j) {
      rhs -= a[i][j] * x[j];
    }
    x[i] = rhs / a[i][i];
  }
  return x;
}

struct ScopeKGmresResult {
  std::vector<double> x;
  int iterations = 0;
  double initial_residual = 0.0;
  double final_residual = 0.0;
  double matvec_ms = 0.0;
  bool converged = false;
  bool finite_outputs = true;
};

ScopeKGmresResult run_scope_k_gmres(
    const std::function<std::vector<double>(const std::vector<double> &)>
        &matvec,
    const std::vector<double> &rhs, int restart, int max_iterations,
    double tolerance) {
  if (rhs.empty()) {
    throw std::invalid_argument("Scope K GMRES requires a non-empty RHS");
  }
  restart = std::max(1, restart);
  max_iterations = std::max(1, max_iterations);
  tolerance = tolerance > 0.0 ? tolerance : 1.0e-6;
  const std::size_t dim = rhs.size();
  ScopeKGmresResult result;
  result.x.assign(dim, 0.0);
  result.initial_residual = 1.0;
  result.final_residual = 1.0;
  const double rhs_norm = std::max(l2_norm(rhs), 1.0e-30);

  while (result.iterations < max_iterations && !result.converged) {
    const std::vector<double> ax = matvec(result.x);
    std::vector<double> residual = subtract_vectors(rhs, ax);
    const double beta = l2_norm(residual);
    const double relative = beta / rhs_norm;
    if (result.iterations == 0) {
      result.initial_residual = relative;
    }
    result.final_residual = relative;
    if (relative <= tolerance) {
      result.converged = true;
      break;
    }

    const int cycle = std::min(restart, max_iterations - result.iterations);
    std::vector<std::vector<double>> v(static_cast<std::size_t>(cycle) + 1,
                                       std::vector<double>(dim, 0.0));
    for (std::size_t i = 0; i < dim; ++i) {
      v[0][i] = residual[i] / beta;
    }
    std::vector<std::vector<double>> h(
        static_cast<std::size_t>(cycle) + 1,
        std::vector<double>(static_cast<std::size_t>(cycle), 0.0));
    std::vector<double> best_x = result.x;
    double best_residual = relative;
    int best_j = -1;

    for (int j = 0; j < cycle && result.iterations < max_iterations; ++j) {
      const auto mv_begin = Clock::now();
      std::vector<double> w = matvec(v[static_cast<std::size_t>(j)]);
      const auto mv_end = Clock::now();
      result.matvec_ms += elapsed_seconds(mv_begin, mv_end) * 1000.0;
      result.finite_outputs = result.finite_outputs && finite_vector(w);

      for (int i = 0; i <= j; ++i) {
        double dot = 0.0;
        for (std::size_t k = 0; k < dim; ++k) {
          dot += w[k] * v[static_cast<std::size_t>(i)][k];
        }
        h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = dot;
        for (std::size_t k = 0; k < dim; ++k) {
          w[k] -= dot * v[static_cast<std::size_t>(i)][k];
        }
      }
      const double h_next = l2_norm(w);
      h[static_cast<std::size_t>(j + 1)][static_cast<std::size_t>(j)] = h_next;
      if (h_next > 1.0e-30 && j + 1 < cycle + 1) {
        for (std::size_t k = 0; k < dim; ++k) {
          v[static_cast<std::size_t>(j + 1)][k] = w[k] / h_next;
        }
      }

      const std::size_t m = static_cast<std::size_t>(j + 1);
      std::vector<std::vector<double>> normal(m, std::vector<double>(m, 0.0));
      std::vector<double> normal_rhs(m, 0.0);
      for (std::size_t row = 0; row <= m; ++row) {
        const double g = row == 0 ? beta : 0.0;
        for (std::size_t col = 0; col < m; ++col) {
          normal_rhs[col] += h[row][col] * g;
          for (std::size_t col2 = 0; col2 < m; ++col2) {
            normal[col][col2] += h[row][col] * h[row][col2];
          }
        }
      }
      std::vector<double> y =
          solve_dense_system(std::move(normal), std::move(normal_rhs));
      std::vector<double> candidate = result.x;
      for (std::size_t col = 0; col < m; ++col) {
        for (std::size_t k = 0; k < dim; ++k) {
          candidate[k] += y[col] * v[col][k];
        }
      }
      const auto residual_begin = Clock::now();
      const std::vector<double> candidate_ax = matvec(candidate);
      const auto residual_end = Clock::now();
      result.matvec_ms +=
          elapsed_seconds(residual_begin, residual_end) * 1000.0;
      result.finite_outputs = result.finite_outputs &&
                              finite_vector(candidate) &&
                              finite_vector(candidate_ax);
      const double candidate_residual =
          relative_norm(subtract_vectors(rhs, candidate_ax), rhs);
      ++result.iterations;
      if (candidate_residual < best_residual) {
        best_residual = candidate_residual;
        best_x = std::move(candidate);
        best_j = j;
      }
      if (candidate_residual <= tolerance) {
        result.x = best_x;
        result.final_residual = candidate_residual;
        result.converged = true;
        break;
      }
      if (!(h_next > 1.0e-30)) {
        break;
      }
    }
    if (!result.converged) {
      if (best_j < 0) {
        break;
      }
      result.x = std::move(best_x);
      result.final_residual = best_residual;
    }
  }
  return result;
}

} // namespace

JcpGuardedGmresMetrics
run_jcp_fmm_m2l_guarded_gmres_probe(const PrototypeConfig &input_config,
                                    const std::vector<PrototypePanel> &panels,
                                    double eps_ratio, double kappa, int restart,
                                    int max_iterations, double tolerance) {
  if (panels.empty()) {
    throw std::invalid_argument("Scope K GMRES probe requires panels");
  }
  PrototypeConfig config = input_config;
  config.kernel = KernelKind::Yukawa;
  config.kappa = kappa > 0.0 ? kappa : input_config.kappa;
  config.max_leaf_size = std::max(1, config.max_leaf_size);
  config.threads = std::max(1, config.threads);

  JcpGuardedGmresMetrics metrics;
  metrics.panels = panels.size();
  const std::vector<double> x_true = make_scope_k_truth_vector(panels.size());
  const auto direct_matvec = [&](const std::vector<double> &x) {
    return direct_jcp_scope_k_matvec(panels, x, eps_ratio, config.kappa);
  };
  AdapterMatvecOutcome representative_adapter;
  const auto adapter_matvec = [&](const std::vector<double> &x) {
    AdapterMatvecOutcome outcome =
        adapter_jcp_scope_k_matvec(config, panels, x, eps_ratio, config.kappa);
    if (representative_adapter.out.empty()) {
      representative_adapter = outcome;
    }
    metrics.finite_outputs = metrics.finite_outputs && outcome.finite_outputs;
    return outcome.out;
  };

  const std::vector<double> rhs = direct_matvec(x_true);
  const ScopeKGmresResult direct =
      run_scope_k_gmres(direct_matvec, rhs, restart, max_iterations, tolerance);
  const ScopeKGmresResult adapter = run_scope_k_gmres(
      adapter_matvec, rhs, restart, max_iterations, tolerance);

  const std::vector<double> adapter_direct_ax = direct_matvec(adapter.x);
  const std::vector<double> solution_delta =
      subtract_vectors(adapter.x, x_true);
  metrics.direct_iterations = direct.iterations;
  metrics.adapter_iterations = adapter.iterations;
  metrics.direct_initial_residual = direct.initial_residual;
  metrics.direct_final_residual = direct.final_residual;
  metrics.adapter_initial_residual = adapter.initial_residual;
  metrics.adapter_final_residual = adapter.final_residual;
  metrics.direct_true_residual =
      relative_norm(subtract_vectors(rhs, adapter_direct_ax), rhs);
  metrics.solution_rel_error =
      l2_norm(solution_delta) / std::max(l2_norm(x_true), 1.0e-30);
  metrics.direct_matvec_ms =
      direct.iterations > 0
          ? direct.matvec_ms / static_cast<double>(direct.iterations)
          : direct.matvec_ms;
  metrics.adapter_matvec_ms =
      adapter.iterations > 0
          ? adapter.matvec_ms / static_cast<double>(adapter.iterations)
          : adapter.matvec_ms;
  metrics.speedup_vs_direct_matvec =
      metrics.adapter_matvec_ms > 0.0
          ? metrics.direct_matvec_ms / metrics.adapter_matvec_ms
          : 0.0;
  metrics.direct_converged = direct.converged;
  metrics.adapter_converged = adapter.converged;
  metrics.adapter_residual_decreased =
      adapter.final_residual <
      0.99 * std::max(adapter.initial_residual, 1.0e-30);
  metrics.finite_outputs = metrics.finite_outputs && direct.finite_outputs &&
                           adapter.finite_outputs && finite_vector(rhs) &&
                           finite_vector(adapter_direct_ax);
  if (!representative_adapter.out.empty()) {
    metrics.near_pairs = representative_adapter.near_pairs;
    metrics.far_pairs = representative_adapter.far_pairs;
    metrics.adapter_setup_ms = representative_adapter.setup_ms;
    metrics.adapter_apply_ms = representative_adapter.apply_ms;
  }
  return metrics;
}

JcpChannelPrototypeMetrics
run_jcp_channel_probe(const PrototypeConfig &config,
                      const std::vector<PrototypePanel> &panels) {
  const auto channels = make_jcp_channel_point_clouds(panels);
  JcpChannelPrototypeMetrics out;
  out.panels = panels.size();
  out.channels = channels.size();

  double weighted_err2 = 0.0;
  double weighted_ref2 = 0.0;
  for (const PrototypePointCloud &cloud : channels) {
    const PrototypeMetrics m = run_point_cloud_probe(config, cloud);
    out.near_pairs += m.near_pairs;
    out.far_pairs += m.far_pairs;
    out.setup_seconds += m.setup_seconds;
    out.apply_seconds += m.apply_seconds;
    out.direct_seconds += m.direct_seconds;
    out.max_channel_rel_rms_error =
        std::max(out.max_channel_rel_rms_error, m.rel_rms_error);
    out.max_abs_error = std::max(out.max_abs_error, m.max_abs_error);
    out.finite_outputs = out.finite_outputs && m.finite_outputs;

    const double samples =
        static_cast<double>(std::max<std::size_t>(1, m.targets));
    const double err2 = m.rms_abs_error * m.rms_abs_error * samples;
    const double ref2 =
        err2 / std::max(m.rel_rms_error * m.rel_rms_error, 1.0e-30);
    weighted_err2 += err2;
    weighted_ref2 += ref2;
  }
  out.aggregate_rel_rms_error =
      std::sqrt(weighted_err2 / std::max(weighted_ref2, 1.0e-30));
  return out;
}

std::string
classify_scope_d_decision(const JcpChannelPrototypeMetrics &metrics) {
  if (!metrics.finite_outputs ||
      !std::isfinite(metrics.aggregate_rel_rms_error)) {
    return "accuracy_fail";
  }
  if (metrics.aggregate_rel_rms_error > 5.0e-2 ||
      metrics.max_channel_rel_rms_error > 1.0e-1) {
    return "accuracy_fail";
  }
  if (metrics.far_pairs == 0) {
    return "no_far_work";
  }
  if (metrics.setup_seconds > metrics.direct_seconds) {
    return "setup_limited";
  }
  if (!(metrics.apply_seconds < metrics.direct_seconds)) {
    return "no_speedup";
  }
  return "pass";
}

std::string classify_scope_a_decision(const PrototypeMetrics &metrics,
                                      const PrototypeConfig &config) {
  if (!metrics.finite_outputs || !std::isfinite(metrics.rel_rms_error)) {
    return "accuracy_fail";
  }
  const double rel_gate = config.kernel == KernelKind::Yukawa ? 3.0e-3 : 1.0e-3;
  if (metrics.rel_rms_error > rel_gate) {
    return "accuracy_fail";
  }
  if (metrics.setup_seconds > metrics.direct_seconds) {
    return "setup_limited";
  }
  if (!(metrics.apply_seconds < metrics.direct_seconds)) {
    return "no_speedup";
  }
  if (metrics.apply_seconds <= 0.5 * metrics.direct_seconds) {
    return "promote_to_scope_b";
  }
  return "promising_scalar";
}

} // namespace titan::tabi::fmm_m2l

namespace TitanPBE::Tabi {

std::shared_ptr<titan::tabi::fmm_m2l::FmmM2lAnalyticContext>
ensure_jcp_fmm_m2l_context(
    const std::vector<Panel> &panels, double eps_ratio, double kappa,
    std::size_t leaf_size, double opening_theta, int expansion_order,
    TitanExecutor &executor,
    std::shared_ptr<titan::tabi::fmm_m2l::FmmM2lAnalyticContext>
        *context_cache) {
  titan::tabi::fmm_m2l::PrototypeConfig config;
  config.kernel = titan::tabi::fmm_m2l::KernelKind::Yukawa;
  config.kappa = kappa;
  config.max_leaf_size = static_cast<int>(
      std::clamp<std::size_t>(leaf_size == 0 ? 64 : leaf_size, 8, 512));
  config.expansion_order = std::clamp(expansion_order, 2, 5);
  config.separation_theta = std::clamp(
      std::isfinite(opening_theta) && opening_theta > 0.0 ? opening_theta : 0.8,
      0.05, 2.0);
  config.threads =
      static_cast<int>(std::max<std::size_t>(1, executor.capacity()));

  std::shared_ptr<titan::tabi::fmm_m2l::FmmM2lAnalyticContext> local_context;
  std::shared_ptr<titan::tabi::fmm_m2l::FmmM2lAnalyticContext> &context =
      context_cache != nullptr ? *context_cache : local_context;
  if (context == nullptr ||
      !context->matches(config, panels.size(), eps_ratio, kappa)) {
    std::vector<titan::tabi::fmm_m2l::PrototypePanel> prototype;
    prototype.reserve(panels.size());
    for (const Panel &panel : panels) {
      titan::tabi::fmm_m2l::PrototypePanel converted;
      converted.x = panel.x;
      converted.y = panel.y;
      converted.z = panel.z;
      converted.nx = panel.nx;
      converted.ny = panel.ny;
      converted.nz = panel.nz;
      converted.area = panel.area;
      converted.source_x = panel.source_x;
      converted.source_y = panel.source_y;
      converted.source_z = panel.source_z;
      converted.source_nx = panel.source_nx;
      converted.source_ny = panel.source_ny;
      converted.source_nz = panel.source_nz;
      converted.has_source_quadrature_center =
          panel.has_source_quadrature_center;
      prototype.push_back(converted);
    }
    context = std::make_shared<titan::tabi::fmm_m2l::FmmM2lAnalyticContext>(
        config, std::move(prototype), eps_ratio, kappa);
  }
  return context;
}

void apply_jcp_fmm_m2l_single_matvec(
    const std::vector<Panel> &panels, const std::vector<double> &unknowns,
    double eps_ratio, double kappa, double lhs_phi, double lhs_phin,
    std::size_t leaf_size, double opening_theta, int expansion_order,
    TitanExecutor &executor, std::vector<double> &out,
    FmmM2lSingleMatvecStats *stats, bool coupled,
    std::shared_ptr<titan::tabi::fmm_m2l::FmmM2lAnalyticContext> *context_cache,
    const std::string &near_integration, const std::string &quad_buffer,
    FmmM2lSingleMatvecWorkspace *workspace) {
  const std::size_t n = panels.size();
  if (n == 0) {
    out.clear();
    if (stats != nullptr) {
      *stats = FmmM2lSingleMatvecStats{};
    }
    return;
  }
  if (unknowns.size() < (coupled ? 2 * n : n)) {
    throw std::invalid_argument("PBE BIE FMM/M2L single-matvec received an "
                                "incomplete JCP unknown vector");
  }

  FmmM2lSingleMatvecWorkspace local_workspace;
  FmmM2lSingleMatvecWorkspace &active_workspace =
      workspace != nullptr ? *workspace : local_workspace;
  active_workspace.phi.resize(n);
  active_workspace.phin.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    active_workspace.phi[i] = unknowns[i];
    active_workspace.phin[i] = coupled ? unknowns[n + i] : 0.0;
  }

  const std::shared_ptr<titan::tabi::fmm_m2l::FmmM2lAnalyticContext> context =
      ensure_jcp_fmm_m2l_context(panels, eps_ratio, kappa, leaf_size,
                                 opening_theta, expansion_order, executor,
                                 context_cache);

  const titan::tabi::fmm_m2l::JcpAnalyticOperatorResult result =
      context->apply_into(
          active_workspace.phi, active_workspace.phin, executor,
          active_workspace.interaction_phi,
          active_workspace.interaction_phin, coupled, near_integration,
          quad_buffer, &panels);

  out.resize(coupled ? 2 * n : n);
  bool finite = result.finite_outputs;
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = jcp_lhs_phi_for_panel(panels[i], eps_ratio, lhs_phi) *
                 unknowns[i] -
             active_workspace.interaction_phi[i];
    finite = finite && std::isfinite(out[i]);
    if (coupled) {
      out[n + i] = jcp_lhs_phin_for_panel(panels[i], eps_ratio, lhs_phin) *
                       unknowns[n + i] -
                   active_workspace.interaction_phin[i];
      finite = finite && std::isfinite(out[n + i]);
    }
  }

  if (stats != nullptr) {
    stats->near_pairs = result.near_pairs;
    stats->far_pairs = result.far_pairs;
    stats->near_self_pairs = result.near_self_pairs;
    stats->near_one_ring_pairs = result.near_one_ring_pairs;
    stats->near_duffy_pairs = result.near_duffy_pairs;
    stats->near_centroid_pairs = result.near_centroid_pairs;
    stats->near_fallbacks = result.near_fallbacks;
    stats->duffy_subdivisions_max = result.duffy_subdivisions_max;
    stats->duffy_subdivisions_mean = result.duffy_subdivisions_mean;
    stats->near_kernel_policy = result.near_kernel_policy;
    stats->expansion_order = std::clamp(expansion_order, 2, 5);
    stats->setup_ms = 1000.0 * result.setup_seconds;
    stats->apply_ms = 1000.0 * result.apply_seconds;
    stats->cached = context_cache != nullptr;
    stats->context_nodes = result.context_nodes;
    stats->context_work_items = result.context_work_items;
    stats->finite_outputs = finite;
  }
}

} // namespace TitanPBE::Tabi
