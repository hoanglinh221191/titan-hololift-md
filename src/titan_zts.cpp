#include "titan_zts.h"
#include "titan_executor.h"
#include "titan_ui.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>

namespace TitanZTS {
namespace {

constexpr double kDefaultFreeEnergy = 20.0;
constexpr double kMinimumSigma = 1e-8;
constexpr double kMinimumArcLength = 1e-12;

struct PathSoA {
  AlignedDoubleVector pc1;
  AlignedDoubleVector pc2;
  AlignedDoubleVector fes;
  AlignedDoubleVector arc;

  void assign(size_t n, double value = 0.0) {
    pc1.assign(n, value);
    pc2.assign(n, value);
    fes.assign(n, value);
    arc.assign(n, value);
  }

  size_t size() const { return pc1.size(); }
  bool empty() const { return pc1.empty(); }
};

size_t choose_worker_count(size_t requested_threads, size_t work_items,
                           size_t min_items_per_worker) {
  if (work_items == 0)
    return 1;
  const size_t capped_threads = std::max<size_t>(1, requested_threads);
  const size_t max_useful =
      std::max<size_t>(1, work_items / std::max<size_t>(size_t(1), min_items_per_worker));
  return std::max<size_t>(1, std::min(capped_threads, max_useful));
}

size_t grid_index(const TitanPCA::FES2DResult &fes, int ix, int iy) {
  return static_cast<size_t>(ix) * static_cast<size_t>(fes.grid_size) +
         static_cast<size_t>(iy);
}

double first_center_x(const TitanPCA::FES2DResult &fes) {
  return fes.min_x + 0.5 * fes.step_x;
}

double first_center_y(const TitanPCA::FES2DResult &fes) {
  return fes.min_y + 0.5 * fes.step_y;
}

double max_center_x(const TitanPCA::FES2DResult &fes) {
  return first_center_x(fes) +
         fes.step_x * static_cast<double>(std::max(fes.grid_size - 1, 0));
}

double max_center_y(const TitanPCA::FES2DResult &fes) {
  return first_center_y(fes) +
         fes.step_y * static_cast<double>(std::max(fes.grid_size - 1, 0));
}

void clamp_to_bounds(const TitanPCA::FES2DResult &fes, double &x, double &y) {
  x = std::clamp(x, first_center_x(fes), max_center_x(fes));
  y = std::clamp(y, first_center_y(fes), max_center_y(fes));
}

void gather_fes_field(const TitanPCA::FES2DResult &fes, AlignedDoubleVector &field,
                      AlignedIntVector &visited_mask) {
  field.assign(fes.bins.size(), kDefaultFreeEnergy);
  visited_mask.assign(fes.bins.size(), 0);
  for (size_t idx = 0; idx < fes.bins.size(); ++idx) {
    const auto &bin = fes.bins[idx];
    const bool visited = (bin.count_w > 0.0 || bin.count > 0);
    field[idx] = bin.free_energy;
    visited_mask[idx] = visited ? 1 : 0;
  }
}

AlignedDoubleVector build_smoothed_field(const TitanPCA::FES2DResult &fes,
                                         const AlignedDoubleVector &field,
                                         const AlignedIntVector &visited_mask,
                                         double sigma, TitanExecutor &executor) {
  if (sigma <= kMinimumSigma || fes.grid_size <= 1)
    return field;

  const int radius = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
  AlignedDoubleVector smoothed(field.size(), kDefaultFreeEnergy);
  const size_t worker_count =
      choose_worker_count(executor.capacity(), field.size(), 256);
  executor.parallel_do(worker_count, [&](size_t worker) {
    const size_t begin = field.size() * worker / worker_count;
    const size_t end = field.size() * (worker + 1) / worker_count;
    for (size_t linear = begin; linear < end; ++linear) {
      if (visited_mask[linear] == 0) {
        smoothed[linear] = field[linear];
        continue;
      }
      const int ix = static_cast<int>(linear / static_cast<size_t>(fes.grid_size));
      const int iy = static_cast<int>(linear % static_cast<size_t>(fes.grid_size));
      double weighted_sum = 0.0;
      double weight_total = 0.0;
      for (int dx = -radius; dx <= radius; ++dx) {
        const int nx = ix + dx;
        if (nx < 0 || nx >= fes.grid_size)
          continue;
        for (int dy = -radius; dy <= radius; ++dy) {
          const int ny = iy + dy;
          if (ny < 0 || ny >= fes.grid_size)
            continue;
          const size_t neighbor_idx = grid_index(fes, nx, ny);
          if (visited_mask[neighbor_idx] == 0)
            continue;
          const double dist2 = static_cast<double>(dx * dx + dy * dy);
          const double weight = std::exp(-0.5 * dist2 / (sigma * sigma));
          weighted_sum += weight * field[neighbor_idx];
          weight_total += weight;
        }
      }
      smoothed[linear] =
          (weight_total > 0.0) ? (weighted_sum / weight_total) : field[linear];
    }
  });
  return smoothed;
}

struct InterpolationStencil {
  int ix0{0};
  int ix1{0};
  int iy0{0};
  int iy1{0};
  double tx{0.0};
  double ty{0.0};
};

InterpolationStencil make_stencil(const TitanPCA::FES2DResult &fes, double x,
                                  double y) {
  InterpolationStencil stencil;
  if (fes.grid_size <= 1)
    return stencil;

  clamp_to_bounds(fes, x, y);
  const double gx = (x - first_center_x(fes)) / fes.step_x;
  const double gy = (y - first_center_y(fes)) / fes.step_y;
  const double gx_clamped =
      std::clamp(gx, 0.0, static_cast<double>(fes.grid_size - 1));
  const double gy_clamped =
      std::clamp(gy, 0.0, static_cast<double>(fes.grid_size - 1));

  stencil.ix0 = static_cast<int>(std::floor(gx_clamped));
  stencil.iy0 = static_cast<int>(std::floor(gy_clamped));
  stencil.ix1 = std::min(stencil.ix0 + 1, fes.grid_size - 1);
  stencil.iy1 = std::min(stencil.iy0 + 1, fes.grid_size - 1);
  stencil.tx = gx_clamped - static_cast<double>(stencil.ix0);
  stencil.ty = gy_clamped - static_cast<double>(stencil.iy0);
  return stencil;
}

double bilinear_value(const TitanPCA::FES2DResult &fes,
                      const AlignedDoubleVector &field, double x, double y) {
  if (field.empty())
    return 0.0;
  const InterpolationStencil s = make_stencil(fes, x, y);
  const double v00 = field[grid_index(fes, s.ix0, s.iy0)];
  const double v10 = field[grid_index(fes, s.ix1, s.iy0)];
  const double v01 = field[grid_index(fes, s.ix0, s.iy1)];
  const double v11 = field[grid_index(fes, s.ix1, s.iy1)];
  const double vx0 = (1.0 - s.tx) * v00 + s.tx * v10;
  const double vx1 = (1.0 - s.tx) * v01 + s.tx * v11;
  return (1.0 - s.ty) * vx0 + s.ty * vx1;
}

std::pair<double, double> bilinear_gradient(const TitanPCA::FES2DResult &fes,
                                            const AlignedDoubleVector &field,
                                            double x, double y) {
  if (field.empty() || fes.grid_size <= 1)
    return {0.0, 0.0};
  const InterpolationStencil s = make_stencil(fes, x, y);
  const double v00 = field[grid_index(fes, s.ix0, s.iy0)];
  const double v10 = field[grid_index(fes, s.ix1, s.iy0)];
  const double v01 = field[grid_index(fes, s.ix0, s.iy1)];
  const double v11 = field[grid_index(fes, s.ix1, s.iy1)];

  double d_dx = 0.0;
  double d_dy = 0.0;
  if (s.ix1 != s.ix0 && fes.step_x > 0.0) {
    d_dx = ((1.0 - s.ty) * (v10 - v00) + s.ty * (v11 - v01)) / fes.step_x;
  }
  if (s.iy1 != s.iy0 && fes.step_y > 0.0) {
    d_dy = ((1.0 - s.tx) * (v01 - v00) + s.tx * (v11 - v10)) / fes.step_y;
  }
  return {d_dx, d_dy};
}

double compute_arc_lengths(const PathSoA &path, AlignedDoubleVector &arc_out) {
  arc_out.assign(path.size(), 0.0);
  if (path.size() < 2)
    return 0.0;
  for (size_t i = 1; i < path.size(); ++i) {
    const double dx = path.pc1[i] - path.pc1[i - 1];
    const double dy = path.pc2[i] - path.pc2[i - 1];
    arc_out[i] = arc_out[i - 1] + std::sqrt(dx * dx + dy * dy);
  }
  return arc_out.back();
}

void initialize_linear_path(PathSoA &path, const TitanPCA::ClusterNode &source,
                            const TitanPCA::ClusterNode &sink) {
  const size_t n = path.size();
  if (n == 0)
    return;
  for (size_t i = 0; i < n; ++i) {
    const double t = (n == 1) ? 0.0
                              : static_cast<double>(i) /
                                    static_cast<double>(n - 1);
    path.pc1[i] = source.min_pc1 + t * (sink.min_pc1 - source.min_pc1);
    path.pc2[i] = source.min_pc2 + t * (sink.min_pc2 - source.min_pc2);
  }
}

void reparameterize_path(const TitanPCA::FES2DResult &fes, PathSoA &path,
                         const TitanPCA::ClusterNode &source,
                         const TitanPCA::ClusterNode &sink) {
  if (path.size() < 3)
    return;

  AlignedDoubleVector arc;
  const double total_length = compute_arc_lengths(path, arc);
  if (total_length <= kMinimumArcLength) {
    initialize_linear_path(path, source, sink);
    return;
  }

  AlignedDoubleVector new_pc1(path.size(), 0.0);
  AlignedDoubleVector new_pc2(path.size(), 0.0);
  new_pc1.front() = source.min_pc1;
  new_pc2.front() = source.min_pc2;
  new_pc1.back() = sink.min_pc1;
  new_pc2.back() = sink.min_pc2;

  size_t seg = 0;
  for (size_t i = 1; i + 1 < path.size(); ++i) {
    const double target =
        total_length * static_cast<double>(i) / static_cast<double>(path.size() - 1);
    while (seg + 1 < arc.size() && arc[seg + 1] < target)
      ++seg;
    const size_t next_seg = std::min(seg + 1, arc.size() - 1);
    const double seg_len = arc[next_seg] - arc[seg];
    const double alpha =
        (seg_len > kMinimumArcLength) ? (target - arc[seg]) / seg_len : 0.0;
    new_pc1[i] = (1.0 - alpha) * path.pc1[seg] + alpha * path.pc1[next_seg];
    new_pc2[i] = (1.0 - alpha) * path.pc2[seg] + alpha * path.pc2[next_seg];
    clamp_to_bounds(fes, new_pc1[i], new_pc2[i]);
  }

  path.pc1.swap(new_pc1);
  path.pc2.swap(new_pc2);
}

const TitanPCA::ClusterNode *find_cluster_by_id(
    const std::vector<TitanPCA::ClusterNode> &clusters, int id) {
  for (const auto &cluster : clusters) {
    if (cluster.id == id)
      return &cluster;
  }
  return nullptr;
}

bool select_endpoints(const ZtsArgs &zts_args,
                      const std::vector<TitanPCA::ClusterNode> &clusters,
                      TitanPCA::ClusterNode &source,
                      TitanPCA::ClusterNode &sink) {
  if (clusters.size() < 2) {
    TitanUI::print_error("ZTS: need at least two FES basins to build a path.");
    return false;
  }

  if (zts_args.zts_source >= 0 && zts_args.zts_sink >= 0) {
    const TitanPCA::ClusterNode *source_ptr =
        find_cluster_by_id(clusters, zts_args.zts_source);
    const TitanPCA::ClusterNode *sink_ptr =
        find_cluster_by_id(clusters, zts_args.zts_sink);
    if (!source_ptr || !sink_ptr) {
      TitanUI::print_error(
          "ZTS: requested source/sink basin ID is not present in the detected basin list.");
      return false;
    }
    source = *source_ptr;
    sink = *sink_ptr;
    return true;
  }

  std::vector<const TitanPCA::ClusterNode *> ranked;
  ranked.reserve(clusters.size());
  for (const auto &cluster : clusters)
    ranked.push_back(&cluster);
  std::sort(ranked.begin(), ranked.end(),
            [](const TitanPCA::ClusterNode *lhs,
               const TitanPCA::ClusterNode *rhs) {
              if (lhs->min_energy != rhs->min_energy)
                return lhs->min_energy < rhs->min_energy;
              return lhs->id < rhs->id;
            });
  source = *ranked[0];
  sink = *ranked[1];
  return true;
}

void evaluate_path_energies(const TitanPCA::FES2DResult &fes,
                            const AlignedDoubleVector &field, PathSoA &path,
                            TitanExecutor &executor) {
  if (path.empty())
    return;
  const size_t worker_count =
      (path.size() >= 128)
          ? choose_worker_count(executor.capacity(), path.size(), 32)
          : size_t(1);
  if (worker_count <= 1) {
    for (size_t i = 0; i < path.size(); ++i) {
      path.fes[i] = bilinear_value(fes, field, path.pc1[i], path.pc2[i]);
    }
    return;
  }
  executor.parallel_do(worker_count, [&](size_t worker) {
    const size_t begin = path.size() * worker / worker_count;
    const size_t end = path.size() * (worker + 1) / worker_count;
    for (size_t i = begin; i < end; ++i)
      path.fes[i] = bilinear_value(fes, field, path.pc1[i], path.pc2[i]);
  });
}

std::string convergence_status(const ZtsResult &result) {
  return result.converged ? "yes" : "no (max iter reached)";
}

} // namespace

ZtsResult run_zts(const ZtsArgs &zts_args, const TitanPCA::FES2DResult &fes,
                  const std::vector<TitanPCA::ClusterNode> &clusters,
                  TitanExecutor &executor, std::fstream &log) {
  ZtsResult result;
  if (fes.grid_size <= 1 || fes.bins.empty())
    return result;

  TitanPCA::ClusterNode source;
  TitanPCA::ClusterNode sink;
  if (!select_endpoints(zts_args, clusters, source, sink))
    return result;

  TitanUI::log_kv(log, "ZTS Images", std::to_string(zts_args.zts_images));
  TitanUI::log_kv(log, "ZTS Max Iter", std::to_string(zts_args.zts_max_iter));
  TitanUI::log_kv(log, "ZTS Step", std::to_string(zts_args.zts_step));
  TitanUI::log_kv(log, "ZTS Tol", std::to_string(zts_args.zts_tol));
  TitanUI::log_kv(log, "ZTS Smooth FES", zts_args.zts_smooth_fes ? "yes" : "no");
  TitanUI::log_kv(log, "ZTS Sigma", std::to_string(zts_args.zts_fes_sigma));
  TitanUI::log_kv(log, "ZTS Source Basin", std::to_string(source.id));
  TitanUI::log_kv(log, "ZTS Sink Basin", std::to_string(sink.id));

  AlignedDoubleVector base_field;
  AlignedIntVector visited_mask;
  gather_fes_field(fes, base_field, visited_mask);
  const AlignedDoubleVector working_field =
      zts_args.zts_smooth_fes
          ? build_smoothed_field(fes, base_field, visited_mask,
                                 zts_args.zts_fes_sigma, executor)
          : base_field;

  PathSoA path;
  path.assign(static_cast<size_t>(zts_args.zts_images), 0.0);
  initialize_linear_path(path, source, sink);

  AlignedDoubleVector prev_pc1(path.size(), 0.0);
  AlignedDoubleVector prev_pc2(path.size(), 0.0);

  for (int iter = 0; iter < zts_args.zts_max_iter; ++iter) {
    prev_pc1 = path.pc1;
    prev_pc2 = path.pc2;

    for (size_t image_idx = 1; image_idx + 1 < path.size(); ++image_idx) {
      auto [grad_x, grad_y] = bilinear_gradient(fes, working_field,
                                                path.pc1[image_idx],
                                                path.pc2[image_idx]);
      path.pc1[image_idx] -= zts_args.zts_step * grad_x;
      path.pc2[image_idx] -= zts_args.zts_step * grad_y;
      clamp_to_bounds(fes, path.pc1[image_idx], path.pc2[image_idx]);
    }

    reparameterize_path(fes, path, source, sink);
    path.pc1.front() = source.min_pc1;
    path.pc2.front() = source.min_pc2;
    path.pc1.back() = sink.min_pc1;
    path.pc2.back() = sink.min_pc2;

    double max_displacement = 0.0;
    for (size_t image_idx = 0; image_idx < path.size(); ++image_idx) {
      const double dx = path.pc1[image_idx] - prev_pc1[image_idx];
      const double dy = path.pc2[image_idx] - prev_pc2[image_idx];
      max_displacement =
          std::max(max_displacement, std::sqrt(dx * dx + dy * dy));
    }

    result.iterations_run = iter + 1;
    if (max_displacement < zts_args.zts_tol) {
      result.converged = true;
      break;
    }
  }

  evaluate_path_energies(fes, working_field, path, executor);
  compute_arc_lengths(path, path.arc);

  double max_energy = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < path.size(); ++i) {
    if (path.fes[i] > max_energy) {
      max_energy = path.fes[i];
      result.saddle_image_idx = static_cast<int>(i);
    }
  }

  if (result.saddle_image_idx >= 0) {
    const size_t saddle_idx = static_cast<size_t>(result.saddle_image_idx);
    result.saddle_pc1 = path.pc1[saddle_idx];
    result.saddle_pc2 = path.pc2[saddle_idx];
    result.saddle_energy = path.fes[saddle_idx];
    if (!path.fes.empty()) {
      result.barrier_fwd = result.saddle_energy - path.fes.front();
      result.barrier_rev = result.saddle_energy - path.fes.back();
    }
  }

  result.path.reserve(path.size());
  for (size_t i = 0; i < path.size(); ++i) {
    result.path.push_back({path.pc1[i], path.pc2[i], path.fes[i]});
  }

  TitanUI::log_kv(log, "ZTS Iterations", std::to_string(result.iterations_run));
  TitanUI::log_kv(log, "ZTS Converged", convergence_status(result));
  TitanUI::log_kv(log, "ZTS Saddle Image", std::to_string(result.saddle_image_idx));
  return result;
}

void write_zts_path(const std::string &filename, const ZtsResult &result) {
  if (result.path.empty())
    return;

  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_error("ZTS: cannot open " + filename + " for writing.");
    return;
  }

  out << "Image,PC1,PC2,FreeEnergy,ArcLength,IsSaddle\n";
  double arc_length = 0.0;
  for (size_t i = 0; i < result.path.size(); ++i) {
    if (i > 0) {
      const double dx = result.path[i].pc1 - result.path[i - 1].pc1;
      const double dy = result.path[i].pc2 - result.path[i - 1].pc2;
      arc_length += std::sqrt(dx * dx + dy * dy);
    }
    out << i << "," << result.path[i].pc1 << "," << result.path[i].pc2 << ","
        << result.path[i].fes << "," << arc_length << ","
        << ((static_cast<int>(i) == result.saddle_image_idx) ? 1 : 0) << "\n";
  }
}

} // namespace TitanZTS
