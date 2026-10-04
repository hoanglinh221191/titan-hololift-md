#include "pbe/grid/ses_mask.h"

#include "titan_executor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace TitanPBE {
namespace {

struct DistanceScratch {
  std::vector<double> source;
  std::vector<double> transformed;
  std::vector<double> boundaries;
  std::vector<std::size_t> sites;

  void resize(std::size_t n) {
    source.resize(n);
    transformed.resize(n);
    boundaries.resize(n + 1);
    sites.resize(n);
  }
};

void distance_transform_1d(std::span<const double> source,
                           std::span<double> transformed,
                           std::span<std::size_t> sites,
                           std::span<double> boundaries) {
  const std::size_t n = source.size();
  if (n == 0)
    return;

  std::size_t envelope_size = 0;
  sites[0] = 0;
  boundaries[0] = -std::numeric_limits<double>::infinity();
  boundaries[1] = std::numeric_limits<double>::infinity();

  for (std::size_t q = 1; q < n; ++q) {
    double intersection = 0.0;
    while (true) {
      const std::size_t site = sites[envelope_size];
      const double qd = static_cast<double>(q);
      const double sd = static_cast<double>(site);
      intersection =
          ((source[q] + qd * qd) - (source[site] + sd * sd)) /
          (2.0 * (qd - sd));
      if (intersection > boundaries[envelope_size] || envelope_size == 0)
        break;
      --envelope_size;
    }
    ++envelope_size;
    sites[envelope_size] = q;
    boundaries[envelope_size] = intersection;
    boundaries[envelope_size + 1] =
        std::numeric_limits<double>::infinity();
  }

  std::size_t envelope_index = 0;
  for (std::size_t q = 0; q < n; ++q) {
    const double qd = static_cast<double>(q);
    while (boundaries[envelope_index + 1] < qd)
      ++envelope_index;
    const double delta = qd - static_cast<double>(sites[envelope_index]);
    transformed[q] = delta * delta + source[sites[envelope_index]];
  }
}

template <typename Body>
void parallel_lines(std::size_t line_count, TitanExecutor *executor,
                    Body &&body) {
  if (executor == nullptr || executor->capacity() <= 1 || line_count <= 1) {
    for (std::size_t line = 0; line < line_count; ++line)
      body(0, line);
    return;
  }
  const std::size_t workers =
      std::min<std::size_t>(executor->capacity(), line_count);
  executor->parallel_for(
      line_count, workers, 1,
      [&](std::size_t worker, std::size_t begin, std::size_t end) {
        for (std::size_t line = begin; line < end; ++line)
          body(worker, line);
      });
}

} // namespace

void erode_expanded_mask_to_ses(const SesMaskShape &shape,
                                std::span<const std::uint8_t> expanded_mask,
                                double probe_radius,
                                std::span<std::uint8_t> ses_mask,
                                TitanExecutor *executor) {
  if (shape.nx == 0 || shape.ny == 0 || shape.nz == 0 ||
      !(shape.spacing > 0.0) || !std::isfinite(shape.spacing) ||
      !(probe_radius >= 0.0) || !std::isfinite(probe_radius)) {
    throw std::invalid_argument("PBE Grid SES: invalid mask geometry.");
  }
  if (shape.nx > std::numeric_limits<std::size_t>::max() / shape.ny ||
      shape.nx * shape.ny >
          std::numeric_limits<std::size_t>::max() / shape.nz) {
    throw std::overflow_error("PBE Grid SES: mask dimensions overflow size_t.");
  }
  const std::size_t points = shape.points();
  if (expanded_mask.size() != points || ses_mask.size() != points) {
    throw std::invalid_argument("PBE Grid SES: mask size does not match shape.");
  }
  if (probe_radius == 0.0) {
    std::copy(expanded_mask.begin(), expanded_mask.end(), ses_mask.begin());
    return;
  }

  const std::size_t max_line = std::max({shape.nx, shape.ny, shape.nz});
  const std::size_t worker_count =
      executor == nullptr ? 1 : std::max<std::size_t>(1, executor->capacity());
  std::vector<DistanceScratch> scratch(worker_count);
  for (DistanceScratch &worker : scratch)
    worker.resize(max_line);

  const double nx = static_cast<double>(shape.nx);
  const double ny = static_cast<double>(shape.ny);
  const double nz = static_cast<double>(shape.nz);
  const double infinity = nx * nx + ny * ny + nz * nz + 1.0;
  std::vector<double> distance(points, infinity);
  const std::size_t xy = shape.nx * shape.ny;
  for (std::size_t k = 0; k < shape.nz; ++k) {
    for (std::size_t j = 0; j < shape.ny; ++j) {
      for (std::size_t i = 0; i < shape.nx; ++i) {
        const std::size_t id = (k * shape.ny + j) * shape.nx + i;
        const bool boundary = i == 0 || j == 0 || k == 0 ||
                              i + 1 == shape.nx || j + 1 == shape.ny ||
                              k + 1 == shape.nz;
        if (expanded_mask[id] == 0U || boundary)
          distance[id] = 0.0;
      }
    }
  }

  parallel_lines(shape.ny * shape.nz, executor,
                 [&](std::size_t worker_id, std::size_t line) {
                   DistanceScratch &work = scratch[worker_id];
                   const std::size_t j = line % shape.ny;
                   const std::size_t k = line / shape.ny;
                   const std::size_t base = (k * shape.ny + j) * shape.nx;
                   std::copy_n(distance.data() + base, shape.nx,
                               work.source.data());
                   distance_transform_1d(
                       std::span<const double>(work.source.data(), shape.nx),
                       std::span<double>(work.transformed.data(), shape.nx),
                       std::span<std::size_t>(work.sites.data(), shape.nx),
                       std::span<double>(work.boundaries.data(), shape.nx + 1));
                   std::copy_n(work.transformed.data(), shape.nx,
                               distance.data() + base);
                 });

  parallel_lines(shape.nx * shape.nz, executor,
                 [&](std::size_t worker_id, std::size_t line) {
                   DistanceScratch &work = scratch[worker_id];
                   const std::size_t i = line % shape.nx;
                   const std::size_t k = line / shape.nx;
                   for (std::size_t j = 0; j < shape.ny; ++j)
                     work.source[j] = distance[(k * shape.ny + j) * shape.nx + i];
                   distance_transform_1d(
                       std::span<const double>(work.source.data(), shape.ny),
                       std::span<double>(work.transformed.data(), shape.ny),
                       std::span<std::size_t>(work.sites.data(), shape.ny),
                       std::span<double>(work.boundaries.data(), shape.ny + 1));
                   for (std::size_t j = 0; j < shape.ny; ++j)
                     distance[(k * shape.ny + j) * shape.nx + i] =
                         work.transformed[j];
                 });

  parallel_lines(shape.nx * shape.ny, executor,
                 [&](std::size_t worker_id, std::size_t line) {
                   DistanceScratch &work = scratch[worker_id];
                   const std::size_t i = line % shape.nx;
                   const std::size_t j = line / shape.nx;
                   for (std::size_t k = 0; k < shape.nz; ++k)
                     work.source[k] = distance[k * xy + j * shape.nx + i];
                   distance_transform_1d(
                       std::span<const double>(work.source.data(), shape.nz),
                       std::span<double>(work.transformed.data(), shape.nz),
                       std::span<std::size_t>(work.sites.data(), shape.nz),
                       std::span<double>(work.boundaries.data(), shape.nz + 1));
                   for (std::size_t k = 0; k < shape.nz; ++k)
                     distance[k * xy + j * shape.nx + i] = work.transformed[k];
                 });

  const double probe_grid = probe_radius / shape.spacing;
  const double threshold2 = probe_grid * probe_grid;
  for (std::size_t id = 0; id < points; ++id) {
    ses_mask[id] = expanded_mask[id] != 0U && distance[id] > threshold2
                       ? std::uint8_t{1}
                       : std::uint8_t{0};
  }
}

} // namespace TitanPBE
