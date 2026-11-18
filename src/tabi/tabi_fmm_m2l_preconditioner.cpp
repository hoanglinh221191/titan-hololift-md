#include "tabi_fmm_m2l_preconditioner.h"

#include "../titan_executor.h"
#include "tabi_near.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace TitanPBE::Tabi {

namespace {
constexpr std::size_t kMinFmmBlockJacobiPanelCap = 8;
constexpr std::size_t kMaxFmmBlockJacobiPanelCap = 128;
constexpr std::size_t kMaxFmmOverlapOwnedPanelCap = 32;
constexpr std::size_t kMaxFmmOverlapHaloPanelCap = 128;
constexpr std::size_t kMinFmmIluNeighborCap = 32;
constexpr std::size_t kMaxFmmIluNeighborCap = 1024;
constexpr double kMinIluPivotDet = 1.0e-18;
constexpr double kMinIluReciprocalPivotRatio = 1.0e-10;
constexpr double kMaxIluFactorGrowth = 1.0e8;
constexpr double kMaxIluProbeGain = 1.0e8;
constexpr double kMaxIluProbeRelativeResidual = 1.0;

using PanelGraph = std::vector<std::vector<std::size_t>>;
using Block2 = JcpSparseIluPreconditioner::Block2;

std::shared_ptr<const PanelGraph> make_normalized_ilu_graph(
    const titan::tabi::fmm_m2l::FmmM2lAnalyticContext &context,
    std::size_t cap, const std::string &graph_source,
    TitanExecutor *executor = nullptr) {
  auto graph = std::make_shared<PanelGraph>(
      graph_source == "distance-knn"
          ? context.distance_knn_panel_graph(cap)
          : context.near_panel_graph(cap, executor));
  for (std::size_t row_index = 0; row_index < graph->size(); ++row_index) {
    auto &row = (*graph)[row_index];
    if (!std::binary_search(row.begin(), row.end(), row_index))
      row.insert(std::lower_bound(row.begin(), row.end(), row_index),
                 row_index);
    if (row.size() > cap) {
      row.resize(cap);
      if (!std::binary_search(row.begin(), row.end(), row_index)) {
        row.back() = row_index;
        std::sort(row.begin(), row.end());
      }
    }
  }
  return graph;
}

std::size_t estimate_fmm_ilu_bytes_from_graph(
    const PanelGraph &graph, std::size_t diagonal_size, bool coupled,
    bool include_regularization_retry) {
  std::size_t nnz = 0;
  for (const auto &row : graph)
    nnz += row.size();
  const std::size_t panel_count = coupled ? diagonal_size / 2 : diagonal_size;
  const std::size_t retry_bytes =
      include_regularization_retry ? nnz * sizeof(Block2) : 0;
  return diagonal_size * sizeof(double) +
         (graph.size() + 1) * sizeof(std::size_t) +
         nnz * (sizeof(std::size_t) + sizeof(Block2)) +
         graph.size() * sizeof(Block2) + 4 * panel_count * sizeof(double) +
         4 * graph.size() * sizeof(std::size_t) + retry_bytes;
}

std::vector<JcpOwnedFactorPanelGroup> to_owned_factor_groups(
    const std::vector<titan::tabi::fmm_m2l::FmmM2lOverlapPanelGroup>
        &overlap_groups) {
  std::vector<JcpOwnedFactorPanelGroup> converted;
  converted.reserve(overlap_groups.size());
  for (const auto &group : overlap_groups) {
    JcpOwnedFactorPanelGroup out;
    out.owned_panels = group.owned_panels;
    out.factor_panels = group.factor_panels;
    converted.push_back(std::move(out));
  }
  return converted;
}

Block2 add(Block2 lhs, const Block2 &rhs) noexcept {
  lhs.a00 += rhs.a00;
  lhs.a01 += rhs.a01;
  lhs.a10 += rhs.a10;
  lhs.a11 += rhs.a11;
  return lhs;
}

Block2 sub(Block2 lhs, const Block2 &rhs) noexcept {
  lhs.a00 -= rhs.a00;
  lhs.a01 -= rhs.a01;
  lhs.a10 -= rhs.a10;
  lhs.a11 -= rhs.a11;
  return lhs;
}

Block2 mul(const Block2 &lhs, const Block2 &rhs) noexcept {
  return {lhs.a00 * rhs.a00 + lhs.a01 * rhs.a10,
          lhs.a00 * rhs.a01 + lhs.a01 * rhs.a11,
          lhs.a10 * rhs.a00 + lhs.a11 * rhs.a10,
          lhs.a10 * rhs.a01 + lhs.a11 * rhs.a11};
}

std::pair<double, double> mul_vec(const Block2 &block, double x0,
                                  double x1) noexcept {
  return {block.a00 * x0 + block.a01 * x1,
          block.a10 * x0 + block.a11 * x1};
}

double max_abs(const Block2 &block) noexcept {
  return std::max({std::abs(block.a00), std::abs(block.a01),
                   std::abs(block.a10), std::abs(block.a11)});
}

bool invert_block2(const Block2 &block, Block2 &inverse) noexcept {
  const double det = block.a00 * block.a11 - block.a01 * block.a10;
  const double scale = std::max(1.0, max_abs(block));
  if (!std::isfinite(det) || std::abs(det) <= kMinIluPivotDet * scale * scale)
    return false;
  const double inv_det = 1.0 / det;
  inverse = {block.a11 * inv_det, -block.a01 * inv_det,
             -block.a10 * inv_det, block.a00 * inv_det};
  return std::isfinite(inverse.a00) && std::isfinite(inverse.a01) &&
         std::isfinite(inverse.a10) && std::isfinite(inverse.a11);
}

std::size_t find_block(const std::vector<std::size_t> &row_ptr,
                       const std::vector<std::size_t> &col_index,
                       std::size_t row, std::size_t col) noexcept {
  if (row + 1 >= row_ptr.size())
    return static_cast<std::size_t>(-1);
  const std::size_t begin = row_ptr[row];
  const std::size_t end = row_ptr[row + 1];
  const auto it = std::lower_bound(col_index.begin() +
                                       static_cast<std::ptrdiff_t>(begin),
                                   col_index.begin() +
                                       static_cast<std::ptrdiff_t>(end),
                                   col);
  if (it == col_index.begin() + static_cast<std::ptrdiff_t>(end) ||
      *it != col)
    return static_cast<std::size_t>(-1);
  return static_cast<std::size_t>(std::distance(col_index.begin(), it));
}

Block2 make_jcp_block(const Panel &target, std::size_t target_index,
                      const Panel &source, std::size_t source_index,
                      double eps_ratio, double kappa, bool coupled,
                      const std::string &near_integration,
                      const std::string &quad_buffer) {
  const JcpNearPairKernel selected = evaluate_jcp_near_pair_kernel(
      target, target_index, source, source_index, eps_ratio, kappa, coupled,
      near_integration, quad_buffer, nullptr);
  if (!selected.valid)
    return {};
  if (coupled) {
    return {-selected.k2, -selected.k1, -selected.k4, -selected.k3};
  }
  return {-selected.k2, 0.0, 0.0, 0.0};
}

double vector_norm2(const std::vector<double> &values) noexcept {
  double sum = 0.0;
  for (const double value : values)
    sum += value * value;
  return std::sqrt(sum);
}

bool sparse_block_matvec(const JcpSparseIluPreconditioner &precond,
                         const std::vector<Block2> &matrix,
                         const std::vector<double> &in,
                         std::vector<double> &out) {
  const std::size_t n = precond.panel_count;
  if (!precond.coupled || precond.row_ptr.size() != n + 1 ||
      matrix.size() != precond.col_index.size() || in.size() != 2 * n)
    return false;
  out.assign(2 * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t p = precond.row_ptr[i]; p < precond.row_ptr[i + 1];
         ++p) {
      const std::size_t col = precond.col_index[p];
      if (col >= n)
        return false;
      const auto [v0, v1] = mul_vec(matrix[p], in[col], in[n + col]);
      out[i] += v0;
      out[n + i] += v1;
    }
  }
  return std::all_of(out.begin(), out.end(),
                     [](double value) { return std::isfinite(value); });
}

struct SparseIluQualityGateResult {
  bool pass{false};
  std::string failure_stage{"off"};
  std::int64_t failure_probe{-1};
  double failure_gain{0.0};
  double failure_rel_residual{0.0};
};

SparseIluQualityGateResult passes_sparse_ilu_quality_gate(
    JcpSparseIluPreconditioner &precond,
    const std::vector<Block2> &original_matrix) {
  SparseIluQualityGateResult result;
  const std::size_t n = precond.panel_count;
  if (!precond.coupled || n == 0) {
    result.failure_stage = "probe_shape";
    return result;
  }
  std::vector<std::vector<double>> probes(2, std::vector<double>(2 * n, 1.0));
  for (std::size_t i = 0; i < n; ++i) {
    probes[0][n + i] = (i % 3 == 0) ? -1.0 : 1.0;
    probes[1][i] = (i % 2 == 0) ? 1.0 : -1.0;
    probes[1][n + i] = (i % 5 < 2) ? -1.0 : 1.0;
  }
  std::vector<double> rhs;
  std::vector<double> solved;
  std::vector<double> reconstructed_rhs;
  for (std::size_t probe_index = 0; probe_index < probes.size();
       ++probe_index) {
    const std::vector<double> &probe = probes[probe_index];
    if (!sparse_block_matvec(precond, original_matrix, probe, rhs)) {
      result.failure_stage = "probe_matvec";
      result.failure_probe = static_cast<std::int64_t>(probe_index);
      return result;
    }
    precond.apply(rhs, solved);
    if (solved.size() != probe.size() ||
        !std::all_of(solved.begin(), solved.end(),
                     [](double value) { return std::isfinite(value); })) {
      result.failure_stage = "probe_non_finite";
      result.failure_probe = static_cast<std::int64_t>(probe_index);
      return result;
    }
    const double rhs_norm = std::max(1.0e-300, vector_norm2(rhs));
    const double gain = vector_norm2(solved) / rhs_norm;
    precond.quality_probe_max_gain =
        std::max(precond.quality_probe_max_gain, gain);
    // ILU(0) is an approximate sparse solve, not an exact inverse. Requiring
    // M^{-1} A v to be close to v rejects useful preconditioners whenever the
    // approximation is non-normal. Grade the factor by the residual it leaves
    // on the original, unshifted near-field matrix instead.
    if (!sparse_block_matvec(precond, original_matrix, solved,
                             reconstructed_rhs)) {
      result.failure_stage = "probe_reconstruction";
      result.failure_probe = static_cast<std::int64_t>(probe_index);
      return result;
    }
    for (std::size_t i = 0; i < reconstructed_rhs.size(); ++i)
      reconstructed_rhs[i] -= rhs[i];
    const double relative_residual =
        vector_norm2(reconstructed_rhs) / rhs_norm;
    precond.quality_probe_rel_residual =
        std::max(precond.quality_probe_rel_residual, relative_residual);
    if (!std::isfinite(gain) || gain > kMaxIluProbeGain) {
      result.failure_stage = "probe_gain";
      result.failure_probe = static_cast<std::int64_t>(probe_index);
      result.failure_gain = gain;
      result.failure_rel_residual = relative_residual;
      return result;
    }
    if (!std::isfinite(relative_residual) ||
        relative_residual > kMaxIluProbeRelativeResidual) {
      result.failure_stage = "probe_residual";
      result.failure_probe = static_cast<std::int64_t>(probe_index);
      result.failure_gain = gain;
      result.failure_rel_residual = relative_residual;
      return result;
    }
  }
  result.pass = true;
  return result;
}

void build_sparse_ilu_levels(JcpSparseIluPreconditioner &precond) {
  const std::size_t n = precond.panel_count;
  if (precond.row_ptr.size() != n + 1)
    return;
  precond.lower_end.resize(n);
  precond.upper_begin.resize(n);
  auto append_level = [](std::vector<std::vector<std::size_t>> &levels,
                         std::size_t level, std::size_t row) {
    if (levels.size() <= level)
      levels.resize(level + 1);
    levels[level].push_back(row);
  };
  std::vector<std::size_t> forward_level(n, 0);
  for (std::size_t row = 0; row < n; ++row) {
    std::size_t level = 0;
    std::size_t lower_end = precond.row_ptr[row + 1];
    std::size_t upper_begin = precond.row_ptr[row + 1];
    for (std::size_t p = precond.row_ptr[row];
         p < precond.row_ptr[row + 1]; ++p) {
      const std::size_t col = precond.col_index[p];
      if (col >= row) {
        lower_end = p;
        upper_begin = col == row ? p + 1u : p;
        break;
      }
      level = std::max(level, forward_level[col] + 1);
    }
    precond.lower_end[row] = lower_end;
    precond.upper_begin[row] = upper_begin;
    forward_level[row] = level;
    append_level(precond.forward_levels, level, row);
  }
  std::vector<std::size_t> backward_level(n, 0);
  for (std::size_t rev = 0; rev < n; ++rev) {
    const std::size_t row = n - 1 - rev;
    std::size_t level = 0;
    for (std::size_t p = precond.row_ptr[row];
         p < precond.row_ptr[row + 1]; ++p) {
      const std::size_t col = precond.col_index[p];
      if (col <= row)
        continue;
      level = std::max(level, backward_level[col] + 1);
    }
    backward_level[row] = level;
    append_level(precond.backward_levels, level, row);
  }
  for (const auto &level : precond.forward_levels)
    precond.parallel_level_max_width =
        std::max(precond.parallel_level_max_width, level.size());
  for (const auto &level : precond.backward_levels)
    precond.parallel_level_max_width =
        std::max(precond.parallel_level_max_width, level.size());
}
}

void JcpSparseIluPreconditioner::apply(const std::vector<double> &in,
                                       std::vector<double> &out) const {
  const std::size_t n = panel_count;
  const std::size_t expected = coupled ? 2 * n : n;
  if (n == 0 || in.size() < expected || row_ptr.size() != n + 1 ||
      diag_inverse.size() != n) {
    out.assign(in.size(), 0.0);
    for (std::size_t i = 0; i < in.size(); ++i)
      out[i] = in[i] * (i < inv_diag.size() ? inv_diag[i] : 1.0);
    return;
  }

  std::lock_guard<std::mutex> lock(workspace_mutex);
  if (workspace.size() != 4 * n)
    workspace.resize(4 * n);
  double *const y0 = workspace.data();
  double *const y1 = y0 + n;
  double *const x0 = y1 + n;
  double *const x1 = x0 + n;
  const auto forward_solve_row = [&](std::size_t i) {
    double rhs0 = i < in.size() ? in[i] : 0.0;
    double rhs1 = coupled && n + i < in.size() ? in[n + i] : 0.0;
    const std::size_t end = lower_end.size() == n ? lower_end[i]
                                                  : row_ptr[i + 1];
    for (std::size_t p = row_ptr[i]; p < end; ++p) {
      const std::size_t col = col_index[p];
      if (lower_end.size() != n && col >= i)
        break;
      const auto [v0, v1] = mul_vec(values[p], y0[col], y1[col]);
      rhs0 -= v0;
      rhs1 -= v1;
    }
    y0[i] = rhs0;
    y1[i] = rhs1;
  };
  const auto backward_solve_row = [&](std::size_t i) {
    double rhs0 = y0[i];
    double rhs1 = y1[i];
    const std::size_t begin =
        upper_begin.size() == n ? upper_begin[i] : row_ptr[i];
    for (std::size_t p = begin; p < row_ptr[i + 1]; ++p) {
      const std::size_t col = col_index[p];
      if (upper_begin.size() != n && col <= i)
        continue;
      const auto [v0, v1] = mul_vec(values[p], x0[col], x1[col]);
      rhs0 -= v0;
      rhs1 -= v1;
    }
    const auto [s0, s1] = mul_vec(diag_inverse[i], rhs0, rhs1);
    x0[i] = s0;
    x1[i] = s1;
  };

  const auto apply_levels = [&](const std::vector<std::vector<std::size_t>> &levels,
                                const auto &solve_row) {
    const std::size_t available_workers =
        executor == nullptr ? 1 : executor->capacity();
    for (const auto &level : levels) {
      if (level.empty())
        continue;
      const std::size_t workers = std::min(available_workers, level.size());
      // Short triangular levels are cheaper to execute serially than to fan
      // out through the executor. Retain the original small-problem fast path
      // and only parallelize sufficiently wide levels on production meshes.
      const bool parallel_level =
          n >= 4096 && workers > 1 && level.size() >= 16 * workers;
      if (!parallel_level) {
        for (const std::size_t row : level)
          solve_row(row);
      } else {
        executor->parallel_for(
            level.size(), workers, 1,
            [&](std::size_t, std::size_t begin, std::size_t end) {
              for (std::size_t position = begin; position < end; ++position)
                solve_row(level[position]);
            });
      }
    }
  };
  if (!forward_levels.empty() && !backward_levels.empty()) {
    apply_levels(forward_levels, forward_solve_row);
    apply_levels(backward_levels, backward_solve_row);
  } else {
    for (std::size_t i = 0; i < n; ++i)
      forward_solve_row(i);
    for (std::size_t rev = 0; rev < n; ++rev)
      backward_solve_row(n - 1 - rev);
  }

  out.resize(expected);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = x0[i];
    if (coupled)
      out[n + i] = x1[i];
  }
}

std::size_t JcpSparseIluPreconditioner::estimated_bytes() const noexcept {
  std::size_t level_bytes =
      (forward_levels.capacity() + backward_levels.capacity()) *
      sizeof(std::vector<std::size_t>);
  for (const auto &level : forward_levels)
    level_bytes += level.capacity() * sizeof(std::size_t);
  for (const auto &level : backward_levels)
    level_bytes += level.capacity() * sizeof(std::size_t);
  return inv_diag.size() * sizeof(double) + row_ptr.size() * sizeof(std::size_t) +
         col_index.size() * sizeof(std::size_t) +
         (lower_end.size() + upper_begin.size()) * sizeof(std::size_t) +
         values.size() * sizeof(Block2) + diag_inverse.size() * sizeof(Block2) +
         workspace.capacity() * sizeof(double) + level_bytes;
}

std::size_t estimate_fmm_m2l_block_jacobi_bytes(
    const titan::tabi::fmm_m2l::FmmM2lAnalyticContext &context,
    std::size_t diagonal_size, bool coupled, std::size_t block_panel_cap) {
  const std::size_t cap = std::clamp<std::size_t>(
      block_panel_cap, kMinFmmBlockJacobiPanelCap,
      kMaxFmmBlockJacobiPanelCap);
  const std::vector<std::vector<std::size_t>> panel_groups =
      context.block_panel_ids(cap);
  return estimate_jcp_block_preconditioner_bytes_from_panel_groups(
      panel_groups, diagonal_size, coupled);
}

std::shared_ptr<JcpBlockPreconditioner>
build_fmm_m2l_block_jacobi_preconditioner(
    const std::vector<Panel> &panels,
    const titan::tabi::fmm_m2l::FmmM2lAnalyticContext &context, bool coupled,
    double eps_ratio, double kappa, double lhs_phi, double lhs_phin,
    const std::vector<double> &diagonal, std::size_t block_panel_cap,
    TitanExecutor &executor, const std::string &near_integration,
    const std::string &quad_buffer) {
  if (panels.empty())
    return {};
  if (context.panel_count() != panels.size()) {
    throw std::invalid_argument(
        "FMM/M2L block-Jacobi preconditioner context does not match panels");
  }
  const std::size_t cap = std::clamp<std::size_t>(
      block_panel_cap, kMinFmmBlockJacobiPanelCap,
      kMaxFmmBlockJacobiPanelCap);
  const std::vector<std::vector<std::size_t>> panel_groups =
      context.block_panel_ids(cap);
  if (panel_groups.empty())
    return {};
  return build_jcp_block_preconditioner_from_panel_groups(
      panels, coupled, eps_ratio, kappa, lhs_phi, lhs_phin, diagonal,
      near_integration, quad_buffer, "operator-near", panel_groups, executor);
}

std::size_t estimate_fmm_m2l_overlap_schwarz_bytes(
    const titan::tabi::fmm_m2l::FmmM2lAnalyticContext &context,
    std::size_t diagonal_size, bool coupled, std::size_t owned_panel_cap,
    std::size_t overlap_panel_cap, const std::string &graph_source) {
  const std::size_t owned_cap = std::clamp<std::size_t>(
      owned_panel_cap, kMinFmmBlockJacobiPanelCap,
      kMaxFmmOverlapOwnedPanelCap);
  const std::size_t halo_cap =
      std::clamp<std::size_t>(overlap_panel_cap, 0,
                              kMaxFmmOverlapHaloPanelCap);
  const auto overlap_groups =
      graph_source == "distance-knn"
          ? context.distance_knn_overlap_panel_groups(owned_cap, halo_cap)
          : context.overlap_panel_groups(owned_cap, halo_cap);
  const auto panel_groups = to_owned_factor_groups(overlap_groups);
  return estimate_jcp_block_preconditioner_bytes_from_owned_factor_groups(
      panel_groups, diagonal_size, coupled);
}

std::shared_ptr<JcpBlockPreconditioner>
build_fmm_m2l_overlap_schwarz_preconditioner(
    const std::vector<Panel> &panels,
    const titan::tabi::fmm_m2l::FmmM2lAnalyticContext &context, bool coupled,
    double eps_ratio, double kappa, double lhs_phi, double lhs_phin,
    const std::vector<double> &diagonal, std::size_t owned_panel_cap,
    std::size_t overlap_panel_cap, TitanExecutor &executor,
    const std::string &near_integration, const std::string &quad_buffer,
    const std::string &graph_source) {
  if (panels.empty())
    return {};
  if (context.panel_count() != panels.size()) {
    throw std::invalid_argument(
        "FMM/M2L overlap-Schwarz preconditioner context does not match panels");
  }
  const std::size_t owned_cap = std::clamp<std::size_t>(
      owned_panel_cap, kMinFmmBlockJacobiPanelCap,
      kMaxFmmOverlapOwnedPanelCap);
  const std::size_t halo_cap =
      std::clamp<std::size_t>(overlap_panel_cap, 0,
                              kMaxFmmOverlapHaloPanelCap);
  const auto overlap_groups =
      graph_source == "distance-knn"
          ? context.distance_knn_overlap_panel_groups(owned_cap, halo_cap)
          : context.overlap_panel_groups(owned_cap, halo_cap);
  if (overlap_groups.empty())
    return {};
  const auto panel_groups = to_owned_factor_groups(overlap_groups);
  return build_jcp_block_preconditioner_from_owned_factor_groups(
      panels, coupled, eps_ratio, kappa, lhs_phi, lhs_phin, diagonal,
      near_integration, quad_buffer, "operator-near", panel_groups, executor);
}

std::size_t estimate_fmm_m2l_nearfield_ilu0_bytes(
    const titan::tabi::fmm_m2l::FmmM2lAnalyticContext &context,
    std::size_t diagonal_size, bool coupled, std::size_t neighbor_cap,
    bool include_regularization_retry, const std::string &graph_source) {
  const std::size_t cap = std::clamp<std::size_t>(
      neighbor_cap, kMinFmmIluNeighborCap, kMaxFmmIluNeighborCap);
  const auto graph = make_normalized_ilu_graph(context, cap, graph_source);
  return estimate_fmm_ilu_bytes_from_graph(
      *graph, diagonal_size, coupled, include_regularization_retry);
}

FmmIluBudgetSelection select_fmm_m2l_nearfield_ilu0_budget(
    const titan::tabi::fmm_m2l::FmmM2lAnalyticContext &context,
    std::size_t diagonal_size, bool coupled, std::size_t neighbor_cap,
    std::size_t memory_budget_bytes, bool include_regularization_retry,
    const std::string &graph_source, TitanExecutor *executor) {
  FmmIluBudgetSelection selection;
  selection.requested_neighbor_cap = std::clamp<std::size_t>(
      neighbor_cap, kMinFmmIluNeighborCap, kMaxFmmIluNeighborCap);
  selection.effective_neighbor_cap = selection.requested_neighbor_cap;

  while (true) {
    const auto graph_begin = std::chrono::steady_clock::now();
    selection.normalized_graph = make_normalized_ilu_graph(
        context, selection.effective_neighbor_cap, graph_source, executor);
    selection.graph_build_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - graph_begin)
            .count();
    selection.estimated_bytes = estimate_fmm_ilu_bytes_from_graph(
        *selection.normalized_graph, diagonal_size, coupled,
        include_regularization_retry);
    if (selection.estimated_bytes <= memory_budget_bytes) {
      selection.fits_budget = true;
      return selection;
    }
    if (selection.effective_neighbor_cap == kMinFmmIluNeighborCap)
      return selection;
    selection.effective_neighbor_cap = std::max<std::size_t>(
        kMinFmmIluNeighborCap, selection.effective_neighbor_cap / 2);
  }
}

std::shared_ptr<JcpSparseIluPreconditioner>
build_fmm_m2l_nearfield_ilu0_preconditioner(
    const std::vector<Panel> &panels,
    const titan::tabi::fmm_m2l::FmmM2lAnalyticContext &context, bool coupled,
    double eps_ratio, double kappa, double lhs_phi, double lhs_phin,
    const std::vector<double> &diagonal, std::size_t neighbor_cap,
    double drop_tol, TitanExecutor &executor, const std::string &near_integration,
    const std::string &quad_buffer, const std::string &regularization_mode,
    double regularization_max_relative, const std::string &graph_source,
    JcpCoarseGalerkinPreconditioner *coarse_near_capture,
    std::shared_ptr<const PanelGraph> normalized_graph) {
  if (panels.empty())
    return {};
  if (!coupled)
    return {};
  if (context.panel_count() != panels.size()) {
    throw std::invalid_argument(
        "FMM/M2L near-field ILU context does not match panels");
  }
  const std::size_t n = panels.size();
  const std::size_t cap = std::clamp<std::size_t>(
      neighbor_cap, kMinFmmIluNeighborCap, kMaxFmmIluNeighborCap);
  const double effective_drop =
      std::isfinite(drop_tol) && drop_tol > 0.0 ? drop_tol : 0.0;
  if (regularization_mode != "off" && regularization_mode != "adaptive") {
    throw std::invalid_argument(
        "FMM/M2L near-field ILU regularization must be off or adaptive");
  }
  if (!std::isfinite(regularization_max_relative) ||
      regularization_max_relative < 0.0 ||
      regularization_max_relative > 1.0e-2) {
    throw std::invalid_argument(
        "FMM/M2L near-field ILU regularization max must be in 0..1e-2");
  }
  auto precond = std::make_shared<JcpSparseIluPreconditioner>();
  precond->panel_count = n;
  precond->coupled = coupled;
  precond->executor = &executor;
  precond->regularization_mode = regularization_mode;
  precond->regularization_max_relative = regularization_max_relative;
  precond->inv_diag.assign(diagonal.size(), 1.0);
  for (std::size_t i = 0; i < diagonal.size(); ++i) {
    const double diag = std::abs(diagonal[i]);
    precond->inv_diag[i] =
        std::isfinite(diag) && diag > 1.0e-12 ? 1.0 / diag : 1.0;
  }

  const auto graph_begin = std::chrono::steady_clock::now();
  const bool graph_cache_reused =
      normalized_graph != nullptr && normalized_graph->size() == n &&
      std::all_of(normalized_graph->begin(), normalized_graph->end(),
                  [cap](const auto &row) {
                    return !row.empty() && row.size() <= cap;
                  });
  if (!graph_cache_reused)
    normalized_graph =
        make_normalized_ilu_graph(context, cap, graph_source, &executor);
  precond->graph_cache_reused = graph_cache_reused;
  const PanelGraph &graph = *normalized_graph;
  if (graph.size() != n)
    return {};
  precond->row_ptr.assign(n + 1, 0);
  std::size_t nnz = 0;
  std::size_t max_degree = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto &row = graph[i];
    precond->row_ptr[i] = nnz;
    nnz += row.size();
    max_degree = std::max(max_degree, row.size());
  }
  precond->row_ptr[n] = nnz;
  precond->col_index.resize(nnz);
  precond->values.resize(nnz);
  precond->graph_setup_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - graph_begin)
          .count();
  const auto assembly_begin = std::chrono::steady_clock::now();
  JcpCoarseGalerkinPreconditioner *capture = coarse_near_capture;
  std::size_t capture_dim = 0;
  std::size_t capture_aggregates = 0;
  if (capture != nullptr) {
    capture_dim = capture->dimension();
    capture_aggregates = capture->aggregates.size();
    const bool compatible =
        capture->coupled && capture->panel_count == n &&
        capture->panel_aggregate.size() == n &&
        capture_dim == 2 * capture_aggregates && capture_dim > 0 &&
        n <= std::numeric_limits<std::size_t>::max() / 2 / capture_dim;
    if (compatible) {
      capture->fine_operator_basis.assign(2 * n * capture_dim, 0.0);
    } else {
      capture = nullptr;
    }
  }
  const std::size_t assembly_workers =
      std::max<std::size_t>(1, std::min(executor.capacity(), n));
  executor.parallel_for(
      n, assembly_workers, 1,
      [&](std::size_t, std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
          std::size_t position = precond->row_ptr[i];
          for (const std::size_t col : graph[i]) {
            precond->col_index[position] = col;
            Block2 block{};
            if (i == col) {
              block.a00 += lhs_phi;
              block.a11 += lhs_phin;
            }
            block = add(block,
                        make_jcp_block(panels[i], i, panels[col], col,
                                       eps_ratio, kappa, coupled,
                                       near_integration, quad_buffer));
            if (i != col && effective_drop > 0.0 &&
                max_abs(block) < effective_drop)
              block = {};
            if (capture != nullptr) {
              const std::size_t aggregate = capture->panel_aggregate[col];
              if (aggregate < capture_aggregates) {
                double *const phi_row = capture->fine_operator_basis.data() +
                                        i * capture_dim;
                double *const phin_row =
                    capture->fine_operator_basis.data() +
                    (n + i) * capture_dim;
                phi_row[aggregate] += block.a00;
                phi_row[capture_aggregates + aggregate] += block.a01;
                phin_row[aggregate] += block.a10;
                phin_row[capture_aggregates + aggregate] += block.a11;
              }
            }
            precond->values[position] = block;
            ++position;
          }
        }
      });
  precond->near_assembly_setup_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - assembly_begin)
          .count();
  precond->nnz_blocks = precond->values.size();
  precond->valid_blocks = precond->nnz_blocks;
  precond->valid_panel_count = n;
  precond->max_degree = max_degree;
  precond->avg_degree =
      n > 0 ? static_cast<double>(precond->nnz_blocks) / static_cast<double>(n)
            : 0.0;
  precond->diag_inverse.assign(n, {});
  precond->workspace.assign(4 * n, 0.0);
  const auto factor_begin = std::chrono::steady_clock::now();
  const std::vector<Block2> original_values = precond->values;
  double original_peak = 0.0;
  for (const Block2 &block : original_values)
    original_peak = std::max(original_peak, max_abs(block));
  const auto factor_attempt = [&](double relative_shift) {
    precond->values = original_values;
    precond->diag_inverse.assign(n, {});
    precond->forward_levels.clear();
    precond->backward_levels.clear();
    precond->parallel_level_max_width = 0;
    precond->ill_conditioned_blocks = 0;
    precond->quality_rejected_blocks = 0;
    precond->pivot_ratio_min = 1.0;
    precond->factor_growth_max = 0.0;
    precond->pivot_ratio_min_row = -1;
    precond->pivot_ratio_min_column = -1;
    precond->factor_growth_row = -1;
    precond->factor_growth_column = -1;
    precond->quality_probe_max_gain = 0.0;
    precond->quality_probe_rel_residual = 0.0;
    precond->quality_failure_stage = "off";
    precond->quality_failure_row = -1;
    precond->quality_failure_column = -1;
    precond->quality_failure_probe = -1;
    precond->quality_failure_probe_gain = 0.0;
    precond->quality_failure_probe_rel_residual = 0.0;
    precond->quality_status = "off";
    precond->regularization_shifted_rows = 0;
    precond->regularization_selected_absolute_max = 0.0;

    if (relative_shift > 0.0) {
      for (std::size_t i = 0; i < n; ++i) {
        const std::size_t diag_pos =
            find_block(precond->row_ptr, precond->col_index, i, i);
        if (diag_pos == static_cast<std::size_t>(-1))
          continue;
        double row_scale = max_abs(original_values[diag_pos]);
        double off_diagonal_sum = 0.0;
        for (std::size_t p = precond->row_ptr[i];
             p < precond->row_ptr[i + 1]; ++p) {
          if (precond->col_index[p] != i)
            off_diagonal_sum += max_abs(original_values[p]);
        }
        row_scale = std::max(row_scale, off_diagonal_sum);
        const double shift =
            relative_shift * std::max(1.0e-12, row_scale);
        precond->values[diag_pos].a00 += shift;
        precond->values[diag_pos].a11 += shift;
        precond->regularization_selected_absolute_max =
            std::max(precond->regularization_selected_absolute_max, shift);
        ++precond->regularization_shifted_rows;
      }
    }

    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t p = precond->row_ptr[i]; p < precond->row_ptr[i + 1];
           ++p) {
        const std::size_t k = precond->col_index[p];
        if (k >= i)
          break;
        const Block2 diag_inv_k = precond->diag_inverse[k];
        precond->values[p] = mul(precond->values[p], diag_inv_k);
        const Block2 lik = precond->values[p];
        // Both CSR rows are sorted. Merge the upper part of row k with the
        // remainder of row i instead of performing a binary search for every
        // candidate fill position. The update order for existing blocks is
        // unchanged, preserving deterministic ILU(0) arithmetic.
        std::size_t ij = p + 1;
        for (std::size_t q = precond->row_ptr[k];
             q < precond->row_ptr[k + 1]; ++q) {
          const std::size_t j = precond->col_index[q];
          if (j <= k)
            continue;
          while (ij < precond->row_ptr[i + 1] &&
                 precond->col_index[ij] < j)
            ++ij;
          if (ij == precond->row_ptr[i + 1] ||
              precond->col_index[ij] != j)
            continue;
          precond->values[ij] =
              sub(precond->values[ij], mul(lik, precond->values[q]));
        }
      }
      const std::size_t diag_pos =
          find_block(precond->row_ptr, precond->col_index, i, i);
      if (diag_pos == static_cast<std::size_t>(-1) ||
          !invert_block2(precond->values[diag_pos], precond->diag_inverse[i])) {
        ++precond->ill_conditioned_blocks;
        precond->quality_status = "factor_fallback";
        precond->quality_failure_stage = "pivot_inversion";
        precond->quality_failure_row = static_cast<std::int64_t>(i);
        precond->quality_failure_column = static_cast<std::int64_t>(i);
        return false;
      }
      const double pivot_scale = max_abs(precond->values[diag_pos]) *
                                 max_abs(precond->diag_inverse[i]);
      const double pivot_ratio =
          pivot_scale > 0.0 ? 1.0 / pivot_scale : 0.0;
      if (pivot_ratio < precond->pivot_ratio_min) {
        precond->pivot_ratio_min = pivot_ratio;
        precond->pivot_ratio_min_row = static_cast<std::int64_t>(i);
        precond->pivot_ratio_min_column = static_cast<std::int64_t>(i);
      }
    }
    build_sparse_ilu_levels(*precond);
    double factor_peak = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t p = precond->row_ptr[i]; p < precond->row_ptr[i + 1];
           ++p) {
        const double magnitude = max_abs(precond->values[p]);
        if (magnitude > factor_peak) {
          factor_peak = magnitude;
          precond->factor_growth_row = static_cast<std::int64_t>(i);
          precond->factor_growth_column =
              static_cast<std::int64_t>(precond->col_index[p]);
        }
      }
    }
    precond->factor_growth_max =
        factor_peak / std::max(1.0e-300, original_peak);
    if (!std::isfinite(precond->pivot_ratio_min) ||
        precond->pivot_ratio_min < kMinIluReciprocalPivotRatio) {
      ++precond->ill_conditioned_blocks;
      precond->quality_status = "factor_fallback";
      precond->quality_failure_stage = "pivot_condition";
      precond->quality_failure_row = precond->pivot_ratio_min_row;
      precond->quality_failure_column = precond->pivot_ratio_min_column;
      return false;
    }
    if (!std::isfinite(precond->factor_growth_max) ||
        precond->factor_growth_max > kMaxIluFactorGrowth) {
      ++precond->ill_conditioned_blocks;
      precond->quality_status = "factor_fallback";
      precond->quality_failure_stage = "factor_growth";
      precond->quality_failure_row = precond->factor_growth_row;
      precond->quality_failure_column = precond->factor_growth_column;
      return false;
    }
    const SparseIluQualityGateResult quality =
        passes_sparse_ilu_quality_gate(*precond, original_values);
    if (!quality.pass) {
      ++precond->quality_rejected_blocks;
      precond->quality_status = "quality_fallback";
      precond->quality_failure_stage = quality.failure_stage;
      precond->quality_failure_probe = quality.failure_probe;
      precond->quality_failure_probe_gain = quality.failure_gain;
      precond->quality_failure_probe_rel_residual =
          quality.failure_rel_residual;
      return false;
    }
    precond->quality_status = "pass";
    return true;
  };

  std::vector<double> shifts = {0.0};
  if (precond->regularization_mode == "adaptive" &&
      precond->regularization_max_relative > 0.0) {
    constexpr std::array<double, 4> candidates = {1.0e-8, 1.0e-6, 1.0e-4,
                                                   1.0e-2};
    for (const double candidate : candidates) {
      if (candidate <= precond->regularization_max_relative)
        shifts.push_back(candidate);
    }
    if (shifts.back() < precond->regularization_max_relative)
      shifts.push_back(precond->regularization_max_relative);
  }
  for (const double shift : shifts) {
    ++precond->regularization_attempts;
    const bool pass = factor_attempt(shift);
    if (shift == 0.0 && !pass) {
      precond->regularization_unshifted_failure_stage =
          precond->quality_failure_stage;
    }
    if (pass) {
      precond->regularization_selected_relative = shift;
      precond->factor_setup_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - factor_begin)
              .count();
      return precond;
    }
    ++precond->regularization_rejected_attempts;
  }
  precond->factor_setup_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - factor_begin)
          .count();
  return precond;
}

std::size_t estimate_jcp_sparse_ilu0_graph_bytes(
    const std::vector<std::vector<std::size_t>> &graph,
    std::size_t diagonal_size, bool include_regularization_retry) {
  std::size_t nnz = 0;
  for (const auto &row : graph)
    nnz += row.size();
  const std::size_t retry_bytes =
      include_regularization_retry ? nnz * sizeof(Block2) : 0;
  return diagonal_size * sizeof(double) +
         (graph.size() + 1) * sizeof(std::size_t) +
         nnz * (sizeof(std::size_t) + sizeof(Block2)) +
         graph.size() * sizeof(Block2) +
         2 * graph.size() * sizeof(std::size_t) + retry_bytes;
}

std::shared_ptr<JcpSparseIluPreconditioner>
build_jcp_sparse_ilu0_preconditioner_from_graph(
    const std::vector<Panel> &panels,
    const std::vector<std::vector<std::size_t>> &graph, bool coupled,
    double eps_ratio, double kappa, double lhs_phi, double lhs_phin,
    const std::vector<double> &diagonal, double drop_tol,
    const std::string &near_integration,
    const std::string &quad_buffer, const std::string &regularization_mode,
    double regularization_max_relative) {
  if (panels.empty())
    return {};
  if (!coupled)
    return {};
  const std::size_t n = panels.size();
  if (graph.size() != n)
    return {};
  const double effective_drop =
      std::isfinite(drop_tol) && drop_tol > 0.0 ? drop_tol : 0.0;
  if (regularization_mode != "off" && regularization_mode != "adaptive")
    throw std::invalid_argument(
        "sparse graph ILU regularization must be off or adaptive");
  if (!std::isfinite(regularization_max_relative) ||
      regularization_max_relative < 0.0 ||
      regularization_max_relative > 1.0e-2)
    throw std::invalid_argument(
        "sparse graph ILU regularization max must be in 0..1e-2");
  auto precond = std::make_shared<JcpSparseIluPreconditioner>();
  precond->panel_count = n;
  precond->coupled = coupled;
  precond->regularization_mode = regularization_mode;
  precond->regularization_max_relative = regularization_max_relative;
  precond->inv_diag.assign(diagonal.size(), 1.0);
  for (std::size_t i = 0; i < diagonal.size(); ++i) {
    const double diag = std::abs(diagonal[i]);
    precond->inv_diag[i] =
        std::isfinite(diag) && diag > 1.0e-12 ? 1.0 / diag : 1.0;
  }

  std::vector<std::vector<std::size_t>> normalized = graph;
  precond->row_ptr.assign(n + 1, 0);
  std::size_t nnz = 0;
  std::size_t max_degree = 0;
  for (std::size_t i = 0; i < n; ++i) {
    auto &row = normalized[i];
    row.push_back(i);
    row.erase(std::remove_if(row.begin(), row.end(),
                             [n](std::size_t col) { return col >= n; }),
              row.end());
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
    precond->row_ptr[i] = nnz;
    nnz += row.size();
    max_degree = std::max(max_degree, row.size());
  }
  precond->row_ptr[n] = nnz;
  precond->col_index.reserve(nnz);
  precond->values.reserve(nnz);
  for (std::size_t i = 0; i < n; ++i) {
    for (const std::size_t col : normalized[i]) {
      precond->col_index.push_back(col);
      Block2 block{};
      if (i == col) {
        block.a00 += lhs_phi;
        block.a11 += lhs_phin;
      }
      block = add(block,
                  make_jcp_block(panels[i], i, panels[col], col, eps_ratio,
                                 kappa, coupled, near_integration,
                                 quad_buffer));
      if (i != col && effective_drop > 0.0 && max_abs(block) < effective_drop)
        block = {};
      precond->values.push_back(block);
    }
  }
  precond->nnz_blocks = precond->values.size();
  precond->valid_blocks = precond->nnz_blocks;
  precond->valid_panel_count = n;
  precond->max_degree = max_degree;
  precond->avg_degree =
      n > 0 ? static_cast<double>(precond->nnz_blocks) / static_cast<double>(n)
            : 0.0;
  precond->diag_inverse.assign(n, {});
  precond->workspace.assign(4 * n, 0.0);
  const std::vector<Block2> original_values = precond->values;
  double original_peak = 0.0;
  for (const Block2 &block : original_values)
    original_peak = std::max(original_peak, max_abs(block));

  const auto factor_attempt = [&](double relative_shift) {
    precond->values = original_values;
    precond->diag_inverse.assign(n, {});
    precond->forward_levels.clear();
    precond->backward_levels.clear();
    precond->parallel_level_max_width = 0;
    precond->ill_conditioned_blocks = 0;
    precond->quality_rejected_blocks = 0;
    precond->pivot_ratio_min = 1.0;
    precond->factor_growth_max = 0.0;
    precond->pivot_ratio_min_row = -1;
    precond->pivot_ratio_min_column = -1;
    precond->factor_growth_row = -1;
    precond->factor_growth_column = -1;
    precond->quality_probe_max_gain = 0.0;
    precond->quality_probe_rel_residual = 0.0;
    precond->quality_failure_stage = "off";
    precond->quality_failure_row = -1;
    precond->quality_failure_column = -1;
    precond->quality_failure_probe = -1;
    precond->quality_failure_probe_gain = 0.0;
    precond->quality_failure_probe_rel_residual = 0.0;
    precond->quality_status = "off";
    precond->regularization_shifted_rows = 0;
    precond->regularization_selected_absolute_max = 0.0;

    if (relative_shift > 0.0) {
      for (std::size_t i = 0; i < n; ++i) {
        const std::size_t diag_pos =
            find_block(precond->row_ptr, precond->col_index, i, i);
        if (diag_pos == static_cast<std::size_t>(-1))
          continue;
        double row_scale = max_abs(original_values[diag_pos]);
        double off_diagonal_sum = 0.0;
        for (std::size_t p = precond->row_ptr[i];
             p < precond->row_ptr[i + 1]; ++p) {
          if (precond->col_index[p] != i)
            off_diagonal_sum += max_abs(original_values[p]);
        }
        row_scale = std::max(row_scale, off_diagonal_sum);
        const double shift =
            relative_shift * std::max(1.0e-12, row_scale);
        precond->values[diag_pos].a00 += shift;
        precond->values[diag_pos].a11 += shift;
        precond->regularization_selected_absolute_max =
            std::max(precond->regularization_selected_absolute_max, shift);
        ++precond->regularization_shifted_rows;
      }
    }

    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t p = precond->row_ptr[i]; p < precond->row_ptr[i + 1];
           ++p) {
        const std::size_t k = precond->col_index[p];
        if (k >= i)
          break;
        precond->values[p] =
            mul(precond->values[p], precond->diag_inverse[k]);
        const Block2 lik = precond->values[p];
        for (std::size_t q = precond->row_ptr[k];
             q < precond->row_ptr[k + 1]; ++q) {
          const std::size_t j = precond->col_index[q];
          if (j <= k)
            continue;
          const std::size_t ij = find_block(
              precond->row_ptr, precond->col_index, i, j);
          if (ij != static_cast<std::size_t>(-1))
            precond->values[ij] =
                sub(precond->values[ij], mul(lik, precond->values[q]));
        }
      }
      const std::size_t diag_pos =
          find_block(precond->row_ptr, precond->col_index, i, i);
      if (diag_pos == static_cast<std::size_t>(-1) ||
          !invert_block2(precond->values[diag_pos], precond->diag_inverse[i])) {
        ++precond->ill_conditioned_blocks;
        precond->quality_status = "factor_fallback";
        precond->quality_failure_stage = "pivot_inversion";
        precond->quality_failure_row = static_cast<std::int64_t>(i);
        precond->quality_failure_column = static_cast<std::int64_t>(i);
        return false;
      }
      const double pivot_scale = max_abs(precond->values[diag_pos]) *
                                 max_abs(precond->diag_inverse[i]);
      const double pivot_ratio =
          pivot_scale > 0.0 ? 1.0 / pivot_scale : 0.0;
      if (pivot_ratio < precond->pivot_ratio_min) {
        precond->pivot_ratio_min = pivot_ratio;
        precond->pivot_ratio_min_row = static_cast<std::int64_t>(i);
        precond->pivot_ratio_min_column = static_cast<std::int64_t>(i);
      }
    }
    build_sparse_ilu_levels(*precond);
    double factor_peak = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t p = precond->row_ptr[i]; p < precond->row_ptr[i + 1];
           ++p) {
        const double magnitude = max_abs(precond->values[p]);
        if (magnitude > factor_peak) {
          factor_peak = magnitude;
          precond->factor_growth_row = static_cast<std::int64_t>(i);
          precond->factor_growth_column =
              static_cast<std::int64_t>(precond->col_index[p]);
        }
      }
    }
    precond->factor_growth_max =
        factor_peak / std::max(1.0e-300, original_peak);
    if (!std::isfinite(precond->pivot_ratio_min) ||
        precond->pivot_ratio_min < kMinIluReciprocalPivotRatio) {
      ++precond->ill_conditioned_blocks;
      precond->quality_status = "factor_fallback";
      precond->quality_failure_stage = "pivot_condition";
      precond->quality_failure_row = precond->pivot_ratio_min_row;
      precond->quality_failure_column = precond->pivot_ratio_min_column;
      return false;
    }
    if (!std::isfinite(precond->factor_growth_max) ||
        precond->factor_growth_max > kMaxIluFactorGrowth) {
      ++precond->ill_conditioned_blocks;
      precond->quality_status = "factor_fallback";
      precond->quality_failure_stage = "factor_growth";
      precond->quality_failure_row = precond->factor_growth_row;
      precond->quality_failure_column = precond->factor_growth_column;
      return false;
    }
    const SparseIluQualityGateResult quality =
        passes_sparse_ilu_quality_gate(*precond, original_values);
    if (!quality.pass) {
      ++precond->quality_rejected_blocks;
      precond->quality_status = "quality_fallback";
      precond->quality_failure_stage = quality.failure_stage;
      precond->quality_failure_probe = quality.failure_probe;
      precond->quality_failure_probe_gain = quality.failure_gain;
      precond->quality_failure_probe_rel_residual =
          quality.failure_rel_residual;
      return false;
    }
    precond->quality_status = "pass";
    return true;
  };

  std::vector<double> shifts = {0.0};
  if (regularization_mode == "adaptive" &&
      regularization_max_relative > 0.0) {
    constexpr std::array<double, 4> candidates = {1.0e-8, 1.0e-6, 1.0e-4,
                                                   1.0e-2};
    for (const double candidate : candidates) {
      if (candidate <= regularization_max_relative)
        shifts.push_back(candidate);
    }
    if (shifts.back() < regularization_max_relative)
      shifts.push_back(regularization_max_relative);
  }
  for (const double shift : shifts) {
    ++precond->regularization_attempts;
    const bool pass = factor_attempt(shift);
    if (shift == 0.0 && !pass)
      precond->regularization_unshifted_failure_stage =
          precond->quality_failure_stage;
    if (pass) {
      precond->regularization_selected_relative = shift;
      return precond;
    }
    ++precond->regularization_rejected_attempts;
  }
  return precond;
}

} // namespace TitanPBE::Tabi
