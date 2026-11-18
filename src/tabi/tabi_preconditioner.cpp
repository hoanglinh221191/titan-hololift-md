#include "tabi_preconditioner.h"

#include "tabi_near.h"

#include "../titan_executor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace TitanPBE::Tabi {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

std::size_t solver_workers(const TitanExecutor &executor,
                           std::size_t work_items) noexcept {
  if (work_items == 0)
    return 1;
  return std::max<std::size_t>(
      1, std::min<std::size_t>(executor.capacity(), work_items));
}

double green_kernel(double r, double kappa) noexcept {
  if (r <= 1.0e-14)
    return 0.0;
  return std::exp(-kappa * r) / (4.0 * kPi * r);
}

double source_normal_derivative_green(double tx, double ty, double tz,
                                      const Panel &source,
                                      double kappa) noexcept {
  const double ux = tx - source.x;
  const double uy = ty - source.y;
  const double uz = tz - source.z;
  const double r2 = ux * ux + uy * uy + uz * uz;
  if (r2 <= 1.0e-28)
    return 0.0;
  const double r = std::sqrt(r2);
  const double ndotu = source.nx * ux + source.ny * uy + source.nz * uz;
  return std::exp(-kappa * r) * (1.0 + kappa * r) * ndotu /
         (4.0 * kPi * r2 * r);
}

double target_normal_derivative_green(const Panel &target, double sx,
                                      double sy, double sz,
                                      double kappa) noexcept {
  const double ux = target.x - sx;
  const double uy = target.y - sy;
  const double uz = target.z - sz;
  const double r2 = ux * ux + uy * uy + uz * uz;
  if (r2 <= 1.0e-28)
    return 0.0;
  const double r = std::sqrt(r2);
  const double ndotu = target.nx * ux + target.ny * uy + target.nz * uz;
  return -std::exp(-kappa * r) * (1.0 + kappa * r) * ndotu /
         (4.0 * kPi * r2 * r);
}

double mixed_normal_derivative_green(const Panel &target, const Panel &source,
                                     double kappa) noexcept {
  const double ux = target.x - source.x;
  const double uy = target.y - source.y;
  const double uz = target.z - source.z;
  const double r2 = ux * ux + uy * uy + uz * uz;
  if (r2 <= 1.0e-28)
    return 0.0;
  const double r = std::sqrt(r2);
  const double inv_r = 1.0 / r;
  const double nxu =
      (target.nx * ux + target.ny * uy + target.nz * uz) * inv_r;
  const double nyu =
      (source.nx * ux + source.ny * uy + source.nz * uz) * inv_r;
  const double nxny = target.nx * source.nx + target.ny * source.ny +
                      target.nz * source.nz;
  const double expkr = std::exp(-kappa * r);
  const double inv_r2 = inv_r * inv_r;
  const double fprime =
      -expkr * (1.0 + kappa * r) * inv_r2 / (4.0 * kPi);
  const double fsecond =
      expkr * (kappa * kappa * inv_r + 2.0 * kappa * inv_r2 +
               2.0 * inv_r2 * inv_r) /
      (4.0 * kPi);
  // Mixed target/source derivative, not the target/target Hessian.
  return -fsecond * nxu * nyu - (fprime * inv_r) * (nxny - nxu * nyu);
}

struct JcpKernels {
  double k1{0.0};
  double k2{0.0};
  double k3{0.0};
  double k4{0.0};
};

JcpKernels jcp_surface_kernels(const Panel &target, const Panel &source,
                               double eps_ratio, double kappa) noexcept {
  const double dx = target.x - source.x;
  const double dy = target.y - source.y;
  const double dz = target.z - source.z;
  const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
  if (r <= 1.0e-14)
    return {};
  const double g0 = green_kernel(r, 0.0);
  const double gk = green_kernel(r, kappa);
  const double d0_src =
      source_normal_derivative_green(target.x, target.y, target.z, source, 0.0);
  const double dk_src = source_normal_derivative_green(
      target.x, target.y, target.z, source, kappa);
  const double d0_tgt = target_normal_derivative_green(
      target, source.x, source.y, source.z, 0.0);
  const double dk_tgt = target_normal_derivative_green(
      target, source.x, source.y, source.z, kappa);
  const double d2_0 = mixed_normal_derivative_green(target, source, 0.0);
  const double d2_k = mixed_normal_derivative_green(target, source, kappa);
  const double inv_eps_ratio = 1.0 / std::max(1.0e-12, eps_ratio);
  return {g0 - gk, eps_ratio * dk_src - d0_src,
          d0_tgt - inv_eps_ratio * dk_tgt, d2_k - d2_0};
}

int shared_vertex_count(const Panel &a, const Panel &b) noexcept {
  if (!a.has_triangle_geometry || !b.has_triangle_geometry)
    return 0;
  auto has_vertex = [](const Panel &panel, std::size_t vertex) noexcept {
    return vertex != static_cast<std::size_t>(-1) &&
           (panel.v0_index == vertex || panel.v1_index == vertex ||
            panel.v2_index == vertex);
  };
  int count = 0;
  if (has_vertex(b, a.v0_index))
    ++count;
  if (a.v1_index != a.v0_index && has_vertex(b, a.v1_index))
    ++count;
  if (a.v2_index != a.v0_index && a.v2_index != a.v1_index &&
      has_vertex(b, a.v2_index))
    ++count;
  return count;
}

// A node patch carries no triangle vertex indices, so the vertex-sharing test
// would leave it only its self pair and every block of a block preconditioner
// would be the 2x2 point block.  Its counterpart of a triangle's
// vertex-sharing neighbours is its one-ring: the nodes adjacent to it.
bool node_patch_one_ring(const Panel &target, const Panel &source) noexcept {
  if (!target.node_patch_geometry ||
      target.node_patch_geometry != source.node_patch_geometry)
    return false;
  const NodePatchGeometry &g = *source.node_patch_geometry;
  const std::size_t s = source.node_patch_index;
  if (s + 1 >= g.adjacent_offsets.size())
    return false;
  return std::binary_search(
      g.adjacent_nodes.begin() + static_cast<std::ptrdiff_t>(g.adjacent_offsets[s]),
      g.adjacent_nodes.begin() +
          static_cast<std::ptrdiff_t>(g.adjacent_offsets[s + 1]),
      target.node_patch_index);
}

bool is_singular_near_pair(const Panel &target, std::size_t target_index,
                           const Panel &source,
                           std::size_t source_index) noexcept {
  if (source_index == target_index)
    return true;
  if (source.node_patch_geometry)
    return node_patch_one_ring(target, source);
  return shared_vertex_count(target, source) > 0;
}

bool factor_dense_lu(std::vector<double> &matrix, std::vector<int> &pivots,
                     std::size_t dim, double *pivot_ratio = nullptr) {
  pivots.assign(dim, 0);
  double min_pivot = std::numeric_limits<double>::infinity();
  double max_pivot = 0.0;
  for (std::size_t k = 0; k < dim; ++k) {
    std::size_t pivot = k;
    double best = std::abs(matrix[k * dim + k]);
    for (std::size_t row = k + 1; row < dim; ++row) {
      const double value = std::abs(matrix[row * dim + k]);
      if (value > best) {
        best = value;
        pivot = row;
      }
    }
    if (!std::isfinite(best) || best <= 1.0e-14)
      return false;
    min_pivot = std::min(min_pivot, best);
    max_pivot = std::max(max_pivot, best);
    pivots[k] = static_cast<int>(pivot);
    if (pivot != k) {
      for (std::size_t col = 0; col < dim; ++col)
        std::swap(matrix[k * dim + col], matrix[pivot * dim + col]);
    }
    const double diag = matrix[k * dim + k];
    for (std::size_t row = k + 1; row < dim; ++row) {
      matrix[row * dim + k] /= diag;
      const double multiplier = matrix[row * dim + k];
      for (std::size_t col = k + 1; col < dim; ++col)
        matrix[row * dim + col] -= multiplier * matrix[k * dim + col];
    }
  }
  if (pivot_ratio != nullptr)
    *pivot_ratio = max_pivot > 0.0 && std::isfinite(min_pivot)
                       ? min_pivot / max_pivot
                       : 0.0;
  return true;
}

std::vector<double> make_equilibration_scale(const std::vector<double> &matrix,
                                             std::size_t dim) {
  std::vector<double> scale(dim, 1.0);
  for (std::size_t i = 0; i < dim; ++i) {
    const double diag = std::abs(matrix[i * dim + i]);
    if (std::isfinite(diag) && diag > 1.0e-24)
      scale[i] = 1.0 / std::sqrt(diag);
  }
  return scale;
}

void apply_symmetric_scale(std::vector<double> &matrix,
                           const std::vector<double> &scale,
                           std::size_t dim) {
  if (scale.size() != dim)
    return;
  for (std::size_t row = 0; row < dim; ++row) {
    const double sr = scale[row];
    for (std::size_t col = 0; col < dim; ++col)
      matrix[row * dim + col] *= sr * scale[col];
  }
}

void solve_dense_lu(const std::vector<double> &lu,
                    const std::vector<int> &pivots, std::size_t dim,
                    std::vector<double> &rhs) {
  for (std::size_t k = 0; k < dim; ++k) {
    const std::size_t pivot = static_cast<std::size_t>(pivots[k]);
    if (pivot != k)
      std::swap(rhs[k], rhs[pivot]);
  }
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < row; ++col)
      rhs[row] -= lu[row * dim + col] * rhs[col];
  }
  for (std::size_t rev = 0; rev < dim; ++rev) {
    const std::size_t row = dim - 1 - rev;
    for (std::size_t col = row + 1; col < dim; ++col)
      rhs[row] -= lu[row * dim + col] * rhs[col];
    const double diag = lu[row * dim + row];
    rhs[row] = std::abs(diag) > 1.0e-14 ? rhs[row] / diag : 0.0;
  }
}

double norm2(const std::vector<double> &values) noexcept {
  double sum = 0.0;
  for (const double value : values)
    sum += value * value;
  return std::sqrt(sum);
}

void dense_matvec(const std::vector<double> &matrix,
                  const std::vector<double> &x, std::size_t dim,
                  std::vector<double> &out) {
  out.assign(dim, 0.0);
  for (std::size_t row = 0; row < dim; ++row) {
    double sum = 0.0;
    for (std::size_t col = 0; col < dim; ++col)
      sum += matrix[row * dim + col] * x[col];
    out[row] = sum;
  }
}

void solve_scaled_lu(const std::vector<double> &lu,
                     const std::vector<int> &pivots,
                     const std::vector<double> &scale, std::size_t dim,
                     const std::vector<double> &rhs,
                     std::vector<double> &solution) {
  solution.assign(dim, 0.0);
  if (lu.size() != dim * dim || pivots.size() != dim)
    return;
  solution = rhs;
  if (scale.size() == dim) {
    for (std::size_t i = 0; i < dim; ++i)
      solution[i] *= scale[i];
  }
  solve_dense_lu(lu, pivots, dim, solution);
  if (scale.size() == dim) {
    for (std::size_t i = 0; i < dim; ++i)
      solution[i] *= scale[i];
  }
}

bool passes_block_quality_gate(const std::vector<double> &original_matrix,
                               const std::vector<double> &lu,
                               const std::vector<int> &pivots,
                               const std::vector<double> &scale,
                               std::size_t dim, double *max_gain) {
  if (max_gain != nullptr)
    *max_gain = 0.0;
  if (dim == 0 || original_matrix.size() != dim * dim)
    return false;
  std::vector<std::vector<double>> probes;
  probes.push_back(std::vector<double>(dim, 1.0));
  std::vector<double> alternating(dim, 1.0);
  for (std::size_t i = 1; i < dim; i += 2)
    alternating[i] = -1.0;
  probes.push_back(std::move(alternating));
  std::vector<double> solution;
  std::vector<double> applied;
  for (const std::vector<double> &probe : probes) {
    solve_scaled_lu(lu, pivots, scale, dim, probe, solution);
    for (const double value : solution) {
      if (!std::isfinite(value))
        return false;
    }
    const double gain = norm2(solution) / std::max(1.0e-300, norm2(probe));
    if (max_gain != nullptr)
      *max_gain = std::max(*max_gain, gain);
    if (!std::isfinite(gain) || gain > 1.0e8)
      return false;
    dense_matvec(original_matrix, solution, dim, applied);
    for (std::size_t i = 0; i < dim; ++i)
      applied[i] -= probe[i];
    const double rel =
        norm2(applied) / std::max(1.0e-300, norm2(probe));
    if (!std::isfinite(rel) || rel > 1.0e-6)
      return false;
  }
  return true;
}

std::vector<JcpCoarseGalerkinPreconditioner::Aggregate>
make_jcp_coarse_aggregates(const std::vector<Panel> &panels,
                           const TreeWorkspace &tree,
                           std::size_t aggregate_cap) {
  std::vector<std::size_t> selected;
  if (tree.root_index < tree.nodes.size())
    selected.push_back(tree.root_index);
  if (selected.empty() || tree.order.size() != panels.size()) {
    JcpCoarseGalerkinPreconditioner::Aggregate aggregate;
    aggregate.panels.resize(panels.size());
    std::iota(aggregate.panels.begin(), aggregate.panels.end(), 0);
    for (const Panel &panel : panels)
      aggregate.area_sum += panel.area;
    return {std::move(aggregate)};
  }
  const std::size_t cap = std::clamp<std::size_t>(aggregate_cap, 4, 64);
  while (selected.size() < cap) {
    std::size_t split_pos = selected.size();
    std::size_t split_panels = 0;
    for (std::size_t pos = 0; pos < selected.size(); ++pos) {
      const TreeNode &node = tree.nodes[selected[pos]];
      if (node.child_count < 2 ||
          selected.size() - 1 + node.child_count > cap)
        continue;
      if (node.panel_count > split_panels) {
        split_panels = node.panel_count;
        split_pos = pos;
      }
    }
    if (split_pos >= selected.size())
      break;
    const TreeNode node = tree.nodes[selected[split_pos]];
    selected.erase(selected.begin() + static_cast<std::ptrdiff_t>(split_pos));
    for (std::size_t child = 0; child < node.child_count; ++child) {
      const std::size_t child_pos = node.first_child + child;
      if (child_pos < tree.child_indices.size() &&
          tree.child_indices[child_pos] < tree.nodes.size()) {
        selected.insert(selected.begin() + static_cast<std::ptrdiff_t>(split_pos),
                        tree.child_indices[child_pos]);
        ++split_pos;
      }
    }
  }
  std::vector<JcpCoarseGalerkinPreconditioner::Aggregate> aggregates;
  aggregates.reserve(selected.size());
  for (const std::size_t node_index : selected) {
    const TreeNode &node = tree.nodes[node_index];
    JcpCoarseGalerkinPreconditioner::Aggregate aggregate;
    aggregate.panels.reserve(node.panel_count);
    const std::size_t end =
        std::min(tree.order.size(), node.first_panel + node.panel_count);
    for (std::size_t pos = node.first_panel; pos < end; ++pos) {
      const std::size_t panel = tree.order[pos];
      if (panel < panels.size()) {
        aggregate.panels.push_back(panel);
        aggregate.area_sum += panels[panel].area;
      }
    }
    if (!aggregate.panels.empty())
      aggregates.push_back(std::move(aggregate));
  }
  return aggregates;
}

} // namespace

void JcpBlockPreconditioner::apply_block_with_scratch(
    const Block &block, const std::vector<double> &in,
    std::vector<double> &out, std::vector<double> &scratch) const {
  const std::size_t n = panel_count;
  const std::size_t m = block.panels.size();
  const std::size_t owned =
      block.owned_count == 0 ? m : std::min(block.owned_count, m);
  const std::size_t dim = coupled ? 2 * m : m;
  if (block.valid && block.lu.size() == dim * dim) {
    scratch.assign(dim, 0.0);
    for (std::size_t r = 0; r < m; ++r) {
      const std::size_t panel = block.panels[r];
      scratch[r] = panel < in.size() ? in[panel] : 0.0;
      if (coupled && n + panel < in.size())
        scratch[m + r] = in[n + panel];
    }
    if (block.scale.size() == dim) {
      for (std::size_t i = 0; i < dim; ++i)
        scratch[i] *= block.scale[i];
    }
    solve_dense_lu(block.lu, block.pivots, dim, scratch);
    if (block.scale.size() == dim) {
      for (std::size_t i = 0; i < dim; ++i)
        scratch[i] *= block.scale[i];
    }
    for (std::size_t r = 0; r < owned; ++r) {
      const std::size_t panel = block.panels[r];
      if (panel < out.size())
        out[panel] = scratch[r];
      if (coupled && n + panel < out.size())
        out[n + panel] = scratch[m + r];
    }
  } else {
    for (std::size_t r = 0; r < owned; ++r) {
      const std::size_t panel = block.panels[r];
      if (panel < out.size())
        out[panel] =
            in[panel] * (panel < inv_diag.size() ? inv_diag[panel] : 1.0);
      if (coupled && n + panel < out.size())
        out[n + panel] =
            in[n + panel] *
            (n + panel < inv_diag.size() ? inv_diag[n + panel] : 1.0);
    }
  }
}

void JcpBlockPreconditioner::apply_block(const Block &block,
                                         const std::vector<double> &in,
                                         std::vector<double> &out) const {
  std::vector<double> scratch;
  apply_block_with_scratch(block, in, out, scratch);
}

void JcpBlockPreconditioner::apply(const std::vector<double> &in,
                                   std::vector<double> &out) const {
  out.assign(in.size(), 0.0);
  std::vector<double> scratch;
  for (const Block &block : blocks)
    apply_block_with_scratch(block, in, out, scratch);
}

void JcpBlockPreconditioner::apply_parallel(const std::vector<double> &in,
                                            std::vector<double> &out,
                                            TitanExecutor &executor) const {
  out.assign(in.size(), 0.0);
  if (blocks.size() < 4 || executor.capacity() <= 1) {
    apply(in, out);
    return;
  }
  const std::size_t workers = solver_workers(executor, blocks.size());
  executor.parallel_for(
      blocks.size(), workers, 1,
      [&](std::size_t, std::size_t begin, std::size_t end) {
        std::vector<double> scratch;
        for (std::size_t i = begin; i < end; ++i)
          apply_block_with_scratch(blocks[i], in, out, scratch);
      });
}

std::size_t JcpBlockPreconditioner::estimated_bytes() const noexcept {
  std::size_t bytes = inv_diag.size() * sizeof(double);
  for (const Block &block : blocks) {
    bytes += block.panels.size() * sizeof(std::size_t);
    bytes += block.lu.size() * sizeof(double);
    bytes += block.pivots.size() * sizeof(int);
  }
  return bytes;
}

std::size_t JcpCoarseGalerkinPreconditioner::dimension() const noexcept {
  return coupled ? 2 * aggregates.size() : aggregates.size();
}

void JcpCoarseGalerkinPreconditioner::restrict_area_weighted(
    const std::vector<Panel> &panels, const std::vector<double> &fine,
    std::vector<double> &coarse) const {
  const std::size_t q = aggregates.size();
  coarse.assign(dimension(), 0.0);
  for (std::size_t a = 0; a < q; ++a) {
    const Aggregate &aggregate = aggregates[a];
    const double denom = std::max(1.0e-300, aggregate.area_sum);
    double phi = 0.0;
    double phin = 0.0;
    for (const std::size_t panel : aggregate.panels) {
      const double weight = panel < panels.size() ? panels[panel].area : 0.0;
      phi += weight * (panel < fine.size() ? fine[panel] : 0.0);
      if (coupled && panel_count + panel < fine.size())
        phin += weight * fine[panel_count + panel];
    }
    coarse[a] = phi / denom;
    if (coupled)
      coarse[q + a] = phin / denom;
  }
}

void JcpCoarseGalerkinPreconditioner::prolongate(
    const std::vector<double> &coarse, std::vector<double> &fine) const {
  const std::size_t q = aggregates.size();
  fine.assign(coupled ? 2 * panel_count : panel_count, 0.0);
  for (std::size_t a = 0; a < q; ++a) {
    for (const std::size_t panel : aggregates[a].panels) {
      if (panel < fine.size())
        fine[panel] = a < coarse.size() ? coarse[a] : 0.0;
      if (coupled && panel_count + panel < fine.size())
        fine[panel_count + panel] =
            q + a < coarse.size() ? coarse[q + a] : 0.0;
    }
  }
}

std::size_t JcpCoarseGalerkinPreconditioner::estimated_bytes(
    bool include_operator_basis) const noexcept {
  const std::size_t dim = dimension();
  const std::size_t fine_dim = coupled ? 2 * panel_count : panel_count;
  std::size_t bytes = inv_diag.size() * sizeof(double) +
                      panel_aggregate.size() * sizeof(std::size_t) +
                      aggregates.size() * sizeof(Aggregate) +
                      dim * dim * sizeof(double) + dim * sizeof(int) +
                      fine_operator_basis.size() * sizeof(double);
  if (include_operator_basis && fine_operator_basis.empty()) {
    if (dim > 0 && fine_dim <=
                       (std::numeric_limits<std::size_t>::max() - bytes) /
                           (dim * sizeof(double))) {
      bytes += fine_dim * dim * sizeof(double);
    } else {
      return std::numeric_limits<std::size_t>::max();
    }
  }
  const std::size_t residual_workspace =
      balanced_residual_workspace.empty()
          ? fine_dim
          : balanced_residual_workspace.size();
  const std::size_t coarse_workspace =
      balanced_coarse_workspace.empty() ? dim
                                        : balanced_coarse_workspace.size();
  if (residual_workspace >
          (std::numeric_limits<std::size_t>::max() - bytes) /
              sizeof(double) ||
      coarse_workspace >
          (std::numeric_limits<std::size_t>::max() - bytes -
           residual_workspace * sizeof(double)) /
              sizeof(double))
    return std::numeric_limits<std::size_t>::max();
  bytes += (residual_workspace + coarse_workspace) * sizeof(double);
  for (const Aggregate &aggregate : aggregates)
    bytes += aggregate.panels.size() * sizeof(std::size_t);
  return bytes;
}

void JcpCoarseGalerkinPreconditioner::apply(
    const std::vector<Panel> &panels, const std::vector<double> &in,
    std::vector<double> &out, TitanExecutor &executor) const {
  std::vector<double> coarse_rhs;
  restrict_area_weighted(panels, in, coarse_rhs);
  std::vector<double> projected;
  prolongate(coarse_rhs, projected);
  std::vector<double> complement(in.size(), 0.0);
  for (std::size_t i = 0; i < in.size(); ++i)
    complement[i] = in[i] - (i < projected.size() ? projected[i] : 0.0);
  if (block_base != nullptr) {
    block_base->apply_parallel(complement, out, executor);
  } else {
    out.resize(in.size());
    for (std::size_t i = 0; i < in.size(); ++i)
      out[i] = complement[i] * (i < inv_diag.size() ? inv_diag[i] : 1.0);
  }
  if (valid && lu.size() == dimension() * dimension()) {
    solve_dense_lu(lu, pivots, dimension(), coarse_rhs);
    prolongate(coarse_rhs, projected);
    for (std::size_t i = 0; i < out.size(); ++i)
      out[i] += i < projected.size() ? projected[i] : 0.0;
  }
}

void JcpCoarseGalerkinPreconditioner::apply_balanced(
    const std::vector<Panel> &panels, const std::vector<double> &in,
    std::vector<double> &out,
    const std::function<void(const std::vector<double> &,
                             std::vector<double> &)> &base_apply) const {
  const std::size_t dim = dimension();
  const std::size_t fine_dim = coupled ? 2 * panel_count : panel_count;
  const bool ready = valid && lu.size() == dim * dim &&
                     fine_operator_basis.size() == dim * fine_dim;
  if (!ready) {
    if (base_apply) {
      base_apply(in, out);
    } else {
      out.resize(in.size());
      for (std::size_t i = 0; i < in.size(); ++i)
        out[i] = in[i] * (i < inv_diag.size() ? inv_diag[i] : 1.0);
    }
    return;
  }

  std::lock_guard<std::mutex> lock(balanced_workspace_mutex);
  std::vector<double> &coarse_solution = balanced_coarse_workspace;
  restrict_area_weighted(panels, in, coarse_solution);
  solve_dense_lu(lu, pivots, dim, coarse_solution);

  std::vector<double> &residual = balanced_residual_workspace;
  residual.assign(in.begin(), in.end());
  if (residual.size() < fine_dim)
    residual.resize(fine_dim, 0.0);
  for (std::size_t row = 0; row < fine_dim; ++row) {
    const double *const images =
        fine_operator_basis.data() + row * dim;
    double coarse_image = 0.0;
    for (std::size_t col = 0; col < dim; ++col)
      coarse_image += images[col] * coarse_solution[col];
    residual[row] -= coarse_image;
  }

  if (base_apply) {
    base_apply(residual, out);
  } else {
    out.resize(residual.size());
    for (std::size_t i = 0; i < residual.size(); ++i)
      out[i] = residual[i] * (i < inv_diag.size() ? inv_diag[i] : 1.0);
  }
  if (out.size() < fine_dim)
    out.resize(fine_dim, 0.0);
  if (panel_aggregate.size() == panel_count) {
    const std::size_t q = aggregates.size();
    for (std::size_t panel = 0; panel < panel_count; ++panel) {
      const std::size_t aggregate = panel_aggregate[panel];
      if (aggregate >= q)
        continue;
      out[panel] += coarse_solution[aggregate];
      if (coupled)
        out[panel_count + panel] += coarse_solution[q + aggregate];
    }
  } else {
    for (std::size_t aggregate = 0; aggregate < aggregates.size();
         ++aggregate) {
      for (const std::size_t panel : aggregates[aggregate].panels) {
        if (panel >= panel_count)
          continue;
        out[panel] += coarse_solution[aggregate];
        if (coupled)
          out[panel_count + panel] +=
              coarse_solution[aggregates.size() + aggregate];
      }
    }
  }
}

std::vector<JcpBlockRange>
make_jcp_block_ranges(const TreeWorkspace &tree, std::size_t panel_count,
                      std::size_t block_panel_cap) {
  std::vector<JcpBlockRange> ranges;
  const std::size_t cap = std::clamp<std::size_t>(block_panel_cap, 8, 32);
  auto append_range = [&](std::size_t begin, std::size_t end) {
    if (begin < end && end <= tree.order.size())
      ranges.push_back({begin, end});
  };
  bool emitted_leaf_block = false;
  for (const TreeNode &node : tree.nodes) {
    if (node.child_count != 0 || node.panel_count == 0)
      continue;
    emitted_leaf_block = true;
    const std::size_t leaf_end = node.first_panel + node.panel_count;
    for (std::size_t begin = node.first_panel; begin < leaf_end; begin += cap)
      append_range(begin, std::min(leaf_end, begin + cap));
  }
  if (!emitted_leaf_block) {
    for (std::size_t begin = 0; begin < panel_count; begin += cap)
      append_range(begin, std::min(panel_count, begin + cap));
  }
  return ranges;
}

std::size_t estimate_jcp_block_preconditioner_bytes(
    const std::vector<JcpBlockRange> &ranges, std::size_t diagonal_size,
    bool coupled) noexcept {
  std::size_t bytes = diagonal_size * sizeof(double);
  for (const JcpBlockRange &range : ranges) {
    const std::size_t m = range.second - range.first;
    const std::size_t dim = coupled ? 2 * m : m;
    bytes += m * sizeof(std::size_t);
    bytes += dim * dim * sizeof(double);
    bytes += dim * sizeof(int);
  }
  return bytes;
}

std::size_t estimate_jcp_block_preconditioner_bytes_from_panel_groups(
    const std::vector<std::vector<std::size_t>> &panel_groups,
    std::size_t diagonal_size, bool coupled) noexcept {
  std::vector<JcpOwnedFactorPanelGroup> owned_factor_groups;
  owned_factor_groups.reserve(panel_groups.size());
  for (const auto &group : panel_groups) {
    JcpOwnedFactorPanelGroup converted;
    converted.owned_panels = group;
    converted.factor_panels = group;
    owned_factor_groups.push_back(std::move(converted));
  }
  return estimate_jcp_block_preconditioner_bytes_from_owned_factor_groups(
      owned_factor_groups, diagonal_size, coupled);
}

std::size_t estimate_jcp_block_preconditioner_bytes_from_owned_factor_groups(
    const std::vector<JcpOwnedFactorPanelGroup> &panel_groups,
    std::size_t diagonal_size, bool coupled) noexcept {
  std::size_t bytes = diagonal_size * sizeof(double);
  for (const auto &group : panel_groups) {
    const std::size_t m = group.factor_panels.size();
    const std::size_t dim = coupled ? 2 * m : m;
    bytes += m * sizeof(std::size_t);
    bytes += dim * dim * sizeof(double);
    bytes += dim * sizeof(int);
  }
  return bytes;
}

std::shared_ptr<JcpBlockPreconditioner>
build_jcp_block_preconditioner_from_panel_groups(
    const std::vector<Panel> &panels, bool coupled,
    double eps_ratio, double kappa, double lhs_phi, double lhs_phin,
    const std::vector<double> &diagonal, const std::string &near_integration,
    const std::string &quad_buffer, const std::string &block_near_mode,
    const std::vector<std::vector<std::size_t>> &panel_groups,
    TitanExecutor &executor) {
  std::vector<JcpOwnedFactorPanelGroup> owned_factor_groups;
  owned_factor_groups.reserve(panel_groups.size());
  for (const auto &group : panel_groups) {
    JcpOwnedFactorPanelGroup converted;
    converted.owned_panels = group;
    converted.factor_panels = group;
    owned_factor_groups.push_back(std::move(converted));
  }
  return build_jcp_block_preconditioner_from_owned_factor_groups(
      panels, coupled, eps_ratio, kappa, lhs_phi, lhs_phin, diagonal,
      near_integration, quad_buffer, block_near_mode, owned_factor_groups,
      executor);
}

std::shared_ptr<JcpBlockPreconditioner>
build_jcp_block_preconditioner_from_owned_factor_groups(
    const std::vector<Panel> &panels, bool coupled,
    double eps_ratio, double kappa, double lhs_phi, double lhs_phin,
    const std::vector<double> &diagonal, const std::string &near_integration,
    const std::string &quad_buffer, const std::string &block_near_mode,
    const std::vector<JcpOwnedFactorPanelGroup> &panel_groups,
    TitanExecutor &executor) {
  const std::size_t n = panels.size();
  if (n == 0)
    return {};
  auto precond = std::make_shared<JcpBlockPreconditioner>();
  precond->panel_count = n;
  precond->coupled = coupled;
  precond->inv_diag.assign(diagonal.size(), 1.0);
  for (std::size_t i = 0; i < diagonal.size(); ++i) {
    const double diag = std::abs(diagonal[i]);
    precond->inv_diag[i] =
        std::isfinite(diag) && diag > 1.0e-12 ? 1.0 / diag : 1.0;
  }

  const bool buffer_aware = block_near_mode == "buffer-aware" ||
                            block_near_mode == "operator-near";
  const bool operator_near = block_near_mode == "operator-near";
  auto build_block = [&](const JcpOwnedFactorPanelGroup &panel_group) {
    JcpBlockPreconditioner::Block block;
    block.panels.reserve(panel_group.factor_panels.size());
    std::vector<unsigned char> owned_seen(n, 0);
    std::vector<unsigned char> factor_seen(n, 0);
    for (const std::size_t panel : panel_group.owned_panels) {
      if (panel < n && !owned_seen[panel]) {
        owned_seen[panel] = 1;
        factor_seen[panel] = 1;
        block.panels.push_back(panel);
      }
    }
    block.owned_count = block.panels.size();
    for (const std::size_t panel : panel_group.factor_panels) {
      if (panel < n && !factor_seen[panel]) {
        factor_seen[panel] = 1;
        block.panels.push_back(panel);
      }
    }
    block.halo_count = block.panels.size() - block.owned_count;
    const std::size_t m = block.panels.size();
    const std::size_t dim = coupled ? 2 * m : m;
    if (dim == 0 || block.owned_count == 0)
      return block;
    block.lu.assign(dim * dim, 0.0);
    for (std::size_t r = 0; r < m; ++r) {
      const std::size_t target_index = block.panels[r];
      block.lu[r * dim + r] += lhs_phi;
      if (coupled)
        block.lu[(m + r) * dim + (m + r)] += lhs_phin;
      for (std::size_t c = 0; c < m; ++c) {
        const std::size_t source_index = block.panels[c];
        const bool singular = is_singular_near_pair(
            panels[target_index], target_index, panels[source_index],
            source_index);
        if (!singular && !buffer_aware)
          continue;
        if (!singular) {
          const GeometricNearDecision geometric =
              classify_geometric_near_buffer_pair(
                  panels[target_index], panels[source_index], quad_buffer);
          if (!geometric.active)
            continue;
        }
        double k1 = 0.0;
        double k2 = 0.0;
        double k3 = 0.0;
        double k4 = 0.0;
        double weight = panels[source_index].area;
        if (operator_near) {
          const JcpNearPairKernel selected = evaluate_jcp_near_pair_kernel(
              panels[target_index], target_index, panels[source_index],
              source_index, eps_ratio, kappa, coupled, near_integration,
              quad_buffer, nullptr);
          if (!selected.valid)
            continue;
          k1 = selected.k1;
          k2 = selected.k2;
          k3 = selected.k3;
          k4 = selected.k4;
          weight = 1.0;
        } else {
          const JcpKernels kernels = jcp_surface_kernels(
              panels[target_index], panels[source_index], eps_ratio, kappa);
          k1 = kernels.k1;
          k2 = kernels.k2;
          k3 = kernels.k3;
          k4 = kernels.k4;
        }
        if (coupled) {
          block.lu[r * dim + c] -= weight * k2;
          block.lu[r * dim + (m + c)] -= weight * k1;
          block.lu[(m + r) * dim + c] -= weight * k4;
          block.lu[(m + r) * dim + (m + c)] -= weight * k3;
        } else {
          block.lu[r * dim + c] -= weight * k2;
        }
      }
    }
    constexpr double kMinBlockPivotRatio = 1.0e-10;
    const std::vector<double> original_matrix = block.lu;
    block.scale = make_equilibration_scale(block.lu, dim);
    apply_symmetric_scale(block.lu, block.scale, dim);
    block.valid =
        factor_dense_lu(block.lu, block.pivots, dim, &block.pivot_ratio);
    block.ill_conditioned =
        !std::isfinite(block.pivot_ratio) ||
        block.pivot_ratio < kMinBlockPivotRatio;
    if (block.ill_conditioned)
      block.valid = false;
    if (block.valid) {
      block.valid = passes_block_quality_gate(
          original_matrix, block.lu, block.pivots, block.scale, dim,
          &block.quality_gain);
      block.quality_rejected = !block.valid;
    }
    return block;
  };

  precond->blocks.resize(panel_groups.size());
  const std::size_t workers = solver_workers(executor, panel_groups.size());
  executor.parallel_for(
      panel_groups.size(), workers, 1,
      [&](std::size_t, std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i)
          precond->blocks[i] = build_block(panel_groups[i]);
      });
  for (const auto &block : precond->blocks) {
    if (block.ill_conditioned)
      ++precond->ill_conditioned_blocks;
    if (block.quality_rejected)
      ++precond->quality_rejected_blocks;
    precond->quality_probe_max_gain =
        std::max(precond->quality_probe_max_gain, block.quality_gain);
    if (block.valid) {
      ++precond->valid_blocks;
      precond->valid_panel_count +=
          block.owned_count == 0 ? block.panels.size() : block.owned_count;
    }
  }
  precond->quality_status =
      precond->quality_rejected_blocks == 0 ? "pass" : "partial_reject";
  if (precond->valid_blocks == 0 && !precond->blocks.empty())
    precond->quality_status = "all_rejected";
  return precond;
}

std::shared_ptr<JcpBlockPreconditioner> build_jcp_block_preconditioner(
    const std::vector<Panel> &panels, const TreeWorkspace &tree, bool coupled,
    double eps_ratio, double kappa, double lhs_phi, double lhs_phin,
    const std::vector<double> &diagonal, const std::string &near_integration,
    const std::string &quad_buffer, const std::string &block_near_mode,
    const std::vector<JcpBlockRange> &ranges, TitanExecutor &executor) {
  if (tree.order.size() != panels.size())
    return {};
  std::vector<std::vector<std::size_t>> panel_groups;
  panel_groups.reserve(ranges.size());
  for (const JcpBlockRange &range : ranges) {
    if (range.first >= range.second || range.second > tree.order.size())
      continue;
    std::vector<std::size_t> group;
    group.reserve(range.second - range.first);
    for (std::size_t pos = range.first; pos < range.second; ++pos)
      group.push_back(tree.order[pos]);
    if (!group.empty())
      panel_groups.push_back(std::move(group));
  }
  return build_jcp_block_preconditioner_from_panel_groups(
      panels, coupled, eps_ratio, kappa, lhs_phi, lhs_phin, diagonal,
      near_integration, quad_buffer, block_near_mode, panel_groups, executor);
}

std::shared_ptr<JcpCoarseGalerkinPreconditioner>
make_jcp_coarse_structure(const std::vector<Panel> &panels,
                          const TreeWorkspace &tree, bool coupled,
                          const std::vector<double> &diagonal,
                          std::size_t aggregate_cap) {
  auto coarse = std::make_shared<JcpCoarseGalerkinPreconditioner>();
  coarse->panel_count = panels.size();
  coarse->coupled = coupled;
  coarse->aggregates =
      make_jcp_coarse_aggregates(panels, tree, aggregate_cap);
  coarse->panel_aggregate.assign(panels.size(),
                                 static_cast<std::size_t>(-1));
  for (std::size_t aggregate = 0; aggregate < coarse->aggregates.size();
       ++aggregate) {
    for (const std::size_t panel : coarse->aggregates[aggregate].panels) {
      if (panel < coarse->panel_aggregate.size())
        coarse->panel_aggregate[panel] = aggregate;
    }
  }
  coarse->inv_diag.assign(diagonal.size(), 1.0);
  for (std::size_t i = 0; i < diagonal.size(); ++i) {
    const double diag = std::abs(diagonal[i]);
    coarse->inv_diag[i] =
        std::isfinite(diag) && diag > 1.0e-12 ? 1.0 / diag : 1.0;
  }
  return coarse;
}

bool build_jcp_exact_galerkin(
    const std::vector<Panel> &panels, JcpCoarseGalerkinPreconditioner &coarse,
    const std::function<void(const std::vector<double> &,
                             std::vector<double> &)> &setup_matvec,
    bool verify, bool retain_operator_basis) {
  const std::size_t dim = coarse.dimension();
  if (dim == 0 || !setup_matvec)
    return false;
  coarse.fine_operator_basis.clear();
  const std::size_t fine_dim =
      coarse.coupled ? 2 * coarse.panel_count : coarse.panel_count;
  if (retain_operator_basis) {
    if (fine_dim > std::numeric_limits<std::size_t>::max() / dim)
      return false;
    coarse.fine_operator_basis.assign(fine_dim * dim, 0.0);
  }
  std::vector<double> matrix(dim * dim, 0.0);
  std::vector<double> basis(dim, 0.0);
  std::vector<double> fine;
  std::vector<double> applied;
  std::vector<double> restricted;
  for (std::size_t col = 0; col < dim; ++col) {
    std::fill(basis.begin(), basis.end(), 0.0);
    basis[col] = 1.0;
    coarse.prolongate(basis, fine);
    setup_matvec(fine, applied);
    ++coarse.setup_matvec_calls;
    if (retain_operator_basis) {
      if (applied.size() < fine_dim)
        return false;
      for (std::size_t row = 0; row < fine_dim; ++row)
        coarse.fine_operator_basis[row * dim + col] = applied[row];
    }
    coarse.restrict_area_weighted(panels, applied, restricted);
    for (std::size_t row = 0; row < dim; ++row)
      matrix[row * dim + col] = restricted[row];
  }
  for (std::size_t col = 0; col < dim; ++col) {
    std::fill(basis.begin(), basis.end(), 0.0);
    basis[col] = 1.0;
    coarse.prolongate(basis, fine);
    coarse.restrict_area_weighted(panels, fine, restricted);
    for (std::size_t row = 0; row < dim; ++row) {
      const double target = row == col ? 1.0 : 0.0;
      coarse.rp_identity_error_max =
          std::max(coarse.rp_identity_error_max,
                   std::abs(restricted[row] - target));
    }
  }
  if (verify) {
    double diff_sq = 0.0;
    double ref_sq = 0.0;
    for (std::size_t col = 0; col < dim; ++col) {
      std::fill(basis.begin(), basis.end(), 0.0);
      basis[col] = 1.0;
      coarse.prolongate(basis, fine);
      setup_matvec(fine, applied);
      ++coarse.setup_matvec_calls;
      coarse.restrict_area_weighted(panels, applied, restricted);
      for (std::size_t row = 0; row < dim; ++row) {
        const double expected = matrix[row * dim + col];
        const double delta = restricted[row] - expected;
        diff_sq += delta * delta;
        ref_sq += expected * expected;
      }
    }
    coarse.verify_rel_error_max =
        std::sqrt(diff_sq) / std::max(1.0e-300, std::sqrt(ref_sq));
  }
  coarse.lu = std::move(matrix);
  coarse.valid =
      factor_dense_lu(coarse.lu, coarse.pivots, dim, &coarse.pivot_ratio);
  constexpr double kMinCoarsePivotRatio = 1.0e-12;
  if (!std::isfinite(coarse.pivot_ratio) ||
      coarse.pivot_ratio < kMinCoarsePivotRatio)
    coarse.valid = false;
  return coarse.valid;
}

bool complete_jcp_hybrid_galerkin_monopole(
    const std::vector<Panel> &panels, JcpCoarseGalerkinPreconditioner &coarse,
    const std::vector<std::size_t> &near_row_ptr,
    const std::vector<std::size_t> &near_col_index, double eps_ratio,
    double kappa,
    const std::function<void(const std::vector<double> &,
                             std::vector<double> &)> &audit_matvec,
    std::size_t audit_probes, double audit_tolerance,
    TitanExecutor &executor) {
  struct AggregateMoment {
    double area{0.0};
    double x{0.0};
    double y{0.0};
    double z{0.0};
    double nx{0.0};
    double ny{0.0};
    double nz{0.0};
  };

  const std::size_t n = panels.size();
  const std::size_t q = coarse.aggregates.size();
  const std::size_t dim = coarse.dimension();
  const std::size_t fine_dim = coarse.coupled ? 2 * n : n;
  constexpr std::size_t kOctantsPerAggregate = 8;
  const bool dimensions_safe =
      dim > 0 && fine_dim <= std::numeric_limits<std::size_t>::max() / dim &&
      q <= std::numeric_limits<std::size_t>::max() / kOctantsPerAggregate &&
      n <= std::numeric_limits<std::size_t>::max() -
               q * kOctantsPerAggregate;
  const bool capture_ready =
      dimensions_safe && coarse.coupled && coarse.panel_count == n && q > 0 &&
      dim == 2 * q && coarse.panel_aggregate.size() == n &&
      near_row_ptr.size() == n + 1 &&
      coarse.fine_operator_basis.size() == fine_dim * dim &&
      std::all_of(coarse.panel_aggregate.begin(),
                  coarse.panel_aggregate.end(),
                  [q](std::size_t aggregate) { return aggregate < q; });
  if (!capture_ready || !audit_matvec || !std::isfinite(audit_tolerance) ||
      audit_tolerance <= 0.0)
    return false;

  const auto moments_begin = std::chrono::steady_clock::now();
  std::vector<AggregateMoment> aggregate_moments(q);
  auto source_x = [](const Panel &panel) noexcept {
    return panel.has_source_quadrature_center ? panel.source_x : panel.x;
  };
  auto source_y = [](const Panel &panel) noexcept {
    return panel.has_source_quadrature_center ? panel.source_y : panel.y;
  };
  auto source_z = [](const Panel &panel) noexcept {
    return panel.has_source_quadrature_center ? panel.source_z : panel.z;
  };
  auto subtract_panel = [&](AggregateMoment &moment,
                            const Panel &panel) noexcept {
    moment.area -= panel.area;
    moment.x -= panel.area * source_x(panel);
    moment.y -= panel.area * source_y(panel);
    moment.z -= panel.area * source_z(panel);
    moment.nx -= panel.area * panel_source_nx(panel);
    moment.ny -= panel.area * panel_source_ny(panel);
    moment.nz -= panel.area * panel_source_nz(panel);
  };
  for (std::size_t aggregate = 0; aggregate < q; ++aggregate) {
    AggregateMoment &moment = aggregate_moments[aggregate];
    for (const std::size_t panel_index : coarse.aggregates[aggregate].panels) {
      if (panel_index >= n)
        return false;
      const Panel &panel = panels[panel_index];
      moment.area += panel.area;
      moment.x += panel.area * source_x(panel);
      moment.y += panel.area * source_y(panel);
      moment.z += panel.area * source_z(panel);
      moment.nx += panel.area * panel_source_nx(panel);
      moment.ny += panel.area * panel_source_ny(panel);
      moment.nz += panel.area * panel_source_nz(panel);
    }
  }

  coarse.rp_identity_error_max = 0.0;
  for (std::size_t aggregate = 0; aggregate < q; ++aggregate) {
    const double denom = coarse.aggregates[aggregate].area_sum;
    if (!(denom > 1.0e-300) || !std::isfinite(denom))
      return false;
    coarse.rp_identity_error_max =
        std::max(coarse.rp_identity_error_max,
                 std::abs(aggregate_moments[aggregate].area / denom - 1.0));
  }

  std::vector<AggregateMoment> octant_moments(q * kOctantsPerAggregate);
  std::vector<std::size_t> panel_octant(n, 0);
  for (std::size_t aggregate = 0; aggregate < q; ++aggregate) {
    const AggregateMoment &total = aggregate_moments[aggregate];
    if (!(total.area > 1.0e-14))
      continue;
    const double center_x = total.x / total.area;
    const double center_y = total.y / total.area;
    const double center_z = total.z / total.area;
    for (const std::size_t panel_index : coarse.aggregates[aggregate].panels) {
      const Panel &panel = panels[panel_index];
      const std::size_t octant =
          (source_x(panel) >= center_x ? 1U : 0U) |
          (source_y(panel) >= center_y ? 2U : 0U) |
          (source_z(panel) >= center_z ? 4U : 0U);
      panel_octant[panel_index] = octant;
      AggregateMoment &moment =
          octant_moments[aggregate * kOctantsPerAggregate + octant];
      moment.area += panel.area;
      moment.x += panel.area * source_x(panel);
      moment.y += panel.area * source_y(panel);
      moment.z += panel.area * source_z(panel);
      moment.nx += panel.area * panel_source_nx(panel);
      moment.ny += panel.area * panel_source_ny(panel);
      moment.nz += panel.area * panel_source_nz(panel);
    }
  }
  coarse.moments_setup_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - moments_begin)
          .count();

  // A_near*P was captured while the ILU matrix was assembled. Complete each
  // fine row with octant monopoles after subtracting graph sources already
  // represented exactly. The target's own octant is omitted because a single
  // monopole is not reliable there; the full-FMM audit below decides whether
  // that controlled approximation is accurate enough for coarse correction.
  const auto far_begin = std::chrono::steady_clock::now();
  const std::size_t workers = solver_workers(executor, n);
  executor.parallel_for(
      n, workers, 4,
      [&](std::size_t, std::size_t begin, std::size_t end) {
        std::vector<AggregateMoment> remaining(q * kOctantsPerAggregate);
        for (std::size_t target = begin; target < end; ++target) {
          remaining = octant_moments;
          const std::size_t row_begin = near_row_ptr[target];
          const std::size_t row_end = near_row_ptr[target + 1];
          if (row_begin > row_end || row_end > near_col_index.size())
            continue;
          for (std::size_t position = row_begin; position < row_end;
               ++position) {
            const std::size_t source = near_col_index[position];
            if (source >= n)
              continue;
            const std::size_t aggregate = coarse.panel_aggregate[source];
            if (aggregate < q) {
              const std::size_t octant = panel_octant[source];
              subtract_panel(
                  remaining[aggregate * kOctantsPerAggregate + octant],
                  panels[source]);
            }
          }
          const std::size_t target_aggregate =
              coarse.panel_aggregate[target];
          const std::size_t target_octant = panel_octant[target];
          double *const phi_row = coarse.fine_operator_basis.data() +
                                  target * dim;
          double *const phin_row = coarse.fine_operator_basis.data() +
                                   (n + target) * dim;
          for (std::size_t aggregate = 0; aggregate < q; ++aggregate) {
            for (std::size_t octant = 0; octant < kOctantsPerAggregate;
                 ++octant) {
              if (aggregate == target_aggregate && octant == target_octant)
                continue;
              const AggregateMoment &moment =
                  remaining[aggregate * kOctantsPerAggregate + octant];
              if (!(moment.area > 1.0e-14) || !std::isfinite(moment.area))
                continue;
              Panel source;
              source.area = moment.area;
              source.x = moment.x / moment.area;
              source.y = moment.y / moment.area;
              source.z = moment.z / moment.area;
              source.nx = moment.nx / moment.area;
              source.ny = moment.ny / moment.area;
              source.nz = moment.nz / moment.area;
              const JcpNearPairKernel kernel =
                  evaluate_jcp_centroid_monopole_kernel(
                      panels[target], source, eps_ratio, kappa);
              if (!kernel.valid)
                continue;
              phi_row[aggregate] -= kernel.k2;
              phi_row[q + aggregate] -= kernel.k1;
              phin_row[aggregate] -= kernel.k4;
              phin_row[q + aggregate] -= kernel.k3;
            }
          }
        }
      });

  // Form R(A_hybrid P) in a single fine-row pass. The previous column-wise
  // implementation reconstructed a full fine vector and rescanned every
  // aggregate for each coarse basis column.
  std::vector<double> matrix(dim * dim, 0.0);
  for (std::size_t target = 0; target < n; ++target) {
    const std::size_t aggregate = coarse.panel_aggregate[target];
    const double weight =
        panels[target].area / coarse.aggregates[aggregate].area_sum;
    double *const phi_coarse_row = matrix.data() + aggregate * dim;
    double *const phin_coarse_row = matrix.data() + (q + aggregate) * dim;
    const double *const phi_fine_row =
        coarse.fine_operator_basis.data() + target * dim;
    const double *const phin_fine_row =
        coarse.fine_operator_basis.data() + (n + target) * dim;
    for (std::size_t col = 0; col < dim; ++col) {
      phi_coarse_row[col] += weight * phi_fine_row[col];
      phin_coarse_row[col] += weight * phin_fine_row[col];
    }
  }
  coarse.far_setup_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - far_begin)
          .count();

  coarse.verify_rel_error_max = 0.0;
  const auto audit_begin = std::chrono::steady_clock::now();
  const std::size_t probes = std::clamp<std::size_t>(audit_probes, 1, 2);
  std::vector<double> basis(dim, 0.0);
  std::vector<double> fine;
  std::vector<double> exact_fine;
  std::vector<double> exact_coarse;
  std::vector<double> approximate_coarse(dim, 0.0);
  for (std::size_t probe = 0; probe < probes; ++probe) {
    for (std::size_t col = 0; col < dim; ++col) {
      const std::size_t code = (col + 1) * (probe == 0 ? 5 : 11) + 3 * probe;
      basis[col] = (code & 1U) == 0U ? 1.0 : -1.0;
    }
    coarse.prolongate(basis, fine);
    audit_matvec(fine, exact_fine);
    ++coarse.setup_matvec_calls;
    if (exact_fine.size() < fine_dim) {
      coarse.audit_setup_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - audit_begin)
              .count();
      return false;
    }
    coarse.restrict_area_weighted(panels, exact_fine, exact_coarse);
    for (std::size_t row = 0; row < dim; ++row) {
      double value = 0.0;
      const double *const coarse_row = matrix.data() + row * dim;
      for (std::size_t col = 0; col < dim; ++col)
        value += coarse_row[col] * basis[col];
      approximate_coarse[row] = value;
    }
    double diff_sq = 0.0;
    double ref_sq = 0.0;
    for (std::size_t row = 0; row < dim; ++row) {
      const double delta = approximate_coarse[row] - exact_coarse[row];
      diff_sq += delta * delta;
      ref_sq += exact_coarse[row] * exact_coarse[row];
    }
    const double relative_error =
        std::sqrt(diff_sq) / std::max(1.0e-300, std::sqrt(ref_sq));
    if (!std::isfinite(relative_error)) {
      coarse.audit_setup_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - audit_begin)
              .count();
      return false;
    }
    coarse.verify_rel_error_max =
        std::max(coarse.verify_rel_error_max, relative_error);
  }
  coarse.audit_setup_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - audit_begin)
          .count();
  if (coarse.verify_rel_error_max > audit_tolerance)
    return false;

  coarse.lu = std::move(matrix);
  coarse.valid =
      factor_dense_lu(coarse.lu, coarse.pivots, dim, &coarse.pivot_ratio);
  constexpr double kMinCoarsePivotRatio = 1.0e-12;
  if (!std::isfinite(coarse.pivot_ratio) ||
      coarse.pivot_ratio < kMinCoarsePivotRatio)
    coarse.valid = false;
  return coarse.valid;
}

} // namespace TitanPBE::Tabi
