#include "tabi_calderon_prepared.h"

#include "tabi_fmm_m2l_prototype.h"
#include "tabi_types.h"
#include "titan_executor.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace TitanPBE::Tabi::calderon {
namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] double elapsed_seconds(Clock::time_point begin,
                                     Clock::time_point end) noexcept {
  return std::chrono::duration<double>(end - begin).count();
}

[[nodiscard]] bool all_finite(std::span<const double> values) noexcept {
  for (const double value : values) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

struct Vec3 {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

[[nodiscard]] Vec3 operator+(const Vec3 &a, const Vec3 &b) noexcept {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
[[nodiscard]] Vec3 operator-(const Vec3 &a, const Vec3 &b) noexcept {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
[[nodiscard]] Vec3 operator*(double scale, const Vec3 &v) noexcept {
  return {scale * v.x, scale * v.y, scale * v.z};
}
[[nodiscard]] double dot(const Vec3 &a, const Vec3 &b) noexcept {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
[[nodiscard]] Vec3 cross(const Vec3 &a, const Vec3 &b) noexcept {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}
[[nodiscard]] double norm(const Vec3 &v) noexcept {
  return std::sqrt(dot(v, v));
}
[[nodiscard]] Vec3 mesh_vertex(const SurfaceMesh &mesh,
                               std::size_t index) noexcept {
  const Vertex &v = mesh.vertices[index];
  return {v.x, v.y, v.z};
}

struct PatchMeta {
  std::size_t owner{0};
  std::array<std::size_t, 3> parent_vertex{};
  std::array<double, 3> parent_lambda{};
};

struct PatchPoint {
  Vec3 point{};
  std::array<double, 3> lambda{};
  std::size_t id{0};
};

[[nodiscard]] std::pair<std::size_t, std::size_t>
edge_key(std::size_t a, std::size_t b) noexcept {
  return std::minmax(a, b);
}

void append_patch_panel(std::vector<Panel> &panels,
                        std::vector<PatchMeta> &metadata,
                        std::vector<std::map<std::size_t, double>> &pairing,
                        const Triangle &parent,
                        const std::array<std::size_t, 3> &parent_vertex,
                        std::size_t owner, const PatchPoint &a,
                        const PatchPoint &b, const PatchPoint &c) {
  const Vec3 normal_vector = cross(b.point - a.point, c.point - a.point);
  const double twice_area = norm(normal_vector);
  if (!(twice_area > 1.0e-18)) {
    throw std::runtime_error("degenerate Calderon dual patch");
  }
  const double area = 0.5 * twice_area;
  const Vec3 centroid = (1.0 / 3.0) * (a.point + b.point + c.point);
  const std::array<double, 3> lambda = {
      (a.lambda[0] + b.lambda[0] + c.lambda[0]) / 3.0,
      (a.lambda[1] + b.lambda[1] + c.lambda[1]) / 3.0,
      (a.lambda[2] + b.lambda[2] + c.lambda[2]) / 3.0};
  Panel panel;
  panel.x = centroid.x;
  panel.y = centroid.y;
  panel.z = centroid.z;
  panel.nx = parent.nx;
  panel.ny = parent.ny;
  panel.nz = parent.nz;
  panel.area = area;
  panel.atom_index = parent.atom_index;
  panel.v0x = a.point.x;
  panel.v0y = a.point.y;
  panel.v0z = a.point.z;
  panel.v1x = b.point.x;
  panel.v1y = b.point.y;
  panel.v1z = b.point.z;
  panel.v2x = c.point.x;
  panel.v2y = c.point.y;
  panel.v2z = c.point.z;
  panel.v0_index = a.id;
  panel.v1_index = b.id;
  panel.v2_index = c.id;
  panel.has_triangle_geometry = true;
  panels.push_back(panel);
  metadata.push_back({owner, parent_vertex, lambda});
  for (std::size_t local = 0; local < 3; ++local) {
    pairing[owner][parent_vertex[local]] += area * lambda[local];
  }
}

void build_dual_patch_panels(
    const SurfaceMesh &mesh, std::vector<Panel> &panels,
    std::vector<PatchMeta> &metadata,
    std::vector<std::map<std::size_t, double>> &pairing) {
  pairing.assign(mesh.vertices.size(), {});
  panels.clear();
  metadata.clear();
  panels.reserve(6 * mesh.triangles.size());
  metadata.reserve(6 * mesh.triangles.size());
  std::map<std::pair<std::size_t, std::size_t>, std::size_t> midpoint_ids;
  std::size_t next_id = mesh.vertices.size();
  for (const Triangle &triangle : mesh.triangles) {
    const std::array<std::size_t, 3> vertex = {triangle.v0, triangle.v1,
                                               triangle.v2};
    const Vec3 p0 = mesh_vertex(mesh, vertex[0]);
    const Vec3 p1 = mesh_vertex(mesh, vertex[1]);
    const Vec3 p2 = mesh_vertex(mesh, vertex[2]);
    const auto midpoint_id = [&](std::size_t x, std::size_t y) {
      const auto key = edge_key(x, y);
      const auto [it, inserted] = midpoint_ids.emplace(key, next_id);
      if (inserted) {
        ++next_id;
      }
      return it->second;
    };
    const PatchPoint v0{p0, {1.0, 0.0, 0.0}, vertex[0]};
    const PatchPoint v1{p1, {0.0, 1.0, 0.0}, vertex[1]};
    const PatchPoint v2{p2, {0.0, 0.0, 1.0}, vertex[2]};
    const PatchPoint m01{0.5 * (p0 + p1), {0.5, 0.5, 0.0},
                         midpoint_id(vertex[0], vertex[1])};
    const PatchPoint m12{0.5 * (p1 + p2), {0.0, 0.5, 0.5},
                         midpoint_id(vertex[1], vertex[2])};
    const PatchPoint m20{0.5 * (p2 + p0), {0.5, 0.0, 0.5},
                         midpoint_id(vertex[2], vertex[0])};
    const PatchPoint center{(1.0 / 3.0) * (p0 + p1 + p2),
                            {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}, next_id++};
    append_patch_panel(panels, metadata, pairing, triangle, vertex, vertex[0],
                       v0, m01, center);
    append_patch_panel(panels, metadata, pairing, triangle, vertex, vertex[0],
                       v0, center, m20);
    append_patch_panel(panels, metadata, pairing, triangle, vertex, vertex[1],
                       v1, m12, center);
    append_patch_panel(panels, metadata, pairing, triangle, vertex, vertex[1],
                       v1, center, m01);
    append_patch_panel(panels, metadata, pairing, triangle, vertex, vertex[2],
                       v2, m20, center);
    append_patch_panel(panels, metadata, pairing, triangle, vertex, vertex[2],
                       v2, center, m12);
  }
}

} // namespace

struct PreparedCalderonOperator::State {
  std::string status{"not_prepared"};
  std::string geometry_status{"not_built"};
  PreparedCalderonConfig config;
  std::vector<Panel> auxiliary_panels;
  std::vector<PatchMeta> patch_metadata;
  std::vector<double> auxiliary_lumped_mass;
  std::vector<std::size_t> pairing_row_ptr;
  std::vector<std::size_t> pairing_col_index;
  std::vector<double> pairing_values;
  std::vector<double> pairing_diag_inverse;
  MassCompatibleTransferResult transfer;
  std::shared_ptr<titan::tabi::fmm_m2l::FmmM2lAnalyticContext> fmm_context;
  std::vector<double> triangle_unknowns;
  std::vector<double> triangle_action;
  std::vector<double> auxiliary_action;
  std::vector<double> primary_phi;
  std::vector<double> primary_phin;
  std::size_t prepare_build_count{0};
  std::size_t context_build_count{0};
  std::size_t apply_count{0};
  double prepare_seconds{0.0};
  double transfer_seconds{0.0};
  double fmm_setup_seconds{0.0};
  std::size_t estimated_bytes{0};
};

namespace {

void apply_pairing(const PreparedCalderonOperator::State &state,
                   std::span<const double> input,
                   std::vector<double> &output) {
  output.assign(input.size(), 0.0);
  for (std::size_t row = 0; row < input.size(); ++row) {
    for (std::size_t p = state.pairing_row_ptr[row];
         p < state.pairing_row_ptr[row + 1]; ++p) {
      output[row] += state.pairing_values[p] *
                     input[state.pairing_col_index[p]];
    }
  }
}

std::size_t solve_pairing(const PreparedCalderonOperator::State &state,
                          std::span<const double> rhs,
                          std::vector<double> &solution) {
  const std::size_t n = rhs.size();
  solution.assign(n, 0.0);
  std::vector<double> residual(rhs.begin(), rhs.end());
  std::vector<double> z(n, 0.0);
  std::vector<double> direction(n, 0.0);
  std::vector<double> product(n, 0.0);
  double rhs_sq = 0.0;
  double rz = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    rhs_sq += rhs[i] * rhs[i];
    z[i] = state.pairing_diag_inverse[i] * residual[i];
    direction[i] = z[i];
    rz += residual[i] * z[i];
  }
  if (rhs_sq <= 1.0e-300) {
    return 0;
  }
  constexpr std::size_t kMaxIterations = 256;
  constexpr double kTolerance = 1.0e-12;
  for (std::size_t iteration = 0; iteration < kMaxIterations; ++iteration) {
    apply_pairing(state, direction, product);
    double denominator = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      denominator += direction[i] * product[i];
    }
    if (!(std::abs(denominator) > 1.0e-300) || !std::isfinite(denominator)) {
      return kMaxIterations;
    }
    const double alpha = rz / denominator;
    double residual_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      solution[i] += alpha * direction[i];
      residual[i] -= alpha * product[i];
      residual_sq += residual[i] * residual[i];
    }
    if (std::sqrt(residual_sq / rhs_sq) <= kTolerance) {
      return iteration + 1;
    }
    double next_rz = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      z[i] = state.pairing_diag_inverse[i] * residual[i];
      next_rz += residual[i] * z[i];
    }
    const double beta = next_rz / std::max(1.0e-300, rz);
    for (std::size_t i = 0; i < n; ++i) {
      direction[i] = z[i] + beta * direction[i];
    }
    rz = next_rz;
  }
  return kMaxIterations;
}

} // namespace

PreparedCalderonOperator::PreparedCalderonOperator()
    : state_(std::make_unique<State>()) {}

PreparedCalderonOperator::PreparedCalderonOperator(std::unique_ptr<State> state)
    : state_(std::move(state)) {}

PreparedCalderonOperator::~PreparedCalderonOperator() = default;
PreparedCalderonOperator::PreparedCalderonOperator(
    PreparedCalderonOperator &&) noexcept = default;
PreparedCalderonOperator &PreparedCalderonOperator::operator=(
    PreparedCalderonOperator &&) noexcept = default;

const std::string &PreparedCalderonOperator::status() const noexcept {
  return state_->status;
}
const std::string &PreparedCalderonOperator::geometry_status() const noexcept {
  return state_->geometry_status;
}
std::size_t PreparedCalderonOperator::primary_panels() const noexcept {
  return state_->transfer.primary_panels;
}
std::size_t PreparedCalderonOperator::auxiliary_vertices() const noexcept {
  return state_->transfer.auxiliary_vertices;
}
std::size_t PreparedCalderonOperator::fmm_quadrature_panels() const noexcept {
  return state_->auxiliary_panels.size();
}
std::size_t PreparedCalderonOperator::prepare_build_count() const noexcept {
  return state_->prepare_build_count;
}
std::size_t PreparedCalderonOperator::context_build_count() const noexcept {
  return state_->context_build_count;
}
std::size_t PreparedCalderonOperator::apply_count() const noexcept {
  return state_->apply_count;
}
double PreparedCalderonOperator::prepare_seconds() const noexcept {
  return state_->prepare_seconds;
}
double PreparedCalderonOperator::transfer_seconds() const noexcept {
  return state_->transfer_seconds;
}
double PreparedCalderonOperator::fmm_setup_seconds() const noexcept {
  return state_->fmm_setup_seconds;
}
std::size_t PreparedCalderonOperator::estimated_bytes() const noexcept {
  return state_->estimated_bytes;
}

PreparedCalderonApplyResult PreparedCalderonOperator::apply_auxiliary(
    std::span<const double> auxiliary_unknowns, TitanExecutor &executor) {
  PreparedCalderonApplyResult result;
  if (state_->status != "prepared") {
    result.status = "operator_not_prepared";
    return result;
  }
  const std::size_t n = state_->transfer.auxiliary_vertices;
  const std::size_t triangle_count = state_->auxiliary_panels.size();
  if (auxiliary_unknowns.size() != 2 * n) {
    result.status = "invalid_auxiliary_unknown_size";
    return result;
  }
  const auto begin = Clock::now();
  state_->triangle_unknowns.assign(2 * triangle_count, 0.0);
  for (std::size_t triangle_index = 0; triangle_index < triangle_count;
       ++triangle_index) {
    const PatchMeta &patch = state_->patch_metadata[triangle_index];
    double phi = 0.0;
    for (std::size_t local = 0; local < 3; ++local) {
      phi += patch.parent_lambda[local] *
             auxiliary_unknowns[patch.parent_vertex[local]];
    }
    state_->triangle_unknowns[triangle_index] = phi;
    state_->triangle_unknowns[triangle_count + triangle_index] =
        auxiliary_unknowns[n + patch.owner];
  }

  FmmM2lSingleMatvecStats stats;
  const auto fmm_begin = Clock::now();
  apply_jcp_fmm_m2l_single_matvec(
      state_->auxiliary_panels, state_->triangle_unknowns,
      state_->config.eps_ratio, state_->config.kappa,
      0.5 * (1.0 + state_->config.eps_ratio),
      0.5 * (1.0 + 1.0 / state_->config.eps_ratio),
      state_->config.leaf_size, state_->config.opening_theta,
      state_->config.expansion_order, executor, state_->triangle_action, &stats,
      true, &state_->fmm_context, state_->config.near_integration,
      state_->config.quad_buffer);
  result.fmm_seconds = elapsed_seconds(fmm_begin, Clock::now());
  result.near_pairs = stats.near_pairs;
  result.far_pairs = stats.far_pairs;
  if (state_->triangle_action.size() != 2 * triangle_count ||
      !all_finite(state_->triangle_action)) {
    result.status = "fmm_nonfinite_or_shape_gap";
    return result;
  }

  state_->auxiliary_action.assign(2 * n, 0.0);
  for (std::size_t triangle_index = 0; triangle_index < triangle_count;
       ++triangle_index) {
    const Panel &panel = state_->auxiliary_panels[triangle_index];
    const PatchMeta &patch = state_->patch_metadata[triangle_index];
    state_->auxiliary_action[patch.owner] +=
        panel.area * state_->triangle_action[triangle_index];
    for (std::size_t local = 0; local < 3; ++local) {
      state_->auxiliary_action[n + patch.parent_vertex[local]] +=
          panel.area * patch.parent_lambda[local] *
          state_->triangle_action[triangle_count + triangle_index];
    }
  }
  const auto pairing_begin = Clock::now();
  std::vector<double> strong_phi;
  std::vector<double> strong_phin;
  const std::size_t phi_iterations = solve_pairing(
      *state_, std::span<const double>(state_->auxiliary_action.data(), n),
      strong_phi);
  const std::size_t phin_iterations = solve_pairing(
      *state_, std::span<const double>(state_->auxiliary_action.data() + n, n),
      strong_phin);
  result.pairing_seconds = elapsed_seconds(pairing_begin, Clock::now());
  result.pairing_iterations_max = std::max(phi_iterations, phin_iterations);
  if (result.pairing_iterations_max >= 256) {
    result.status = "pairing_solve_gap";
    return result;
  }
  std::copy(strong_phi.begin(), strong_phi.end(),
            state_->auxiliary_action.begin());
  std::copy(strong_phin.begin(), strong_phin.end(),
            state_->auxiliary_action.begin() + static_cast<std::ptrdiff_t>(n));
  result.finite = all_finite(state_->auxiliary_action);
  result.status = result.finite ? "ok" : "nonfinite";
  result.values = state_->auxiliary_action;
  ++state_->apply_count;
  result.apply_count = state_->apply_count;
  result.apply_seconds = elapsed_seconds(begin, Clock::now());
  return result;
}

PreparedCalderonApplyResult PreparedCalderonOperator::apply_primary(
    std::span<const double> primary_unknowns, TitanExecutor &executor) {
  PreparedCalderonApplyResult result;
  if (state_->status != "prepared") {
    result.status = "operator_not_prepared";
    return result;
  }
  const std::size_t primary_count = state_->transfer.primary_panels;
  const std::size_t auxiliary_count = state_->transfer.auxiliary_vertices;
  if (primary_unknowns.size() != 2 * primary_count) {
    result.status = "invalid_primary_unknown_size";
    return result;
  }
  const auto begin = Clock::now();
  const auto transfer_in_begin = Clock::now();
  const std::vector<double> auxiliary_phi =
      apply_mass_compatible_primary_to_auxiliary(
          state_->transfer, primary_unknowns.first(primary_count));
  const std::vector<double> auxiliary_phin =
      apply_mass_compatible_primary_to_auxiliary(
          state_->transfer, primary_unknowns.subspan(primary_count));
  std::vector<double> auxiliary_unknowns(2 * auxiliary_count, 0.0);
  std::copy(auxiliary_phi.begin(), auxiliary_phi.end(),
            auxiliary_unknowns.begin());
  std::copy(auxiliary_phin.begin(), auxiliary_phin.end(),
            auxiliary_unknowns.begin() + static_cast<std::ptrdiff_t>(auxiliary_count));
  result.transfer_in_seconds =
      elapsed_seconds(transfer_in_begin, Clock::now());

  PreparedCalderonApplyResult auxiliary =
      apply_auxiliary(auxiliary_unknowns, executor);
  if (auxiliary.status != "ok") {
    return auxiliary;
  }
  result.fmm_seconds = auxiliary.fmm_seconds;
  result.pairing_seconds = auxiliary.pairing_seconds;
  result.pairing_iterations_max = auxiliary.pairing_iterations_max;
  result.near_pairs = auxiliary.near_pairs;
  result.far_pairs = auxiliary.far_pairs;
  const auto transfer_out_begin = Clock::now();
  const std::vector<double> primary_phi =
      apply_mass_compatible_auxiliary_to_primary(
          state_->transfer,
          std::span<const double>(auxiliary.values.data(), auxiliary_count));
  const std::vector<double> primary_phin =
      apply_mass_compatible_auxiliary_to_primary(
          state_->transfer,
          std::span<const double>(auxiliary.values.data() + auxiliary_count,
                                  auxiliary_count));
  result.values.assign(2 * primary_count, 0.0);
  std::copy(primary_phi.begin(), primary_phi.end(), result.values.begin());
  std::copy(primary_phin.begin(), primary_phin.end(),
            result.values.begin() + static_cast<std::ptrdiff_t>(primary_count));
  result.transfer_out_seconds =
      elapsed_seconds(transfer_out_begin, Clock::now());
  result.finite = all_finite(result.values);
  result.status = result.finite ? "ok" : "nonfinite";
  result.apply_count = state_->apply_count;
  result.apply_seconds = elapsed_seconds(begin, Clock::now());
  return result;
}

PreparedCalderonOperator prepare_fmm_native_calderon_operator(
    std::span<const Panel> primary_panels,
    const AuxiliarySurfaceResult &auxiliary,
    const PreparedCalderonConfig &config, TitanExecutor &executor) {
  auto state = std::make_unique<PreparedCalderonOperator::State>();
  const auto begin = Clock::now();
  state->config = config;
  if (primary_panels.empty() || auxiliary.status != "pass" ||
      !auxiliary.trace_spaces.quality.valid || !(config.eps_ratio > 0.0) ||
      !(config.kappa > 0.0) || config.leaf_size < 8 ||
      config.expansion_order < 2 || config.expansion_order > 5 ||
      !(config.opening_theta > 0.0) ||
      config.near_integration != "analytic_singular" ||
      config.quad_buffer != "off") {
    state->status = "invalid_config_or_geometry";
    state->prepare_seconds = elapsed_seconds(begin, Clock::now());
    return PreparedCalderonOperator(std::move(state));
  }

  MassCompatibleTransferConfig transfer_config = config.transfer;
  transfer_config.geometry.normal_matching = GeometryNormalMatching::OwnerSdfAware;
  transfer_config.geometry.require_area_weighted_normal_alignment = true;
  const auto transfer_begin = Clock::now();
  state->transfer = build_mass_compatible_primary_auxiliary_trace_transfer(
      primary_panels, auxiliary, transfer_config);
  state->transfer_seconds = elapsed_seconds(transfer_begin, Clock::now());
  state->geometry_status = state->transfer.geometry_status;
  if (state->transfer.status != "pass" || state->geometry_status != "pass") {
    state->status = "transfer_" + state->transfer.status;
    state->prepare_seconds = elapsed_seconds(begin, Clock::now());
    return PreparedCalderonOperator(std::move(state));
  }

  std::vector<std::map<std::size_t, double>> pairing_rows;
  build_dual_patch_panels(auxiliary.mesh, state->auxiliary_panels,
                          state->patch_metadata, pairing_rows);
  state->auxiliary_lumped_mass =
      auxiliary.trace_spaces.p1_vertex_lumped_mass;
  if (state->auxiliary_panels.empty() ||
      state->patch_metadata.size() != state->auxiliary_panels.size() ||
      pairing_rows.size() != auxiliary.mesh.vertices.size() ||
      state->auxiliary_lumped_mass.size() != auxiliary.mesh.vertices.size()) {
    state->status = "auxiliary_discretization_gap";
    state->prepare_seconds = elapsed_seconds(begin, Clock::now());
    return PreparedCalderonOperator(std::move(state));
  }
  state->pairing_row_ptr.resize(pairing_rows.size() + 1, 0);
  state->pairing_diag_inverse.resize(pairing_rows.size(), 0.0);
  for (std::size_t row = 0; row < pairing_rows.size(); ++row) {
    state->pairing_row_ptr[row] = state->pairing_values.size();
    for (const auto &[column, value] : pairing_rows[row]) {
      state->pairing_col_index.push_back(column);
      state->pairing_values.push_back(value);
      if (column == row) {
        state->pairing_diag_inverse[row] = 1.0 / value;
      }
    }
    if (!(state->pairing_diag_inverse[row] > 0.0) ||
        !std::isfinite(state->pairing_diag_inverse[row])) {
      state->status = "pairing_diagonal_gap";
      state->prepare_seconds = elapsed_seconds(begin, Clock::now());
      return PreparedCalderonOperator(std::move(state));
    }
  }
  state->pairing_row_ptr.back() = state->pairing_values.size();
  const auto fmm_begin = Clock::now();
  state->fmm_context = ensure_jcp_fmm_m2l_context(
      state->auxiliary_panels, config.eps_ratio, config.kappa, config.leaf_size,
      config.opening_theta, config.expansion_order, executor,
      &state->fmm_context);
  state->fmm_setup_seconds = elapsed_seconds(fmm_begin, Clock::now());
  if (!state->fmm_context) {
    state->status = "fmm_context_gap";
    state->prepare_seconds = elapsed_seconds(begin, Clock::now());
    return PreparedCalderonOperator(std::move(state));
  }
  state->prepare_build_count = 1;
  state->context_build_count = 1;
  const std::size_t stencil_bytes = [&]() {
    std::size_t bytes = 0;
    for (const auto &row : state->transfer.primary_to_auxiliary_p1) {
      bytes += row.size() * sizeof(TraceTransferWeight);
    }
    for (const auto &row : state->transfer.auxiliary_to_primary_balanced_adjoint) {
      bytes += row.size() * sizeof(TraceTransferWeight);
    }
    return bytes;
  }();
  state->estimated_bytes = state->auxiliary_panels.size() * sizeof(Panel) +
                           state->patch_metadata.size() * sizeof(PatchMeta) +
                           state->auxiliary_lumped_mass.size() * sizeof(double) +
                           state->pairing_row_ptr.size() * sizeof(std::size_t) +
                           state->pairing_col_index.size() * sizeof(std::size_t) +
                           state->pairing_values.size() * sizeof(double) +
                           state->pairing_diag_inverse.size() * sizeof(double) +
                           stencil_bytes +
                           4 * state->auxiliary_panels.size() * sizeof(double) +
                           2 * state->auxiliary_lumped_mass.size() * sizeof(double);
  state->status = "prepared";
  state->prepare_seconds = elapsed_seconds(begin, Clock::now());
  return PreparedCalderonOperator(std::move(state));
}

} // namespace TitanPBE::Tabi::calderon
