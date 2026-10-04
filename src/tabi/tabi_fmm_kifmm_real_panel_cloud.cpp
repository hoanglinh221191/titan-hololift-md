#include "tabi_fmm_kifmm_real_panel_cloud.h"

#include "tabi_near.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace titan::tabi::fmm_kifmm {
namespace {

using TitanPBE::Tabi::JcpNearPairKernel;
using TitanPBE::Tabi::Panel;

struct RealNode {
  std::vector<std::size_t> indices{};
  Vec3 center{};
  double half_width{0.0};
  double radius{0.0};
  std::size_t node_id{0};
  int level{0};
  int left{-1};
  int right{-1};
};

struct RealTree {
  std::vector<RealNode> nodes{};
  std::vector<int> leaves{};
};

struct RealWorkPair {
  int target_node{-1};
  int source_node{-1};
  bool far{false};
};

struct ProductionNearStats {
  std::size_t valid_pairs{0};
  std::size_t invalid_pairs{0};
};

double coord(Vec3 p, int axis) noexcept {
  if (axis == 0) {
    return p.x;
  }
  if (axis == 1) {
    return p.y;
  }
  return p.z;
}

Vec3 position_of(const Panel &panel) noexcept {
  return {panel.x, panel.y, panel.z};
}

Vec3 normal_of(const Panel &panel) noexcept {
  return {panel.nx, panel.ny, panel.nz};
}

std::array<Vec3, 3> vertices_of(const Panel &panel) noexcept {
  return {{{panel.v0x, panel.v0y, panel.v0z},
           {panel.v1x, panel.v1y, panel.v1z},
           {panel.v2x, panel.v2y, panel.v2z}}};
}

Vec3 subtract(Vec3 a, Vec3 b) noexcept {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

double dot(Vec3 a, Vec3 b) noexcept {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

double norm(Vec3 v) noexcept {
  return std::sqrt(std::max(0.0, dot(v, v)));
}

bool finite_vec(Vec3 v) noexcept {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

bool valid_panel(const Panel &panel, bool require_unit_normals) noexcept {
  const Vec3 position = position_of(panel);
  const Vec3 normal = normal_of(panel);
  const auto vertices = vertices_of(panel);
  const double normal_norm = norm(normal);
  if (!finite_vec(position) || !finite_vec(normal) ||
      !finite_vec(vertices[0]) || !finite_vec(vertices[1]) ||
      !finite_vec(vertices[2]) || normal_norm <= 1.0e-12 ||
      !std::isfinite(panel.area) || panel.area <= 0.0 ||
      !panel.has_triangle_geometry) {
    return false;
  }
  if (require_unit_normals && std::abs(normal_norm - 1.0) > 1.0e-6) {
    return false;
  }
  const Vec3 e01 = subtract(vertices[1], vertices[0]);
  const Vec3 e02 = subtract(vertices[2], vertices[0]);
  const Vec3 cross{e01.y * e02.z - e01.z * e02.y,
                   e01.z * e02.x - e01.x * e02.z,
                   e01.x * e02.y - e01.y * e02.x};
  const double triangle_area = 0.5 * norm(cross);
  return triangle_area > 1.0e-14 && std::isfinite(triangle_area);
}

bool valid_unknowns(std::span<const double> unknowns) noexcept {
  for (const double value : unknowns) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

RealNode make_node(const std::vector<std::size_t> &indices,
                   std::span<const Panel> panels, int level,
                   std::size_t node_id) {
  Vec3 min_p{std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity()};
  Vec3 max_p{-std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity()};
  for (const std::size_t index : indices) {
    const Vec3 p = position_of(panels[index]);
    min_p.x = std::min(min_p.x, p.x);
    min_p.y = std::min(min_p.y, p.y);
    min_p.z = std::min(min_p.z, p.z);
    max_p.x = std::max(max_p.x, p.x);
    max_p.y = std::max(max_p.y, p.y);
    max_p.z = std::max(max_p.z, p.z);
  }
  RealNode node;
  node.indices = indices;
  node.center = {(min_p.x + max_p.x) * 0.5, (min_p.y + max_p.y) * 0.5,
                 (min_p.z + max_p.z) * 0.5};
  const double hx = (max_p.x - min_p.x) * 0.5;
  const double hy = (max_p.y - min_p.y) * 0.5;
  const double hz = (max_p.z - min_p.z) * 0.5;
  node.half_width = std::max({hx, hy, hz, 1.0e-9}) * (1.0 + 1.0e-12);
  node.radius = std::sqrt(3.0) * node.half_width;
  node.node_id = node_id;
  node.level = level;
  return node;
}

int build_node(RealTree &tree, std::vector<std::size_t> indices,
               std::span<const Panel> panels, int level, int leaf_size) {
  const int node_index = static_cast<int>(tree.nodes.size());
  RealNode node =
      make_node(indices, panels, level, static_cast<std::size_t>(node_index));
  tree.nodes.push_back(std::move(node));
  if (static_cast<int>(indices.size()) <= leaf_size) {
    tree.leaves.push_back(node_index);
    return node_index;
  }

  Vec3 min_p{std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity()};
  Vec3 max_p{-std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity()};
  for (const std::size_t index : tree.nodes[node_index].indices) {
    const Vec3 p = position_of(panels[index]);
    min_p.x = std::min(min_p.x, p.x);
    min_p.y = std::min(min_p.y, p.y);
    min_p.z = std::min(min_p.z, p.z);
    max_p.x = std::max(max_p.x, p.x);
    max_p.y = std::max(max_p.y, p.y);
    max_p.z = std::max(max_p.z, p.z);
  }
  const double extent[3] = {max_p.x - min_p.x, max_p.y - min_p.y,
                            max_p.z - min_p.z};
  int axis = 0;
  if (extent[1] > extent[axis]) {
    axis = 1;
  }
  if (extent[2] > extent[axis]) {
    axis = 2;
  }
  std::sort(indices.begin(), indices.end(), [&](std::size_t a,
                                                std::size_t b) {
    const double ca = coord(position_of(panels[a]), axis);
    const double cb = coord(position_of(panels[b]), axis);
    if (ca != cb) {
      return ca < cb;
    }
    return a < b;
  });
  const auto mid = indices.begin() +
                   static_cast<std::ptrdiff_t>(indices.size() / 2);
  std::vector<std::size_t> left(indices.begin(), mid);
  std::vector<std::size_t> right(mid, indices.end());
  tree.nodes[node_index].left =
      build_node(tree, std::move(left), panels, level + 1, leaf_size);
  tree.nodes[node_index].right =
      build_node(tree, std::move(right), panels, level + 1, leaf_size);
  return node_index;
}

RealTree build_tree(std::span<const Panel> panels, int leaf_size) {
  RealTree tree;
  std::vector<std::size_t> indices(panels.size());
  std::iota(indices.begin(), indices.end(), std::size_t{0});
  build_node(tree, std::move(indices), panels, 0, leaf_size);
  return tree;
}

bool accept_far(const RealNode &target, const RealNode &source,
                double theta) noexcept {
  const double center_distance = norm(subtract(target.center, source.center));
  const double separation =
      center_distance - target.radius - source.radius;
  if (!(separation > 0.0)) {
    return false;
  }
  return source.radius / separation <= theta;
}

bool target_node_outside_source_check_box(const RealNode &target,
                                          const RealNode &source,
                                          std::span<const Panel> panels,
                                          double check_scale) noexcept {
  const double h = source.half_width * check_scale;
  if (!(h > 0.0) || !std::isfinite(h)) {
    return false;
  }
  for (const std::size_t index : target.indices) {
    for (const Vec3 p : vertices_of(panels[index])) {
      if (std::abs(p.x - source.center.x) <= h &&
          std::abs(p.y - source.center.y) <= h &&
          std::abs(p.z - source.center.z) <= h) {
        return false;
      }
    }
  }
  return true;
}

bool share_vertex(const Panel &a, const Panel &b) noexcept {
  const std::size_t invalid = static_cast<std::size_t>(-1);
  const std::array<std::size_t, 3> av{a.v0_index, a.v1_index, a.v2_index};
  const std::array<std::size_t, 3> bv{b.v0_index, b.v1_index, b.v2_index};
  for (const std::size_t x : av) {
    if (x == invalid) {
      continue;
    }
    for (const std::size_t y : bv) {
      if (x == y) {
        return true;
      }
    }
  }
  return false;
}

bool node_pairs_share_vertex(const RealNode &target, const RealNode &source,
                             std::span<const Panel> panels) noexcept {
  for (const std::size_t ti : target.indices) {
    for (const std::size_t si : source.indices) {
      if (share_vertex(panels[ti], panels[si])) {
        return true;
      }
    }
  }
  return false;
}

void collect_work(const RealTree &target_tree, const RealTree &source_tree,
                  int target_node, int source_node, double theta,
                  double check_scale, std::span<const Panel> panels,
                  std::vector<RealWorkPair> &pairs) {
  const RealNode &target =
      target_tree.nodes[static_cast<std::size_t>(target_node)];
  const RealNode &source =
      source_tree.nodes[static_cast<std::size_t>(source_node)];
  if (accept_far(target, source, theta) &&
      target_node_outside_source_check_box(target, source, panels,
                                           check_scale) &&
      !node_pairs_share_vertex(target, source, panels)) {
    pairs.push_back({target_node, source_node, true});
    return;
  }
  if (source.left < 0 || source.right < 0) {
    pairs.push_back({target_node, source_node, false});
    return;
  }
  collect_work(target_tree, source_tree, target_node, source.left, theta,
               check_scale, panels, pairs);
  collect_work(target_tree, source_tree, target_node, source.right, theta,
               check_scale, panels, pairs);
}

std::vector<RealWorkPair> build_worklist(const RealTree &target_tree,
                                         const RealTree &source_tree,
                                         const KifmmLiteConfig &config,
                                         std::span<const Panel> panels) {
  std::vector<RealWorkPair> pairs;
  for (const int target_leaf : target_tree.leaves) {
    collect_work(target_tree, source_tree, target_leaf, 0,
                 config.opening_theta, config.check_scale, panels, pairs);
  }
  std::sort(pairs.begin(), pairs.end(),
            [](const RealWorkPair &a, const RealWorkPair &b) {
              if (a.target_node != b.target_node) {
                return a.target_node < b.target_node;
              }
              if (a.source_node != b.source_node) {
                return a.source_node < b.source_node;
              }
              return a.far > b.far;
            });
  return pairs;
}

void hash_u64(std::uint64_t &hash, std::uint64_t value) noexcept {
  constexpr std::uint64_t prime = 1099511628211ull;
  hash ^= value;
  hash *= prime;
}

void hash_double(std::uint64_t &hash, double value) noexcept {
  hash_u64(hash, std::bit_cast<std::uint64_t>(value));
}

void hash_string(std::uint64_t &hash, const std::string &value) noexcept {
  hash_u64(hash, static_cast<std::uint64_t>(value.size()));
  for (const char c : value) {
    hash_u64(hash, static_cast<unsigned char>(c));
  }
}

void hash_vec3(std::uint64_t &hash, Vec3 value) noexcept {
  hash_double(hash, value.x);
  hash_double(hash, value.y);
  hash_double(hash, value.z);
}

std::uint64_t worklist_hash(const std::vector<RealWorkPair> &pairs,
                            const RealTree &target_tree,
                            const RealTree &source_tree,
                            const KifmmLiteConfig &config) noexcept {
  std::uint64_t hash = 1469598103934665603ull;
  hash_u64(hash, static_cast<std::uint64_t>(config.order));
  hash_u64(hash, static_cast<std::uint64_t>(config.leaf_size));
  hash_double(hash, config.opening_theta);
  hash_double(hash, config.equiv_scale);
  hash_double(hash, config.check_scale);
  hash_double(hash, config.kappa);
  hash_double(hash, config.eps_ratio);
  hash_double(hash, config.max_condition_estimate);
  hash_double(hash, config.max_equivalent_relative_residual);
  hash_u64(hash, config.require_far_work ? 1ull : 0ull);
  hash_u64(hash, config.require_unit_normals ? 1ull : 0ull);
  hash_string(hash, config.solver);
  hash_string(hash, config.condition_policy);
  hash_u64(hash, static_cast<std::uint64_t>(pairs.size()));
  for (const RealWorkPair &pair : pairs) {
    const RealNode &target =
        target_tree.nodes[static_cast<std::size_t>(pair.target_node)];
    const RealNode &source =
        source_tree.nodes[static_cast<std::size_t>(pair.source_node)];
    hash_u64(hash, static_cast<std::uint64_t>(pair.target_node));
    hash_u64(hash, static_cast<std::uint64_t>(pair.source_node));
    hash_u64(hash, pair.far ? 1ull : 0ull);
    hash_u64(hash, static_cast<std::uint64_t>(target.indices.size()));
    hash_u64(hash, static_cast<std::uint64_t>(source.indices.size()));
    hash_vec3(hash, target.center);
    hash_double(hash, target.half_width);
    hash_vec3(hash, source.center);
    hash_double(hash, source.half_width);
  }
  return hash;
}

std::uint64_t policy_hash(const KifmmRealPanelCloudConfig &config,
                          std::uint64_t worklist_hash_value) noexcept {
  std::uint64_t hash = 1469598103934665603ull;
  hash_u64(hash, worklist_hash_value);
  hash_u64(hash, static_cast<std::uint64_t>(config.lite.order));
  hash_u64(hash, static_cast<std::uint64_t>(config.lite.leaf_size));
  hash_double(hash, config.lite.opening_theta);
  hash_double(hash, config.lite.equiv_scale);
  hash_double(hash, config.lite.check_scale);
  hash_double(hash, config.lite.kappa);
  hash_double(hash, config.lite.eps_ratio);
  hash_double(hash, config.lite.max_condition_estimate);
  hash_double(hash, config.lite.max_equivalent_relative_residual);
  hash_double(hash, config.lite.max_aggregate_relative_rms);
  hash_double(hash, config.lite.max_channel_relative_rms);
  hash_double(hash, config.lite.min_far_panel_fraction);
  hash_u64(hash, config.lite.require_far_work ? 1ull : 0ull);
  hash_u64(hash, config.lite.require_unit_normals ? 1ull : 0ull);
  hash_string(hash, config.lite.solver);
  hash_string(hash, config.lite.condition_policy);
  hash_string(hash, config.near_integration);
  hash_string(hash, config.quad_buffer);
  return hash;
}

KifmmPanelSource source_from_panel(const Panel &panel, double phi,
                                   double phin) noexcept {
  return {position_of(panel), normal_of(panel), panel.area, phi, phin};
}

KifmmLiteTargetPanel target_from_panel(const Panel &panel) noexcept {
  return {position_of(panel), normal_of(panel)};
}

std::vector<KifmmPanelSource>
gather_sources(std::span<const KifmmPanelSource> sources,
               const RealNode &node) {
  std::vector<KifmmPanelSource> subset;
  subset.reserve(node.indices.size());
  for (const std::size_t index : node.indices) {
    subset.push_back(sources[index]);
  }
  return subset;
}

KifmmBox node_box(const RealNode &node) noexcept {
  return {node.center, node.half_width, node.level, node.node_id};
}

void add_row(KifmmJcpRow &dst, const KifmmJcpRow &src) noexcept {
  dst.phi_row += src.phi_row;
  dst.phin_row += src.phin_row;
}

void add_channel_row(KifmmJcpChannelRow &dst,
                     const KifmmJcpChannelRow &src) noexcept {
  dst.k1 += src.k1;
  dst.k2 += src.k2;
  dst.k3 += src.k3;
  dst.k4 += src.k4;
}

KifmmJcpChannelRow production_channel_row(
    std::span<const Panel> panels, std::span<const double> unknowns,
    std::size_t target_index, const RealNode &source_node, double eps_ratio,
    double kappa, const std::string &near_integration,
    const std::string &quad_buffer, ProductionNearStats *stats) {
  KifmmJcpChannelRow row;
  const std::size_t n = panels.size();
  for (const std::size_t source_index : source_node.indices) {
    const JcpNearPairKernel kernel =
        TitanPBE::Tabi::evaluate_jcp_near_pair_kernel(
            panels[target_index], target_index, panels[source_index],
            source_index, eps_ratio, kappa, true, near_integration,
            quad_buffer, nullptr);
    if (!kernel.valid) {
      if (stats != nullptr) {
        ++stats->invalid_pairs;
      }
      continue;
    }
    if (stats != nullptr) {
      ++stats->valid_pairs;
    }
    const double phi = unknowns[source_index];
    const double phin = unknowns[n + source_index];
    row.k1 += kernel.k1 * phin;
    row.k2 += kernel.k2 * phi;
    row.k3 += kernel.k3 * phin;
    row.k4 += kernel.k4 * phi;
  }
  return row;
}

KifmmJcpRow row_from_channel(const KifmmJcpChannelRow &channel) noexcept {
  return {channel.k1 + channel.k2, channel.k3 + channel.k4};
}

double rel_rms_rows(std::span<const KifmmJcpRow> approx,
                    std::span<const KifmmJcpRow> direct) noexcept {
  double err_sq = 0.0;
  double ref_sq = 0.0;
  for (std::size_t i = 0; i < approx.size(); ++i) {
    const double d_phi = approx[i].phi_row - direct[i].phi_row;
    const double d_phin = approx[i].phin_row - direct[i].phin_row;
    err_sq += d_phi * d_phi + d_phin * d_phin;
    ref_sq += direct[i].phi_row * direct[i].phi_row +
              direct[i].phin_row * direct[i].phin_row;
  }
  return std::sqrt(err_sq / std::max(ref_sq, 1.0e-300));
}

enum class Channel { K1, K2, K3, K4 };

double rel_rms_channel(std::span<const KifmmJcpChannelRow> approx,
                       std::span<const KifmmJcpChannelRow> direct,
                       Channel channel) noexcept {
  double err_sq = 0.0;
  double ref_sq = 0.0;
  for (std::size_t i = 0; i < approx.size(); ++i) {
    double a = 0.0;
    double d = 0.0;
    if (channel == Channel::K1) {
      a = approx[i].k1;
      d = direct[i].k1;
    } else if (channel == Channel::K2) {
      a = approx[i].k2;
      d = direct[i].k2;
    } else if (channel == Channel::K3) {
      a = approx[i].k3;
      d = direct[i].k3;
    } else {
      a = approx[i].k4;
      d = direct[i].k4;
    }
    const double delta = a - d;
    err_sq += delta * delta;
    ref_sq += d * d;
  }
  return std::sqrt(err_sq / std::max(ref_sq, 1.0e-300));
}

bool finite_rows(std::span<const KifmmJcpRow> rows) noexcept {
  for (const KifmmJcpRow &row : rows) {
    if (!std::isfinite(row.phi_row) || !std::isfinite(row.phin_row)) {
      return false;
    }
  }
  return true;
}

bool finite_channel_rows(std::span<const KifmmJcpChannelRow> rows) noexcept {
  for (const KifmmJcpChannelRow &row : rows) {
    if (!std::isfinite(row.k1) || !std::isfinite(row.k2) ||
        !std::isfinite(row.k3) || !std::isfinite(row.k4)) {
      return false;
    }
  }
  return true;
}

KifmmRealPanelCloudResult failed(std::string status) {
  KifmmRealPanelCloudResult result;
  result.status = std::move(status);
  result.lite.status = result.status;
  result.finite = false;
  result.lite.finite = false;
  return result;
}

} // namespace

KifmmRealPanelCloudResult run_kifmm_lite_real_panel_matvec(
    std::span<const Panel> panels, std::span<const double> unknowns,
    const KifmmRealPanelCloudConfig &config) {
  if (panels.empty()) {
    return failed("empty_cloud");
  }
  if (unknowns.size() != 2 * panels.size() || !valid_unknowns(unknowns)) {
    return failed("invalid_unknowns");
  }
  if (config.near_integration != "analytic_singular" ||
      config.quad_buffer != "off") {
    return failed("unsupported_near_policy");
  }
  if (config.lite.order < 2 || (config.lite.order % 2) != 0 ||
      config.lite.leaf_size <= 0 ||
      !std::isfinite(config.lite.opening_theta) ||
      config.lite.opening_theta < 0.0 ||
      !std::isfinite(config.lite.equiv_scale) ||
      !std::isfinite(config.lite.check_scale) ||
      config.lite.equiv_scale <= 1.0 ||
      config.lite.equiv_scale >= config.lite.check_scale ||
      !std::isfinite(config.lite.kappa) || config.lite.kappa < 0.0 ||
      !std::isfinite(config.lite.eps_ratio) || config.lite.eps_ratio <= 0.0 ||
      !std::isfinite(config.lite.max_condition_estimate) ||
      config.lite.max_condition_estimate <= 0.0 ||
      !std::isfinite(config.lite.max_equivalent_relative_residual) ||
      config.lite.max_equivalent_relative_residual <= 0.0 ||
      !std::isfinite(config.lite.max_aggregate_relative_rms) ||
      config.lite.max_aggregate_relative_rms <= 0.0 ||
      !std::isfinite(config.lite.max_channel_relative_rms) ||
      config.lite.max_channel_relative_rms <= 0.0 ||
      !(config.lite.condition_policy == "svd-full-debug" ||
        config.lite.condition_policy == "svd-values" ||
        config.lite.condition_policy == "qr-rank-only" ||
        config.lite.condition_policy == "off")) {
    return failed("invalid_config");
  }
  for (const Panel &panel : panels) {
    if (!valid_panel(panel, config.lite.require_unit_normals)) {
      return failed("invalid_panel");
    }
  }

  KifmmRealPanelCloudResult result;
  result.lite.source_panels = panels.size();
  result.lite.target_panels = panels.size();
  result.lite.rows.assign(panels.size(), {});
  result.lite.channel_rows.assign(panels.size(), {});
  result.lite.direct_rows.assign(panels.size(), {});
  result.lite.direct_channel_rows.assign(panels.size(), {});

  std::vector<KifmmPanelSource> sources;
  sources.reserve(panels.size());
  std::vector<KifmmLiteTargetPanel> targets;
  targets.reserve(panels.size());
  for (std::size_t i = 0; i < panels.size(); ++i) {
    sources.push_back(source_from_panel(panels[i], unknowns[i],
                                        unknowns[panels.size() + i]));
    targets.push_back(target_from_panel(panels[i]));
  }

  const double inv_eps_ratio = 1.0 / config.lite.eps_ratio;
  RealNode all_sources;
  all_sources.indices.resize(panels.size());
  std::iota(all_sources.indices.begin(), all_sources.indices.end(),
            std::size_t{0});
  ProductionNearStats direct_reference_stats;
  for (std::size_t i = 0; i < panels.size(); ++i) {
    result.lite.direct_channel_rows[i] = production_channel_row(
        panels, unknowns, i, all_sources, config.lite.eps_ratio,
        config.lite.kappa, config.near_integration, config.quad_buffer,
        &direct_reference_stats);
    result.lite.direct_rows[i] =
        row_from_channel(result.lite.direct_channel_rows[i]);
  }
  result.production_direct_reference_valid_pairs +=
      direct_reference_stats.valid_pairs;
  result.production_invalid_pairs += direct_reference_stats.invalid_pairs;
  if (result.production_invalid_pairs != 0) {
    result.status = "invalid_production_near_kernel";
    result.lite.status = result.status;
    result.finite = false;
    result.lite.finite = false;
    return result;
  }
  if (!finite_rows(result.lite.direct_rows) ||
      !finite_channel_rows(result.lite.direct_channel_rows)) {
    result.status = "nonfinite_direct_reference";
    result.lite.status = result.status;
    return result;
  }

  const RealTree source_tree = build_tree(panels, config.lite.leaf_size);
  const RealTree target_tree = build_tree(panels, config.lite.leaf_size);
  result.lite.source_nodes = source_tree.nodes.size();
  result.lite.target_nodes = target_tree.nodes.size();
  result.lite.source_leaves = source_tree.leaves.size();
  result.lite.target_leaves = target_tree.leaves.size();

  const std::vector<RealWorkPair> pairs =
      build_worklist(target_tree, source_tree, config.lite, panels);
  result.lite.worklist_hash =
      worklist_hash(pairs, target_tree, source_tree, config.lite);
  result.policy_hash = policy_hash(config, result.lite.worklist_hash);

  const KifmmSurfaceRule surface_rule =
      make_cube_surface_rule(config.lite.order, config.lite.equiv_scale,
                             config.lite.check_scale);
  std::unordered_map<int, KifmmJcpChannels> channel_cache;

  for (const RealWorkPair &pair : pairs) {
    const RealNode &target_node =
        target_tree.nodes[static_cast<std::size_t>(pair.target_node)];
    const RealNode &source_node =
        source_tree.nodes[static_cast<std::size_t>(pair.source_node)];
    const std::size_t panel_pairs =
        target_node.indices.size() * source_node.indices.size();
    if (pair.far) {
      ++result.lite.accepted_far_cell_pairs;
      result.lite.accepted_far_panel_pairs += panel_pairs;
      auto channel_it = channel_cache.find(pair.source_node);
      if (channel_it == channel_cache.end()) {
        const KifmmBox box = node_box(source_node);
        const std::vector<KifmmPanelSource> subset =
            gather_sources(sources, source_node);
        const KifmmJcpChannels channels = assemble_kifmm_jcp_channels(
            surface_rule, box, subset, config.lite.kappa,
            config.lite.eps_ratio, inv_eps_ratio, config.lite.solver, nullptr,
            config.lite.condition_policy);
        result.lite.equivalent_system_builds +=
            channels.equivalent_system_builds;
        result.lite.equivalent_rhs_solves += channels.equivalent_rhs_solves;
        channel_it = channel_cache.emplace(pair.source_node, channels).first;
      }
      const KifmmJcpChannels &channels = channel_it->second;
      result.lite.max_condition_estimate = std::max(
          result.lite.max_condition_estimate,
          channels.max_condition_estimate);
      result.lite.max_equivalent_relative_residual =
          std::max(result.lite.max_equivalent_relative_residual,
                   channels.max_relative_residual);
      if (!channels.finite || channels.status != "pass") {
        result.status = "far_equivalent_failure";
        result.lite.status = result.status;
        return result;
      }
      if (channels.max_condition_estimate >
          config.lite.max_condition_estimate) {
        result.status = "ill_conditioned_equivalent_system";
        result.lite.status = result.status;
        return result;
      }
      if (channels.max_relative_residual >
          config.lite.max_equivalent_relative_residual) {
        result.status = "equivalent_residual_too_large";
        result.lite.status = result.status;
        return result;
      }
      const KifmmBox box = node_box(source_node);
      for (const std::size_t target_index : target_node.indices) {
        const KifmmJcpChannelRow channel = assemble_kifmm_jcp_channel_row(
            surface_rule, box, channels, config.lite.kappa,
            config.lite.eps_ratio, inv_eps_ratio, targets[target_index].position,
            targets[target_index].normal);
        add_channel_row(result.lite.channel_rows[target_index], channel);
        add_row(result.lite.rows[target_index], row_from_channel(channel));
      }
    } else {
      ++result.lite.direct_fallback_cell_pairs;
      result.lite.direct_fallback_panel_pairs += panel_pairs;
      ProductionNearStats fallback_stats;
      for (const std::size_t target_index : target_node.indices) {
        const KifmmJcpChannelRow channel = production_channel_row(
            panels, unknowns, target_index, source_node, config.lite.eps_ratio,
            config.lite.kappa, config.near_integration, config.quad_buffer,
            &fallback_stats);
        add_channel_row(result.lite.channel_rows[target_index], channel);
        add_row(result.lite.rows[target_index], row_from_channel(channel));
      }
      result.production_fallback_valid_pairs += fallback_stats.valid_pairs;
      result.production_invalid_pairs += fallback_stats.invalid_pairs;
      if (result.production_invalid_pairs != 0) {
        result.status = "invalid_production_near_kernel";
        result.lite.status = result.status;
        result.finite = false;
        result.lite.finite = false;
        return result;
      }
    }
  }

  result.lite.aggregate_relative_rms =
      rel_rms_rows(result.lite.rows, result.lite.direct_rows);
  result.lite.k1_relative_rms =
      rel_rms_channel(result.lite.channel_rows, result.lite.direct_channel_rows,
                      Channel::K1);
  result.lite.k2_relative_rms =
      rel_rms_channel(result.lite.channel_rows, result.lite.direct_channel_rows,
                      Channel::K2);
  result.lite.k3_relative_rms =
      rel_rms_channel(result.lite.channel_rows, result.lite.direct_channel_rows,
                      Channel::K3);
  result.lite.k4_relative_rms =
      rel_rms_channel(result.lite.channel_rows, result.lite.direct_channel_rows,
                      Channel::K4);

  if (!finite_rows(result.lite.rows) ||
      !finite_channel_rows(result.lite.channel_rows)) {
    result.status = "nonfinite_kifmm_rows";
    result.lite.status = result.status;
    return result;
  }
  if (result.lite.aggregate_relative_rms >
      config.lite.max_aggregate_relative_rms) {
    result.status = "aggregate_rms_gate_failed";
    result.lite.status = result.status;
    result.finite = false;
    result.lite.finite = false;
    return result;
  }
  const double max_channel_rms =
      std::max({result.lite.k1_relative_rms, result.lite.k2_relative_rms,
                result.lite.k3_relative_rms, result.lite.k4_relative_rms});
  if (max_channel_rms > config.lite.max_channel_relative_rms) {
    result.status = "channel_rms_gate_failed";
    result.lite.status = result.status;
    result.finite = false;
    result.lite.finite = false;
    return result;
  }
  if (config.lite.require_far_work &&
      (result.lite.accepted_far_cell_pairs == 0 ||
       result.lite.accepted_far_panel_pairs == 0)) {
    result.status = "no_far_work";
    result.lite.status = result.status;
    return result;
  }

  result.lite.finite = std::isfinite(result.lite.aggregate_relative_rms) &&
                       std::isfinite(result.lite.k1_relative_rms) &&
                       std::isfinite(result.lite.k2_relative_rms) &&
                       std::isfinite(result.lite.k3_relative_rms) &&
                       std::isfinite(result.lite.k4_relative_rms);
  result.finite = result.lite.finite;
  result.used_nonproduction_fallback = result.nonproduction_fallback_pairs > 0;
  result.status = result.finite ? "pass" : "non_finite";
  if (result.finite && result.lite.accepted_far_cell_pairs == 0 &&
      result.lite.accepted_far_panel_pairs == 0 &&
      !config.lite.require_far_work) {
    result.status = "direct_only_reference";
  }
  result.lite.status = result.status;
  return result;
}

} // namespace titan::tabi::fmm_kifmm
