#include "tabi_fmm_kifmm_lite.h"

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

struct LiteNode {
  std::vector<std::size_t> indices{};
  Vec3 center{};
  double half_width{0.0};
  double radius{0.0};
  std::size_t node_id{0};
  int level{0};
  int left{-1};
  int right{-1};
  std::array<int, 8> children{{-1, -1, -1, -1, -1, -1, -1, -1}};
};

struct LiteTree {
  std::vector<LiteNode> nodes{};
  std::vector<int> leaves{};
};

struct WorkPair {
  int target_node{-1};
  int source_node{-1};
  bool far{false};
};

struct BoxBounds {
  Vec3 min{};
  Vec3 max{};
  Vec3 center{};
  double half_width{0.0};
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

Vec3 subtract(Vec3 a, Vec3 b) noexcept {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

double dot(Vec3 a, Vec3 b) noexcept {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

double norm(Vec3 a) noexcept { return std::sqrt(dot(a, a)); }

bool finite_vec(Vec3 v) noexcept {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

double normal_norm(Vec3 v) noexcept { return norm(v); }

bool valid_normal(Vec3 normal, bool require_unit) noexcept {
  const double n = normal_norm(normal);
  if (!(n > 1.0e-12) || !std::isfinite(n)) {
    return false;
  }
  return !require_unit || std::abs(n - 1.0) <= 1.0e-6;
}

bool valid_source(const KifmmPanelSource &source, bool require_unit) noexcept {
  return finite_vec(source.position) && finite_vec(source.normal) &&
         std::isfinite(source.area) && source.area > 0.0 &&
         std::isfinite(source.phi) && std::isfinite(source.phin) &&
         valid_normal(source.normal, require_unit);
}

bool valid_target(const KifmmLiteTargetPanel &target,
                  bool require_unit) noexcept {
  return finite_vec(target.position) && finite_vec(target.normal) &&
         valid_normal(target.normal, require_unit);
}

bool valid_config(const KifmmLiteConfig &config) noexcept {
  return config.order >= 2 && (config.order % 2) == 0 &&
         config.leaf_size > 0 && std::isfinite(config.opening_theta) &&
         config.opening_theta >= 0.0 && std::isfinite(config.equiv_scale) &&
         std::isfinite(config.check_scale) && config.equiv_scale > 1.0 &&
         config.equiv_scale < config.check_scale && std::isfinite(config.kappa) &&
         config.kappa >= 0.0 && std::isfinite(config.eps_ratio) &&
         config.eps_ratio > 0.0 &&
         std::isfinite(config.max_condition_estimate) &&
         config.max_condition_estimate > 0.0 &&
         std::isfinite(config.max_equivalent_relative_residual) &&
         config.max_equivalent_relative_residual > 0.0 &&
         std::isfinite(config.max_aggregate_relative_rms) &&
         config.max_aggregate_relative_rms > 0.0 &&
         std::isfinite(config.max_channel_relative_rms) &&
         config.max_channel_relative_rms > 0.0 &&
         std::isfinite(config.min_far_panel_fraction) &&
         config.min_far_panel_fraction >= 0.0 &&
         config.min_far_panel_fraction <= 1.0 &&
         (config.condition_policy == "svd-full-debug" ||
          config.condition_policy == "svd-values" ||
          config.condition_policy == "qr-rank-only" ||
          config.condition_policy == "off") &&
         (config.source_box_policy == "tight-bbox" ||
          config.source_box_policy == "level-cube");
}

template <class PositionFn>
BoxBounds tight_bounds(const std::vector<std::size_t> &indices,
                       const PositionFn &position) {
  Vec3 min_p{std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity()};
  Vec3 max_p{-std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity()};
  for (const std::size_t index : indices) {
    const Vec3 p = position(index);
    min_p.x = std::min(min_p.x, p.x);
    min_p.y = std::min(min_p.y, p.y);
    min_p.z = std::min(min_p.z, p.z);
    max_p.x = std::max(max_p.x, p.x);
    max_p.y = std::max(max_p.y, p.y);
    max_p.z = std::max(max_p.z, p.z);
  }
  const double hx = (max_p.x - min_p.x) * 0.5;
  const double hy = (max_p.y - min_p.y) * 0.5;
  const double hz = (max_p.z - min_p.z) * 0.5;
  BoxBounds bounds;
  bounds.min = min_p;
  bounds.max = max_p;
  bounds.center = {(min_p.x + max_p.x) * 0.5, (min_p.y + max_p.y) * 0.5,
                   (min_p.z + max_p.z) * 0.5};
  bounds.half_width = std::max({hx, hy, hz, 1.0e-9}) * (1.0 + 1.0e-12);
  return bounds;
}

template <class PositionFn>
LiteNode make_node(const std::vector<std::size_t> &indices,
                   const PositionFn &position, int level, std::size_t node_id) {
  const BoxBounds bounds = tight_bounds(indices, position);
  LiteNode node;
  node.indices = indices;
  node.center = bounds.center;
  node.half_width = bounds.half_width;
  node.radius = std::sqrt(3.0) * node.half_width;
  node.node_id = node_id;
  node.level = level;
  return node;
}

double enclosing_level_half_width(double root_half_width,
                                  double tight_half_width) noexcept {
  double h = std::max(root_half_width, tight_half_width);
  while (h * 0.5 >= tight_half_width && h > 1.0e-12) {
    h *= 0.5;
  }
  return std::max(h, 1.0e-12);
}

template <class PositionFn>
LiteNode make_level_stable_node(const std::vector<std::size_t> &indices,
                                const PositionFn &position,
                                double root_half_width, int level,
                                std::size_t node_id) {
  const BoxBounds bounds = tight_bounds(indices, position);
  LiteNode node;
  node.indices = indices;
  node.center = bounds.center;
  node.half_width =
      enclosing_level_half_width(root_half_width, bounds.half_width);
  node.radius = std::sqrt(3.0) * node.half_width;
  node.node_id = node_id;
  node.level = level;
  return node;
}

template <class PositionFn>
int build_node(LiteTree &tree, std::vector<std::size_t> indices,
               const PositionFn &position, int level, int leaf_size) {
  const int node_index = static_cast<int>(tree.nodes.size());
  LiteNode node =
      make_node(indices, position, level, static_cast<std::size_t>(node_index));
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
    const Vec3 p = position(index);
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
  std::sort(indices.begin(), indices.end(), [&](std::size_t a, std::size_t b) {
    const double ca = coord(position(a), axis);
    const double cb = coord(position(b), axis);
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
      build_node(tree, std::move(left), position, level + 1, leaf_size);
  tree.nodes[node_index].right =
      build_node(tree, std::move(right), position, level + 1, leaf_size);
  return node_index;
}

template <class PositionFn>
int build_level_stable_binary_node(LiteTree &tree,
                                   std::vector<std::size_t> indices,
                                   const PositionFn &position,
                                   double root_half_width, int level,
                                   int leaf_size) {
  const int node_index = static_cast<int>(tree.nodes.size());
  LiteNode node = make_level_stable_node(
      indices, position, root_half_width, level,
      static_cast<std::size_t>(node_index));
  tree.nodes.push_back(std::move(node));
  if (static_cast<int>(indices.size()) <= leaf_size) {
    tree.leaves.push_back(node_index);
    return node_index;
  }

  const BoxBounds bounds =
      tight_bounds(tree.nodes[static_cast<std::size_t>(node_index)].indices,
                   position);
  const double extent[3] = {bounds.max.x - bounds.min.x,
                            bounds.max.y - bounds.min.y,
                            bounds.max.z - bounds.min.z};
  int axis = 0;
  if (extent[1] > extent[axis]) {
    axis = 1;
  }
  if (extent[2] > extent[axis]) {
    axis = 2;
  }
  std::sort(indices.begin(), indices.end(), [&](std::size_t a, std::size_t b) {
    const double ca = coord(position(a), axis);
    const double cb = coord(position(b), axis);
    if (ca != cb) {
      return ca < cb;
    }
    return a < b;
  });
  const auto mid = indices.begin() +
                   static_cast<std::ptrdiff_t>(indices.size() / 2);
  std::vector<std::size_t> left(indices.begin(), mid);
  std::vector<std::size_t> right(mid, indices.end());
  tree.nodes[static_cast<std::size_t>(node_index)].left =
      build_level_stable_binary_node(tree, std::move(left), position,
                                     root_half_width, level + 1, leaf_size);
  tree.nodes[static_cast<std::size_t>(node_index)].right =
      build_level_stable_binary_node(tree, std::move(right), position,
                                     root_half_width, level + 1, leaf_size);
  return node_index;
}

template <class PositionFn>
LiteTree build_tree(std::size_t count, const PositionFn &position,
                    int leaf_size, const KifmmLiteConfig &config) {
  LiteTree tree;
  if (count == 0) {
    return tree;
  }
  std::vector<std::size_t> indices(count);
  std::iota(indices.begin(), indices.end(), std::size_t{0});
  if (config.source_box_policy == "level-cube") {
    const BoxBounds root = tight_bounds(indices, position);
    build_level_stable_binary_node(tree, std::move(indices), position,
                                   root.half_width, 0, leaf_size);
  } else {
    build_node(tree, std::move(indices), position, 0, leaf_size);
  }
  return tree;
}

bool accept_far(const LiteNode &target, const LiteNode &source,
                double theta) noexcept {
  const double center_distance = norm(subtract(target.center, source.center));
  const double separation =
      center_distance - target.radius - source.radius;
  if (!(separation > 0.0)) {
    return false;
  }
  return source.radius / separation <= theta;
}

bool target_node_outside_source_check_box(
    const LiteNode &target, const LiteNode &source,
    std::span<const KifmmLiteTargetPanel> targets,
    double check_scale) noexcept {
  const double h = source.half_width * check_scale;
  if (!(h > 0.0) || !std::isfinite(h)) {
    return false;
  }
  for (const std::size_t index : target.indices) {
    const Vec3 p = targets[index].position;
    if (std::abs(p.x - source.center.x) <= h &&
        std::abs(p.y - source.center.y) <= h &&
        std::abs(p.z - source.center.z) <= h) {
      return false;
    }
  }
  return true;
}

void collect_work(const LiteTree &target_tree, const LiteTree &source_tree,
                  int target_node, int source_node, double theta,
                  double check_scale,
                  std::span<const KifmmLiteTargetPanel> targets,
                  std::vector<WorkPair> &pairs) {
  const LiteNode &target = target_tree.nodes[static_cast<std::size_t>(target_node)];
  const LiteNode &source = source_tree.nodes[static_cast<std::size_t>(source_node)];
  if (accept_far(target, source, theta) &&
      target_node_outside_source_check_box(target, source, targets,
                                           check_scale)) {
    pairs.push_back({target_node, source_node, true});
    return;
  }
  bool has_octree_children = false;
  for (const int child : source.children) {
    if (child < 0) {
      continue;
    }
    has_octree_children = true;
    collect_work(target_tree, source_tree, target_node, child, theta,
                 check_scale, targets, pairs);
  }
  if (has_octree_children) {
    return;
  }
  if (source.left < 0 || source.right < 0) {
    pairs.push_back({target_node, source_node, false});
    return;
  }
  collect_work(target_tree, source_tree, target_node, source.left, theta,
               check_scale, targets, pairs);
  collect_work(target_tree, source_tree, target_node, source.right, theta,
               check_scale, targets, pairs);
}

std::vector<WorkPair> build_worklist(const LiteTree &target_tree,
                                     const LiteTree &source_tree,
                                     double theta, double check_scale,
                                     std::span<const KifmmLiteTargetPanel> targets) {
  std::vector<WorkPair> pairs;
  for (const int target_leaf : target_tree.leaves) {
    collect_work(target_tree, source_tree, target_leaf, 0, theta, check_scale,
                 targets, pairs);
  }
  std::sort(pairs.begin(), pairs.end(), [](const WorkPair &a, const WorkPair &b) {
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

std::uint64_t worklist_hash(const std::vector<WorkPair> &pairs,
                            const LiteTree &target_tree,
                            const LiteTree &source_tree,
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
  hash_double(hash, config.max_aggregate_relative_rms);
  hash_double(hash, config.max_channel_relative_rms);
  hash_double(hash, config.min_far_panel_fraction);
  hash_u64(hash, config.require_far_work ? 1ull : 0ull);
  hash_u64(hash, config.require_unit_normals ? 1ull : 0ull);
  hash_string(hash, config.solver);
  hash_string(hash, config.condition_policy);
  hash_string(hash, config.source_box_policy);
  hash_u64(hash, static_cast<std::uint64_t>(pairs.size()));
  for (const WorkPair &pair : pairs) {
    const LiteNode &target =
        target_tree.nodes[static_cast<std::size_t>(pair.target_node)];
    const LiteNode &source =
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

std::vector<KifmmPanelSource>
gather_sources(std::span<const KifmmPanelSource> sources,
               const LiteNode &node) {
  std::vector<KifmmPanelSource> subset;
  subset.reserve(node.indices.size());
  for (const std::size_t index : node.indices) {
    subset.push_back(sources[index]);
  }
  return subset;
}

KifmmBox node_box(const LiteNode &node) noexcept {
  return {node.center, node.half_width, node.level, node.node_id};
}

std::size_t tree_levels(const LiteTree &tree) {
  int max_level = -1;
  for (const LiteNode &node : tree.nodes) {
    max_level = std::max(max_level, node.level);
  }
  return max_level < 0 ? 0 : static_cast<std::size_t>(max_level + 1);
}

std::size_t unique_node_half_widths(const LiteTree &tree) {
  std::vector<std::uint64_t> values;
  values.reserve(tree.nodes.size());
  for (const LiteNode &node : tree.nodes) {
    values.push_back(std::bit_cast<std::uint64_t>(node.half_width));
  }
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values.size();
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

enum class KifmmJcpChannel { K1, K2, K3, K4 };

double rel_rms_channel(std::span<const KifmmJcpChannelRow> approx,
                       std::span<const KifmmJcpChannelRow> direct,
                       KifmmJcpChannel channel) noexcept {
  double err_sq = 0.0;
  double ref_sq = 0.0;
  for (std::size_t i = 0; i < approx.size(); ++i) {
    double a = 0.0;
    double d = 0.0;
    if (channel == KifmmJcpChannel::K1) {
      a = approx[i].k1;
      d = direct[i].k1;
    } else if (channel == KifmmJcpChannel::K2) {
      a = approx[i].k2;
      d = direct[i].k2;
    } else if (channel == KifmmJcpChannel::K3) {
      a = approx[i].k3;
      d = direct[i].k3;
    } else if (channel == KifmmJcpChannel::K4) {
      a = approx[i].k4;
      d = direct[i].k4;
    } else {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const double diff = a - d;
    err_sq += diff * diff;
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

KifmmLiteResult failed(std::string status) {
  KifmmLiteResult result;
  result.status = std::move(status);
  result.finite = false;
  return result;
}

} // namespace

KifmmLiteResult
run_kifmm_lite_matvec(std::span<const KifmmPanelSource> sources,
                      std::span<const KifmmLiteTargetPanel> targets,
                      const KifmmLiteConfig &config) {
  if (!valid_config(config)) {
    return failed("invalid_config");
  }
  if (sources.empty() || targets.empty()) {
    return failed("empty_cloud");
  }
  for (const KifmmPanelSource &source : sources) {
    if (!valid_source(source, config.require_unit_normals)) {
      return failed("invalid_source_panel");
    }
  }
  for (const KifmmLiteTargetPanel &target : targets) {
    if (!valid_target(target, config.require_unit_normals)) {
      return failed("invalid_target_panel");
    }
  }

  KifmmLiteResult result;
  result.source_panels = sources.size();
  result.target_panels = targets.size();
  result.min_far_panel_fraction = config.min_far_panel_fraction;
  result.rows.assign(targets.size(), {});
  result.channel_rows.assign(targets.size(), {});
  result.direct_rows.assign(targets.size(), {});
  result.direct_channel_rows.assign(targets.size(), {});

  const double inv_eps_ratio = 1.0 / config.eps_ratio;
  for (std::size_t i = 0; i < targets.size(); ++i) {
    result.direct_rows[i] = assemble_direct_jcp_row(
        sources, config.kappa, config.eps_ratio, inv_eps_ratio,
        targets[i].position, targets[i].normal);
    result.direct_channel_rows[i] = assemble_direct_jcp_channel_row(
        sources, config.kappa, config.eps_ratio, inv_eps_ratio,
        targets[i].position, targets[i].normal);
  }
  if (!finite_rows(result.direct_rows) ||
      !finite_channel_rows(result.direct_channel_rows)) {
    result.status = "nonfinite_direct_reference";
    result.finite = false;
    return result;
  }

  const LiteTree source_tree = build_tree(
      sources.size(),
      [&](std::size_t index) noexcept { return sources[index].position; },
      config.leaf_size, config);
  KifmmLiteConfig target_tree_config = config;
  target_tree_config.source_box_policy = "tight-bbox";
  const LiteTree target_tree = build_tree(
      targets.size(),
      [&](std::size_t index) noexcept { return targets[index].position; },
      config.leaf_size, target_tree_config);
  result.source_nodes = source_tree.nodes.size();
  result.target_nodes = target_tree.nodes.size();
  result.source_leaves = source_tree.leaves.size();
  result.target_leaves = target_tree.leaves.size();
  result.source_tree_levels = tree_levels(source_tree);
  result.unique_source_node_half_widths =
      unique_node_half_widths(source_tree);
  result.source_root_half_width =
      source_tree.nodes.empty() ? 0.0 : source_tree.nodes.front().half_width;
  result.level_cube_geometry = config.source_box_policy == "level-cube";

  const std::vector<WorkPair> pairs =
      build_worklist(target_tree, source_tree, config.opening_theta,
                     config.check_scale, targets);
  result.worklist_hash = worklist_hash(pairs, target_tree, source_tree, config);

  const KifmmSurfaceRule surface_rule =
      make_cube_surface_rule(config.order, config.equiv_scale,
                             config.check_scale);
  std::unordered_map<int, KifmmJcpChannels> channel_cache;

  for (const WorkPair &pair : pairs) {
    const LiteNode &target_node =
        target_tree.nodes[static_cast<std::size_t>(pair.target_node)];
    const LiteNode &source_node =
        source_tree.nodes[static_cast<std::size_t>(pair.source_node)];
    const std::vector<KifmmPanelSource> subset =
        gather_sources(sources, source_node);
    const std::size_t panel_pairs =
        target_node.indices.size() * source_node.indices.size();
    if (pair.far) {
      ++result.accepted_far_cell_pairs;
      result.accepted_far_panel_pairs += panel_pairs;
      auto channel_it = channel_cache.find(pair.source_node);
      if (channel_it == channel_cache.end()) {
        const KifmmBox box = node_box(source_node);
        const KifmmJcpChannels channels = assemble_kifmm_jcp_channels(
            surface_rule, box, subset, config.kappa, config.eps_ratio,
            inv_eps_ratio, config.solver, nullptr, config.condition_policy);
        result.equivalent_system_builds += channels.equivalent_system_builds;
        result.equivalent_rhs_solves += channels.equivalent_rhs_solves;
        channel_it = channel_cache.emplace(pair.source_node, channels).first;
      }
      const KifmmJcpChannels &channels = channel_it->second;
      result.max_condition_estimate =
          std::max(result.max_condition_estimate,
                   channels.max_condition_estimate);
      result.max_equivalent_relative_residual =
          std::max(result.max_equivalent_relative_residual,
                   channels.max_relative_residual);
      if (!channels.finite || channels.status != "pass") {
        result.status = "far_equivalent_failure";
        result.finite = false;
        return result;
      }
      if (channels.max_condition_estimate > config.max_condition_estimate) {
        result.status = "ill_conditioned_equivalent_system";
        result.finite = false;
        return result;
      }
      if (channels.max_relative_residual >
          config.max_equivalent_relative_residual) {
        result.status = "equivalent_residual_too_large";
        result.finite = false;
        return result;
      }
      const KifmmBox box = node_box(source_node);
      for (const std::size_t target_index : target_node.indices) {
        add_row(result.rows[target_index],
                assemble_kifmm_jcp_row(surface_rule, box, channels,
                                       config.kappa, config.eps_ratio,
                                       inv_eps_ratio,
                                       targets[target_index].position,
                                       targets[target_index].normal));
        add_channel_row(
            result.channel_rows[target_index],
            assemble_kifmm_jcp_channel_row(
                surface_rule, box, channels, config.kappa, config.eps_ratio,
                inv_eps_ratio, targets[target_index].position,
                targets[target_index].normal));
      }
    } else {
      ++result.direct_fallback_cell_pairs;
      result.direct_fallback_panel_pairs += panel_pairs;
      for (const std::size_t target_index : target_node.indices) {
        add_row(result.rows[target_index],
                assemble_direct_jcp_row(subset, config.kappa, config.eps_ratio,
                                        inv_eps_ratio,
                                        targets[target_index].position,
                                        targets[target_index].normal));
        add_channel_row(
            result.channel_rows[target_index],
            assemble_direct_jcp_channel_row(
                subset, config.kappa, config.eps_ratio, inv_eps_ratio,
                targets[target_index].position, targets[target_index].normal));
      }
    }
  }

  result.aggregate_relative_rms =
      rel_rms_rows(result.rows, result.direct_rows);
  result.k1_relative_rms =
      rel_rms_channel(result.channel_rows, result.direct_channel_rows,
                      KifmmJcpChannel::K1);
  result.k2_relative_rms =
      rel_rms_channel(result.channel_rows, result.direct_channel_rows,
                      KifmmJcpChannel::K2);
  result.k3_relative_rms =
      rel_rms_channel(result.channel_rows, result.direct_channel_rows,
                      KifmmJcpChannel::K3);
  result.k4_relative_rms =
      rel_rms_channel(result.channel_rows, result.direct_channel_rows,
                      KifmmJcpChannel::K4);

  if (!finite_rows(result.rows) || !finite_channel_rows(result.channel_rows)) {
    result.status = "nonfinite_kifmm_rows";
    result.finite = false;
    return result;
  }
  if (result.aggregate_relative_rms > config.max_aggregate_relative_rms) {
    result.status = "aggregate_rms_gate_failed";
    result.finite = false;
    return result;
  }
  const double max_channel_rms =
      std::max({result.k1_relative_rms, result.k2_relative_rms,
                result.k3_relative_rms, result.k4_relative_rms});
  if (max_channel_rms > config.max_channel_relative_rms) {
    result.status = "channel_rms_gate_failed";
    result.finite = false;
    return result;
  }
  if (config.require_far_work &&
      (result.accepted_far_cell_pairs == 0 ||
       result.accepted_far_panel_pairs == 0)) {
    result.status = "no_far_work";
    result.finite = false;
    return result;
  }
  if (config.require_far_work && config.min_far_panel_fraction > 0.0) {
    const std::size_t total_panel_pairs =
        result.accepted_far_panel_pairs + result.direct_fallback_panel_pairs;
    const double far_fraction =
        total_panel_pairs == 0
            ? 0.0
            : static_cast<double>(result.accepted_far_panel_pairs) /
                  static_cast<double>(total_panel_pairs);
    if (far_fraction < config.min_far_panel_fraction) {
      result.status = "fallback_dominated";
      result.finite = false;
      return result;
    }
  }

  result.finite = std::isfinite(result.aggregate_relative_rms) &&
                  std::isfinite(result.k1_relative_rms) &&
                  std::isfinite(result.k2_relative_rms) &&
                  std::isfinite(result.k3_relative_rms) &&
                  std::isfinite(result.k4_relative_rms);
  result.status = result.finite ? "pass" : "non_finite";
  if (result.finite && result.accepted_far_cell_pairs == 0 &&
      result.accepted_far_panel_pairs == 0 && !config.require_far_work) {
    result.status = "direct_only_reference";
  }
  return result;
}

} // namespace titan::tabi::fmm_kifmm
