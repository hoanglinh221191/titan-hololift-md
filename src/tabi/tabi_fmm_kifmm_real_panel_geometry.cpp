#include "tabi_fmm_kifmm_real_panel_geometry.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace titan::tabi::fmm_kifmm {
namespace {

using TitanPBE::Tabi::Panel;

std::uint64_t mix_hash(std::uint64_t seed, std::uint64_t value) noexcept {
  value += 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31;
  return seed ^ value;
}

std::uint64_t hash_double(std::uint64_t seed, double value) noexcept {
  return mix_hash(seed, std::bit_cast<std::uint64_t>(value));
}

bool finite_vec(Vec3 v) noexcept {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

} // namespace

Vec3 position_of(const Panel &panel) noexcept { return {panel.x, panel.y, panel.z}; }

Vec3 normal_of(const Panel &panel) noexcept {
  return {panel.nx, panel.ny, panel.nz};
}

Vec3 vertex0(const Panel &panel) noexcept {
  return {panel.v0x, panel.v0y, panel.v0z};
}

Vec3 vertex1(const Panel &panel) noexcept {
  return {panel.v1x, panel.v1y, panel.v1z};
}

Vec3 vertex2(const Panel &panel) noexcept {
  return {panel.v2x, panel.v2y, panel.v2z};
}

Vec3 min_vec() noexcept {
  const double inf = std::numeric_limits<double>::infinity();
  return {inf, inf, inf};
}

Vec3 max_vec() noexcept {
  const double inf = std::numeric_limits<double>::infinity();
  return {-inf, -inf, -inf};
}

void include_point(Vec3 p, Vec3 &lo, Vec3 &hi) noexcept {
  lo.x = std::min(lo.x, p.x);
  lo.y = std::min(lo.y, p.y);
  lo.z = std::min(lo.z, p.z);
  hi.x = std::max(hi.x, p.x);
  hi.y = std::max(hi.y, p.y);
  hi.z = std::max(hi.z, p.z);
}

Vec3 sub(Vec3 a, Vec3 b) noexcept {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 cross(Vec3 a, Vec3 b) noexcept {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}

double dot(Vec3 a, Vec3 b) noexcept {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

double norm(Vec3 a) noexcept { return std::sqrt(std::max(0.0, dot(a, a))); }

double max_abs_component(Vec3 a) noexcept {
  return std::max({std::abs(a.x), std::abs(a.y), std::abs(a.z)});
}

double triangle_area(const Panel &panel) noexcept {
  return 0.5 * norm(cross(sub(vertex1(panel), vertex0(panel)),
                          sub(vertex2(panel), vertex0(panel))));
}

double distance(Vec3 a, Vec3 b) noexcept { return norm(sub(a, b)); }

double panel_radius(const Panel &panel) noexcept {
  const Vec3 c = position_of(panel);
  return std::max({distance(vertex0(panel), c), distance(vertex1(panel), c),
                   distance(vertex2(panel), c)});
}

double triangle_aspect(const Panel &panel) noexcept {
  const double e01 = distance(vertex0(panel), vertex1(panel));
  const double e12 = distance(vertex1(panel), vertex2(panel));
  const double e20 = distance(vertex2(panel), vertex0(panel));
  const double longest = std::max({e01, e12, e20});
  const double shortest = std::min({e01, e12, e20});
  return shortest > 0.0 ? longest / shortest
                        : std::numeric_limits<double>::infinity();
}

bool valid_panel(const Panel &panel) noexcept {
  if (!finite_vec(position_of(panel)) || !finite_vec(normal_of(panel)) ||
      !std::isfinite(panel.area) || !(panel.area > 0.0)) {
    return false;
  }
  const double n = norm(normal_of(panel));
  if (std::abs(n - 1.0) > 1.0e-6) {
    return false;
  }
  if (!panel.has_triangle_geometry || !finite_vec(vertex0(panel)) ||
      !finite_vec(vertex1(panel)) || !finite_vec(vertex2(panel))) {
    return false;
  }
  return triangle_area(panel) > 1.0e-14;
}

std::uint64_t hash_panels(std::span<const Panel> panels) noexcept {
  std::uint64_t h = 0xE255A11C0FFEE123ull;
  h = mix_hash(h, static_cast<std::uint64_t>(panels.size()));
  for (const Panel &panel : panels) {
    h = hash_double(h, panel.x);
    h = hash_double(h, panel.y);
    h = hash_double(h, panel.z);
    h = hash_double(h, panel.nx);
    h = hash_double(h, panel.ny);
    h = hash_double(h, panel.nz);
    h = hash_double(h, panel.area);
    h = hash_double(h, panel.v0x);
    h = hash_double(h, panel.v0y);
    h = hash_double(h, panel.v0z);
    h = hash_double(h, panel.v1x);
    h = hash_double(h, panel.v1y);
    h = hash_double(h, panel.v1z);
    h = hash_double(h, panel.v2x);
    h = hash_double(h, panel.v2y);
    h = hash_double(h, panel.v2z);
    h = mix_hash(h, panel.has_triangle_geometry ? 1ull : 0ull);
  }
  return h;
}

std::uint64_t hash_doubles(std::span<const double> values) noexcept {
  std::uint64_t h = 0xE255D0A8BADC0DEull;
  h = mix_hash(h, static_cast<std::uint64_t>(values.size()));
  for (double value : values) {
    h = hash_double(h, value);
  }
  return h;
}

std::uint64_t hash_indices(std::span<const std::size_t> values) noexcept {
  std::uint64_t h = 0xE255517E5EED123ull;
  h = mix_hash(h, static_cast<std::uint64_t>(values.size()));
  for (std::size_t value : values) {
    h = mix_hash(h, static_cast<std::uint64_t>(value));
  }
  return h;
}

KifmmPanelSource source_from_panel(const Panel &panel, double phi,
                                   double phin) noexcept {
  return {position_of(panel), normal_of(panel), panel.area, phi, phin};
}

std::vector<Vec3>
positions_from_sources(std::span<const KifmmPanelSource> sources) {
  std::vector<Vec3> positions;
  positions.reserve(sources.size());
  for (const KifmmPanelSource &source : sources) {
    positions.push_back(source.position);
  }
  return positions;
}

std::vector<double> panel_radii(std::span<const Panel> panels) {
  std::vector<double> radii;
  radii.reserve(panels.size());
  for (const Panel &panel : panels) {
    radii.push_back(panel_radius(panel));
  }
  return radii;
}

std::vector<double>
sampled_panel_radii(std::span<const Panel> panels,
                    std::span<const std::size_t> sampled_target_indices) {
  std::vector<double> radii;
  radii.reserve(sampled_target_indices.size());
  for (const std::size_t index : sampled_target_indices) {
    radii.push_back(panel_radius(panels[index]));
  }
  return radii;
}

KifmmRealPanelNodeEnvelope empty_envelope() noexcept {
  KifmmRealPanelNodeEnvelope envelope;
  envelope.centroid_min = min_vec();
  envelope.centroid_max = max_vec();
  envelope.vertex_min = min_vec();
  envelope.vertex_max = max_vec();
  return envelope;
}

void include_panel_in_envelope(KifmmRealPanelNodeEnvelope &envelope,
                               const Panel &panel) noexcept {
  include_point(position_of(panel), envelope.centroid_min,
                envelope.centroid_max);
  include_point(vertex0(panel), envelope.vertex_min, envelope.vertex_max);
  include_point(vertex1(panel), envelope.vertex_min, envelope.vertex_max);
  include_point(vertex2(panel), envelope.vertex_min, envelope.vertex_max);
  envelope.max_panel_diagonal =
      std::max(envelope.max_panel_diagonal, panel.diagonal);
  envelope.max_panel_radius =
      std::max(envelope.max_panel_radius, panel_radius(panel));
  envelope.max_triangle_aspect =
      std::max(envelope.max_triangle_aspect, triangle_aspect(panel));
  ++envelope.panel_count;
}

std::vector<KifmmRealPanelNodeEnvelope>
build_source_envelopes(const KifmmHierarchyTree &tree,
                       std::span<const Panel> panels) {
  std::vector<KifmmRealPanelNodeEnvelope> envelopes(tree.nodes.size());
  for (std::size_t node_id = 0; node_id < tree.nodes.size(); ++node_id) {
    envelopes[node_id] = empty_envelope();
    const KifmmHierarchyNode &node = tree.nodes[node_id];
    for (const int panel_int : node.source_indices) {
      include_panel_in_envelope(envelopes[node_id],
                                panels[static_cast<std::size_t>(panel_int)]);
    }
  }
  return envelopes;
}

std::vector<KifmmRealPanelNodeEnvelope> build_target_envelopes(
    const KifmmHierarchyTree &tree, std::span<const Panel> panels,
    std::span<const std::size_t> sampled_target_indices) {
  std::vector<KifmmRealPanelNodeEnvelope> envelopes(tree.nodes.size());
  for (std::size_t node_id = 0; node_id < tree.nodes.size(); ++node_id) {
    envelopes[node_id] = empty_envelope();
    const KifmmHierarchyNode &node = tree.nodes[node_id];
    for (const int target_local_int : node.target_indices) {
      const std::size_t target_local =
          static_cast<std::size_t>(target_local_int);
      include_panel_in_envelope(envelopes[node_id],
                                panels[sampled_target_indices[target_local]]);
    }
  }
  return envelopes;
}

} // namespace titan::tabi::fmm_kifmm
