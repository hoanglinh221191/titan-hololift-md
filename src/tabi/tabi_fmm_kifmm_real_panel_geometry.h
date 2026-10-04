#ifndef TITAN_TABI_FMM_KIFMM_REAL_PANEL_GEOMETRY_H
#define TITAN_TABI_FMM_KIFMM_REAL_PANEL_GEOMETRY_H

#include "tabi_fmm_kifmm_fields.h"
#include "tabi_fmm_kifmm_hierarchy.h"
#include "tabi_types.h"

#include <cstdint>
#include <span>
#include <vector>

namespace titan::tabi::fmm_kifmm {

struct KifmmRealPanelNodeEnvelope {
  Vec3 centroid_min{};
  Vec3 centroid_max{};
  Vec3 vertex_min{};
  Vec3 vertex_max{};
  double max_panel_diagonal{0.0};
  double max_panel_radius{0.0};
  double max_triangle_aspect{0.0};
  std::size_t panel_count{0};
};

Vec3 position_of(const TitanPBE::Tabi::Panel &panel) noexcept;
Vec3 normal_of(const TitanPBE::Tabi::Panel &panel) noexcept;
Vec3 vertex0(const TitanPBE::Tabi::Panel &panel) noexcept;
Vec3 vertex1(const TitanPBE::Tabi::Panel &panel) noexcept;
Vec3 vertex2(const TitanPBE::Tabi::Panel &panel) noexcept;
Vec3 min_vec() noexcept;
Vec3 max_vec() noexcept;
void include_point(Vec3 p, Vec3 &lo, Vec3 &hi) noexcept;
Vec3 sub(Vec3 a, Vec3 b) noexcept;
Vec3 cross(Vec3 a, Vec3 b) noexcept;
double dot(Vec3 a, Vec3 b) noexcept;
double norm(Vec3 a) noexcept;
double max_abs_component(Vec3 a) noexcept;
double triangle_area(const TitanPBE::Tabi::Panel &panel) noexcept;
double distance(Vec3 a, Vec3 b) noexcept;
double panel_radius(const TitanPBE::Tabi::Panel &panel) noexcept;
double triangle_aspect(const TitanPBE::Tabi::Panel &panel) noexcept;
bool valid_panel(const TitanPBE::Tabi::Panel &panel) noexcept;

std::uint64_t hash_panels(
    std::span<const TitanPBE::Tabi::Panel> panels) noexcept;
std::uint64_t hash_doubles(std::span<const double> values) noexcept;
std::uint64_t hash_indices(std::span<const std::size_t> values) noexcept;

KifmmPanelSource source_from_panel(const TitanPBE::Tabi::Panel &panel,
                                   double phi, double phin) noexcept;
std::vector<Vec3>
positions_from_sources(std::span<const KifmmPanelSource> sources);
std::vector<double> panel_radii(
    std::span<const TitanPBE::Tabi::Panel> panels);
std::vector<double> sampled_panel_radii(
    std::span<const TitanPBE::Tabi::Panel> panels,
    std::span<const std::size_t> sampled_target_indices);

KifmmRealPanelNodeEnvelope empty_envelope() noexcept;
void include_panel_in_envelope(KifmmRealPanelNodeEnvelope &envelope,
                               const TitanPBE::Tabi::Panel &panel) noexcept;
std::vector<KifmmRealPanelNodeEnvelope>
build_source_envelopes(const KifmmHierarchyTree &tree,
                       std::span<const TitanPBE::Tabi::Panel> panels);
std::vector<KifmmRealPanelNodeEnvelope> build_target_envelopes(
    const KifmmHierarchyTree &tree,
    std::span<const TitanPBE::Tabi::Panel> panels,
    std::span<const std::size_t> sampled_target_indices);

} // namespace titan::tabi::fmm_kifmm

#endif // TITAN_TABI_FMM_KIFMM_REAL_PANEL_GEOMETRY_H
