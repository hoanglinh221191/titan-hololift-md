#ifndef TITAN_TABI_FMM_KIFMM_REAL_PANEL_WORKLIST_H
#define TITAN_TABI_FMM_KIFMM_REAL_PANEL_WORKLIST_H

#include "tabi_fmm_kifmm_hierarchy_real_panel.h"
#include "tabi_fmm_kifmm_real_panel_geometry.h"

#include <span>

namespace titan::tabi::fmm_kifmm {

int mac_bin(double ratio) noexcept;

double target_vertex_check_excess(const TitanPBE::Tabi::Panel &target,
                                  const KifmmHierarchyNode &source,
                                  double check_scale) noexcept;

double source_boundary_margin(const KifmmPanelSource &source,
                              const KifmmHierarchyNode &node) noexcept;

double target_local_check_excess(const KifmmHierarchyNode &source,
                                 const KifmmHierarchyNode &target,
                                 const KifmmHierarchyPolicy &policy) noexcept;

KifmmHierarchyWorklist build_real_panel_guarded_worklist(
    const KifmmHierarchyTree &tree, const KifmmHierarchyPolicy &policy,
    std::span<const KifmmRealPanelNodeEnvelope> source_envelopes,
    std::span<const KifmmRealPanelNodeEnvelope> target_envelopes,
    double guard_margin_factor, bool local_check_guard,
    KifmmHierarchyRealPanelResult &result);

void update_leaf_panel_telemetry(
    const KifmmHierarchyTree &tree,
    std::span<const KifmmRealPanelNodeEnvelope> source_envelopes,
    std::span<const KifmmRealPanelNodeEnvelope> target_envelopes,
    KifmmHierarchyRealPanelResult &result);

void update_l2l_containment_telemetry(const KifmmHierarchyTree &tree,
                                      const KifmmHierarchyPolicy &policy,
                                      KifmmHierarchyRealPanelResult &result);

} // namespace titan::tabi::fmm_kifmm

#endif // TITAN_TABI_FMM_KIFMM_REAL_PANEL_WORKLIST_H
