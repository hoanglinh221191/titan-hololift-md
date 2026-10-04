#include "tabi_fmm_kifmm_channel_hybrid_operator.h"

namespace titan::tabi::fmm_kifmm {
namespace {

[[maybe_unused]] constexpr const char *kE46OperatorMarker =
    "run_dual_worklist_channel_hybrid_with_geometry";
[[maybe_unused]] constexpr const char *kE46FreezeMarker =
    "frozen_prepared_apply";
[[maybe_unused]] constexpr const char *kE46DoubleCountGuardMarker =
    "channel_hybrid_double_count_guard";

} // namespace

DualWorklistChannelHybridPreparedOperator
prepare_dual_worklist_channel_hybrid_operator_impl(
    std::span<const TitanPBE::Tabi::Panel> panels,
    std::span<const std::size_t> sampled_target_indices,
    const KifmmHierarchyJcpConfig &config);

KifmmHierarchyRealPanelResult
apply_prepared_dual_worklist_channel_hybrid_operator_impl(
    DualWorklistChannelHybridPreparedOperator &prepared,
    std::span<const double> unknowns, bool compute_sampled_direct_audit);

void assemble_dual_worklist_channel_hybrid_tabi_system_matvec_impl(
    const KifmmHierarchyRealPanelResult &applied,
    std::span<const double> unknowns, double lhs_phi, double lhs_phin,
    std::vector<double> &out);

DualWorklistChannelHybridPreparedOperator
prepare_dual_worklist_channel_hybrid_operator(
    std::span<const TitanPBE::Tabi::Panel> panels,
    std::span<const std::size_t> sampled_target_indices,
    const KifmmHierarchyJcpConfig &config) {
  return prepare_dual_worklist_channel_hybrid_operator_impl(
      panels, sampled_target_indices, config);
}

KifmmHierarchyRealPanelResult
apply_prepared_dual_worklist_channel_hybrid_operator(
    DualWorklistChannelHybridPreparedOperator &prepared,
    std::span<const double> unknowns, bool compute_sampled_direct_audit) {
  return apply_prepared_dual_worklist_channel_hybrid_operator_impl(
      prepared, unknowns, compute_sampled_direct_audit);
}

void assemble_dual_worklist_channel_hybrid_tabi_system_matvec(
    const KifmmHierarchyRealPanelResult &applied,
    std::span<const double> unknowns, double lhs_phi, double lhs_phin,
    std::vector<double> &out) {
  assemble_dual_worklist_channel_hybrid_tabi_system_matvec_impl(
      applied, unknowns, lhs_phi, lhs_phin, out);
}

} // namespace titan::tabi::fmm_kifmm
