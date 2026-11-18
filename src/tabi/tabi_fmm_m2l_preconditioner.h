#ifndef TITAN_TABI_FMM_M2L_PRECONDITIONER_H
#define TITAN_TABI_FMM_M2L_PRECONDITIONER_H

#include "tabi_fmm_m2l_prototype.h"
#include "tabi_preconditioner.h"
#include "tabi_types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

class TitanExecutor;

namespace TitanPBE::Tabi {

struct JcpSparseIluPreconditioner {
  struct Block2 {
    double a00{0.0};
    double a01{0.0};
    double a10{0.0};
    double a11{0.0};
  };

  std::size_t panel_count{0};
  bool coupled{true};
  std::vector<double> inv_diag;
  std::vector<std::size_t> row_ptr;
  std::vector<std::size_t> col_index;
  std::vector<std::size_t> lower_end;
  std::vector<std::size_t> upper_begin;
  std::vector<Block2> values;
  std::vector<Block2> diag_inverse;
  std::size_t nnz_blocks{0};
  double avg_degree{0.0};
  std::size_t max_degree{0};
  std::size_t valid_blocks{0};
  std::size_t valid_panel_count{0};
  std::size_t ill_conditioned_blocks{0};
  // Compatibility counter for a rejected quality gate. For sparse ILU this
  // is a global probe/factorization outcome, not a count of bad panel blocks;
  // inspect quality_failure_stage for the precise cause.
  std::size_t quality_rejected_blocks{0};
  double pivot_ratio_min{0.0};
  double factor_growth_max{0.0};
  std::int64_t pivot_ratio_min_row{-1};
  std::int64_t pivot_ratio_min_column{-1};
  std::int64_t factor_growth_row{-1};
  std::int64_t factor_growth_column{-1};
  double quality_probe_max_gain{0.0};
  double quality_probe_rel_residual{0.0};
  // Failure metadata is diagnostic only: ILU remains fail-closed whenever a
  // gate below rejects the factorization. Row/column are panel indices and
  // use -1 when the corresponding stage has no single offending block.
  std::string quality_failure_stage{"off"};
  std::int64_t quality_failure_row{-1};
  std::int64_t quality_failure_column{-1};
  std::int64_t quality_failure_probe{-1};
  double quality_failure_probe_gain{0.0};
  double quality_failure_probe_rel_residual{0.0};
  std::string quality_status{"off"};
  std::string regularization_mode{"off"};
  double regularization_max_relative{0.0};
  double regularization_selected_relative{0.0};
  double regularization_selected_absolute_max{0.0};
  std::size_t regularization_attempts{0};
  std::size_t regularization_rejected_attempts{0};
  std::size_t regularization_shifted_rows{0};
  std::string regularization_unshifted_failure_stage{"off"};
  TitanExecutor *executor{nullptr};
  std::vector<std::vector<std::size_t>> forward_levels;
  std::vector<std::vector<std::size_t>> backward_levels;
  std::size_t parallel_level_max_width{0};
  double graph_setup_ms{0.0};
  bool graph_cache_reused{false};
  double near_assembly_setup_ms{0.0};
  double factor_setup_ms{0.0};

  // GMRES invokes this serial preconditioner repeatedly; retain its four
  // triangular-solve work vectors without exposing mutable state to callers.
  mutable std::mutex workspace_mutex;
  mutable std::vector<double> workspace;

  void apply(const std::vector<double> &in, std::vector<double> &out) const;
  [[nodiscard]] std::size_t estimated_bytes() const noexcept;
};

std::size_t estimate_fmm_m2l_block_jacobi_bytes(
    const titan::tabi::fmm_m2l::FmmM2lAnalyticContext &context,
    std::size_t diagonal_size, bool coupled, std::size_t block_panel_cap);

std::shared_ptr<JcpBlockPreconditioner>
build_fmm_m2l_block_jacobi_preconditioner(
    const std::vector<Panel> &panels,
    const titan::tabi::fmm_m2l::FmmM2lAnalyticContext &context, bool coupled,
    double eps_ratio, double kappa, double lhs_phi, double lhs_phin,
    const std::vector<double> &diagonal, std::size_t block_panel_cap,
    TitanExecutor &executor, const std::string &near_integration = "centroid",
    const std::string &quad_buffer = "off");

std::size_t estimate_fmm_m2l_overlap_schwarz_bytes(
    const titan::tabi::fmm_m2l::FmmM2lAnalyticContext &context,
    std::size_t diagonal_size, bool coupled, std::size_t owned_panel_cap,
    std::size_t overlap_panel_cap,
    const std::string &graph_source = "fmm-near");

std::shared_ptr<JcpBlockPreconditioner>
build_fmm_m2l_overlap_schwarz_preconditioner(
    const std::vector<Panel> &panels,
    const titan::tabi::fmm_m2l::FmmM2lAnalyticContext &context, bool coupled,
    double eps_ratio, double kappa, double lhs_phi, double lhs_phin,
    const std::vector<double> &diagonal, std::size_t owned_panel_cap,
    std::size_t overlap_panel_cap, TitanExecutor &executor,
    const std::string &near_integration = "centroid",
    const std::string &quad_buffer = "off",
    const std::string &graph_source = "fmm-near");

std::size_t estimate_fmm_m2l_nearfield_ilu0_bytes(
    const titan::tabi::fmm_m2l::FmmM2lAnalyticContext &context,
    std::size_t diagonal_size, bool coupled, std::size_t neighbor_cap,
    bool include_regularization_retry = false,
    const std::string &graph_source = "fmm-near");

struct FmmIluBudgetSelection {
  std::size_t requested_neighbor_cap{32};
  std::size_t effective_neighbor_cap{32};
  std::size_t estimated_bytes{0};
  bool fits_budget{false};
  double graph_build_ms{0.0};
  std::shared_ptr<const std::vector<std::vector<std::size_t>>>
      normalized_graph;
};

FmmIluBudgetSelection select_fmm_m2l_nearfield_ilu0_budget(
    const titan::tabi::fmm_m2l::FmmM2lAnalyticContext &context,
    std::size_t diagonal_size, bool coupled, std::size_t neighbor_cap,
    std::size_t memory_budget_bytes,
    bool include_regularization_retry = false,
    const std::string &graph_source = "fmm-near",
    TitanExecutor *executor = nullptr);

std::shared_ptr<JcpSparseIluPreconditioner>
build_fmm_m2l_nearfield_ilu0_preconditioner(
    const std::vector<Panel> &panels,
    const titan::tabi::fmm_m2l::FmmM2lAnalyticContext &context, bool coupled,
    double eps_ratio, double kappa, double lhs_phi, double lhs_phin,
    const std::vector<double> &diagonal, std::size_t neighbor_cap,
    double drop_tol, TitanExecutor &executor,
    const std::string &near_integration = "centroid",
    const std::string &quad_buffer = "off",
    const std::string &regularization_mode = "adaptive",
    double regularization_max_relative = 1.0e-2,
    const std::string &graph_source = "fmm-near",
    JcpCoarseGalerkinPreconditioner *coarse_near_capture = nullptr,
    std::shared_ptr<const std::vector<std::vector<std::size_t>>>
        normalized_graph = {});

std::size_t estimate_jcp_sparse_ilu0_graph_bytes(
    const std::vector<std::vector<std::size_t>> &graph,
    std::size_t diagonal_size, bool include_regularization_retry = false);

std::shared_ptr<JcpSparseIluPreconditioner>
build_jcp_sparse_ilu0_preconditioner_from_graph(
    const std::vector<Panel> &panels,
    const std::vector<std::vector<std::size_t>> &graph, bool coupled,
    double eps_ratio, double kappa, double lhs_phi, double lhs_phin,
    const std::vector<double> &diagonal, double drop_tol,
    const std::string &near_integration = "centroid",
    const std::string &quad_buffer = "off",
    const std::string &regularization_mode = "adaptive",
    double regularization_max_relative = 1.0e-2);

} // namespace TitanPBE::Tabi

#endif // TITAN_TABI_FMM_M2L_PRECONDITIONER_H
