#ifndef TITAN_PBE_TABI_PRECONDITIONER_H
#define TITAN_PBE_TABI_PRECONDITIONER_H

#include "tabi_types.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

class TitanExecutor;

namespace TitanPBE::Tabi {

using JcpBlockRange = std::pair<std::size_t, std::size_t>;

struct JcpOwnedFactorPanelGroup {
  std::vector<std::size_t> owned_panels;
  std::vector<std::size_t> factor_panels;
};

struct JcpBlockPreconditioner {
  struct Block {
    std::vector<std::size_t> panels;
    std::size_t owned_count{0};
    std::size_t halo_count{0};
    std::vector<double> lu;
    std::vector<int> pivots;
    std::vector<double> scale;
    double pivot_ratio{1.0};
    double quality_gain{0.0};
    bool ill_conditioned{false};
    bool quality_rejected{false};
    bool valid{false};
  };

  std::size_t panel_count{0};
  bool coupled{false};
  std::vector<double> inv_diag;
  std::vector<Block> blocks;
  std::size_t valid_blocks{0};
  std::size_t valid_panel_count{0};
  std::size_t ill_conditioned_blocks{0};
  std::size_t quality_rejected_blocks{0};
  double quality_probe_max_gain{0.0};
  std::string quality_status{"off"};

  void apply_block_with_scratch(const Block &block,
                                const std::vector<double> &in,
                                std::vector<double> &out,
                                std::vector<double> &scratch) const;
  void apply_block(const Block &block, const std::vector<double> &in,
                   std::vector<double> &out) const;
  void apply(const std::vector<double> &in, std::vector<double> &out) const;
  void apply_parallel(const std::vector<double> &in, std::vector<double> &out,
                      TitanExecutor &executor) const;
  [[nodiscard]] std::size_t estimated_bytes() const noexcept;
};

struct JcpCoarseGalerkinPreconditioner {
  struct Aggregate {
    std::vector<std::size_t> panels;
    double area_sum{0.0};
  };

  std::size_t panel_count{0};
  bool coupled{false};
  std::vector<Aggregate> aggregates;
  // Fine-panel to aggregate lookup used while the sparse near operator is
  // assembled. Keeping this map lets the ILU builder accumulate A_near*P
  // without retaining a second copy of the unfactored sparse matrix.
  std::vector<std::size_t> panel_aggregate;
  std::vector<double> inv_diag;
  std::vector<double> lu;
  std::vector<int> pivots;
  // Row-major fine-grid images A * P(:,j). The balanced two-level path
  // retains these columns during Galerkin setup so that applying
  // (I - A P A_c^-1 R) does not require another full FMM matvec.
  std::vector<double> fine_operator_basis;
  std::shared_ptr<JcpBlockPreconditioner> block_base;
  double pivot_ratio{0.0};
  bool valid{false};
  std::size_t setup_matvec_calls{0};
  double rp_identity_error_max{0.0};
  double verify_rel_error_max{0.0};
  double moments_setup_ms{0.0};
  double far_setup_ms{0.0};
  double audit_setup_ms{0.0};

  // GMRES applies the balanced correction serially. Reuse the fine residual
  // and small coarse solve vector instead of allocating them every iteration.
  mutable std::mutex balanced_workspace_mutex;
  mutable std::vector<double> balanced_residual_workspace;
  mutable std::vector<double> balanced_coarse_workspace;

  [[nodiscard]] std::size_t dimension() const noexcept;
  void restrict_area_weighted(const std::vector<Panel> &panels,
                              const std::vector<double> &fine,
                              std::vector<double> &coarse) const;
  void prolongate(const std::vector<double> &coarse,
                  std::vector<double> &fine) const;
  [[nodiscard]] std::size_t
  estimated_bytes(bool include_operator_basis = false) const noexcept;
  void apply(const std::vector<Panel> &panels, const std::vector<double> &in,
             std::vector<double> &out, TitanExecutor &executor) const;
  void apply_balanced(
      const std::vector<Panel> &panels, const std::vector<double> &in,
      std::vector<double> &out,
      const std::function<void(const std::vector<double> &,
                               std::vector<double> &)> &base_apply) const;
};

std::vector<JcpBlockRange>
make_jcp_block_ranges(const TreeWorkspace &tree, std::size_t panel_count,
                      std::size_t block_panel_cap);

std::size_t estimate_jcp_block_preconditioner_bytes(
    const std::vector<JcpBlockRange> &ranges, std::size_t diagonal_size,
    bool coupled) noexcept;

std::size_t estimate_jcp_block_preconditioner_bytes_from_panel_groups(
    const std::vector<std::vector<std::size_t>> &panel_groups,
    std::size_t diagonal_size, bool coupled) noexcept;

std::size_t estimate_jcp_block_preconditioner_bytes_from_owned_factor_groups(
    const std::vector<JcpOwnedFactorPanelGroup> &panel_groups,
    std::size_t diagonal_size, bool coupled) noexcept;

std::shared_ptr<JcpBlockPreconditioner>
build_jcp_block_preconditioner_from_panel_groups(
    const std::vector<Panel> &panels, bool coupled, double eps_ratio,
    double kappa, double lhs_phi, double lhs_phin,
    const std::vector<double> &diagonal, const std::string &near_integration,
    const std::string &quad_buffer, const std::string &block_near_mode,
    const std::vector<std::vector<std::size_t>> &panel_groups,
    TitanExecutor &executor);

std::shared_ptr<JcpBlockPreconditioner>
build_jcp_block_preconditioner_from_owned_factor_groups(
    const std::vector<Panel> &panels, bool coupled, double eps_ratio,
    double kappa, double lhs_phi, double lhs_phin,
    const std::vector<double> &diagonal, const std::string &near_integration,
    const std::string &quad_buffer, const std::string &block_near_mode,
    const std::vector<JcpOwnedFactorPanelGroup> &panel_groups,
    TitanExecutor &executor);

std::shared_ptr<JcpBlockPreconditioner> build_jcp_block_preconditioner(
    const std::vector<Panel> &panels, const TreeWorkspace &tree, bool coupled,
    double eps_ratio, double kappa, double lhs_phi, double lhs_phin,
    const std::vector<double> &diagonal, const std::string &near_integration,
    const std::string &quad_buffer, const std::string &block_near_mode,
    const std::vector<JcpBlockRange> &ranges, TitanExecutor &executor);

std::shared_ptr<JcpCoarseGalerkinPreconditioner>
make_jcp_coarse_structure(const std::vector<Panel> &panels,
                          const TreeWorkspace &tree, bool coupled,
                          const std::vector<double> &diagonal,
                          std::size_t aggregate_cap);

bool build_jcp_exact_galerkin(
    const std::vector<Panel> &panels, JcpCoarseGalerkinPreconditioner &coarse,
    const std::function<void(const std::vector<double> &,
                             std::vector<double> &)> &setup_matvec,
    bool verify, bool retain_operator_basis = false);

bool complete_jcp_hybrid_galerkin_monopole(
    const std::vector<Panel> &panels, JcpCoarseGalerkinPreconditioner &coarse,
    const std::vector<std::size_t> &near_row_ptr,
    const std::vector<std::size_t> &near_col_index, double eps_ratio,
    double kappa,
    const std::function<void(const std::vector<double> &,
                             std::vector<double> &)> &audit_matvec,
    std::size_t audit_probes, double audit_tolerance,
    TitanExecutor &executor);

} // namespace TitanPBE::Tabi

#endif // TITAN_PBE_TABI_PRECONDITIONER_H
