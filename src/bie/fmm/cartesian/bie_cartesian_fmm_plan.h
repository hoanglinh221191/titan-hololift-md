#ifndef TITAN_BIE_CARTESIAN_FMM_PLAN_H
#define TITAN_BIE_CARTESIAN_FMM_PLAN_H

#include "bie_cartesian_fmm_types.h"

#include <expected>
#include <memory>
#include <span>
#include <string>
#include <vector>

class TitanExecutor;

namespace titan::bie::fmm::cartesian {

class CartesianFmmPlan {
public:
  CartesianFmmPlan(const CartesianFmmPlan &) = delete;
  CartesianFmmPlan &operator=(const CartesianFmmPlan &) = delete;
  CartesianFmmPlan(CartesianFmmPlan &&) = delete;
  CartesianFmmPlan &operator=(CartesianFmmPlan &&) = delete;
  ~CartesianFmmPlan() = default;

  [[nodiscard]] const CartesianFmmPlanOptions &options() const noexcept {
    return options_;
  }
  [[nodiscard]] const CartesianFmmPlanStats &stats() const noexcept {
    return stats_;
  }
  [[nodiscard]] std::span<const PanelGeometry> panels() const noexcept {
    return panels_;
  }
  [[nodiscard]] std::span<const CartesianFmmNode> nodes() const noexcept {
    return nodes_;
  }
  [[nodiscard]] std::span<const int> source_panel_indices() const noexcept {
    return source_panel_indices_;
  }
  [[nodiscard]] std::span<const int> target_panel_indices() const noexcept {
    return target_panel_indices_;
  }
  [[nodiscard]] std::span<const std::uint32_t> source_leaves() const noexcept {
    return source_leaves_;
  }
  [[nodiscard]] std::span<const std::uint32_t> target_leaves() const noexcept {
    return target_leaves_;
  }
  [[nodiscard]] std::span<const std::uint32_t> nodes_by_level() const noexcept {
    return nodes_by_level_;
  }
  [[nodiscard]] std::span<const std::size_t> level_offsets() const noexcept {
    return level_offsets_;
  }
  [[nodiscard]] std::span<const CartesianFmmPair> m2l_pairs() const noexcept {
    return m2l_pairs_;
  }
  [[nodiscard]] std::span<const std::size_t> m2l_offsets() const noexcept {
    return m2l_offsets_;
  }
  [[nodiscard]] std::span<const std::uint32_t>
  m2l_target_nodes() const noexcept {
    return m2l_target_nodes_;
  }
  [[nodiscard]] std::span<const CartesianFmmPair>
  direct_pairs() const noexcept {
    return direct_pairs_;
  }
  [[nodiscard]] std::span<const std::size_t> direct_offsets() const noexcept {
    return direct_offsets_;
  }
  [[nodiscard]] std::span<const std::uint32_t>
  direct_target_nodes() const noexcept {
    return direct_target_nodes_;
  }
  [[nodiscard]] std::size_t green_term_count() const noexcept {
    return green_term_count_;
  }
  [[nodiscard]] std::span<const double>
  coulomb_translation(std::size_t index) const noexcept;
  [[nodiscard]] std::span<const double>
  yukawa_translation(std::size_t index) const noexcept;

private:
  CartesianFmmPlan() = default;

  CartesianFmmPlanOptions options_{};
  CartesianFmmPlanStats stats_{};
  std::vector<PanelGeometry> panels_{};
  std::vector<CartesianFmmNode> nodes_{};
  std::vector<int> source_panel_indices_{};
  std::vector<int> target_panel_indices_{};
  std::vector<std::uint32_t> source_leaves_{};
  std::vector<std::uint32_t> target_leaves_{};
  std::vector<std::uint32_t> nodes_by_level_{};
  std::vector<std::size_t> level_offsets_{};
  std::vector<CartesianFmmPair> m2l_pairs_{};
  std::vector<std::size_t> m2l_offsets_{};
  std::vector<std::uint32_t> m2l_target_nodes_{};
  std::vector<CartesianFmmPair> direct_pairs_{};
  std::vector<std::size_t> direct_offsets_{};
  std::vector<std::uint32_t> direct_target_nodes_{};
  std::size_t green_term_count_{0};
  CoefficientBuffer coulomb_translations_{};
  CoefficientBuffer yukawa_translations_{};

  friend std::expected<std::shared_ptr<const CartesianFmmPlan>, std::string>
  prepare_cartesian_fmm_plan(std::span<const PanelGeometry>,
                             const CartesianFmmPlanOptions &, TitanExecutor &);
};

std::expected<std::shared_ptr<const CartesianFmmPlan>, std::string>
prepare_cartesian_fmm_plan(std::span<const PanelGeometry> panels,
                           const CartesianFmmPlanOptions &options,
                           TitanExecutor &executor);

} // namespace titan::bie::fmm::cartesian

#endif // TITAN_BIE_CARTESIAN_FMM_PLAN_H
