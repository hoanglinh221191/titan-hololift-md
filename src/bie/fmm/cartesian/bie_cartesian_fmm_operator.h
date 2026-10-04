#ifndef TITAN_BIE_CARTESIAN_FMM_OPERATOR_H
#define TITAN_BIE_CARTESIAN_FMM_OPERATOR_H

#include "bie_cartesian_fmm_plan.h"
#include "bie_cartesian_fmm_workspace.h"

#include <memory>
#include <span>

class TitanExecutor;

namespace titan::bie::fmm::cartesian {

class CartesianFmmOperator {
public:
  explicit CartesianFmmOperator(std::shared_ptr<const CartesianFmmPlan> plan);

  CartesianFmmApplyStats apply_far(std::span<const double> phi,
                                   std::span<const double> phin,
                                   TitanExecutor &executor);

  [[nodiscard]] const CartesianFmmPlan &plan() const noexcept { return *plan_; }
  [[nodiscard]] CartesianFmmWorkspace &workspace() noexcept {
    return workspace_;
  }
  [[nodiscard]] const CartesianFmmWorkspace &workspace() const noexcept {
    return workspace_;
  }

private:
  std::shared_ptr<const CartesianFmmPlan> plan_{};
  CartesianFmmWorkspace workspace_{};
};

} // namespace titan::bie::fmm::cartesian

#endif // TITAN_BIE_CARTESIAN_FMM_OPERATOR_H
