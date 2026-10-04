#ifndef TITAN_BIE_CARTESIAN_FMM_WORKSPACE_H
#define TITAN_BIE_CARTESIAN_FMM_WORKSPACE_H

#include "bie_cartesian_fmm_checked_size.h"
#include "bie_cartesian_fmm_plan.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace titan::bie::fmm::cartesian {

class CartesianFmmWorkspace {
public:
  void prepare(const CartesianFmmPlan &plan) {
    const std::size_t node_count = plan.nodes().size();
    const std::size_t panel_count = plan.panels().size();
    const auto moments_bytes = detail::checked_container_allocation(
        node_count, moments_, "allocating Cartesian workspace moments");
    const auto locals_bytes = detail::checked_container_allocation(
        node_count, locals_, "allocating Cartesian workspace locals");
    const auto phi_bytes = detail::checked_container_allocation(
        panel_count, interaction_phi_,
        "allocating Cartesian workspace phi interaction buffer");
    const auto phin_bytes = detail::checked_container_allocation(
        panel_count, interaction_phin_,
        "allocating Cartesian workspace phin interaction buffer");
    if (!moments_bytes || !locals_bytes || !phi_bytes || !phin_bytes) {
      const std::string message = !moments_bytes
                                      ? moments_bytes.error()
                                      : !locals_bytes
                                            ? locals_bytes.error()
                                            : !phi_bytes ? phi_bytes.error()
                                                         : phin_bytes.error();
      throw std::length_error(message);
    }
    const auto first = detail::checked_add_or_error(
        *moments_bytes, *locals_bytes,
        "summing Cartesian workspace moment and local buffers");
    const auto second = first
                            ? detail::checked_add_or_error(
                                  *first, *phi_bytes,
                                  "summing Cartesian workspace buffers")
                            : first;
    const auto total = second
                           ? detail::checked_add_or_error(
                                 *second, *phin_bytes,
                                 "summing Cartesian workspace buffers")
                           : second;
    if (!total)
      throw std::length_error(total.error());
    moments_.resize(node_count);
    locals_.resize(node_count);
    interaction_phi_.resize(panel_count);
    interaction_phin_.resize(panel_count);
  }

  void clear(const CartesianFmmPlan &plan) {
    const int order = plan.options().expansion_order;
    const std::size_t terms =
        static_cast<std::size_t>((order + 1) * (order + 2) * (order + 3) / 6);
    for (JcpMoments &moments : moments_) {
      moments = {};
      moments.order = order;
      moments.term_count = terms;
    }
    for (JcpLocals &local : locals_) {
      local = {};
      local.order = order;
      local.term_count = terms;
    }
    std::fill(interaction_phi_.begin(), interaction_phi_.end(), 0.0);
    std::fill(interaction_phin_.begin(), interaction_phin_.end(), 0.0);
  }

  [[nodiscard]] std::vector<JcpMoments> &moments() noexcept { return moments_; }
  [[nodiscard]] std::vector<JcpLocals> &locals() noexcept { return locals_; }
  [[nodiscard]] CoefficientBuffer &interaction_phi() noexcept {
    return interaction_phi_;
  }
  [[nodiscard]] CoefficientBuffer &interaction_phin() noexcept {
    return interaction_phin_;
  }
  [[nodiscard]] const CoefficientBuffer &interaction_phi() const noexcept {
    return interaction_phi_;
  }
  [[nodiscard]] const CoefficientBuffer &interaction_phin() const noexcept {
    return interaction_phin_;
  }
  [[nodiscard]] std::size_t retained_bytes() const {
    const auto moments_bytes = detail::checked_container_capacity_bytes(
        moments_, "accounting Cartesian workspace moments");
    const auto locals_bytes = detail::checked_container_capacity_bytes(
        locals_, "accounting Cartesian workspace locals");
    const auto phi_bytes = detail::checked_container_capacity_bytes(
        interaction_phi_, "accounting Cartesian workspace phi buffer");
    const auto phin_bytes = detail::checked_container_capacity_bytes(
        interaction_phin_, "accounting Cartesian workspace phin buffer");
    if (!moments_bytes || !locals_bytes || !phi_bytes || !phin_bytes) {
      const std::string message = !moments_bytes
                                      ? moments_bytes.error()
                                      : !locals_bytes
                                            ? locals_bytes.error()
                                            : !phi_bytes ? phi_bytes.error()
                                                         : phin_bytes.error();
      throw std::overflow_error(message);
    }
    const auto first = detail::checked_add_or_error(
        *moments_bytes, *locals_bytes,
        "summing retained Cartesian workspace buffers");
    const auto second = first
                            ? detail::checked_add_or_error(
                                  *first, *phi_bytes,
                                  "summing retained Cartesian workspace buffers")
                            : first;
    const auto total = second
                           ? detail::checked_add_or_error(
                                 *second, *phin_bytes,
                                 "summing retained Cartesian workspace buffers")
                           : second;
    if (!total)
      throw std::overflow_error(total.error());
    return *total;
  }

private:
  std::vector<JcpMoments> moments_{};
  std::vector<JcpLocals> locals_{};
  CoefficientBuffer interaction_phi_{};
  CoefficientBuffer interaction_phin_{};
};

} // namespace titan::bie::fmm::cartesian

#endif // TITAN_BIE_CARTESIAN_FMM_WORKSPACE_H
