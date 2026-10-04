#ifndef TITAN_PBE_TABI_CALDERON_PREPARED_H
#define TITAN_PBE_TABI_CALDERON_PREPARED_H

#include "tabi_calderon_transfer.h"

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

class TitanExecutor;

namespace TitanPBE::Tabi::calderon {

// C5D.1 candidate configuration. This remains diagnostic-only until the
// mass-lumped FMM action passes the dense mixed-Galerkin oracle.
struct PreparedCalderonConfig {
  double eps_ratio{80.0};
  double kappa{0.125};
  std::size_t leaf_size{64};
  double opening_theta{0.6};
  int expansion_order{4};
  std::string near_integration{"analytic_singular"};
  std::string quad_buffer{"off"};
  MassCompatibleTransferConfig transfer{};
};

struct PreparedCalderonApplyResult {
  std::string status{"not_applied"};
  std::vector<double> values;
  std::size_t apply_count{0};
  std::size_t near_pairs{0};
  std::size_t far_pairs{0};
  std::size_t pairing_iterations_max{0};
  double transfer_in_seconds{0.0};
  double fmm_seconds{0.0};
  double pairing_seconds{0.0};
  double transfer_out_seconds{0.0};
  double apply_seconds{0.0};
  bool finite{false};
};

class PreparedCalderonOperator {
public:
  struct State;
  PreparedCalderonOperator();
  ~PreparedCalderonOperator();
  PreparedCalderonOperator(PreparedCalderonOperator &&) noexcept;
  PreparedCalderonOperator &operator=(PreparedCalderonOperator &&) noexcept;
  PreparedCalderonOperator(const PreparedCalderonOperator &) = delete;
  PreparedCalderonOperator &operator=(const PreparedCalderonOperator &) = delete;

  [[nodiscard]] const std::string &status() const noexcept;
  [[nodiscard]] const std::string &geometry_status() const noexcept;
  [[nodiscard]] std::size_t primary_panels() const noexcept;
  [[nodiscard]] std::size_t auxiliary_vertices() const noexcept;
  [[nodiscard]] std::size_t fmm_quadrature_panels() const noexcept;
  [[nodiscard]] std::size_t prepare_build_count() const noexcept;
  [[nodiscard]] std::size_t context_build_count() const noexcept;
  [[nodiscard]] std::size_t apply_count() const noexcept;
  [[nodiscard]] double prepare_seconds() const noexcept;
  [[nodiscard]] double transfer_seconds() const noexcept;
  [[nodiscard]] double fmm_setup_seconds() const noexcept;
  [[nodiscard]] std::size_t estimated_bytes() const noexcept;

  [[nodiscard]] PreparedCalderonApplyResult
  apply_auxiliary(std::span<const double> auxiliary_unknowns,
                  TitanExecutor &executor);

  [[nodiscard]] PreparedCalderonApplyResult
  apply_primary(std::span<const double> primary_unknowns,
                TitanExecutor &executor);

private:
  std::unique_ptr<State> state_;

  explicit PreparedCalderonOperator(std::unique_ptr<State> state);
  friend PreparedCalderonOperator prepare_fmm_native_calderon_operator(
      std::span<const Panel>, const AuxiliarySurfaceResult &,
      const PreparedCalderonConfig &, TitanExecutor &);
};

[[nodiscard]] PreparedCalderonOperator prepare_fmm_native_calderon_operator(
    std::span<const Panel> primary_panels,
    const AuxiliarySurfaceResult &auxiliary,
    const PreparedCalderonConfig &config,
    TitanExecutor &executor);

} // namespace TitanPBE::Tabi::calderon

#endif // TITAN_PBE_TABI_CALDERON_PREPARED_H
