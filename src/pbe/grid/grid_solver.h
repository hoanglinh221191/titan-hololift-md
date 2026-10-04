#ifndef TITAN_PBE_GRID_SOLVER_H
#define TITAN_PBE_GRID_SOLVER_H

#include <string_view>

namespace TitanPBE {

struct FrameResult;

enum class GridSolveMode {
  Linear,
  Physical,
  Nonlinear,
};

[[nodiscard]] constexpr std::string_view
grid_solve_mode_name(GridSolveMode mode) noexcept {
  switch (mode) {
  case GridSolveMode::Linear:
    return "linear";
  case GridSolveMode::Physical:
    return "physical";
  case GridSolveMode::Nonlinear:
    return "nonlinear";
  }
  return "unknown";
}

struct GridSolveRequest {
  GridSolveMode mode{GridSolveMode::Physical};
  double eps_in{1.0};
  double eps_out{80.0};
  double kappa{0.0};
  const FrameResult *map_result{nullptr};

  [[nodiscard]] static constexpr GridSolveRequest
  linear(double eps_in, double eps_out,
         const FrameResult *map_result = nullptr) noexcept {
    return GridSolveRequest{GridSolveMode::Linear, eps_in, eps_out, 0.0,
                            map_result};
  }

  [[nodiscard]] static constexpr GridSolveRequest
  physical(double eps_in, double eps_out, double kappa,
           const FrameResult *map_result = nullptr) noexcept {
    return GridSolveRequest{GridSolveMode::Physical, eps_in, eps_out, kappa,
                            map_result};
  }

  [[nodiscard]] static constexpr GridSolveRequest
  nonlinear(double eps_in, double eps_out, double kappa,
            const FrameResult *map_result = nullptr) noexcept {
    return GridSolveRequest{GridSolveMode::Nonlinear, eps_in, eps_out, kappa,
                            map_result};
  }
};

} // namespace TitanPBE

#endif // TITAN_PBE_GRID_SOLVER_H
