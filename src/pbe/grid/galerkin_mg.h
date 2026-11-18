#ifndef TITAN_PBE_GRID_GALERKIN_MG_H
#define TITAN_PBE_GRID_GALERKIN_MG_H

// Algebraic-coarse-grid (Galerkin) multigrid for the grid Poisson-Boltzmann
// operators (GRID-03).
//
// The grid solvers write the operator in h^2-scaled 7-point form on an
// nx x ny x nz grid whose outer layer carries Dirichlet values:
//   (A x)_id = diag[id] x[id] - sum over the six faces of eps_face x[nb],
// with eps_x[id] coupling id and id+1, eps_y[id] id and id+nx, eps_z[id] id
// and id+nx*ny. A is symmetric positive definite.
//
// Coarse operators are A_H = P^T A_h P with trilinear P, built from A_h alone
// (no re-discretised geometry), so they stay consistent with the fine
// dielectric map, and the (H/h)^2 scaling of the residual is implicit in
// R = P^T. Coarse operators are 27-point and stored symmetric. Smoothing is
// Gauss-Seidel: red-black on the 7-point fine level, eight-colour on the
// 27-point levels, symmetric (forward colours before the coarse correction,
// reverse colours after), so one V-cycle is an SPD preconditioner for CG.
// Every colour sweep, transfer and reduction is split over z planes with a
// fixed summation order: results do not depend on the thread count.
//
// References: Alcouffe, Brandt, Dendy, Painter, SIAM J. Sci. Stat. Comput.
// 2, 430 (1981) (multigrid for strongly discontinuous coefficients);
// Holst, Saied, J. Comput. Chem. 14, 105 (1993) (multigrid for the
// Poisson-Boltzmann equation).

#include "titan_executor.h"

#include <Eigen/Dense>

#include <cstddef>
#include <functional>
#include <vector>

namespace TitanPBE::galerkin_mg {

struct FineOperator {
  std::size_t nx{0};
  std::size_t ny{0};
  std::size_t nz{0};
  const double *eps_x{nullptr};
  const double *eps_y{nullptr};
  const double *eps_z{nullptr};
  const double *diag{nullptr};
};

struct Options {
  // 0: coarsen until the coarsest fits the direct solve; N > 0: build N
  // levels where the grid allows it (the coarsest is factored only when
  // small enough, otherwise relaxed with coarse_sweeps)
  int max_levels{0};
  int pre_sweeps{2};
  int post_sweeps{2};
  int coarse_sweeps{200}; // coarsest relaxation when it is too large to factor
  // Largest coarsest-level unknown count solved by dense Cholesky; coarsening
  // stops once a level has at most this many unknowns.
  std::size_t direct_unknowns{1000};
  std::size_t workers{1};
};

class Hierarchy {
public:
  void build(const FineOperator &fine, const Options &options,
             TitanExecutor &executor);

  [[nodiscard]] int level_count() const noexcept {
    return static_cast<int>(levels_.size());
  }
  [[nodiscard]] bool direct_coarsest() const noexcept { return direct_; }
  [[nodiscard]] std::size_t memory_bytes() const noexcept;

  // z = M r: one symmetric V-cycle for A z = r from z = 0. r and z are full
  // fine grids; the outer layer of z is left at zero.
  void precondition(const double *r, double *z, TitanExecutor &executor) const;

  // y = A x on the interior, reading the outer layer of x (Dirichlet data);
  // the outer layer of y is zero.
  void apply_fine(const double *x, double *y, TitanExecutor &executor) const;

  [[nodiscard]] const FineOperator &fine() const noexcept { return fine_; }
  [[nodiscard]] std::size_t interior_count() const noexcept;

  struct Level {
    std::size_t nx{0}, ny{0}, nz{0}, n{0};
    std::vector<double> coef; // 14 per node (centre + 13 forward); level >= 1
    mutable std::vector<double> rhs;
    mutable std::vector<double> sol;
    mutable std::vector<double> res;
  };
  [[nodiscard]] const Level &level(int index) const {
    return levels_.at(static_cast<std::size_t>(index));
  }
  // Offsets of the 13 stored forward entries of a coarse 27-point row, in
  // slot order 1..13 (slot 0 is the centre).
  static void forward_offset(int slot, int &dx, int &dy, int &dz) noexcept;

private:
  void smooth_fine(const double *r, double *z, int sweeps, bool forward,
                   TitanExecutor &executor) const;
  void smooth_coarse(const Level &level, int sweeps, bool forward,
                     TitanExecutor &executor) const;
  void residual_coarse(const Level &level, TitanExecutor &executor) const;
  void solve_coarsest(TitanExecutor &executor) const;
  void cycle(std::size_t index, TitanExecutor &executor) const;

  FineOperator fine_{};
  Options options_{};
  std::vector<Level> levels_; // levels_[0] holds only the fine work vectors
  bool direct_{false};
  Eigen::LLT<Eigen::MatrixXd> coarsest_factor_;
  mutable Eigen::VectorXd coarsest_rhs_;
};

struct IterationReport {
  int iteration{0};
  double update_rms{0.0};
  double update_max{0.0};
  double residual_rms{0.0};
};

struct SolveResult {
  int iterations{0};
  bool converged{false};
  bool breakdown{false};
  double residual_rms{0.0};
};

// Solves A x = b for the interior of x (its outer layer is Dirichlet data),
// with CG preconditioned by one V-cycle per iteration (use_cg) or by the
// stationary iteration x += M (b - A x). stop() is called after every
// iteration and returns true to accept x.
SolveResult solve(const Hierarchy &mg, const double *b, double *x,
                  int max_iterations, bool use_cg,
                  const std::function<bool(const IterationReport &)> &stop,
                  TitanExecutor &executor);

// rms over the interior of b - A x.
double residual_rms(const Hierarchy &mg, const double *b, const double *x,
                    TitanExecutor &executor);

} // namespace TitanPBE::galerkin_mg

#endif // TITAN_PBE_GRID_GALERKIN_MG_H
