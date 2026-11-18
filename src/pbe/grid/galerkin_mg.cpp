#include "pbe/grid/galerkin_mg.h"

#include "pbe/grid/grid_common.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace TitanPBE::galerkin_mg {
namespace {

struct Offset {
  int dx;
  int dy;
  int dz;
};

// The 13 "forward" neighbours of a 27-point stencil; a coarse node stores its
// centre and these, the backward entries being those of the neighbour.
constexpr std::array<Offset, 13> kForward = {{{-1, -1, 1},
                                              {0, -1, 1},
                                              {1, -1, 1},
                                              {-1, 0, 1},
                                              {0, 0, 1},
                                              {1, 0, 1},
                                              {-1, 1, 1},
                                              {0, 1, 1},
                                              {1, 1, 1},
                                              {-1, 1, 0},
                                              {0, 1, 0},
                                              {1, 1, 0},
                                              {1, 0, 0}}};
constexpr std::size_t kSlots = 14;
constexpr std::size_t kSerialPoints = 32768;

constexpr int offset_index(int dx, int dy, int dz) noexcept {
  return (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1);
}

std::size_t coarse_dim(std::size_t fine) noexcept { return (fine - 2) / 2 + 2; }

std::size_t interior(std::size_t nx, std::size_t ny, std::size_t nz) noexcept {
  return (nx - 2) * (ny - 2) * (nz - 2);
}

std::ptrdiff_t linear_offset(const Offset &o, std::size_t nx,
                             std::size_t ny) noexcept {
  return static_cast<std::ptrdiff_t>(o.dx) +
         static_cast<std::ptrdiff_t>(o.dy) * static_cast<std::ptrdiff_t>(nx) +
         static_cast<std::ptrdiff_t>(o.dz) *
             static_cast<std::ptrdiff_t>(nx * ny);
}

// Runs fn(t) for t in [0, count), split over the executor when the work is
// large enough. Every fn(t) writes its own plane.
template <class Fn>
void for_planes(TitanExecutor &executor, std::size_t workers,
                std::size_t count, std::size_t plane_points, Fn &&fn) {
  if (count == 0)
    return;
  const std::size_t use = std::min(workers, count);
  if (use <= 1 || count * plane_points < kSerialPoints) {
    for (std::size_t t = 0; t < count; ++t)
      fn(t);
    return;
  }
  executor.parallel_for(count, use, 1,
                        [&](std::size_t, std::size_t begin, std::size_t end) {
                          for (std::size_t t = begin; t < end; ++t)
                            fn(t);
                        });
}

// Interior planes k = 1 .. nz-2 as plane indices t = k-1.
template <class Fn>
void for_interior_planes(TitanExecutor &executor, std::size_t workers,
                         std::size_t nx, std::size_t ny, std::size_t nz,
                         Fn &&fn) {
  if (nz < 3)
    return;
  for_planes(executor, workers, nz - 2, nx * ny,
             [&](std::size_t t) { fn(t + 1); });
}

// Entries A(id, id + offset) of the 7-point fine operator.
template <class Fn>
void fine_entries(const FineOperator &op, std::size_t id, Fn &&fn) {
  const std::size_t sy = op.nx;
  const std::size_t sz = op.nx * op.ny;
  fn(0, 0, 0, op.diag[id]);
  fn(1, 0, 0, -op.eps_x[id]);
  fn(-1, 0, 0, -op.eps_x[id - 1]);
  fn(0, 1, 0, -op.eps_y[id]);
  fn(0, -1, 0, -op.eps_y[id - sy]);
  fn(0, 0, 1, -op.eps_z[id]);
  fn(0, 0, -1, -op.eps_z[id - sz]);
}

// Entries A(id, id + offset) of a stored 27-point level.
template <class Fn>
void coarse_entries(const Hierarchy::Level &level, std::size_t id, Fn &&fn) {
  const double *c = level.coef.data();
  fn(0, 0, 0, c[id * kSlots]);
  for (std::size_t s = 0; s < kForward.size(); ++s) {
    const Offset &o = kForward[s];
    const std::ptrdiff_t lin = linear_offset(o, level.nx, level.ny);
    fn(o.dx, o.dy, o.dz, c[id * kSlots + 1 + s]);
    fn(-o.dx, -o.dy, -o.dz,
       c[static_cast<std::size_t>(static_cast<std::ptrdiff_t>(id) - lin) *
             kSlots +
         1 + s]);
  }
}

// Coarse parents of fine index g along one axis: g = 2I (weight 1) or
// g = 2I +- 1 (weight 1/2 each).
int parents(std::size_t g, std::array<std::size_t, 2> &index,
            std::array<double, 2> &weight) noexcept {
  if ((g & 1U) == 0U) {
    index[0] = g / 2;
    weight[0] = 1.0;
    return 1;
  }
  index[0] = (g - 1) / 2;
  index[1] = (g + 1) / 2;
  weight[0] = weight[1] = 0.5;
  return 2;
}

// A_H = P^T A_h P for the coarse level `coarse` of a fine level with
// dimensions fnx x fny x fnz and entries entries(f, fn).
template <class Entries>
void galerkin(std::size_t fnx, std::size_t fny, std::size_t fnz,
              Entries &&entries, Hierarchy::Level &coarse,
              TitanExecutor &executor, std::size_t workers) {
  const std::size_t cnx = coarse.nx, cny = coarse.ny, cnz = coarse.nz;
  coarse.coef.assign(coarse.n * kSlots, 0.0);
  for_interior_planes(executor, workers, cnx, cny, cnz, [&](std::size_t K) {
    std::array<double, 27> row{};
    for (std::size_t J = 1; J + 1 < cny; ++J) {
      for (std::size_t I = 1; I + 1 < cnx; ++I) {
        row.fill(0.0);
        for (int dz = -1; dz <= 1; ++dz) {
          const std::size_t fk = 2 * K + static_cast<std::size_t>(dz + 1) - 1;
          if (fk < 1 || fk + 2 > fnz)
            continue;
          const double wz = dz == 0 ? 1.0 : 0.5;
          for (int dy = -1; dy <= 1; ++dy) {
            const std::size_t fj = 2 * J + static_cast<std::size_t>(dy + 1) - 1;
            if (fj < 1 || fj + 2 > fny)
              continue;
            const double wy = dy == 0 ? 1.0 : 0.5;
            for (int dx = -1; dx <= 1; ++dx) {
              const std::size_t fi =
                  2 * I + static_cast<std::size_t>(dx + 1) - 1;
              if (fi < 1 || fi + 2 > fnx)
                continue;
              const double w = wz * wy * (dx == 0 ? 1.0 : 0.5);
              const std::size_t f = idx3(fi, fj, fk, fnx, fny);
              entries(f, [&](int ex, int ey, int ez, double a) {
                if (a == 0.0)
                  return;
                const std::size_t gi = fi + static_cast<std::size_t>(ex + 1) - 1;
                const std::size_t gj = fj + static_cast<std::size_t>(ey + 1) - 1;
                const std::size_t gk = fk + static_cast<std::size_t>(ez + 1) - 1;
                if (gi < 1 || gi + 2 > fnx || gj < 1 || gj + 2 > fny ||
                    gk < 1 || gk + 2 > fnz)
                  return; // Dirichlet node: the correction is zero there
                std::array<std::size_t, 2> pi{}, pj{}, pk{};
                std::array<double, 2> wi{}, wj{}, wk{};
                const int ni = parents(gi, pi, wi);
                const int nj = parents(gj, pj, wj);
                const int nk = parents(gk, pk, wk);
                for (int c = 0; c < nk; ++c) {
                  if (pk[c] < 1 || pk[c] + 2 > cnz)
                    continue;
                  for (int b = 0; b < nj; ++b) {
                    if (pj[b] < 1 || pj[b] + 2 > cny)
                      continue;
                    for (int q = 0; q < ni; ++q) {
                      if (pi[q] < 1 || pi[q] + 2 > cnx)
                        continue;
                      const int ox = static_cast<int>(pi[q]) - static_cast<int>(I);
                      const int oy = static_cast<int>(pj[b]) - static_cast<int>(J);
                      const int oz = static_cast<int>(pk[c]) - static_cast<int>(K);
                      row[static_cast<std::size_t>(offset_index(ox, oy, oz))] +=
                          w * a * wi[q] * wj[b] * wk[c];
                    }
                  }
                }
              });
            }
          }
        }
        const std::size_t cid = idx3(I, J, K, cnx, cny);
        coarse.coef[cid * kSlots] =
            row[static_cast<std::size_t>(offset_index(0, 0, 0))];
        for (std::size_t s = 0; s < kForward.size(); ++s)
          coarse.coef[cid * kSlots + 1 + s] = row[static_cast<std::size_t>(
              offset_index(kForward[s].dx, kForward[s].dy, kForward[s].dz))];
      }
    }
  });
}

// Deterministic sum over interior planes: per-plane partial sums, added in
// plane order.
template <class Fn>
double plane_sum(TitanExecutor &executor, std::size_t workers, std::size_t nx,
                 std::size_t ny, std::size_t nz, Fn &&fn) {
  if (nz < 3)
    return 0.0;
  std::vector<double> partial(nz - 2, 0.0);
  for_interior_planes(executor, workers, nx, ny, nz,
                      [&](std::size_t k) { partial[k - 1] = fn(k); });
  double sum = 0.0;
  for (double value : partial)
    sum += value;
  return sum;
}

} // namespace

void Hierarchy::forward_offset(int slot, int &dx, int &dy, int &dz) noexcept {
  const Offset &o = kForward[static_cast<std::size_t>(slot - 1)];
  dx = o.dx;
  dy = o.dy;
  dz = o.dz;
}

std::size_t Hierarchy::interior_count() const noexcept {
  return interior(fine_.nx, fine_.ny, fine_.nz);
}

std::size_t Hierarchy::memory_bytes() const noexcept {
  std::size_t bytes = 0;
  for (const Level &level : levels_)
    bytes += sizeof(double) * (level.coef.capacity() + level.rhs.capacity() +
                               level.sol.capacity() + level.res.capacity());
  if (direct_)
    bytes += sizeof(double) * static_cast<std::size_t>(
                                  coarsest_factor_.matrixLLT().size());
  return bytes;
}

void Hierarchy::build(const FineOperator &fine, const Options &options,
                      TitanExecutor &executor) {
  if (fine.nx < 3 || fine.ny < 3 || fine.nz < 3 || fine.eps_x == nullptr ||
      fine.eps_y == nullptr || fine.eps_z == nullptr || fine.diag == nullptr)
    throw std::invalid_argument("Galerkin multigrid needs a 3-D grid with an "
                                "interior and all operator arrays.");
  fine_ = fine;
  options_ = options;
  options_.workers = std::max<std::size_t>(1, options.workers);
  levels_.clear();
  direct_ = false;

  Level top;
  top.nx = fine.nx;
  top.ny = fine.ny;
  top.nz = fine.nz;
  top.n = fine.nx * fine.ny * fine.nz;
  top.res.assign(top.n, 0.0);
  levels_.push_back(std::move(top));

  // An explicit level count is honoured as far as the grid can be coarsened;
  // otherwise coarsening stops at the first level small enough to factor.
  const bool explicit_levels = options.max_levels > 0;
  const int cap = explicit_levels ? options.max_levels : 64;
  while (static_cast<int>(levels_.size()) < cap) {
    const Level &last = levels_.back();
    if (!explicit_levels &&
        interior(last.nx, last.ny, last.nz) <= options_.direct_unknowns)
      break;
    Level next;
    next.nx = coarse_dim(last.nx);
    next.ny = coarse_dim(last.ny);
    next.nz = coarse_dim(last.nz);
    if (next.nx < 3 || next.ny < 3 || next.nz < 3 ||
        (next.nx == last.nx && next.ny == last.ny && next.nz == last.nz))
      break;
    next.n = next.nx * next.ny * next.nz;
    if (levels_.size() == 1) {
      galerkin(last.nx, last.ny, last.nz,
               [&](std::size_t f, auto &&fn) { fine_entries(fine_, f, fn); },
               next, executor, options_.workers);
    } else {
      const Level &source = levels_.back();
      galerkin(source.nx, source.ny, source.nz,
               [&](std::size_t f, auto &&fn) { coarse_entries(source, f, fn); },
               next, executor, options_.workers);
    }
    next.rhs.assign(next.n, 0.0);
    next.sol.assign(next.n, 0.0);
    next.res.assign(next.n, 0.0);
    levels_.push_back(std::move(next));
  }

  // Dense Cholesky of the coarsest operator when it is small enough.
  const Level &last = levels_.back();
  const std::size_t m = interior(last.nx, last.ny, last.nz);
  if (m <= options_.direct_unknowns) {
    const std::size_t mx = last.nx - 2, my = last.ny - 2;
    Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(m),
                                                  static_cast<Eigen::Index>(m));
    auto row_of = [&](std::size_t i, std::size_t j, std::size_t k) {
      return static_cast<Eigen::Index>(((k - 1) * my + (j - 1)) * mx + (i - 1));
    };
    for (std::size_t k = 1; k + 1 < last.nz; ++k)
      for (std::size_t j = 1; j + 1 < last.ny; ++j)
        for (std::size_t i = 1; i + 1 < last.nx; ++i) {
          const std::size_t id = idx3(i, j, k, last.nx, last.ny);
          const Eigen::Index row = row_of(i, j, k);
          auto add = [&](int dx, int dy, int dz, double a) {
            const std::size_t gi = i + static_cast<std::size_t>(dx + 1) - 1;
            const std::size_t gj = j + static_cast<std::size_t>(dy + 1) - 1;
            const std::size_t gk = k + static_cast<std::size_t>(dz + 1) - 1;
            if (gi < 1 || gi + 2 > last.nx || gj < 1 || gj + 2 > last.ny ||
                gk < 1 || gk + 2 > last.nz)
              return;
            dense(row, row_of(gi, gj, gk)) += a;
          };
          if (levels_.size() == 1)
            fine_entries(fine_, id, add);
          else
            coarse_entries(last, id, add);
        }
    coarsest_factor_.compute(dense);
    direct_ = coarsest_factor_.info() == Eigen::Success;
    if (direct_)
      coarsest_rhs_.resize(static_cast<Eigen::Index>(m));
  }
}

void Hierarchy::apply_fine(const double *x, double *y,
                           TitanExecutor &executor) const {
  const std::size_t nx = fine_.nx, ny = fine_.ny, nz = fine_.nz;
  const std::size_t sy = nx, sz = nx * ny;
  std::fill(y, y + nx * ny * nz, 0.0);
  for_interior_planes(executor, options_.workers, nx, ny, nz,
                      [&](std::size_t k) {
    for (std::size_t j = 1; j + 1 < ny; ++j)
      for (std::size_t i = 1; i + 1 < nx; ++i) {
        const std::size_t id = idx3(i, j, k, nx, ny);
        y[id] = fine_.diag[id] * x[id] -
                (fine_.eps_x[id] * x[id + 1] + fine_.eps_x[id - 1] * x[id - 1] +
                 fine_.eps_y[id] * x[id + sy] +
                 fine_.eps_y[id - sy] * x[id - sy] +
                 fine_.eps_z[id] * x[id + sz] +
                 fine_.eps_z[id - sz] * x[id - sz]);
      }
  });
}

void Hierarchy::smooth_fine(const double *r, double *z, int sweeps,
                            bool forward, TitanExecutor &executor) const {
  const std::size_t nx = fine_.nx, ny = fine_.ny, nz = fine_.nz;
  const std::size_t sy = nx, sz = nx * ny;
  for (int sweep = 0; sweep < sweeps; ++sweep) {
    for (int pass = 0; pass < 2; ++pass) {
      const std::size_t colour = static_cast<std::size_t>(forward ? pass : 1 - pass);
      for_interior_planes(executor, options_.workers, nx, ny, nz,
                          [&](std::size_t k) {
        for (std::size_t j = 1; j + 1 < ny; ++j) {
          // nodes with (i + j + k) % 2 == colour
          const std::size_t first = ((colour + j + k) & 1U) != 0U ? 1 : 2;
          for (std::size_t i = first; i + 1 < nx; i += 2) {
            const std::size_t id = idx3(i, j, k, nx, ny);
            const double d = fine_.diag[id];
            if (!(d > 0.0))
              continue;
            const double nb =
                fine_.eps_x[id] * z[id + 1] + fine_.eps_x[id - 1] * z[id - 1] +
                fine_.eps_y[id] * z[id + sy] + fine_.eps_y[id - sy] * z[id - sy] +
                fine_.eps_z[id] * z[id + sz] + fine_.eps_z[id - sz] * z[id - sz];
            z[id] = (r[id] + nb) / d;
          }
        }
      });
    }
  }
}

void Hierarchy::smooth_coarse(const Level &level, int sweeps, bool forward,
                              TitanExecutor &executor) const {
  const std::size_t nx = level.nx, ny = level.ny, nz = level.nz;
  std::array<std::ptrdiff_t, 13> lin{};
  for (std::size_t s = 0; s < kForward.size(); ++s)
    lin[s] = linear_offset(kForward[s], nx, ny);
  const double *c = level.coef.data();
  double *x = level.sol.data();
  const double *b = level.rhs.data();
  for (int sweep = 0; sweep < sweeps; ++sweep) {
    for (int pass = 0; pass < 8; ++pass) {
      // eight colours by the parities of (i, j, k): no two nodes of one
      // colour share a 27-point stencil
      const int colour = forward ? pass : 7 - pass;
      const std::size_t pi = static_cast<std::size_t>(colour & 1);
      const std::size_t pj = static_cast<std::size_t>((colour >> 1) & 1);
      const std::size_t pk = static_cast<std::size_t>((colour >> 2) & 1);
      const std::size_t k_first = pk != 0U ? 1 : 2;
      if (k_first + 2 > nz)
        continue;
      const std::size_t planes = (nz - 2 - k_first) / 2 + 1;
      for_planes(executor, options_.workers, planes, nx * ny / 4,
                 [&](std::size_t t) {
        const std::size_t k = k_first + 2 * t;
        for (std::size_t j = pj != 0U ? 1 : 2; j + 1 < ny; j += 2)
          for (std::size_t i = pi != 0U ? 1 : 2; i + 1 < nx; i += 2) {
            const std::size_t id = idx3(i, j, k, nx, ny);
            const double centre = c[id * kSlots];
            if (!(centre > 0.0))
              continue;
            double sum = b[id];
            for (std::size_t s = 0; s < kForward.size(); ++s) {
              const std::size_t up =
                  static_cast<std::size_t>(static_cast<std::ptrdiff_t>(id) + lin[s]);
              const std::size_t down =
                  static_cast<std::size_t>(static_cast<std::ptrdiff_t>(id) - lin[s]);
              sum -= c[id * kSlots + 1 + s] * x[up] +
                     c[down * kSlots + 1 + s] * x[down];
            }
            x[id] = sum / centre;
          }
      });
    }
  }
}

void Hierarchy::residual_coarse(const Level &level,
                                TitanExecutor &executor) const {
  const std::size_t nx = level.nx, ny = level.ny, nz = level.nz;
  std::array<std::ptrdiff_t, 13> lin{};
  for (std::size_t s = 0; s < kForward.size(); ++s)
    lin[s] = linear_offset(kForward[s], nx, ny);
  const double *c = level.coef.data();
  const double *x = level.sol.data();
  std::fill(level.res.begin(), level.res.end(), 0.0);
  for_interior_planes(executor, options_.workers, nx, ny, nz,
                      [&](std::size_t k) {
    for (std::size_t j = 1; j + 1 < ny; ++j)
      for (std::size_t i = 1; i + 1 < nx; ++i) {
        const std::size_t id = idx3(i, j, k, nx, ny);
        double ax = c[id * kSlots] * x[id];
        for (std::size_t s = 0; s < kForward.size(); ++s) {
          const std::size_t up =
              static_cast<std::size_t>(static_cast<std::ptrdiff_t>(id) + lin[s]);
          const std::size_t down =
              static_cast<std::size_t>(static_cast<std::ptrdiff_t>(id) - lin[s]);
          ax += c[id * kSlots + 1 + s] * x[up] + c[down * kSlots + 1 + s] * x[down];
        }
        level.res[id] = level.rhs[id] - ax;
      }
  });
}

namespace {

// rhs_c = P^T res_f over the coarse interior.
void restrict_to(const double *res, std::size_t fnx, std::size_t fny,
                 std::size_t fnz, const Hierarchy::Level &coarse,
                 TitanExecutor &executor, std::size_t workers) {
  std::fill(coarse.rhs.begin(), coarse.rhs.end(), 0.0);
  std::fill(coarse.sol.begin(), coarse.sol.end(), 0.0);
  const std::size_t cnx = coarse.nx, cny = coarse.ny, cnz = coarse.nz;
  for_interior_planes(executor, workers, cnx, cny, cnz, [&](std::size_t K) {
    for (std::size_t J = 1; J + 1 < cny; ++J)
      for (std::size_t I = 1; I + 1 < cnx; ++I) {
        double sum = 0.0;
        for (int dz = -1; dz <= 1; ++dz) {
          const std::size_t fk = 2 * K + static_cast<std::size_t>(dz + 1) - 1;
          if (fk < 1 || fk + 2 > fnz)
            continue;
          for (int dy = -1; dy <= 1; ++dy) {
            const std::size_t fj = 2 * J + static_cast<std::size_t>(dy + 1) - 1;
            if (fj < 1 || fj + 2 > fny)
              continue;
            for (int dx = -1; dx <= 1; ++dx) {
              const std::size_t fi = 2 * I + static_cast<std::size_t>(dx + 1) - 1;
              if (fi < 1 || fi + 2 > fnx)
                continue;
              const double w = (dz == 0 ? 1.0 : 0.5) * (dy == 0 ? 1.0 : 0.5) *
                               (dx == 0 ? 1.0 : 0.5);
              sum += w * res[idx3(fi, fj, fk, fnx, fny)];
            }
          }
        }
        coarse.rhs[idx3(I, J, K, cnx, cny)] = sum;
      }
  });
}

// x_f += P x_c over the fine interior (x_c is zero on its outer layer).
void prolong_add(const Hierarchy::Level &coarse, double *x, std::size_t fnx,
                 std::size_t fny, std::size_t fnz, TitanExecutor &executor,
                 std::size_t workers) {
  const std::size_t cnx = coarse.nx, cny = coarse.ny;
  const double *xc = coarse.sol.data();
  for_interior_planes(executor, workers, fnx, fny, fnz, [&](std::size_t k) {
    std::array<std::size_t, 2> pk{};
    std::array<double, 2> wk{};
    const int nk = parents(k, pk, wk);
    for (std::size_t j = 1; j + 1 < fny; ++j) {
      std::array<std::size_t, 2> pj{};
      std::array<double, 2> wj{};
      const int nj = parents(j, pj, wj);
      for (std::size_t i = 1; i + 1 < fnx; ++i) {
        std::array<std::size_t, 2> pi{};
        std::array<double, 2> wi{};
        const int ni = parents(i, pi, wi);
        double sum = 0.0;
        for (int c = 0; c < nk; ++c)
          for (int b = 0; b < nj; ++b)
            for (int a = 0; a < ni; ++a)
              sum += wk[c] * wj[b] * wi[a] * xc[idx3(pi[a], pj[b], pk[c], cnx, cny)];
        x[idx3(i, j, k, fnx, fny)] += sum;
      }
    }
  });
}

} // namespace

void Hierarchy::solve_coarsest(TitanExecutor &executor) const {
  const Level &last = levels_.back();
  if (!direct_) {
    for (int s = 0; s < options_.coarse_sweeps; ++s) {
      smooth_coarse(last, 1, true, executor);
      smooth_coarse(last, 1, false, executor);
    }
    return;
  }
  const std::size_t mx = last.nx - 2, my = last.ny - 2;
  for (std::size_t k = 1; k + 1 < last.nz; ++k)
    for (std::size_t j = 1; j + 1 < last.ny; ++j)
      for (std::size_t i = 1; i + 1 < last.nx; ++i)
        coarsest_rhs_[static_cast<Eigen::Index>(((k - 1) * my + (j - 1)) * mx +
                                                (i - 1))] =
            last.rhs[idx3(i, j, k, last.nx, last.ny)];
  const Eigen::VectorXd solution = coarsest_factor_.solve(coarsest_rhs_);
  for (std::size_t k = 1; k + 1 < last.nz; ++k)
    for (std::size_t j = 1; j + 1 < last.ny; ++j)
      for (std::size_t i = 1; i + 1 < last.nx; ++i)
        last.sol[idx3(i, j, k, last.nx, last.ny)] =
            solution[static_cast<Eigen::Index>(((k - 1) * my + (j - 1)) * mx +
                                               (i - 1))];
}

void Hierarchy::cycle(std::size_t index, TitanExecutor &executor) const {
  const Level &level = levels_[index];
  if (index + 1 == levels_.size()) {
    solve_coarsest(executor);
    return;
  }
  smooth_coarse(level, options_.pre_sweeps, true, executor);
  residual_coarse(level, executor);
  const Level &coarse = levels_[index + 1];
  restrict_to(level.res.data(), level.nx, level.ny, level.nz, coarse, executor,
              options_.workers);
  cycle(index + 1, executor);
  prolong_add(coarse, level.sol.data(), level.nx, level.ny, level.nz, executor,
              options_.workers);
  smooth_coarse(level, options_.post_sweeps, false, executor);
}

void Hierarchy::precondition(const double *r, double *z,
                             TitanExecutor &executor) const {
  const std::size_t n = fine_.nx * fine_.ny * fine_.nz;
  std::fill(z, z + n, 0.0);
  if (levels_.size() == 1) {
    if (direct_) {
      const std::size_t mx = fine_.nx - 2, my = fine_.ny - 2;
      for (std::size_t k = 1; k + 1 < fine_.nz; ++k)
        for (std::size_t j = 1; j + 1 < fine_.ny; ++j)
          for (std::size_t i = 1; i + 1 < fine_.nx; ++i)
            coarsest_rhs_[static_cast<Eigen::Index>(
                ((k - 1) * my + (j - 1)) * mx + (i - 1))] =
                r[idx3(i, j, k, fine_.nx, fine_.ny)];
      const Eigen::VectorXd solution = coarsest_factor_.solve(coarsest_rhs_);
      for (std::size_t k = 1; k + 1 < fine_.nz; ++k)
        for (std::size_t j = 1; j + 1 < fine_.ny; ++j)
          for (std::size_t i = 1; i + 1 < fine_.nx; ++i)
            z[idx3(i, j, k, fine_.nx, fine_.ny)] = solution[static_cast<Eigen::Index>(
                ((k - 1) * my + (j - 1)) * mx + (i - 1))];
      return;
    }
    for (int s = 0; s < options_.coarse_sweeps; ++s) {
      smooth_fine(r, z, 1, true, executor);
      smooth_fine(r, z, 1, false, executor);
    }
    return;
  }
  smooth_fine(r, z, options_.pre_sweeps, true, executor);
  // fine residual r - A z (z is zero on the outer layer)
  const Level &top = levels_.front();
  apply_fine(z, top.res.data(), executor);
  for_interior_planes(executor, options_.workers, fine_.nx, fine_.ny, fine_.nz,
                      [&](std::size_t k) {
    for (std::size_t j = 1; j + 1 < fine_.ny; ++j)
      for (std::size_t i = 1; i + 1 < fine_.nx; ++i) {
        const std::size_t id = idx3(i, j, k, fine_.nx, fine_.ny);
        top.res[id] = r[id] - top.res[id];
      }
  });
  const Level &coarse = levels_[1];
  restrict_to(top.res.data(), fine_.nx, fine_.ny, fine_.nz, coarse, executor,
              options_.workers);
  cycle(1, executor);
  prolong_add(coarse, z, fine_.nx, fine_.ny, fine_.nz, executor,
              options_.workers);
  smooth_fine(r, z, options_.post_sweeps, false, executor);
}

double residual_rms(const Hierarchy &mg, const double *b, const double *x,
                    TitanExecutor &executor) {
  const FineOperator &op = mg.fine();
  const std::size_t n = op.nx * op.ny * op.nz;
  std::vector<double> ax(n, 0.0);
  mg.apply_fine(x, ax.data(), executor);
  const double sumsq = plane_sum(
      executor, 1, op.nx, op.ny, op.nz, [&](std::size_t k) {
        double s = 0.0;
        for (std::size_t j = 1; j + 1 < op.ny; ++j)
          for (std::size_t i = 1; i + 1 < op.nx; ++i) {
            const std::size_t id = idx3(i, j, k, op.nx, op.ny);
            const double r = b[id] - ax[id];
            s += r * r;
          }
        return s;
      });
  const std::size_t count = mg.interior_count();
  return count > 0 ? std::sqrt(sumsq / static_cast<double>(count)) : 0.0;
}

SolveResult solve(const Hierarchy &mg, const double *b, double *x,
                  int max_iterations, bool use_cg,
                  const std::function<bool(const IterationReport &)> &stop,
                  TitanExecutor &executor) {
  SolveResult result;
  const FineOperator &op = mg.fine();
  const std::size_t nx = op.nx, ny = op.ny, nz = op.nz, n = nx * ny * nz;
  const std::size_t count = mg.interior_count();
  if (count == 0 || max_iterations <= 0)
    return result;
  const std::size_t workers = std::max<std::size_t>(1, executor.capacity());
  std::vector<double> r(n, 0.0), z(n, 0.0), p(n, 0.0), q(n, 0.0);

  auto residual_into_r = [&]() {
    mg.apply_fine(x, q.data(), executor);
    for_interior_planes(executor, workers, nx, ny, nz, [&](std::size_t k) {
      for (std::size_t j = 1; j + 1 < ny; ++j)
        for (std::size_t i = 1; i + 1 < nx; ++i) {
          const std::size_t id = idx3(i, j, k, nx, ny);
          r[id] = b[id] - q[id];
        }
    });
  };
  auto dot = [&](const std::vector<double> &u, const std::vector<double> &v) {
    return plane_sum(executor, workers, nx, ny, nz, [&](std::size_t k) {
      double s = 0.0;
      for (std::size_t j = 1; j + 1 < ny; ++j)
        for (std::size_t i = 1; i + 1 < nx; ++i) {
          const std::size_t id = idx3(i, j, k, nx, ny);
          s += u[id] * v[id];
        }
      return s;
    });
  };
  // x += alpha d on the interior; returns (sum of squares, max) of the update
  auto update_x = [&](const std::vector<double> &d, double alpha,
                      double &update_rms, double &update_max) {
    std::vector<double> plane_max(nz, 0.0);
    const double sumsq =
        plane_sum(executor, workers, nx, ny, nz, [&](std::size_t k) {
          double s = 0.0, m = 0.0;
          for (std::size_t j = 1; j + 1 < ny; ++j)
            for (std::size_t i = 1; i + 1 < nx; ++i) {
              const std::size_t id = idx3(i, j, k, nx, ny);
              const double u = alpha * d[id];
              x[id] += u;
              s += u * u;
              m = std::max(m, std::abs(u));
            }
          plane_max[k] = m;
          return s;
        });
    update_rms = std::sqrt(sumsq / static_cast<double>(count));
    update_max = *std::max_element(plane_max.begin(), plane_max.end());
  };

  residual_into_r();
  if (!use_cg) {
    for (int it = 1; it <= max_iterations; ++it) {
      mg.precondition(r.data(), z.data(), executor);
      IterationReport report;
      report.iteration = it;
      update_x(z, 1.0, report.update_rms, report.update_max);
      residual_into_r();
      report.residual_rms = std::sqrt(dot(r, r) / static_cast<double>(count));
      result.iterations = it;
      result.residual_rms = report.residual_rms;
      if (!std::isfinite(report.residual_rms)) {
        result.breakdown = true;
        break;
      }
      if (stop(report)) {
        result.converged = true;
        break;
      }
    }
    return result;
  }

  mg.precondition(r.data(), z.data(), executor);
  p = z;
  double rz = dot(r, z);
  for (int it = 1; it <= max_iterations; ++it) {
    mg.apply_fine(p.data(), q.data(), executor);
    const double pq = dot(p, q);
    if (!(pq > 0.0) || !std::isfinite(pq) || !std::isfinite(rz)) {
      result.breakdown = true;
      break;
    }
    const double alpha = rz / pq;
    IterationReport report;
    report.iteration = it;
    update_x(p, alpha, report.update_rms, report.update_max);
    for_interior_planes(executor, workers, nx, ny, nz, [&](std::size_t k) {
      for (std::size_t j = 1; j + 1 < ny; ++j)
        for (std::size_t i = 1; i + 1 < nx; ++i) {
          const std::size_t id = idx3(i, j, k, nx, ny);
          r[id] -= alpha * q[id];
        }
    });
    report.residual_rms = std::sqrt(dot(r, r) / static_cast<double>(count));
    result.iterations = it;
    result.residual_rms = report.residual_rms;
    if (stop(report)) {
      result.converged = true;
      break;
    }
    mg.precondition(r.data(), z.data(), executor);
    const double rz_next = dot(r, z);
    if (!(rz_next > 0.0) || !std::isfinite(rz_next)) {
      result.breakdown = !(rz_next == 0.0);
      break;
    }
    const double beta = rz_next / rz;
    rz = rz_next;
    for_interior_planes(executor, workers, nx, ny, nz, [&](std::size_t k) {
      for (std::size_t j = 1; j + 1 < ny; ++j)
        for (std::size_t i = 1; i + 1 < nx; ++i) {
          const std::size_t id = idx3(i, j, k, nx, ny);
          p[id] = z[id] + beta * p[id];
        }
    });
  }
  return result;
}

} // namespace TitanPBE::galerkin_mg
