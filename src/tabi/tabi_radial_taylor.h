#pragma once
#include "tabi_treecode_internal.h"

#include <array>
#include <cmath>

namespace TitanPBE::Tabi::detail {

// Taylor coefficients of the treecode far-field kernels about a node centre,
// built from radial derivatives instead of truncated power-series arithmetic.
// With s = r^2, the m-th s-derivative of the Yukawa kernel is
//   f^(m)(s) = (-1)^m exp(-k r) P_m(k r) / (2^m r^(2m+1) 4 pi),
//   P_(m+1) = (2m+1+z) P_m - z P'_m,
// and expanding s(c + d) exactly gives every multivariate coefficient without
// polynomial convolutions. `base` holds the Laplace coefficients and `delta`
// those of screened minus Laplace, which a small-z series evaluates without
// the cancellation of subtracting two kernels.
struct RadialTaylorTensors {
  bool valid{false};
  std::array<double, 512> base{};
  std::array<double, 512> delta{};
};

// Index of the coefficient of d_x^a d_y^b d_z^c is 64a + 8b + c, degree <= 7.
inline RadialTaylorTensors radial_taylor_tensors(const Panel &t,
                                                 const TreeNode &c,
                                                 double kappa, bool coupled,
                                                 int order) noexcept {
  constexpr auto polynomials = [] {
    std::array<std::array<double, 8>, 8> p{};
    p[0][0] = 1;
    for (int m = 0; m < 7; ++m)
      for (int j = 0; j <= m + 1; ++j)
        p[m + 1][j] = (2 * m + 1 - j) * p[m][j] + (j ? p[m][j - 1] : 0);
    return p;
  }();
  constexpr auto reciprocal_factorial = [] {
    std::array<double, 25> f{};
    f[0] = 1;
    for (int j = 1; j < 25; ++j)
      f[j] = f[j - 1] / j;
    return f;
  }();
  constexpr auto gaps = [&] {
    std::array<std::array<double, 25>, 8> g{};
    for (int m = 0; m < 8; ++m)
      for (int n = 1; n < 25; ++n) {
        long double s = 0;
        for (int j = 0; j <= m && j <= n; ++j)
          s += polynomials[m][j] * ((n - j) % 2 ? -1.L : 1.L) *
               reciprocal_factorial[n - j];
        g[m][n] = static_cast<double>(s);
      }
    return g;
  }();
  RadialTaylorTensors out;
  const double d[3] = {c.cx - t.x, c.cy - t.y, c.cz - t.z};
  const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2], r = std::sqrt(r2);
  if (!(r > 1e-14))
    return out;
  out.valid = true;
  const double z = std::max(0., kappa) * r, e = std::exp(-z);
  const int degree = order + (coupled ? 2 : 1);
  std::array<double, 8> radial{}, difference{};
  double common = 1 / (4 * kPi * r);
  for (int m = 0; m <= degree; ++m) {
    radial[m] = common * polynomials[m][0];
    double gap;
    if (z < .25) {
      gap = 0;
      for (int n = 24; n >= (m ? 2 : 1); --n)
        gap = gap * z + gaps[m][n];
      gap *= m ? z * z : z;
    } else {
      double tail = 0;
      for (int j = m; j >= 1; --j)
        tail = tail * z + polynomials[m][j];
      gap = std::expm1(-z) * polynomials[m][0] + e * z * tail;
    }
    difference[m] = common * gap;
    common /= -2 * r2;
  }
  double powers[3][8]{};
  for (int j = 0; j < 3; ++j) {
    powers[j][0] = 1;
    for (int a = 1; a <= degree; ++a)
      powers[j][a] = powers[j][a - 1] * (2 * d[j]) / a;
  }
  for (int a = 0; a <= degree; ++a)
    for (int b = 0; a + b <= degree; ++b)
      for (int c0 = 0; a + b + c0 <= degree; ++c0) {
        const int index = 64 * a + 8 * b + c0;
        for (int i = 0; i <= a / 2; ++i)
          for (int j = 0; j <= b / 2; ++j)
            for (int k = 0; k <= c0 / 2; ++k) {
              const double factor =
                  powers[0][a - 2 * i] * powers[1][b - 2 * j] *
                  powers[2][c0 - 2 * k] * reciprocal_factorial[i] *
                  reciprocal_factorial[j] * reciprocal_factorial[k];
              const int m = a + b + c0 - i - j - k;
              out.base[index] += factor * radial[m];
              out.delta[index] += factor * difference[m];
            }
      }
  return out;
}

// Contracts the tensors with a node's moments into the four kernel channels
// of the JCP rows (K1, K2 into phi; K3, K4 into phin). `add(channel, row,
// value)` receives every term in the order the rows have always summed them.
template <typename Add>
inline void radial_jcp_contract(const Panel &t, const RadialTaylorTensors &w,
                                const NodeMoments &scalar,
                                const std::array<NodeMoments, 3> &normal,
                                double eps, bool coupled, int order,
                                Add &&add) noexcept {
  const auto &base = w.base;
  const auto &delta = w.delta;
  const double nt[3] = {t.nx, t.ny, t.nz};
  const double inverse_eps = 1 / std::max(1e-12, eps);
  const int stride[3] = {64, 8, 1};
  const auto &indices = moment_indices(order);
  for (std::size_t m = 0; m < indices.size(); ++m) {
    const auto &a = indices[m];
    const int index = 64 * a.ax + 8 * a.ay + a.az,
              alpha[3] = {a.ax, a.ay, a.az};
    if (coupled)
      add(kJcpK1PhiChannel, false, -(scalar.values[m] * delta[index]));
    for (int j = 0; j < 3; ++j) {
      const double factor = alpha[j] + 1;
      const int k = index + stride[j];
      add(kJcpK2PhiNormalChannel, false,
          normal[j].values[m] * factor *
              ((eps - 1) * base[k] + (coupled ? eps * delta[k] : 0)));
      if (coupled) {
        add(kJcpK3PhinChannel, true,
            scalar.values[m] * nt[j] * factor *
                ((inverse_eps - 1) * base[k] + delta[k] * inverse_eps));
        for (int i = 0; i < 3; ++i)
          add(kJcpK4MixedChannel, true,
              -(normal[j].values[m] * nt[i] * factor *
                (alpha[i] + 1 + (i == j)) * delta[k + stride[i]]));
      }
    }
  }
}

inline JcpAccum radial_jcp_node_sum(const Panel &t, const TreeNode &c,
                                    const NodeMoments &scalar,
                                    const std::array<NodeMoments, 3> &normal,
                                    double eps, double kappa, bool coupled,
                                    int order) noexcept {
  JcpAccum out;
  const RadialTaylorTensors w =
      radial_taylor_tensors(t, c, kappa, coupled, order);
  if (!w.valid)
    return out;
  radial_jcp_contract(t, w, scalar, normal, eps, coupled, order,
                      [&](std::size_t, bool phin, double value) {
                        (phin ? out.phin : out.phi) += value;
                      });
  return out;
}

// The far-field kernel of the treecode apply: radial_jcp_node_sum rearranged
// for speed, with the same radial derivatives and the same terms.
//  - The coefficient cube is built by sum factorisation along z, y, then x
//    (about 3x fewer multiplications), and only the degrees the contraction
//    reads: B up to order+1, D up to order+2, D not at all without salt.
//    Nothing is zeroed.
//  - The four channels are contracted through combined tensors
//      E = (eps-1) B + eps D,  F = (1/eps-1) B + D/eps,
//      Dn[k] = sum_i n_i (k_i+1) D[k+e_i],
//    so  phi  = sum_m -S_m D[a] + sum_j (a_j+1) N_jm E[a+e_j]
//        phin = sum_m S_m sum_j n_j (a_j+1) F[a+e_j] - sum_j (a_j+1) N_jm Dn[a+e_j]
//    (K1, K2 into phi; K3, K4 into phin), with independent partial sums.
//  - Moments are read in place.
//  - node_sum4 evaluates four nodes for one target in the lanes of a vector
//    type. The one-node kernel, node_sum1, runs node_sum4 with the node in
//    every lane, so a node's term does not depend on whether it went through
//    a group of four (tabi_treecode_radial_oracle checks it). A separate
//    scalar kernel cannot promise that: g++ 13 fuses multiply-adds
//    differently in the scalar and vector instantiations of contract (on
//    Linux CI, 31% of the oracle's lanes differed, by up to 3e-14 relative).
//    node_sum, the same arithmetic on doubles, is the fallback without
//    vector types.
// The summation order differs from radial_jcp_node_sum, so the result equals
// it to rounding, not bit for bit; TITAN_FORCE_SCALAR=1 keeps the old kernel.
namespace radial_fast {

// Every loop below has a trip count fixed by the order, and the helpers are
// forced inline into node_sum and node_sum4: unrolled and inlined, the
// coefficients stay in registers. Without either (the inliner keeps
// contract and factorised_cube out of line in the engine build) the kernels
// run about 1.8x slower.
#define TITAN_RADIAL_UNROLL _Pragma("GCC unroll 64")

template <int Order> struct MomentTable {
  static constexpr int count = (Order + 1) * (Order + 2) * (Order + 3) / 6;
  std::array<std::array<int, 3>, count> alpha{};
  // Same order as moment_indices(): by degree, then ax and ay descending.
  constexpr MomentTable() {
    int m = 0;
    for (int degree = 0; degree <= Order; ++degree)
      for (int ax = degree; ax >= 0; --ax)
        for (int ay = degree - ax; ay >= 0; --ay)
          alpha[m++] = {ax, ay, degree - ax - ay};
  }
};

// Degrees read by the contraction: B to Order+1, D to Order+2 (coupled only).
template <int Order, bool Coupled> struct Degrees {
  static constexpr int MB = Order + 1, MD = Coupled ? Order + 2 : 0;
  static constexpr int top = Coupled ? MD : MB;
  static constexpr int N = top + 1, H = top / 2 + 1;
};

inline constexpr auto kPolynomials = [] {
  std::array<std::array<double, 8>, 8> p{};
  p[0][0] = 1;
  for (int m = 0; m < 7; ++m)
    for (int j = 0; j <= m + 1; ++j)
      p[m + 1][j] = (2 * m + 1 - j) * p[m][j] + (j ? p[m][j - 1] : 0);
  return p;
}();
inline constexpr auto kReciprocalFactorial = [] {
  std::array<double, 25> f{};
  f[0] = 1;
  for (int j = 1; j < 25; ++j)
    f[j] = f[j - 1] / j;
  return f;
}();
inline constexpr auto kGaps = [] {
  std::array<std::array<double, 25>, 8> g{};
  for (int m = 0; m < 8; ++m)
    for (int n = 1; n < 25; ++n) {
      long double s = 0;
      for (int j = 0; j <= m && j <= n; ++j)
        s += kPolynomials[m][j] * ((n - j) % 2 ? -1.L : 1.L) *
             kReciprocalFactorial[n - j];
      g[m][n] = static_cast<double>(s);
    }
  return g;
}();

// d = centre - target and the radial derivatives of one pair, computed as
// radial_taylor_tensors computes them. False when the pair is degenerate.
template <int Order, bool Coupled>
[[gnu::always_inline]] inline bool radial_derivatives(const Panel &t, const TreeNode &c,
                               double (&d)[3], double *radial,
                               double *difference, double kappa) noexcept {
  constexpr int top = Degrees<Order, Coupled>::top;
  d[0] = c.cx - t.x;
  d[1] = c.cy - t.y;
  d[2] = c.cz - t.z;
  const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2], r = std::sqrt(r2);
  if (!(r > 1e-14))
    return false;
  const double z = std::max(0., kappa) * r;
  const double e = Coupled ? std::exp(-z) : 0.0;
  double common = 1 / (4 * kPi * r);
  TITAN_RADIAL_UNROLL
  for (int m = 0; m <= top; ++m) {
    radial[m] = common * kPolynomials[m][0];
    if constexpr (Coupled) {
      double gap;
      if (z < .25) {
        gap = 0;
        TITAN_RADIAL_UNROLL
        for (int n = 24; n >= (m ? 2 : 1); --n)
          gap = gap * z + kGaps[m][n];
        gap *= m ? z * z : z;
      } else {
        double tail = 0;
        TITAN_RADIAL_UNROLL
        for (int j = m; j >= 1; --j)
          tail = tail * z + kPolynomials[m][j];
        gap = std::expm1(-z) * kPolynomials[m][0] + e * z * tail;
      }
      difference[m] = common * gap;
    }
    common /= -2 * r2;
  }
  return true;
}

// T[a][b][c] = sum_{i<=a/2, j<=b/2, k<=c/2} X[a][i] Y[b][j] Z[c][k]
//              R[a+b+c-i-j-k]  for a+b+c <= M, stored at (a*N+b)*N+c.
template <int M, int N, int H, typename V>
[[gnu::always_inline]] inline void factorised_cube(const V (&X)[N][H], const V (&Y)[N][H],
                            const V (&Z)[N][H], const V *R, V *T) noexcept {
  V w1[M + 1][M + 1];
  TITAN_RADIAL_UNROLL
  for (int c = 0; c <= M; ++c)
    TITAN_RADIAL_UNROLL
    for (int s = 0; s + c <= M; ++s) {
      V v = Z[c][0] * R[s + c];
      TITAN_RADIAL_UNROLL
      for (int k = 1; 2 * k <= c; ++k)
        v += Z[c][k] * R[s + c - k];
      w1[c][s] = v;
    }
  V w2[M + 1][M + 1][M + 1];
  TITAN_RADIAL_UNROLL
  for (int b = 0; b <= M; ++b)
    TITAN_RADIAL_UNROLL
    for (int c = 0; b + c <= M; ++c)
      TITAN_RADIAL_UNROLL
      for (int u = 0; u + b + c <= M; ++u) {
        V v = Y[b][0] * w1[c][u + b];
        TITAN_RADIAL_UNROLL
        for (int j = 1; 2 * j <= b; ++j)
          v += Y[b][j] * w1[c][u + b - j];
        w2[b][c][u] = v;
      }
  TITAN_RADIAL_UNROLL
  for (int a = 0; a <= M; ++a)
    TITAN_RADIAL_UNROLL
    for (int b = 0; a + b <= M; ++b)
      TITAN_RADIAL_UNROLL
      for (int c = 0; a + b + c <= M; ++c) {
        V v = X[a][0] * w2[b][c][a];
        TITAN_RADIAL_UNROLL
        for (int i = 1; 2 * i <= a; ++i)
          v += X[a][i] * w2[b][c][a - i];
        T[(a * N + b) * N + c] = v;
      }
}

// Coefficients and contraction for one pair per lane of V (double, or a
// vector of doubles). `moment(kind, m)` returns moment m of the scalar (kind
// 0) or normal x, y, z (kinds 1-3) density, one value per lane.
template <int Order, bool Coupled, typename V, typename Moment>
[[gnu::always_inline]] inline void contract(const V (&d)[3], const V *radial, const V *difference,
                     const double (&nt)[3], double eps, Moment &&moment,
                     V &phi, V &phin) noexcept {
  using Deg = Degrees<Order, Coupled>;
  constexpr int MB = Deg::MB, MD = Deg::MD, N = Deg::N, H = Deg::H;
  constexpr int S1 = N * N, S2 = N, S3 = 1;
  static constexpr MomentTable<Order> moments{};
  // Per-axis factors (2 d)^(a-2i) / (a-2i)! / i!.
  V X[N][H], Y[N][H], Z[N][H];
  {
    V powers[3][N];
    TITAN_RADIAL_UNROLL
    for (int j = 0; j < 3; ++j) {
      powers[j][0] = d[j] * 0.0 + 1.0;
      TITAN_RADIAL_UNROLL
      for (int a = 1; a < N; ++a)
        powers[j][a] = powers[j][a - 1] * (2.0 * d[j]) / double(a);
    }
    TITAN_RADIAL_UNROLL
    for (int a = 0; a < N; ++a)
      TITAN_RADIAL_UNROLL
      for (int i = 0; 2 * i <= a; ++i) {
        X[a][i] = powers[0][a - 2 * i] * kReciprocalFactorial[i];
        Y[a][i] = powers[1][a - 2 * i] * kReciprocalFactorial[i];
        Z[a][i] = powers[2][a - 2 * i] * kReciprocalFactorial[i];
      }
  }
  V B[N * N * N];
  factorised_cube<MB, N, H>(X, Y, Z, radial, B);
  if constexpr (!Coupled) {
    V phi_x = B[0] * 0.0, phi_y = phi_x, phi_z = phi_x;
    TITAN_RADIAL_UNROLL
    for (int m = 0; m < moments.count; ++m) {
      const auto &a = moments.alpha[m];
      const int i0 = a[0] * S1 + a[1] * S2 + a[2] * S3;
      phi_x += moment(1, m) * double(a[0] + 1) * ((eps - 1) * B[i0 + S1]);
      phi_y += moment(2, m) * double(a[1] + 1) * ((eps - 1) * B[i0 + S2]);
      phi_z += moment(3, m) * double(a[2] + 1) * ((eps - 1) * B[i0 + S3]);
    }
    phi = (phi_x + phi_y) + phi_z;
    phin = phi * 0.0;
  } else {
    V D[N * N * N];
    factorised_cube<MD, N, H>(X, Y, Z, difference, D);
    const double inverse_eps = 1 / std::max(1e-12, eps);
    V E[N * N * N], F[N * N * N], Dn[N * N * N];
    TITAN_RADIAL_UNROLL
    for (int a = 0; a <= MB; ++a)
      TITAN_RADIAL_UNROLL
      for (int b = 0; a + b <= MB; ++b)
        TITAN_RADIAL_UNROLL
        for (int c0 = 0; a + b + c0 <= MB; ++c0) {
          const int k = a * S1 + b * S2 + c0 * S3;
          E[k] = (eps - 1) * B[k] + eps * D[k];
          F[k] = (inverse_eps - 1) * B[k] + D[k] * inverse_eps;
          Dn[k] = nt[0] * double(a + 1) * D[k + S1] +
                  nt[1] * double(b + 1) * D[k + S2] +
                  nt[2] * double(c0 + 1) * D[k + S3];
        }
    V phi_s = B[0] * 0.0, phi_x = phi_s, phi_y = phi_s, phi_z = phi_s;
    V phin_s = phi_s, phin_x = phi_s, phin_y = phi_s, phin_z = phi_s;
    TITAN_RADIAL_UNROLL
    for (int m = 0; m < moments.count; ++m) {
      const auto &a = moments.alpha[m];
      const int i0 = a[0] * S1 + a[1] * S2 + a[2] * S3;
      const double fx = a[0] + 1, fy = a[1] + 1, fz = a[2] + 1;
      const V s = moment(0, m);
      const V wx = moment(1, m) * fx, wy = moment(2, m) * fy,
              wz = moment(3, m) * fz;
      phi_s -= s * D[i0];
      phi_x += wx * E[i0 + S1];
      phi_y += wy * E[i0 + S2];
      phi_z += wz * E[i0 + S3];
      phin_s += s * (nt[0] * fx * F[i0 + S1] + nt[1] * fy * F[i0 + S2] +
                     nt[2] * fz * F[i0 + S3]);
      phin_x -= wx * Dn[i0 + S1];
      phin_y -= wy * Dn[i0 + S2];
      phin_z -= wz * Dn[i0 + S3];
    }
    phi = (phi_s + phi_x) + (phi_y + phi_z);
    phin = (phin_s + phin_x) + (phin_y + phin_z);
  }
}

template <int Order, bool Coupled>
JcpAccum node_sum(const Panel &t, const TreeNode &c, const double *scalar,
                  const double *normal_x, const double *normal_y,
                  const double *normal_z, double eps, double kappa) noexcept {
  constexpr int top = Degrees<Order, Coupled>::top;
  JcpAccum out;
  double d[3], radial[top + 1], difference[top + 1];
  if (!radial_derivatives<Order, Coupled>(t, c, d, radial, difference, kappa))
    return out;
  const double nt[3] = {t.nx, t.ny, t.nz};
  const double *const density[4] = {scalar, normal_x, normal_y, normal_z};
  contract<Order, Coupled>(
      d, radial, difference, nt, eps,
      [&](int kind, int m) { return density[kind][m]; }, out.phi, out.phin);
  return out;
}

#if defined(__GNUC__) || defined(__clang__)
typedef double Lanes4 __attribute__((vector_size(32)));

// Four nodes seen from one target. A degenerate lane, or a null node,
// contributes zero, as node_sum returns zero for a degenerate pair. Neither
// inlined nor cloned (g++'s noipa), so node_sum1 and the apply run the same
// code.
#if defined(__clang__)
#define TITAN_RADIAL_ONE_BODY [[gnu::noinline]]
#else
#define TITAN_RADIAL_ONE_BODY [[gnu::noipa]]
#endif
template <int Order, bool Coupled>
TITAN_RADIAL_ONE_BODY void node_sum4(const Panel &t,
                                     const TreeNode *const *nodes,
                                     const double *const *density, double eps,
                                     double kappa, JcpAccum *out) noexcept {
  constexpr int top = Degrees<Order, Coupled>::top;
  Lanes4 d[3], radial[top + 1], difference[top + 1];
  for (int lane = 0; lane < 4; ++lane) {
    double dl[3], rl[top + 1], fl[top + 1];
    if (nodes[lane] == nullptr ||
        !radial_derivatives<Order, Coupled>(t, *nodes[lane], dl, rl, fl,
                                            kappa)) {
      for (int j = 0; j < 3; ++j)
        dl[j] = 0;
      for (int m = 0; m <= top; ++m)
        rl[m] = fl[m] = 0;
    }
    for (int j = 0; j < 3; ++j)
      d[j][lane] = dl[j];
    for (int m = 0; m <= top; ++m) {
      radial[m][lane] = rl[m];
      difference[m][lane] = Coupled ? fl[m] : 0.0;
    }
  }
  const double nt[3] = {t.nx, t.ny, t.nz};
  Lanes4 phi, phin;
  contract<Order, Coupled>(
      d, radial, difference, nt, eps,
      [&](int kind, int m) {
        return Lanes4{density[kind][m], density[4 + kind][m],
                      density[8 + kind][m], density[12 + kind][m]};
      },
      phi, phin);
  for (int lane = 0; lane < 4; ++lane)
    out[lane] = {phi[lane], phin[lane]};
}
#undef TITAN_RADIAL_ONE_BODY

// One node through node_sum4, in lane 0; the other lanes are empty (their
// moments are read and multiplied by zero).
template <int Order, bool Coupled>
JcpAccum node_sum1(const Panel &t, const TreeNode &c, const double *scalar,
                   const double *normal_x, const double *normal_y,
                   const double *normal_z, double eps, double kappa) noexcept {
  const TreeNode *const nodes[4] = {&c, nullptr, nullptr, nullptr};
  const double *const density[16] = {
      scalar, normal_x, normal_y, normal_z, scalar, normal_x, normal_y, normal_z,
      scalar, normal_x, normal_y, normal_z, scalar, normal_x, normal_y, normal_z};
  JcpAccum out[4];
  node_sum4<Order, Coupled>(t, nodes, density, eps, kappa, out);
  return out[0];
}
#define TITAN_RADIAL_FAST_LANES 1
#define TITAN_RADIAL_ONE_NODE node_sum1
#else
#define TITAN_RADIAL_ONE_NODE node_sum
#endif

#undef TITAN_RADIAL_UNROLL

} // namespace radial_fast

// The fast kernel for a Taylor order (2-5), or nullptr outside that range.
inline RadialJcpFarKernel radial_jcp_far_kernel(int order,
                                                bool coupled) noexcept {
  switch (order) {
  case 2:
    return coupled ? &radial_fast::TITAN_RADIAL_ONE_NODE<2, true>
                   : &radial_fast::TITAN_RADIAL_ONE_NODE<2, false>;
  case 3:
    return coupled ? &radial_fast::TITAN_RADIAL_ONE_NODE<3, true>
                   : &radial_fast::TITAN_RADIAL_ONE_NODE<3, false>;
  case 4:
    return coupled ? &radial_fast::TITAN_RADIAL_ONE_NODE<4, true>
                   : &radial_fast::TITAN_RADIAL_ONE_NODE<4, false>;
  case 5:
    return coupled ? &radial_fast::TITAN_RADIAL_ONE_NODE<5, true>
                   : &radial_fast::TITAN_RADIAL_ONE_NODE<5, false>;
  default:
    return nullptr;
  }
}
#undef TITAN_RADIAL_ONE_NODE

// The four-node kernel for a Taylor order (2-5): per lane equal to
// radial_jcp_far_kernel. nullptr outside that range or without vector types.
inline RadialJcpFarKernel4 radial_jcp_far_kernel4(int order,
                                                  bool coupled) noexcept {
#ifdef TITAN_RADIAL_FAST_LANES
  switch (order) {
  case 2:
    return coupled ? &radial_fast::node_sum4<2, true>
                   : &radial_fast::node_sum4<2, false>;
  case 3:
    return coupled ? &radial_fast::node_sum4<3, true>
                   : &radial_fast::node_sum4<3, false>;
  case 4:
    return coupled ? &radial_fast::node_sum4<4, true>
                   : &radial_fast::node_sum4<4, false>;
  case 5:
    return coupled ? &radial_fast::node_sum4<5, true>
                   : &radial_fast::node_sum4<5, false>;
  default:
    return nullptr;
  }
#else
  (void)order;
  (void)coupled;
  return nullptr;
#endif
}

// The same sum kept apart by kernel channel, for the expansion audits.
inline JcpChannelAccum
radial_jcp_node_channel_sum(const Panel &t, const TreeNode &c,
                            const NodeMoments &scalar,
                            const std::array<NodeMoments, 3> &normal,
                            double eps, double kappa, bool coupled,
                            int order) noexcept {
  JcpChannelAccum out;
  const RadialTaylorTensors w =
      radial_taylor_tensors(t, c, kappa, coupled, order);
  if (!w.valid)
    return out;
  radial_jcp_contract(t, w, scalar, normal, eps, coupled, order,
                      [&](std::size_t channel, bool phin, double value) {
                        (phin ? out.channel[channel].phin
                              : out.channel[channel].phi) += value;
                      });
  return out;
}

} // namespace TitanPBE::Tabi::detail
