// KIN-08 / X-06: seeded random start blocks for the iterative eigensolvers
// (partial ITS subspace iteration, partial PCA subspace iteration,
// randomized TICA range finder).
//
// Eigen::MatrixXd::Random draws from the C library rand(): its state is
// process-wide, unseeded, and shared by the threads of the ITS lag scan, so
// the start block (and the last digits of the converged eigenvalues) depended
// on what ran before and on thread timing. Each call here builds its own
// std::mt19937_64 from the seed and fills the matrix column by column, and
// the uniform doubles come from the raw 64-bit output (53 bits), so the
// block is the same for a seed on every platform and in every thread. The
// Gaussian block uses Box-Muller (std::normal_distribution is
// implementation-defined); its last bit may follow the platform's log/cos.
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <random>

namespace titan_random {

// seed of -msm-seed (default 42), shared by the MSM/PCA/TICA solvers
inline std::uint64_t solver_seed(long long msm_seed) {
  return static_cast<std::uint64_t>(msm_seed);
}

// uniform double in [0, 1) from the top 53 bits of one 64-bit draw
inline double unit_uniform(std::mt19937_64 &rng) {
  return static_cast<double>(rng() >> 11) * 0x1.0p-53;
}

// rows x cols, entries uniform in [-1, 1) (the range of Eigen's Random)
inline Eigen::MatrixXd uniform_block(Eigen::Index rows, Eigen::Index cols,
                                     std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  Eigen::MatrixXd m(rows, cols);
  for (Eigen::Index c = 0; c < cols; ++c)
    for (Eigen::Index r = 0; r < rows; ++r)
      m(r, c) = 2.0 * unit_uniform(rng) - 1.0;
  return m;
}

// rows x cols, standard normal entries (Box-Muller, both values used)
inline Eigen::MatrixXd gaussian_block(Eigen::Index rows, Eigen::Index cols,
                                      std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  constexpr double kTwoPi = 6.283185307179586476925;
  Eigen::MatrixXd m(rows, cols);
  double spare = 0.0;
  bool have_spare = false;
  for (Eigen::Index c = 0; c < cols; ++c)
    for (Eigen::Index r = 0; r < rows; ++r) {
      if (have_spare) {
        m(r, c) = spare;
        have_spare = false;
        continue;
      }
      const double u1 = 1.0 - unit_uniform(rng); // (0, 1]
      const double u2 = unit_uniform(rng);
      const double radius = std::sqrt(-2.0 * std::log(u1));
      m(r, c) = radius * std::cos(kTwoPi * u2);
      spare = radius * std::sin(kTwoPi * u2);
      have_spare = true;
    }
  return m;
}

} // namespace titan_random
