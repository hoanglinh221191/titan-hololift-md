#pragma once
// Implied timescales (ITS) from the eigenvalues of a transition matrix: one
// convention for every ITS path (KIN-06): TitanMSM::compute_implied_timescales,
// msm_timescales.dat, the multi-lag ITS scan and the validation bootstrap.
//
// Order: the eigenvalue nearest to +1 is rank 0 (the stationary mode); the
// others follow by |lambda| descending (ties: larger real part first, then
// index). Rank k >= 1 gives t_k = -lag_time / ln|lambda_k|, or NaN when
// lambda_k is within 1e-8 of 1 (a second stationary mode: disconnected sets),
// |lambda_k| within 1e-8 of 1 or above (e.g. -1 of a periodic chain, which a
// symmetric eigensolver returns as -1 + 2e-16), |Im lambda_k| > 1e-6
// (oscillatory mode) or |lambda_k| < 1e-15. A NaN keeps its rank, so the
// k-th timescale is the k-th process in every path.
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace titan_its {

constexpr double kUnitEigenvalueTol = 1e-8;
constexpr double kComplexTol = 1e-6;
constexpr double kZeroTol = 1e-15;

// Number of eigenvalues within 1e-8 of 1 (distance in the complex plane).
// More than one: the matrix has more than one stationary distribution.
inline int count_unit_eigenvalues(const Eigen::VectorXcd &evals) {
  int count = 0;
  for (Eigen::Index i = 0; i < evals.size(); ++i)
    if (std::abs(evals[i] - std::complex<double>(1.0, 0.0)) <= kUnitEigenvalueTol)
      ++count;
  return count;
}

// Eigenvalue indices in ITS rank order (see above).
inline std::vector<int> its_order(const Eigen::VectorXcd &evals) {
  const int n = static_cast<int>(evals.size());
  std::vector<int> idx(static_cast<std::size_t>(n));
  std::iota(idx.begin(), idx.end(), 0);
  if (n == 0)
    return idx;
  const std::complex<double> one(1.0, 0.0);
  const auto stationary = std::min_element(idx.begin(), idx.end(), [&](int a, int b) {
    const double da = std::abs(evals[a] - one), db = std::abs(evals[b] - one);
    return da < db || (da == db && a < b);
  });
  std::iter_swap(idx.begin(), stationary);
  std::sort(idx.begin() + 1, idx.end(), [&](int a, int b) {
    const double ma = std::abs(evals[a]), mb = std::abs(evals[b]);
    if (ma != mb)
      return ma > mb;
    if (evals[a].real() != evals[b].real())
      return evals[a].real() > evals[b].real();
    return a < b;
  });
  return idx;
}

// ITS of one non-stationary eigenvalue (NaN in the cases listed above).
inline double implied_timescale(std::complex<double> lambda, double lag_time) {
  const double modulus = std::abs(lambda);
  if (std::abs(lambda - std::complex<double>(1.0, 0.0)) <= kUnitEigenvalueTol ||
      modulus >= 1.0 - kUnitEigenvalueTol ||
      std::abs(lambda.imag()) > kComplexTol ||
      modulus < kZeroTol)
    return std::numeric_limits<double>::quiet_NaN();
  return -lag_time / std::log(modulus);
}

// Timescales of ranks 1 .. n_timescales (fewer when the matrix has fewer
// eigenvalues; pad = true fills up to n_timescales with NaN).
inline std::vector<double> implied_timescales(const Eigen::VectorXcd &evals,
                                              double lag_time, int n_timescales,
                                              bool pad = false) {
  std::vector<double> out;
  if (n_timescales <= 0)
    return out;
  const auto order = its_order(evals);
  for (std::size_t rank = 1;
       rank < order.size() && static_cast<int>(out.size()) < n_timescales; ++rank)
    out.push_back(implied_timescale(evals[order[rank]], lag_time));
  while (pad && static_cast<int>(out.size()) < n_timescales)
    out.push_back(std::numeric_limits<double>::quiet_NaN());
  return out;
}

} // namespace titan_its
