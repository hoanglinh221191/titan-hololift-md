// ============================================================================
// TitanTram - TRAM over temperature ensembles (see titan_tram.h)
// ============================================================================
#include "titan_tram.h"

#include "titan_executor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

double log_add_exp(double a, double b) noexcept {
  if (a == -kInf)
    return b;
  if (b == -kInf)
    return a;
  const double m = std::max(a, b);
  return m + std::log1p(std::exp(-std::abs(a - b)));
}

// Streaming log-sum-exp in two passes over the same terms.
template <class Terms> double log_sum_exp(std::size_t count, Terms &&term) {
  double m = -kInf;
  for (std::size_t t = 0; t < count; ++t)
    m = std::max(m, term(t));
  if (m == -kInf)
    return -kInf;
  double s = 0.0;
  for (std::size_t t = 0; t < count; ++t)
    s += std::exp(term(t) - m);
  return m + std::log(s);
}

} // namespace

TitanTram::TitanTram(int n_states, const std::vector<double> &temperatures_K,
                     double target_K)
    : n_states_(n_states) {
  if (n_states <= 0 || temperatures_K.empty())
    throw std::invalid_argument("TRAM needs states and at least one ensemble");
  if (!std::isfinite(target_K) || target_K <= 0.0)
    throw std::invalid_argument("TRAM target temperature must be positive");
  beta_target_ = 1.0 / (kBoltzmannKJ * target_K);
  for (std::size_t k = 0; k < temperatures_K.size(); ++k) {
    const double t = temperatures_K[k];
    if (!std::isfinite(t) || t <= 0.0)
      throw std::invalid_argument("TRAM ensemble temperatures must be positive");
    betas_.push_back(1.0 / (kBoltzmannKJ * t));
    if (target_ensemble_ < 0 && std::abs(t - target_K) <= 1e-6)
      target_ensemble_ = static_cast<int>(k);
  }
  counts_.resize(temperatures_K.size());
}

void TitanTram::add_sample(int ensemble, int state, double energy_kJ,
                           double multiplicity) {
  if (ensemble < 0 || ensemble >= n_ensembles() || state < 0 ||
      state >= n_states_)
    throw std::out_of_range("TRAM sample ensemble/state out of range");
  if (!std::isfinite(energy_kJ))
    throw std::invalid_argument("TRAM sample energy must be finite");
  if (!std::isfinite(multiplicity) || multiplicity <= 0.0)
    throw std::invalid_argument("TRAM sample multiplicity must be positive");
  sample_ensemble_.push_back(ensemble);
  sample_state_.push_back(state);
  sample_energy_.push_back(energy_kJ);
  sample_multiplicity_.push_back(multiplicity);
  solved_ = false;
}

void TitanTram::add_transition(int ensemble, int from, int to, double count) {
  if (ensemble < 0 || ensemble >= n_ensembles() || from < 0 ||
      from >= n_states_ || to < 0 || to >= n_states_)
    throw std::out_of_range("TRAM transition ensemble/state out of range");
  if (!std::isfinite(count) || count < 0.0)
    throw std::invalid_argument("TRAM transition count must be non-negative");
  if (count == 0.0)
    return;
  counts_[static_cast<std::size_t>(ensemble)]
         [static_cast<std::uint64_t>(from) * static_cast<std::uint64_t>(n_states_) +
          static_cast<std::uint64_t>(to)] += count;
  solved_ = false;
}

TitanTram::Report TitanTram::solve(int max_iterations, double tolerance,
                                   TitanExecutor *executor) {
  solved_ = false;
  Report report;
  const int K = n_ensembles();
  const int n = n_states_;
  report.samples = sample_state_.size();
  if (max_iterations <= 0 || !std::isfinite(tolerance) || tolerance <= 0.0) {
    report.message = "TRAM needs a positive iteration limit and tolerance";
    return report;
  }
  if (sample_state_.empty()) {
    report.message = "TRAM has no samples with energies";
    return report;
  }

  // samples grouped by state; sample counts per ensemble and state
  std::vector<std::vector<std::size_t>> by_state(static_cast<std::size_t>(n));
  Eigen::MatrixXd N = Eigen::MatrixXd::Zero(K, n);
  double weight_sum = 0.0, energy_sum = 0.0;
  for (std::size_t x = 0; x < sample_state_.size(); ++x) {
    by_state[static_cast<std::size_t>(sample_state_[x])].push_back(x);
    N(sample_ensemble_[x], sample_state_[x]) += sample_multiplicity_[x];
    weight_sum += sample_multiplicity_[x];
    energy_sum += sample_multiplicity_[x] * sample_energy_[x];
  }
  // A constant energy shift changes every f^k_i of one ensemble by the same
  // amount; it keeps the exponents small.
  energy_shift_ = energy_sum / weight_sum;
  std::vector<unsigned char> has_samples(static_cast<std::size_t>(n), 0);
  for (int i = 0; i < n; ++i)
    if (!by_state[static_cast<std::size_t>(i)].empty()) {
      has_samples[static_cast<std::size_t>(i)] = 1;
      ++report.states_with_samples;
    }

  // symmetric count pairs per ensemble (ordered for determinism), row and
  // column sums
  pairs_.assign(static_cast<std::size_t>(K), {});
  Eigen::MatrixXd row_sum = Eigen::MatrixXd::Zero(K, n);
  Eigen::MatrixXd col_sum = Eigen::MatrixXd::Zero(K, n);
  std::vector<std::vector<int>> adjacency(static_cast<std::size_t>(n));
  for (int k = 0; k < K; ++k) {
    std::map<std::pair<int, int>, double> sym;
    for (const auto &[key, c] : counts_[static_cast<std::size_t>(k)]) {
      const int i = static_cast<int>(key / static_cast<std::uint64_t>(n));
      const int j = static_cast<int>(key % static_cast<std::uint64_t>(n));
      row_sum(k, i) += c;
      col_sum(k, j) += c;
      report.transitions += c;
      sym[{std::min(i, j), std::max(i, j)}] += (i == j ? 2.0 * c : c);
      if (i != j) {
        adjacency[static_cast<std::size_t>(i)].push_back(j);
        adjacency[static_cast<std::size_t>(j)].push_back(i);
      }
    }
    for (const auto &[ij, s] : sym)
      pairs_[static_cast<std::size_t>(k)].push_back({ij.first, ij.second, s});
  }
  if (report.transitions <= 0.0) {
    report.message = "TRAM needs transition counts: without them it cannot "
                     "relate the populations of different states";
    return report;
  }
  for (int k = 0; k < K; ++k)
    for (int i = 0; i < n; ++i) {
      const double slack = 1e-9 * std::max(1.0, N(k, i));
      if (row_sum(k, i) > N(k, i) + slack || col_sum(k, i) > N(k, i) + slack) {
        report.message = "TRAM: ensemble " + std::to_string(k) + " has more "
                         "transitions from or into state " + std::to_string(i) +
                         " than samples of it; every transition endpoint must "
                         "be a sample with an energy";
        return report;
      }
    }
  {
    // the states with samples must form one set connected by transitions
    int first = -1;
    for (int i = 0; i < n && first < 0; ++i)
      if (has_samples[static_cast<std::size_t>(i)])
        first = i;
    std::vector<unsigned char> seen(static_cast<std::size_t>(n), 0);
    std::vector<int> stack{first};
    seen[static_cast<std::size_t>(first)] = 1;
    int reached = 1;
    while (!stack.empty()) {
      const int i = stack.back();
      stack.pop_back();
      for (int j : adjacency[static_cast<std::size_t>(i)])
        if (!seen[static_cast<std::size_t>(j)]) {
          seen[static_cast<std::size_t>(j)] = 1;
          ++reached;
          stack.push_back(j);
        }
    }
    if (reached != report.states_with_samples) {
      report.message = "TRAM: only " + std::to_string(reached) + " of " +
                       std::to_string(report.states_with_samples) +
                       " states with samples are connected by transitions; "
                       "reduce the number of states or the lag";
      return report;
    }
  }

  // reduced bias b_l(x) = (beta_l - beta_target) (E(x) - shift)
  auto bias = [&](int l, std::size_t x) {
    return (betas_[static_cast<std::size_t>(l)] - beta_target_) *
           (sample_energy_[x] - energy_shift_);
  };

  f_ = Eigen::MatrixXd::Zero(K, n);
  log_v_ = Eigen::MatrixXd::Constant(K, n, -kInf);
  for (int k = 0; k < K; ++k)
    for (int i = 0; i < n; ++i) {
      if (!has_samples[static_cast<std::size_t>(i)])
        f_(k, i) = kInf;
      const double row = 0.5 * (row_sum(k, i) + col_sum(k, i));
      if (row > 0.0)
        log_v_(k, i) = std::log(row);
    }
  R_ = N;
  f_target_ = Eigen::VectorXd::Constant(n, kInf);

  // f-update for every state (parallel over states; each state's sums run
  // in a fixed order, so the result does not depend on the thread count)
  Eigen::MatrixXd f_next(K, n);
  auto update_state = [&](int i) {
    const auto &xs = by_state[static_cast<std::size_t>(i)];
    if (xs.empty()) {
      f_next.col(i).setConstant(kInf);
      f_target_[i] = kInf;
      return;
    }
    std::vector<double> denominator(xs.size());
    for (std::size_t t = 0; t < xs.size(); ++t) {
      const std::size_t x = xs[t];
      denominator[t] = log_sum_exp(static_cast<std::size_t>(K), [&](std::size_t l) {
        const int ll = static_cast<int>(l);
        return R_(ll, i) > 0.0 ? std::log(R_(ll, i)) + f_(ll, i) - bias(ll, x)
                               : -kInf;
      });
    }
    for (int k = 0; k < K; ++k)
      f_next(k, i) = -log_sum_exp(xs.size(), [&](std::size_t t) {
        const std::size_t x = xs[t];
        return std::log(sample_multiplicity_[x]) - bias(k, x) - denominator[t];
      });
    f_target_[i] = -log_sum_exp(xs.size(), [&](std::size_t t) {
      return std::log(sample_multiplicity_[xs[t]]) - denominator[t];
    });
  };
  auto update_all_states = [&]() {
    if (executor != nullptr && executor->capacity() > 1 && n > 1) {
      executor->parallel_for(
          static_cast<std::size_t>(n), executor->capacity(), 1,
          [&](std::size_t, std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i)
              update_state(static_cast<int>(i));
          });
    } else {
      for (int i = 0; i < n; ++i)
        update_state(i);
    }
    // gauge: the target distribution sums to one
    const double g = log_sum_exp(static_cast<std::size_t>(n), [&](std::size_t i) {
      return -f_target_[static_cast<Eigen::Index>(i)];
    });
    for (int i = 0; i < n; ++i) {
      if (!std::isfinite(f_target_[i]))
        continue;
      f_target_[i] += g;
      for (int k = 0; k < K; ++k)
        if (std::isfinite(f_next(k, i)))
          f_next(k, i) += g;
    }
  };

  Eigen::MatrixXd log_v_next(K, n);
  for (int it = 1; it <= max_iterations; ++it) {
    // Lagrange multipliers
    log_v_next.setConstant(-kInf);
    for (int k = 0; k < K; ++k) {
      const auto &pairs = pairs_[static_cast<std::size_t>(k)];
      std::vector<std::vector<double>> terms(static_cast<std::size_t>(n));
      for (const Pair &p : pairs) {
        const double fi = f_(k, p.i), fj = f_(k, p.j);
        const double li = log_v_(k, p.i), lj = log_v_(k, p.j);
        if (p.i == p.j) {
          terms[static_cast<std::size_t>(p.i)].push_back(std::log(0.5 * p.s));
          continue;
        }
        const double d = log_add_exp(li - fj, lj - fi);
        terms[static_cast<std::size_t>(p.i)].push_back(std::log(p.s) + li - fj - d);
        terms[static_cast<std::size_t>(p.j)].push_back(std::log(p.s) + lj - fi - d);
      }
      for (int i = 0; i < n; ++i) {
        const auto &t = terms[static_cast<std::size_t>(i)];
        if (!t.empty())
          log_v_next(k, i) = log_sum_exp(t.size(), [&](std::size_t q) { return t[q]; });
      }
    }
    // effective sample counts with the new multipliers
    for (int k = 0; k < K; ++k) {
      for (int i = 0; i < n; ++i)
        R_(k, i) = N(k, i) - col_sum(k, i);
      for (const Pair &p : pairs_[static_cast<std::size_t>(k)]) {
        if (p.i == p.j) {
          R_(k, p.i) += 0.5 * p.s;
          continue;
        }
        const double fi = f_(k, p.i), fj = f_(k, p.j);
        const double li = log_v_next(k, p.i), lj = log_v_next(k, p.j);
        // v_j pi_i / (v_i pi_j + v_j pi_i) and its mirror
        const double a = (li - fj) - (lj - fi);
        const double to_i = 1.0 / (1.0 + std::exp(a));
        R_(k, p.i) += p.s * to_i;
        R_(k, p.j) += p.s * (1.0 - to_i);
      }
      for (int i = 0; i < n; ++i)
        R_(k, i) = std::max(0.0, R_(k, i));
    }
    update_all_states();

    double change = 0.0;
    for (int k = 0; k < K; ++k)
      for (int i = 0; i < n; ++i) {
        if (std::isfinite(f_next(k, i)) && std::isfinite(f_(k, i)))
          change = std::max(change, std::abs(f_next(k, i) - f_(k, i)));
        if (std::isfinite(log_v_next(k, i)) && std::isfinite(log_v_(k, i)))
          change = std::max(change, std::abs(log_v_next(k, i) - log_v_(k, i)));
      }
    f_ = f_next;
    log_v_ = log_v_next;
    report.iterations = it;
    report.max_change = change;
    if (!std::isfinite(change)) {
      report.message = "TRAM iteration produced non-finite free energies";
      return report;
    }
    if (change < tolerance) {
      report.converged = true;
      break;
    }
  }
  if (!report.converged) {
    report.message = "TRAM did not reach the tolerance within the iteration limit";
    return report;
  }

  // final consistent target distribution from (R, f)
  update_all_states();
  target_pi_ = Eigen::VectorXd::Zero(n);
  for (int i = 0; i < n; ++i)
    if (std::isfinite(f_target_[i]))
      target_pi_[i] = std::exp(-f_target_[i]);
  target_pi_ /= target_pi_.sum();
  for (int k = 0; k < K; ++k) {
    const Eigen::MatrixXd p = transition_matrix(k);
    for (int i = 0; i < n; ++i)
      if (std::isfinite(p(i, 0)))
        report.max_row_sum_error =
            std::max(report.max_row_sum_error, std::abs(p.row(i).sum() - 1.0));
  }
  solved_ = true;
  return report;
}

Eigen::VectorXd TitanTram::target_free_energy_kJ() const {
  Eigen::VectorXd out = Eigen::VectorXd::Constant(n_states_, kInf);
  if (target_pi_.size() != n_states_)
    return out;
  const double kT = 1.0 / beta_target_;
  const double p_max = target_pi_.maxCoeff();
  for (int i = 0; i < n_states_; ++i)
    if (target_pi_[i] > 0.0)
      out[i] = -kT * std::log(target_pi_[i] / p_max);
  return out;
}

Eigen::MatrixXd TitanTram::transition_matrix(int ensemble) const {
  if (ensemble < 0 || ensemble >= n_ensembles() || f_.rows() != n_ensembles())
    throw std::out_of_range("TRAM transition matrix ensemble out of range");
  const int k = ensemble;
  Eigen::MatrixXd p = Eigen::MatrixXd::Constant(
      n_states_, n_states_, std::numeric_limits<double>::quiet_NaN());
  for (int i = 0; i < n_states_; ++i)
    if (std::isfinite(log_v_(k, i)))
      p.row(i).setZero();
  for (const Pair &q : pairs_[static_cast<std::size_t>(k)]) {
    const double fi = f_(k, q.i), fj = f_(k, q.j);
    const double li = log_v_(k, q.i), lj = log_v_(k, q.j);
    if (q.i == q.j) {
      p(q.i, q.i) = 0.5 * q.s / std::exp(li);
      continue;
    }
    const double d = log_add_exp(li - fj, lj - fi);
    p(q.i, q.j) = q.s * std::exp(-fj - d);
    p(q.j, q.i) = q.s * std::exp(-fi - d);
  }
  return p;
}

std::vector<double> TitanTram::sample_weights() const {
  std::vector<double> w(sample_state_.size(), 0.0);
  if (!solved_)
    return w;
  const int K = n_ensembles();
  double total = 0.0;
  for (std::size_t x = 0; x < sample_state_.size(); ++x) {
    const int i = sample_state_[x];
    const double d = log_sum_exp(static_cast<std::size_t>(K), [&](std::size_t l) {
      const int ll = static_cast<int>(l);
      return R_(ll, i) > 0.0
                 ? std::log(R_(ll, i)) + f_(ll, i) -
                       (betas_[l] - beta_target_) * (sample_energy_[x] - energy_shift_)
                 : -kInf;
    });
    w[x] = sample_multiplicity_[x] * std::exp(-d);
    total += w[x];
  }
  if (total > 0.0)
    for (double &value : w)
      value /= total;
  return w;
}

std::vector<std::string> TitanTram::references() {
  return {
      "Wu, Paul, Wehmeyer, Noe, Multiensemble Markov models of molecular "
      "thermodynamics and kinetics, Proc. Natl. Acad. Sci. USA 113, E3221 "
      "(2016), doi:10.1073/pnas.1525092113 (TRAM, the estimator used here)",
      "Wu, Mey, Rosta, Noe, Statistically optimal analysis of "
      "state-discretized trajectory data from multiple thermodynamic states, "
      "J. Chem. Phys. 141, 214106 (2014), doi:10.1063/1.4902240 (dTRAM)",
      "Shirts, Chodera, Statistically optimal analysis of samples from "
      "multiple equilibrium states, J. Chem. Phys. 129, 124105 (2008), "
      "doi:10.1063/1.2978177 (MBAR, the binless reweighting in TRAM)",
      "Stelzl, Hummer, Kinetics from replica exchange molecular dynamics "
      "simulations, J. Chem. Theory Comput. 13, 3927 (2017), "
      "doi:10.1021/acs.jctc.7b00372 (transitions between exchanges in REMD)"};
}
