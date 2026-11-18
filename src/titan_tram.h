// ============================================================================
// TitanTram - transition-based reweighting analysis method (TRAM) over a set
// of temperature ensembles (KIN-13).
//
// Every sample carries its own energy E(x): the potential energy, or U + P V
// in NPT with one pressure for all ensembles. The reduced bias of ensemble l
// relative to the target temperature T_t is b_l(x) = (beta_l - beta_t) E(x),
// so reweighting between temperatures is binless: it uses each sample's
// energy, not a state-averaged one. Transitions are counted within one
// ensemble only (for replica exchange: within one exchange-free window).
//
// Self-consistent equations (Wu, Paul, Wehmeyer, Noe, PNAS 113, E3221
// (2016)), with pi^k_i = exp(-f^k_i), c^k the lag-time transition counts of
// ensemble k, s^k_ij = c^k_ij + c^k_ji and N^k_i the samples of state i in k:
//   v^k_i = sum_j s^k_ij v^k_i pi^k_j / (v^k_i pi^k_j + v^k_j pi^k_i)
//   R^k_i = sum_j s^k_ij v^k_j pi^k_i / (v^k_i pi^k_j + v^k_j pi^k_i)
//           + N^k_i - sum_j c^k_ji
//   f^k_i = -ln sum_{x in i} exp(-b_k(x)) / sum_l R^l_i exp(f^l_i - b_l(x))
// The reversible transition matrix of ensemble k is
//   p^k_ij = s^k_ij pi^k_j / (v^k_i pi^k_j + v^k_j pi^k_i),
// and the unbiased (target) weight of a sample is
//   mu(x) = 1 / sum_l R^l_i exp(f^l_i - b_l(x)).
// States are coupled through transitions only: without transitions TRAM
// does not fix the relative populations of states (it does not assume
// global equilibrium within an ensemble), so the states with samples must be
// connected by transitions.
// ============================================================================
#pragma once

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class TitanExecutor;

class TitanTram {
public:
  static constexpr double kBoltzmannKJ = 0.008314463; // kJ/(mol K)

  TitanTram(int n_states, const std::vector<double> &temperatures_K,
            double target_K);

  // One sample (frame) of `state` drawn in `ensemble`, with its energy in
  // kJ/mol; `multiplicity` > 0 counts identical samples (tests use exact
  // populations this way).
  void add_sample(int ensemble, int state, double energy_kJ,
                  double multiplicity = 1.0);
  // Transition from -> to at the lag time inside one ensemble. Its end
  // frame must also be a sample of that ensemble.
  void add_transition(int ensemble, int from, int to, double count = 1.0);

  struct Report {
    bool converged{false};
    int iterations{0};
    double max_change{0.0};
    double max_row_sum_error{0.0};
    std::size_t samples{0};
    double transitions{0.0};
    int states_with_samples{0};
    std::string message;
  };
  Report solve(int max_iterations, double tolerance,
               TitanExecutor *executor = nullptr);

  [[nodiscard]] bool solved() const noexcept { return solved_; }
  [[nodiscard]] int n_states() const noexcept { return n_states_; }
  [[nodiscard]] int n_ensembles() const noexcept {
    return static_cast<int>(betas_.size());
  }
  // Index of the ensemble at the target temperature, -1 if none.
  [[nodiscard]] int target_ensemble() const noexcept { return target_ensemble_; }

  // Stationary distribution at the target temperature (sums to 1).
  [[nodiscard]] const Eigen::VectorXd &target_stationary() const {
    return target_pi_;
  }
  // -kT ln pi at the target, minimum 0, +inf for states without samples.
  [[nodiscard]] Eigen::VectorXd target_free_energy_kJ() const;
  // f^k_i in kT units (rows: ensembles), gauge: the target pi sums to 1.
  [[nodiscard]] const Eigen::MatrixXd &ensemble_free_energies() const {
    return f_;
  }
  // Reversible transition matrix of one ensemble. Rows of states without
  // transitions in that ensemble are NaN.
  [[nodiscard]] Eigen::MatrixXd transition_matrix(int ensemble) const;
  // Target weights mu(x) of the samples, in insertion order, summing to 1.
  [[nodiscard]] std::vector<double> sample_weights() const;

  // Citation lines for output headers ("# " is prepended by the writers).
  static std::vector<std::string> references();

private:
  struct Pair {
    int i;
    int j;
    double s; // c_ij + c_ji (2 c_ii on the diagonal)
  };

  int n_states_{0};
  std::vector<double> betas_;
  double beta_target_{0.0};
  int target_ensemble_{-1};

  std::vector<int> sample_ensemble_;
  std::vector<int> sample_state_;
  std::vector<double> sample_energy_;
  std::vector<double> sample_multiplicity_;
  // per ensemble: from * n_states + to -> count
  std::vector<std::unordered_map<std::uint64_t, double>> counts_;

  // solution
  bool solved_{false};
  double energy_shift_{0.0};
  Eigen::MatrixXd f_;     // K x n, kT
  Eigen::MatrixXd log_v_; // K x n
  Eigen::MatrixXd R_;     // K x n
  Eigen::VectorXd f_target_;
  Eigen::VectorXd target_pi_;
  std::vector<std::vector<Pair>> pairs_;
};
