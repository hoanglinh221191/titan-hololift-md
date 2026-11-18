#include "arguments.h"
#include "folder.h"
#include "titan_interaction_entropy.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <vector>


using namespace std;
double entropy_duan_zhang(std::span<const double> E_e,
                          std::span<const double> E_v, arguments &Arguments) {
  if (Arguments.entropy == false || E_e.empty())
    return 1;

  const double e_con{1.38064e-3 * Arguments.gbsa_temperature * 6.02214 / Arguments.eu_par};
  TitanEntropy::InteractionEntropyOptions options;
  options.rt_energy = e_con;
  // KIN-22: no trimming by default (was a fixed 2-sigma cut, which removes
  // the tail that decides the exponential average: Gaussian sigma = 5 kJ/mol
  // gave 3.4 instead of 5.0 kJ/mol)
  options.filter_sigma = Arguments.entropy_filter_sigma;
  options.min_samples = 1;
  const TitanEntropy::InteractionEntropyResult entropy =
      TitanEntropy::compute_interaction_entropy(E_e, E_v, options);
  if (entropy.status == "no_filtered_samples") {
    return 0.0; // Safe fallback to prevent Segfault if no frames satisfy the
                // sigma filter
  }
  if (entropy.status != "ok")
    return 0.0;
  return entropy.dimensionless;
}

int print_average_entropy_duan_zhange_time(vector<double> &E_e,
                                           vector<double> &E_v,
                                           vector<double> &time_of_snap,
                                           fstream &output,
                                           arguments &Arguments) {
  output << "# Time (" << Arguments.tu << ")     Interaction entropy ("
         << Arguments.eu << ")\n";

  size_t SIZE{E_e.size() > time_of_snap.size() ? time_of_snap.size()
                                               : E_e.size()};
  SIZE = std::min(SIZE, E_v.size());
  const double kt_out{1.38064e-3 * Arguments.gbsa_temperature * 6.02214 /
                      Arguments.eu_par};
  if (Arguments.entropy == false || SIZE == 0)
    return 0;

  if (Arguments.entropy_filter_sigma > 0.0) {
    // trimming depends on the mean and sigma of each prefix: recompute
    // (O(N^2), only when -entropy-filter-sigma > 0 is requested)
    for (size_t i{}; i < SIZE; ++i) {
      std::span<const double> ee_span(E_e.data(), i + 1);
      std::span<const double> ev_span(E_v.data(), i + 1);
      const double entropy{entropy_duan_zhang(ee_span, ev_span, Arguments)};
      output << setw(7) << left << time_of_snap[i] << right << setw(14)
             << kt_out * entropy << '\n';
    }
    return 0;
  }

  // KIN-22: the estimate on every prefix [0, i] from running sums, O(N):
  //   -TdS(n) = kT [ln sum_j exp(x_j) - ln n - mean_n(x)],  x_j = E_j / kT
  // (x taken relative to the first sample; the log-sum-exp kept as a running
  // maximum and a rescaled sum). Same estimator as entropy_duan_zhang on the
  // prefix; a non-finite sample makes every later prefix 0, as before.
  const double x0 = (E_e[0] + E_v[0]) / kt_out;
  double running_max = -std::numeric_limits<double>::infinity();
  long double scaled_sum = 0.0L; // sum_j exp(y_j - running_max)
  long double y_sum = 0.0L;
  bool nonfinite = false;
  for (size_t i{}; i < SIZE; ++i) {
    const double y = (E_e[i] + E_v[i]) / kt_out - x0;
    double dimensionless = 0.0;
    if (!std::isfinite(y))
      nonfinite = true;
    if (!nonfinite) {
      if (y > running_max) {
        scaled_sum = scaled_sum * std::exp(static_cast<long double>(running_max - y)) + 1.0L;
        running_max = y;
      } else {
        scaled_sum += std::exp(static_cast<long double>(y - running_max));
      }
      y_sum += y;
      const long double n = static_cast<long double>(i + 1);
      dimensionless = static_cast<double>(running_max + std::log(scaled_sum) -
                                          std::log(n) - y_sum / n);
    }
    output << setw(7) << left << time_of_snap[i] << right << setw(14)
           << kt_out * dimensionless << '\n';
  }
  return 0;
}

namespace {
// KIN-22: the settings of the estimate, written into entropy_result.dat
std::string entropy_settings_line(const arguments &Arguments) {
  std::ostringstream line;
  line << "Settings: T = " << Arguments.gbsa_temperature
       << " K, filter_sigma = " << Arguments.entropy_filter_sigma;
  if (Arguments.entropy_filter_sigma > 0.0)
    line << " (samples farther than " << Arguments.entropy_filter_sigma
         << " standard deviations from the mean removed)";
  else
    line << " (no trimming)";
  return line.str();
}
} // namespace

int impl_entropy_duan_zhang(folders &paths, vector<vector<double>> &E_e_vector,
                            vector<vector<double>> &E_v_vector,
                            vector<double> &time_of_snap, arguments &Arguments,
                            vector<double> &E_e, vector<double> &E_v) {
  if (Arguments.entropy == false || E_e_vector.empty() || E_e_vector[0].empty())
    return 1;

  fstream out_entropy("entropy_result.dat", ios::out),
      out_ave_entropy_time("entropy_result_time.dat", ios::out);
  print_average_entropy_duan_zhange_time(E_e, E_v, time_of_snap,
                                         out_ave_entropy_time, Arguments);

  out_entropy
      << "Average entropy (from time dependence of mean entropy values): "
      << 1.38064e-3 * Arguments.gbsa_temperature * 6.02214 * entropy_duan_zhang(E_e, E_v, Arguments) /
             Arguments.eu_par
      << " " << Arguments.eu << "\n"
      << "(calculated from average energy of " << paths.get_size()
      << " trajectories)\n"
      << entropy_settings_line(Arguments) << "\n"
      << "-------------------------------------o0o-----------------------------"
         "--------\n";

  vector<double> entropy_values;

  for (size_t i{}; i < paths.get_size(); ++i) {
    vector<double> e_ele, e_vdw;

    for (size_t j{}; j < E_v_vector.size(); ++j)
      if (!E_e_vector[j].empty()) {
        if (E_e_vector[j].size() <= i ||
            (fabs(E_e_vector[j][i]) < 10e-9 && fabs(E_v_vector[j][i]) < 10e-9))
          continue;

        e_ele.emplace_back(E_e_vector[j][i]);
        e_vdw.emplace_back(E_v_vector[j][i]);
      }

    const double exp_beta_deltaE{entropy_duan_zhang(e_ele, e_vdw, Arguments)};
    double entropy_1traj{1.38064e-3 * Arguments.gbsa_temperature * 6.02214 * exp_beta_deltaE /
                         Arguments.eu_par};
    entropy_values.emplace_back(entropy_1traj);

    out_entropy << setw(25) << paths.get_path(i)
                << ": Interaction entropy (-T*DeltaS)= " << entropy_1traj << " "
                << Arguments.eu << "\n";
    out_entropy << "-------------------------------------o0o-------------------"
                   "------------------\n";
    if (paths.get_size() > 1 || paths.get_path(0) != ".") {
      fstream out_per_file(paths.get_path(i) + "/entropy_result.dat", ios::out);
      out_per_file << "Interaction entropy (-T*DeltaS)= " << entropy_1traj
                   << " " << Arguments.eu << "\n";
      out_per_file << entropy_settings_line(Arguments) << "\n";
      out_per_file << "=-------------------------------------------------------"
                      "----------------------------=\n";
      out_per_file
          << "Please cite \"Duan, L., Liu, X., &Zhang, J.Z. (2016).Interaction "
             "Entropy : \nA New Paradigm for Highly Efficient and Reliable "
             "Computation of ProteinLigand Binding Free Energy.\nJournal of "
             "the American Chemical Society, 138(17), 5722 - 5728.\" \n"
          << "and \"Sun, Z., Yan, Y. N., Yang, M., & Zhang, J. Z. (2017). "
             "Interaction entropy for protein-protein binding. The Journal of "
             "chemical physics, 146(12), 124124.\" "
          << "When you use this result!\n";
      out_per_file.close();

      fstream out_entropy_time(paths.get_path(i) + "/out_entropy_time.dat",
                               ios::out);
      print_average_entropy_duan_zhange_time(e_ele, e_vdw, time_of_snap,
                                             out_entropy_time, Arguments);
    }
  }

  double mean_entropy{};
  for (auto &en : entropy_values)
    mean_entropy += en;

  out_entropy << "------------------------o0o------------------------\n";
  out_entropy << "MEAN entropy: " << mean_entropy / entropy_values.size() << " "
              << Arguments.eu << '\n';
  out_entropy << "DONE!\n";
  out_entropy << "=------------------------------------------------------------"
                 "-----------------------=\n";
  out_entropy
      << "Please cite \"Duan, L., Liu, X., &Zhang, J.Z. (2016).Interaction "
         "Entropy : \nA New Paradigm for Highly Efficient and Reliable "
         "Computation of ProteinLigand Binding Free Energy.\nJournal of the "
         "American Chemical Society, 138(17), 5722 - 5728.\" \n"
      << "and \"Sun, Z., Yan, Y. N., Yang, M., & Zhang, J. Z. (2017). "
         "Interaction entropy for protein-protein binding. The Journal of "
         "chemical physics, 146(12), 124124.\" "
      << "When you use this result!\n";
  out_entropy.close();
  return 0;
}
