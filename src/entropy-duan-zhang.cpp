#include "arguments.h"
#include "folder.h"
#include "titan_interaction_entropy.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <span>
#include <vector>


using namespace std;
double entropy_duan_zhang(std::span<const double> E_e,
                          std::span<const double> E_v, arguments &Arguments) {
  if (Arguments.entropy == false || E_e.empty())
    return 1;

  const double e_con{1.38064 * 0.3 * 6.02214 / Arguments.eu_par};
  TitanEntropy::InteractionEntropyOptions options;
  options.rt_energy = e_con;
  options.filter_sigma = 2.0;
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

  const size_t SIZE{E_e.size() > time_of_snap.size() ? time_of_snap.size()
                                                     : E_e.size()};
  for (size_t i{}; i < SIZE; ++i) {
    std::span<const double> ee_span(E_e.data(), i + 1);
    std::span<const double> ev_span(E_v.data(), i + 1);

    double entropy{entropy_duan_zhang(ee_span, ev_span, Arguments)};
    output << setw(7) << left << time_of_snap[i] << right << setw(14)
           << 1.38064 * 0.3 * 6.02214 * entropy / Arguments.eu_par << '\n';
  }
  return 0;
}

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
      << 1.38064 * 0.3 * 6.02214 * entropy_duan_zhang(E_e, E_v, Arguments) /
             Arguments.eu_par
      << " " << Arguments.eu << "\n"
      << "(calculated from average energy of " << paths.get_size()
      << " trajectories)\n"
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
    double entropy_1traj{1.38064 * 0.3 * 6.02214 * exp_beta_deltaE /
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
