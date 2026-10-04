#include "arguments.h"
#include "folder.h"
#include "uti_func.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;
int print_energy_time(vector<double> &E_e, vector<double> &E_v,
                      vector<double> &time_of_snap,
                      vector<vector<double>> &E_e_vector,
                      vector<vector<double>> &E_v_vector, arguments &Arguments,
                      int input_name);

int out(vector<double> &E_e, vector<double> &E_v, vector<double> &time_of_snap,
        fstream &cal_out, arguments &Arguments, int input_name,
        vector<vector<double>> &E_e_vector, vector<vector<double>> &E_v_vector,
        vector<size_t> &size_snap) {
  if (Arguments.eu == "kcal/mol") {
    for (size_t i{}; i < E_e.size(); ++i) {
      E_e[i] /= Arguments.eu_par;
      E_v[i] /= Arguments.eu_par;
    }
    for (size_t i{}; i < E_e_vector.size(); ++i)
      for (size_t j{}; j < E_e_vector[i].size(); ++j) {
        E_e_vector[i][j] /= Arguments.eu_par;
        E_v_vector[i][j] /= Arguments.eu_par;
      }
  }

  double elec_total{}, vdw_total{}, SD_e_temp{}, SD_v_temp{};
  for (size_t i{}; i < E_e.size(); i++) {
    elec_total = elec_total + E_e[i] / size_snap[i];
    vdw_total = vdw_total + E_v[i] / size_snap[i];
  }
  if (E_e.empty()) {
    cout << "DEBUG: E_e is empty in out()\n";
    return 1;
  }
  double elec_mean{elec_total / E_e.size()};
  double vdw_mean{vdw_total / E_v.size()};
  for (int i = 0; i < E_e.size(); i++) {
    SD_e_temp += square(E_e[i] / size_snap[i] - elec_mean);
    SD_v_temp += square(E_v[i] / size_snap[i] - vdw_mean);
  }
  SD_e_temp = sqrt(SD_e_temp / ((E_e.size())));
  SD_v_temp = sqrt(SD_v_temp / ((E_v.size())));

  if ((Arguments.flag_pattern_input == true ||
       Arguments.flag_listfile == true || Arguments.flag_all_folders == true) &&
      input_name != Arguments.b_input) {
    for (size_t i{}; i < E_e.size(); ++i) {
      E_e[i] /= size_snap[i];
      E_v[i] /= size_snap[i];
    }
    cal_out << "Energy is averaged from " << input_name - Arguments.b_input + 1
            << " files.\n";
    cout << "Energy is averaged from " << input_name - Arguments.b_input + 1
         << " files.\n";
    cal_out << "Average Electrostatic energy: " << elec_mean << " "
            << Arguments.eu
            << ". Standard error: " << SD_e_temp / sqrt(E_e.size()) << " "
            << Arguments.eu << ". Confidence interval (95%): "
            << 1.96 * SD_e_temp / sqrt(E_e.size()) << " " << Arguments.eu
            << '\n';
    cal_out << "Average Van der waals energy: " << vdw_mean << " "
            << Arguments.eu
            << ". Standard error: " << SD_v_temp / sqrt(E_v.size()) << " "
            << Arguments.eu << ". Confidence interval (95%): "
            << 1.96 * SD_v_temp / sqrt(E_v.size()) << " " << Arguments.eu
            << '\n';
    cal_out << "Average total energy: " << elec_mean + vdw_mean << " "
            << Arguments.eu << ". Standard error: "
            << SD_e_temp / sqrt(E_e.size()) + SD_v_temp / sqrt(E_v.size())
            << " " << Arguments.eu << ". Confidence interval (95%): "
            << 1.96 * SD_e_temp / sqrt(E_e.size()) +
                   1.96 * SD_v_temp / sqrt(E_v.size())
            << " " << Arguments.eu << '\n';
  } else {
    cal_out << "Average Electrostatic energy: " << elec_mean << " "
            << Arguments.eu << ". Standard deviation: " << SD_e_temp << " "
            << Arguments.eu << ". Confidence interval (95%): "
            << 1.96 * SD_e_temp / sqrt(E_e.size()) << " " << Arguments.eu
            << "\n";
    cal_out << "Average Van der waals energy: " << vdw_mean << " "
            << Arguments.eu << ". Standard deviation: " << SD_v_temp << " "
            << Arguments.eu << ". Confidence interval (95%): "
            << 1.96 * SD_v_temp / sqrt(E_v.size()) << " " << Arguments.eu
            << "\n";
    cal_out << "Average total energy: " << elec_mean + vdw_mean << " "
            << Arguments.eu << ". Standard error: "
            << SD_e_temp / sqrt(E_e.size()) + SD_v_temp / sqrt(E_v.size())
            << " " << Arguments.eu << ". Confidence interval (95%): "
            << 1.96 * SD_e_temp / sqrt(E_e.size()) +
                   1.96 * SD_v_temp / sqrt(E_v.size())
            << " " << Arguments.eu << '\n';
  }
  cal_out.close();
  cout << "Average electrostatic energy: " << elec_mean << " " << Arguments.eu
       << "\n";
  cout << "Average Van der waals energy: " << vdw_mean << " " << Arguments.eu
       << "\n";
  fstream out_xvg;
  out_xvg.open("cal.xvg", ios::out);
  out_xvg << "@    title \"Energy\"\n";
  out_xvg << "@    xaxis label \"Time (" << Arguments.tu << ")\"\n";
  out_xvg << "@    yaxis label \"Energy (" << Arguments.eu << ")\"\n";
  out_xvg << "@ s0 legend \"Time\"\n";
  out_xvg << "@ s1 legend \"Electrostatic\"\n";
  out_xvg << "@ s2 legend \"Van der Waal\"\n";
  out_xvg << "@ s3 legend \"Total energy\"\n";
  out_xvg << setw(14) << "#  " << Arguments.tu << " " << setw(17) << "Elec ("
          << Arguments.eu << ")" << setw(17) << "VdW (" << Arguments.eu << ")"
          << setw(17) << "Total (" << Arguments.eu << ")" << '\n';
  for (int i = 0; i < E_e.size(); ++i)
    out_xvg << setw(14) << fixed << setprecision(1) << time_of_snap[i]
            << setw(17) << scientific << setprecision(6) << E_e[i] << setw(17)
            << E_v[i] << setw(17) << (E_e[i] + E_v[i]) << '\n';
  out_xvg.close();

  print_energy_time(E_e, E_v, time_of_snap, E_e_vector, E_v_vector, Arguments,
                    input_name);
  return 0;
}
int print_energy_time(vector<double> &E_e, vector<double> &E_v,
                      vector<double> &time_of_snap,
                      vector<vector<double>> &E_e_vector,
                      vector<vector<double>> &E_v_vector, arguments &Arguments,
                      int input_name) {
  fstream elec_out("electrostatic.dat", ios::out);
  fstream vdw_out("vanderwaals.dat", ios::out);
  fstream total_e("total_energy.dat", ios::out);
  string error_u{Arguments.flag_use_CI == true ? " CI(95 %) (" : " SD ("};
  total_e << setw(6) << "#   " << Arguments.tu << setw(8) << "TotE ("
          << Arguments.eu << ")" << setw(14) << error_u << Arguments.eu
          << ")\n";
  elec_out << setw(6) << "#   " << Arguments.tu << setw(8) << "ESP ("
           << Arguments.eu << ")" << setw(14) << error_u << Arguments.eu
           << ")\n";
  vdw_out << setw(6) << "#   " << Arguments.tu << setw(8) << "VDW ("
          << Arguments.eu << ")" << setw(14) << error_u << Arguments.eu
          << ")\n";
  if ((Arguments.flag_pattern_input == true ||
       Arguments.flag_listfile == true || Arguments.flag_all_folders == true) &&
      input_name != Arguments.b_input) {
    vector<double> SD_e, SD_v, SD_t;
    for (size_t i{}; i < E_v_vector.size(); ++i) {
      double sde{}, sdv{}, sdt{};
      double limit{10e-9};
      size_t sz_t{};
      for (size_t j{}; j < E_v_vector[i].size(); ++j) {
        if (fabs(E_e_vector[i][j]) < limit && fabs(E_v_vector[i][j]) < limit) {
          ++sz_t;
          continue;
        }
        sde += square(E_e[i] - E_e_vector[i][j]);
        sdv += square(E_v[i] - E_v_vector[i][j]);
        sdt += square(E_e[i] + E_v[i] - E_e_vector[i][j] - E_v_vector[i][j]);
      }
      SD_e.emplace_back(sqrt(sde / (E_e_vector[i].size() - sz_t)));
      SD_v.emplace_back(sqrt(sdv / (E_v_vector[i].size() - sz_t)));
      SD_t.emplace_back(sqrt(sdt / (E_v_vector[i].size() - sz_t)));
    }
    for (size_t i{}; i < E_e.size(); ++i) {
      double sde{Arguments.flag_use_CI == true
                     ? 1.96 * SD_e[i] / sqrt(E_e_vector[i].size())
                     : SD_e[i]},
          sdvdw{Arguments.flag_use_CI == true
                    ? 1.96 * SD_v[i] / sqrt(E_v_vector[i].size())
                    : SD_v[i]},
          sdtot{Arguments.flag_use_CI == true
                    ? 1.96 * SD_t[i] / sqrt(E_v_vector[i].size())
                    : SD_t[i]};

      elec_out << setw(6) << time_of_snap[i] << setw(17) << E_e[i] << setw(17)
               << sde << '\n';
      vdw_out << setw(6) << time_of_snap[i] << setw(17) << E_v[i] << setw(17)
              << sdvdw << '\n';
      total_e << setw(6) << time_of_snap[i] << setw(17) << E_e[i] + E_v[i]
              << setw(17) << sdtot << '\n';
    }
  } else {
    for (size_t i{}; i < E_e.size(); ++i) {
      elec_out << setw(6) << time_of_snap[i] << setw(17) << E_e[i] << '\n';
      vdw_out << setw(6) << time_of_snap[i] << setw(17) << E_v[i] << '\n';
      total_e << setw(6) << time_of_snap[i] << setw(17) << E_e[i] + E_v[i]
              << '\n';
    }
  }
  elec_out.close();
  vdw_out.close();
  total_e.close();
  return 0;
}

int folders::out_to_folders(vector<double> &E_e, vector<double> &E_v,
                            vector<double> &time_of_snap,
                            vector<vector<double>> &E_e_vector,
                            vector<vector<double>> &E_v_vector,
                            arguments &Arguments) {
  if (Arguments.flag_separation_results == false)
    return 1;

  size_t max_line_length{};
  for (size_t i{}; i < paths.size(); ++i)
    if (max_line_length < paths[i].string().length())
      max_line_length = paths[i].string().length();

  fstream stat_out_elec("stat_energy_elec.dat", ios::out),
      stat_out_vdw("stat_energy_vdw.dat", ios::out),
      stat_out_total("stat_total_energy.dat", ios::out),
      cal_out("cal.txt", ios::out | ios::app);
  stat_out_elec << left << setw(max_line_length + 2) << right << "# folder name"
                << "   elec (" << Arguments.eu << ")   " << "SD ("
                << Arguments.eu << ") CI (95%) (" << Arguments.eu << ")\n";
  stat_out_vdw << left << setw(max_line_length + 2) << right << "# folder name"
               << "   vdw (" << Arguments.eu << ")   SD (" << Arguments.eu
               << ") CI (95%) (" << Arguments.eu << ")\n";
  stat_out_total << left << setw(max_line_length + 2) << right
                 << "# folder name" << "   energy (" << Arguments.eu
                 << ")   SD (" << Arguments.eu << ") CI (95%) (" << Arguments.eu
                 << ")\n";
  cal_out << "----------------------------o0o----------------------------\n";
  cal_out << setw(10) << left << "No. traj" << right << setw(8) << "Elec ("
          << Arguments.eu << ")" << setw(8) << "vdW (" << Arguments.eu << ")\n";

  for (size_t i{}; i < paths.size(); ++i) {
    fstream out_elec(paths[i].string() + "/electrostatic.dat", ios::out);
    fstream out_vdw(paths[i].string() + "/vanderwaals.dat", ios::out);
    fstream total_energy(paths[i].string() + "/total_energy.dat", ios::out);
    out_elec << "# Time (" << Arguments.tu << ") " << setw(17)
             << "Electrostatic Energy (" << Arguments.eu << ")\n";
    out_vdw << "# Time (" << Arguments.tu << ") " << setw(17)
            << "van der Waals Energy (" << Arguments.eu << ")\n";
    total_energy << "# Time (" << Arguments.tu << ") " << setw(17)
                 << "Total Energy (" << Arguments.eu << ")\n";

    double sd_e{}, sd_v{};
    double e_mean{}, v_mean{};
    size_t real_size{};
    for (size_t j{}; j < E_e_vector.size(); ++j) {
      if (E_e_vector[j].size() <= i)
        continue;
      if (fabs(E_e_vector[j][i]) < 10e-9 && fabs(E_v_vector[j][i]) < 10e-9) {
        continue;
      }

      e_mean += E_e_vector[j][i];
      v_mean += E_v_vector[j][i];
      ++real_size;
    }
    if (real_size != 0) {
      e_mean /= real_size;
      v_mean /= real_size;
    }
    for (size_t j{}; j < E_e_vector.size(); ++j) {
      if (E_e_vector[j].size() <= i)
        continue;
      if (fabs(E_e_vector[j][i]) < 10e-9 && fabs(E_v_vector[j][i]) < 10e-9) {
        continue;
      }

      out_elec << setw(6) << time_of_snap[j] << setw(17) << E_e_vector[j][i]
               << '\n';
      out_vdw << setw(6) << time_of_snap[j] << setw(17) << E_v_vector[j][i]
              << '\n';
      total_energy << setw(6) << time_of_snap[j] << setw(17)
                   << E_e_vector[j][i] + E_v_vector[j][i] << '\n';
      sd_e += square(E_e_vector[j][i] - e_mean);
      sd_v += square(E_v_vector[j][i] - v_mean);
    }
    out_elec.close();
    out_vdw.close();
    total_energy.close();
    fstream out_average(paths[i].string() + "/cal.dat", ios::out);
    out_average << "Average of electrostatic energy: " << e_mean << " ("
                << Arguments.eu
                << "). Standard deviation: " << sqrt(sd_e / real_size)
                << ". Confidence interval (95%): "
                << 1.96 * sqrt(sd_e) / real_size << "\n"
                << "Average of van der Waals energy: " << v_mean << " ("
                << Arguments.eu
                << "). Standard deviation: " << sqrt(sd_v / real_size)
                << ". Confidence interval (95%): "
                << 1.96 * sqrt(sd_v) / real_size << "\n";
    out_average.close();
    stat_out_elec << setw(25) << paths[i].string() << "  " << setw(17) << e_mean
                  << setw(17) << sqrt(sd_e / real_size) << setw(17)
                  << 1.96 * sqrt(sd_e) / real_size << '\n';
    stat_out_vdw << setw(25) << paths[i].string() << "  " << setw(17) << v_mean
                 << setw(17) << sqrt(sd_v / real_size) << setw(17)
                 << 1.96 * sqrt(sd_v) / real_size << '\n';
    stat_out_total << setw(25) << paths[i].string() << "  " << setw(17)
                   << e_mean + v_mean << setw(17)
                   << sqrt(sd_e / real_size) + sqrt(sd_v / real_size)
                   << setw(17)
                   << 1.96 * sqrt(sd_e) / real_size +
                          1.96 * sqrt(sd_v) / real_size
                   << '\n';
    cal_out << setw(10) << left << Arguments.b_input + 1 + i << right
            << setw(16) << e_mean << setw(16) << v_mean << '\n';
  }
  stat_out_elec.close();
  stat_out_vdw.close();
  stat_out_total.close();
  return 0;
}