#ifndef DE_H
#define DE_H
#include "arguments.h"
#include "miscellaneous.h"
#include <fstream>
#include <iostream>
#include <sstream>


struct de_arg {
  std::string pull_opt{"de"};
  size_t pull_max_evals{0};
  unsigned int pull_seed{42};
  std::string snapshot_source_file{};
  std::string snapshot_input_mode{};
  std::string snapshot_selection_rule{};
  double snapshot_time_ps{0.0};
  size_t snapshot_frame_index{0};
  bool snapshot_metadata_valid{false};
  double diff_weight{0.9}, cross_prob{0.9}, F{0.5}, scan_radius{2.},
      min_scan_radius{0.5}, restrain_angle{90.};
  size_t memory_size{10};
  double p_best_rate{0.2};
  double archive_scale{1.0};
  size_t min_pop_num{4};
  force old_vector{};
  size_t pop_num{2000}, ite_num{100}, algorithm{};
  bool flag_old_vec{false};
  // Adaptive F & early stopping
  bool adaptive_F{false};
  double F_min{0.2}, F_max{0.9};
  int stagnation_limit{20};
  de_arg() = default;
  int parse_input(const arguments &Arguments) {
    pull_opt = Arguments.de.pull_opt;
    pull_max_evals = Arguments.de.pull_max_evals;
    pull_seed = Arguments.de.pull_seed;
    algorithm = Arguments.de.de_algorithm;
    pop_num = Arguments.de.de_pop_num;
    ite_num = Arguments.de.de_ite;
    diff_weight = Arguments.de.de_diff_weight;
    cross_prob = Arguments.de.de_cross_prob;
    F = Arguments.de.de_F;
    scan_radius = fabs(Arguments.de.de_scan_radius);
    min_scan_radius = fabs(Arguments.de.de_min_scan_radius);
    restrain_angle = Arguments.de.de_restrain_angle;
    adaptive_F = Arguments.de.de_adaptive_F;
    stagnation_limit = Arguments.de.de_stagnation_limit;
    memory_size = Arguments.de.de_memory_size;
    p_best_rate = Arguments.de.de_p_best_rate;
    archive_scale = Arguments.de.de_archive_scale;
    min_pop_num = Arguments.de.de_min_pop_num;
    fstream de_old_vec("DE_out_vector.dat", ios::in);
    if (de_old_vec.is_open()) {
      double x{}, y{}, z{};
      for (string temp{}; getline(de_old_vec, temp);) {
        istringstream is(temp);
        if (is >> x >> y >> z) {
          flag_old_vec = true;
          cout << "Found old vector!\n";
          old_vector.Fx = x;
          old_vector.Fy = y;
          old_vector.Fz = z;
        }
      }
    }
    return 0;
  }
};

#endif // !DE_H
