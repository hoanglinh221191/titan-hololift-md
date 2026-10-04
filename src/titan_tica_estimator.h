#ifndef TITAN_TICA_ESTIMATOR_H
#define TITAN_TICA_ESTIMATOR_H

#include <cstddef>
#include <string>
#include <vector>

struct TitanTicaFrameEstimateConfig {
  double begin_time_ps{0.0};
  double end_time_ps{0.0};
  double dt_ps{0.0};
  bool begin_set{false};
  bool end_set{false};
  std::size_t trajectory_count{0};
  std::size_t number_of_files{0};
  long double fallback_estimated_frames{1.0L};
  std::vector<std::string> xtc_paths;
};

struct TitanTicaFrameEstimate {
  std::size_t frames{1};
  std::string source{"fallback"};
  std::string diagnostic;
};

TitanTicaFrameEstimate
estimate_tica_selected_frames(const TitanTicaFrameEstimateConfig &config);

#endif // TITAN_TICA_ESTIMATOR_H
