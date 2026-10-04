#include "titan_tica_estimator.h"

extern "C" {
#include "xdrfile_xtc.h"
}

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace {

std::size_t positive_trajectory_count(const TitanTicaFrameEstimateConfig &config) {
  if (config.trajectory_count > 0)
    return config.trajectory_count;
  if (config.number_of_files > 0)
    return config.number_of_files;
  return 1;
}

std::size_t clamp_positive_frame_count(long double value) {
  if (!std::isfinite(static_cast<double>(value)) || value < 1.0L)
    return 1;
  const long double max_size =
      static_cast<long double>(std::numeric_limits<std::size_t>::max());
  if (value >= max_size)
    return std::numeric_limits<std::size_t>::max();
  return static_cast<std::size_t>(std::ceil(value));
}

} // namespace

TitanTicaFrameEstimate
estimate_tica_selected_frames(const TitanTicaFrameEstimateConfig &config) {
  const std::size_t n_traj = positive_trajectory_count(config);

  std::string xtc_diagnostic;
  if (!config.xtc_paths.empty()) {
    std::size_t total_frames = 0;
    bool scan_ok = true;
    for (const std::string &path : config.xtc_paths) {
      int64_t selected_frames = 0;
      const int rc = estimate_xtc_frame_count(
          path.c_str(), config.begin_time_ps, config.begin_set ? 1 : 0,
          config.end_time_ps, config.end_set ? 1 : 0, config.dt_ps,
          &selected_frames);
      if (rc != exdrOK || selected_frames < 0) {
        xtc_diagnostic = "XTC frame sampling failed for " + path + " (rc=" +
                         std::to_string(rc) + ")";
        scan_ok = false;
        break;
      }
      const auto frames = static_cast<std::uint64_t>(selected_frames);
      if (frames > std::numeric_limits<std::size_t>::max() - total_frames) {
        xtc_diagnostic =
            "XTC frame sampling overflowed the platform frame counter";
        scan_ok = false;
        break;
      }
      total_frames += static_cast<std::size_t>(frames);
    }
    if (scan_ok)
      return {std::max<std::size_t>(1, total_frames), "xtc-header-sample", {}};
  }

  if (config.end_set && config.dt_ps > 0.0 &&
      std::isfinite(config.begin_time_ps) && std::isfinite(config.end_time_ps) &&
      std::isfinite(config.dt_ps) && config.end_time_ps >= config.begin_time_ps) {
    const long double span_ps =
        static_cast<long double>(config.end_time_ps - config.begin_time_ps);
    const long double dt_ps = static_cast<long double>(config.dt_ps);
    const long double frames_per_traj = std::floor(span_ps / dt_ps) + 1.0L;
    const long double total_frames =
        frames_per_traj * static_cast<long double>(n_traj);
    return {clamp_positive_frame_count(total_frames), "time-window",
            xtc_diagnostic};
  }

  return {clamp_positive_frame_count(config.fallback_estimated_frames),
          "fallback", xtc_diagnostic};
}
