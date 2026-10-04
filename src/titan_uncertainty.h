#ifndef TITAN_UNCERTAINTY_H
#define TITAN_UNCERTAINTY_H

#include "arguments.h"
#include "info.h"

#include <cstddef>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace titan_uncertainty {

struct Block {
  std::size_t begin{0};
  std::size_t end{0};
};

struct Stats {
  std::size_t n{0};
  double mean{0.0};
  double variance{0.0};
  double sem{0.0};
  double ci95_low{0.0};
  double ci95_high{0.0};
};

struct XYPoint {
  double x{0.0};
  double y{0.0};
};

struct FitEstimate {
  bool ok{false};
  std::string status{"insufficient_points"};
  std::size_t n{0};
  double slope{0.0};
  double intercept{0.0};
  double r2{0.0};
};

struct BootstrapStatistic {
  Stats samples;
  double ci95_low{0.0};
  double ci95_high{0.0};
};

struct BlockFitResult {
  FitEstimate point;
  Stats slope_stats;
  Stats intercept_stats;
  double slope_bootstrap_ci95_low{0.0};
  double slope_bootstrap_ci95_high{0.0};
  double intercept_bootstrap_ci95_low{0.0};
  double intercept_bootstrap_ci95_high{0.0};
  std::size_t block_count{0};
};

std::vector<Block> partition_blocks(std::size_t frame_count,
                                    const std::string &blocks_spec);
Stats summarize_values(const std::vector<double> &values);
Stats summarize_blocks(const std::vector<double> &values,
                       const std::string &blocks_spec);
std::pair<double, double>
bootstrap_ci95(const std::vector<double> &block_means, int samples,
               unsigned int seed);
BootstrapStatistic bootstrap_statistic(const std::vector<double> &values,
                                       int samples, unsigned int seed);
FitEstimate linear_fit(const std::vector<XYPoint> &points);
BlockFitResult linear_fit_with_blocks(const std::vector<XYPoint> &points,
                                      double tmin, double tmax,
                                      const std::string &blocks_spec,
                                      int bootstrap_samples,
                                      unsigned int seed);
BlockFitResult loglog_fit_with_blocks(const std::vector<XYPoint> &points,
                                      double tmin, double tmax,
                                      const std::string &blocks_spec,
                                      int bootstrap_samples,
                                      unsigned int seed);

void export_uncertainty_outputs(const arguments &args, const info &out_info,
                                std::fstream &log);

} // namespace titan_uncertainty

#endif
