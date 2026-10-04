#ifndef TITAN_ZTS_H
#define TITAN_ZTS_H

#include "arguments.h"
#include "titan_pca.h"
#include <fstream>
#include <string>
#include <vector>

class TitanExecutor;

namespace TitanZTS {

struct ZtsImage {
  double pc1{0.0};
  double pc2{0.0};
  double fes{0.0};
};

struct ZtsResult {
  bool converged{false};
  int iterations_run{0};
  double barrier_fwd{0.0};
  double barrier_rev{0.0};
  int saddle_image_idx{-1};
  double saddle_pc1{0.0};
  double saddle_pc2{0.0};
  double saddle_energy{0.0};
  std::vector<ZtsImage> path;
};

ZtsResult run_zts(const ZtsArgs &zts_args, const TitanPCA::FES2DResult &fes,
                  const std::vector<TitanPCA::ClusterNode> &clusters,
                  TitanExecutor &executor, std::fstream &log);

void write_zts_path(const std::string &filename, const ZtsResult &result);

} // namespace TitanZTS

#endif // TITAN_ZTS_H
