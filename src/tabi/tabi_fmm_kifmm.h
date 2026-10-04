#ifndef TITAN_TABI_FMM_KIFMM_H
#define TITAN_TABI_FMM_KIFMM_H

#include "tabi_fmm_kifmm_surface.h"

#include <string>

namespace titan::tabi::fmm_kifmm {

struct KifmmConfig {
  std::string engine{"cartesian"};
  int order{6};
  double equiv_scale{1.05};
  double check_scale{1.60};
  std::string solver{"qr"};
  std::string profile{"off"};
};

bool is_kifmm_engine(const std::string &engine) noexcept;

KifmmSurfaceRule make_surface_rule(const KifmmConfig &config);

} // namespace titan::tabi::fmm_kifmm

#endif // TITAN_TABI_FMM_KIFMM_H
