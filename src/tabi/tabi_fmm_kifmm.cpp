#include "tabi_fmm_kifmm.h"

namespace titan::tabi::fmm_kifmm {

bool is_kifmm_engine(const std::string &engine) noexcept {
  return engine == "kifmm-lite" || engine == "kifmm-local" ||
         engine == "kifmm-lite-experimental";
}

KifmmSurfaceRule make_surface_rule(const KifmmConfig &config) {
  return make_cube_surface_rule(config.order, config.equiv_scale,
                                config.check_scale);
}

} // namespace titan::tabi::fmm_kifmm
