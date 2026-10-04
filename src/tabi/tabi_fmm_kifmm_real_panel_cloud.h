#ifndef TITAN_TABI_FMM_KIFMM_REAL_PANEL_CLOUD_H
#define TITAN_TABI_FMM_KIFMM_REAL_PANEL_CLOUD_H

#include "tabi_fmm_kifmm_lite.h"
#include "tabi_types.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace titan::tabi::fmm_kifmm {

struct KifmmRealPanelCloudConfig {
  KifmmLiteConfig lite{};
  std::string near_integration{"analytic_singular"};
  std::string quad_buffer{"off"};
  std::string full_apply_policy{"serial"};
};

struct KifmmRealPanelCloudResult {
  KifmmLiteResult lite{};
  std::size_t production_fallback_valid_pairs{0};
  std::size_t production_direct_reference_valid_pairs{0};
  std::size_t production_invalid_pairs{0};
  // Real-panel clouds have no synthetic fallback implementation. This counter
  // guards against accidental future wiring of a non-production fallback path.
  std::size_t nonproduction_fallback_pairs{0};
  bool used_nonproduction_fallback{false};
  std::uint64_t policy_hash{0};
  bool finite{false};
  std::string status{"not_started"};
};

KifmmRealPanelCloudResult run_kifmm_lite_real_panel_matvec(
    std::span<const TitanPBE::Tabi::Panel> panels,
    std::span<const double> unknowns, const KifmmRealPanelCloudConfig &config);

} // namespace titan::tabi::fmm_kifmm

#endif // TITAN_TABI_FMM_KIFMM_REAL_PANEL_CLOUD_H
