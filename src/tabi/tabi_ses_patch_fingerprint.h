#ifndef TITAN_PBE_TABI_SES_PATCH_FINGERPRINT_H
#define TITAN_PBE_TABI_SES_PATCH_FINGERPRINT_H

#include "tabi_analytic_ses.h"

#include <cstdint>

namespace TitanPBE::Tabi::analytic_ses {

// Audit-on identity of one immutable analytic patch-complex snapshot. Two
// independently mixed 64-bit lanes cover every geometry/topology field used
// by analytic patch intersections. This is a numerical consistency token,
// not a security, licensing, or patent token.
struct PatchComplexFingerprint {
  std::uint64_t first{0U};
  std::uint64_t second{0U};
  bool valid{false};

  [[nodiscard]] friend constexpr bool
  operator==(PatchComplexFingerprint,
             PatchComplexFingerprint) noexcept = default;
};

[[nodiscard]] PatchComplexFingerprint
fingerprint_patch_complex(const PatchComplex &complex) noexcept;

} // namespace TitanPBE::Tabi::analytic_ses

#endif // TITAN_PBE_TABI_SES_PATCH_FINGERPRINT_H
