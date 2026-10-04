#ifndef TITAN_PBE_GRID_WORKSPACE_H
#define TITAN_PBE_GRID_WORKSPACE_H

#include "aligned_allocator.h"

#include <cstddef>
#include <vector>

namespace TitanPBE {

struct GridOptions {
  double ext_diel{80.0};
};

struct GridWorkspace {
  AlignedDoubleVector phi;
  AlignedDoubleVector rho;
  AlignedDoubleVector eps_x;
  AlignedDoubleVector eps_y;
  AlignedDoubleVector eps_z;
  AlignedDoubleVector denom;
  AlignedDoubleVector screen;
  std::vector<unsigned char> inside;
  std::vector<unsigned char> ion_accessible;

  void reset(std::size_t n, const GridOptions &options) {
    phi.assign(n, 0.0);
    rho.assign(n, 0.0);
    eps_x.assign(n, options.ext_diel);
    eps_y.assign(n, options.ext_diel);
    eps_z.assign(n, options.ext_diel);
    denom.assign(n, 1.0);
    screen.assign(n, 0.0);
    inside.assign(n, 0U);
    ion_accessible.assign(n, 1U);
  }

  void clear() {
    phi.clear();
    rho.clear();
    eps_x.clear();
    eps_y.clear();
    eps_z.clear();
    denom.clear();
    screen.clear();
    inside.clear();
    ion_accessible.clear();
  }

  [[nodiscard]] bool sized(std::size_t n) const noexcept {
    return phi.size() == n && rho.size() == n && eps_x.size() == n &&
           eps_y.size() == n && eps_z.size() == n && denom.size() == n &&
           screen.size() == n && inside.size() == n &&
           ion_accessible.size() == n;
  }

  [[nodiscard]] std::size_t memory_bytes() const noexcept {
    return (phi.size() + rho.size() + eps_x.size() + eps_y.size() +
            eps_z.size() + denom.size() + screen.size()) *
               sizeof(double) +
           (inside.size() + ion_accessible.size()) * sizeof(unsigned char);
  }
};

} // namespace TitanPBE

#endif // TITAN_PBE_GRID_WORKSPACE_H
