#ifndef TITAN_ENERGY_CUTOFF_BLOCKS_H
#define TITAN_ENERGY_CUTOFF_BLOCKS_H

// ENE-03 (22/10/2026). With -denfo the AVX2 energy kernels still evaluated
// every receptor-ligand pair and masked the ones beyond the cutoff. Here the
// receptor atoms that the kernels process four at a time are wrapped in
// bounding spheres, per block of 4 and per super-block of 64; a block whose
// sphere lies wholly beyond the cutoff is skipped and its pair-cache entries
// are written as the zeros the mask produced. A culled pair is one the mask
// dropped, so energies and caches are unchanged.

#include "MOL.h"
#include "cache_sigma_epsilon.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace titan_cutoff {

class CutoffBlocks {
public:
  static constexpr std::size_t kBlock = 4;
  static constexpr std::size_t kSuper = 64;

  // vector_atoms: receptor atoms handled by the 4-lane loop (a multiple of 4);
  // cutoff_nm <= 0 disables the culling.
  CutoffBlocks(const MOL &rec, std::size_t vector_atoms, double cutoff_nm)
      : vector_atoms_(vector_atoms) {
    if (!(cutoff_nm > 0.0) || vector_atoms == 0)
      return;
    // Kernel distances are sqrt(0.01 * r2[A^2]); the margin keeps the test
    // conservative against that rounding.
    reach_ = cutoff_nm * 10.0 * (1.0 + 1e-9) + 1e-6;
    build(rec, kBlock, blocks_);
    build(rec, kSuper, supers_);
    active_ = true;
  }

  // Receptor atoms from i (a multiple of 4) that are certainly beyond the
  // cutoff of the ligand atom at (x, y, z), in Angstrom: a whole super-block
  // when i starts one, else a block of 4, else 0.
  std::size_t skip(std::size_t i, double x, double y, double z) const {
    if (!active_)
      return 0;
    if (i % kSuper == 0 && far(supers_[i / kSuper], x, y, z))
      return std::min(kSuper, vector_atoms_ - i);
    return far(blocks_[i / kBlock], x, y, z) ? kBlock : 0;
  }

  static void zero_cache(cache_sigma_epsilon &cache, std::size_t pos, std::size_t count) {
    if (cache.elec_ener.empty())
      return;
    std::fill_n(cache.elec_ener.data() + pos, count, 0.0);
    std::fill_n(cache.vdw_ener.data() + pos, count, 0.0);
  }

private:
  struct Sphere {
    double x{}, y{}, z{}, r{};
  };

  void build(const MOL &rec, std::size_t size, std::vector<Sphere> &out) const {
    out.clear();
    for (std::size_t start = 0; start < vector_atoms_; start += size) {
      const std::size_t stop = std::min(start + size, vector_atoms_);
      Sphere s;
      for (std::size_t a = start; a < stop; ++a) {
        s.x += rec.pos.x[a];
        s.y += rec.pos.y[a];
        s.z += rec.pos.z[a];
      }
      const double n = static_cast<double>(stop - start);
      s.x /= n;
      s.y /= n;
      s.z /= n;
      for (std::size_t a = start; a < stop; ++a) {
        const double dx = rec.pos.x[a] - s.x, dy = rec.pos.y[a] - s.y,
                     dz = rec.pos.z[a] - s.z;
        s.r = std::max(s.r, std::sqrt(dx * dx + dy * dy + dz * dz));
      }
      s.r = s.r * (1.0 + 1e-12) + 1e-9;
      out.push_back(s);
    }
  }

  bool far(const Sphere &s, double x, double y, double z) const {
    const double dx = x - s.x, dy = y - s.y, dz = z - s.z;
    const double limit = reach_ + s.r;
    return dx * dx + dy * dy + dz * dz > limit * limit;
  }

  std::size_t vector_atoms_{};
  double reach_{};
  bool active_{false};
  std::vector<Sphere> blocks_, supers_;
};

} // namespace titan_cutoff

#endif // TITAN_ENERGY_CUTOFF_BLOCKS_H
