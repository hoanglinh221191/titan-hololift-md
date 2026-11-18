#ifndef TITAN_PTOPO_POTENTIAL_DSU_H
#define TITAN_PTOPO_POTENTIAL_DSU_H

// Internal: union-find with Z^3 potentials, shared by audit() and the
// incremental filtration so both run the same edge equations.
//
// Node v stores offset(v) = n(v) - n(parent(v)); find() returns
// n(v) - n(root). Path compression and union by size keep every edge
// equation n(v) - n(u) = a_uv; only the tree shape changes. All sums are
// checked: a false return means int64 overflow and the caller must fail
// closed (the structure is then partly updated and must be discarded).

#include "ptopo_types.h"

#include <numeric>
#include <vector>

namespace titan_ptopo::detail {

struct DsuInsert {
  bool merged = false;    // the edge joined two components
  std::uint32_t root = 0; // root of the component now holding u and v
  std::uint32_t absorbed = 0; // former root folded into `root` (if merged)
  // p(u) + a_uv - p(v). Merged: the new relative root potential. Not merged:
  // the winding of the cycle the edge closes, a cycle sum that no later
  // re-rooting can change.
  Z3 residual{};
};

class PotentialDsu {
public:
  explicit PotentialDsu(std::size_t n) : parent_(n), size_(n, 1), offset_(n) {
    std::iota(parent_.begin(), parent_.end(), 0u);
  }

  [[nodiscard]] bool find(std::uint32_t x, std::uint32_t &root,
                          Z3 &potential) noexcept {
    std::uint32_t r = x;
    Z3 acc{};
    while (parent_[r] != r) {
      if (!checked::add3(acc, offset_[r], acc))
        return false;
      r = parent_[r];
    }
    // Point every node on the path at r with its own potential:
    // potential(parent) = potential(child) - offset_old(child).
    std::uint32_t cur = x;
    Z3 pot = acc;
    while (parent_[cur] != r) {
      const std::uint32_t next = parent_[cur];
      const Z3 old = offset_[cur];
      parent_[cur] = r;
      offset_[cur] = pot;
      if (!checked::sub3(pot, old, pot))
        return false;
      cur = next;
    }
    root = r;
    potential = acc;
    return true;
  }

  [[nodiscard]] bool insert(std::uint32_t u, std::uint32_t v, const Z3 &shift,
                            DsuInsert &out) noexcept {
    std::uint32_t ru = 0, rv = 0;
    Z3 pu{}, pv{}, t{};
    if (!find(u, ru, pu) || !find(v, rv, pv) ||
        !checked::add3(pu, shift, t) || !checked::sub3(t, pv, out.residual))
      return false;
    out.merged = ru != rv;
    out.root = ru;
    out.absorbed = ru;
    if (!out.merged)
      return true;
    // Union fixes n(r_v) - n(r_u) = residual.
    if (size_[ru] >= size_[rv]) {
      parent_[rv] = ru;
      offset_[rv] = out.residual;
      size_[ru] += size_[rv];
      out.root = ru;
      out.absorbed = rv;
    } else {
      Z3 negated{};
      if (!checked::neg3(out.residual, negated))
        return false;
      parent_[ru] = rv;
      offset_[ru] = negated;
      size_[rv] += size_[ru];
      out.root = rv;
      out.absorbed = ru;
    }
    return true;
  }

  [[nodiscard]] std::uint32_t component_size(std::uint32_t root) const noexcept {
    return size_[root];
  }

private:
  std::vector<std::uint32_t> parent_;
  std::vector<std::uint32_t> size_;
  std::vector<Z3> offset_;
};

} // namespace titan_ptopo::detail

#endif // TITAN_PTOPO_POTENTIAL_DSU_H
