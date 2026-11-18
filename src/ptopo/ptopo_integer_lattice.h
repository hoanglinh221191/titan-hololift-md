#ifndef TITAN_PTOPO_INTEGER_LATTICE_H
#define TITAN_PTOPO_INTEGER_LATTICE_H

// Exact subgroups of Z^3 in checked int64 (Backbone V.3, VI.2).
//
// Row-HNF (pivots positive, entries above a pivot reduced into [0, pivot))
// is the canonical embedded subgroup in the declared cell basis. Smith
// factors come from determinantal divisors: d1 = D1, d2 = D2/D1, d3 = D3/D2,
// where Dk is the gcd of the k x k minors. These are textbook normal forms,
// not a new method.

#include "ptopo_types.h"

#include <span>

namespace titan_ptopo {

// ROW generators in row-HNF; at most three rows.
using LatticeBasis = std::vector<Z3>;

// Fixed-capacity row-HNF for hot paths: no heap allocation.
struct Hnf3 {
  std::array<Z3, 3> rows{};
  std::uint8_t rank = 0;

  [[nodiscard]] LatticeBasis to_basis() const {
    return LatticeBasis(rows.begin(), rows.begin() + rank);
  }
  friend bool operator==(const Hnf3 &a, const Hnf3 &b) noexcept {
    if (a.rank != b.rank)
      return false;
    for (std::uint8_t i = 0; i < a.rank; ++i)
      if (a.rows[i] != b.rows[i])
        return false;
    return true;
  }
};

struct SmithData {
  std::size_t rank = 0;
  std::size_t free_rank = 3;
  // Positive d_i of coker = Z^(3-r) + sum Z/d_i, INCLUDING factors equal to 1.
  std::vector<Int> factors;
};

// hnf <- HNF(hnf + <g>). `changed` reports whether the subgroup grew; a
// rank-preserving saturation such as 2Z -> Z counts as a change.
Result<void> hnf_insert(Hnf3 &hnf, const Z3 &generator,
                        bool *changed = nullptr);
// into <- HNF(into + other).
Result<void> hnf_merge(Hnf3 &into, const Hnf3 &other, bool *changed = nullptr);
// Membership; `hnf` must be canonical (built by the functions above).
Result<bool> hnf_contains(const Hnf3 &hnf, Z3 v);
Result<Hnf3> hnf_from_rows(std::span<const Z3> generators);

Result<LatticeBasis> row_hnf(std::span<const Z3> generators);
// Recanonicalizes, so arbitrary row generators are accepted.
Result<SmithData> smith_invariants(std::span<const Z3> generators);
Result<bool> contains(std::span<const Z3> generators, Z3 v);
Result<IntMatrix3> inverse_unimodular(const IntMatrix3 &matrix);
Result<Z3> multiply(const IntMatrix3 &matrix, const Z3 &v);
// B_new = B_old U  =>  w_new = U^-1 w_old, det U = +-1. U is never inferred
// from two noisy NPT cells: deformation is not a basis remap.
Result<LatticeBasis> reexpress(std::span<const Z3> old_basis,
                               const IntMatrix3 &U);

} // namespace titan_ptopo

#endif // TITAN_PTOPO_INTEGER_LATTICE_H
