#ifndef TITAN_PBCTOPO_LATTICE_SPAN_H
#define TITAN_PBCTOPO_LATTICE_SPAN_H

// Linearly independent integer lattice vectors (rank <= 3); used for the
// cycle lattice of the contact network and the bonded winding of the lift.

#include "types.h"

#include <array>
#include <cstddef>

namespace titan_pbctopo {

// Each vector has its first nonzero component positive.
struct PbctopoLatticeSpan {
  std::array<Int3, 3> basis{};
  std::size_t rank = 0;
};

inline Int3 pbctopo_canonical_lattice_direction(Int3 value) {
  const bool negate =
      value.x < 0 ||
      (value.x == 0 && (value.y < 0 || (value.y == 0 && value.z < 0)));
  if (negate)
    value = {-value.x, -value.y, -value.z};
  return value;
}

// Adds value if it is independent of the span; returns true if rank grew.
// The tests are exact for the small integers that occur (long double holds
// products of 64-bit lattice indices below 2^31 exactly).
inline bool pbctopo_lattice_span_add(PbctopoLatticeSpan &span, Int3 value) {
  value = pbctopo_canonical_lattice_direction(value);
  if (is_zero_delta(value) || span.rank >= 3)
    return false;
  const auto ld = [](std::int64_t v) { return static_cast<long double>(v); };
  if (span.rank >= 1) {
    const Int3 &a = span.basis[0];
    const bool collinear = ld(a.y) * value.z == ld(a.z) * value.y &&
                           ld(a.z) * value.x == ld(a.x) * value.z &&
                           ld(a.x) * value.y == ld(a.y) * value.x;
    if (span.rank == 1 && collinear)
      return false;
  }
  if (span.rank == 2) {
    const Int3 &a = span.basis[0];
    const Int3 &b = span.basis[1];
    const long double det =
        ld(a.x) * (ld(b.y) * value.z - ld(b.z) * value.y) -
        ld(a.y) * (ld(b.x) * value.z - ld(b.z) * value.x) +
        ld(a.z) * (ld(b.x) * value.y - ld(b.y) * value.x);
    if (det == 0.0L)
      return false;
  }
  span.basis[span.rank++] = value;
  return true;
}

} // namespace titan_pbctopo

#endif
