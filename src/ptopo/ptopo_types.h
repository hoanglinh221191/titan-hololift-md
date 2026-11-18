#ifndef TITAN_PTOPO_TYPES_H
#define TITAN_PTOPO_TYPES_H

// Periodic topology kernel (Backbone v1.1, Part III / VI): shared types.
//
// Port of periodic_topology_cpp23 v0.1.0 without external libraries. The
// reference used Boost cpp_int; here every integer is std::int64_t and every
// operation is checked. An overflow is never wrapped or saturated: the routine
// returns ErrorCode::IntegerOverflow and no partial result. Lattice shifts of
// a molecular graph are small (|a| <= a few), so the checked path never trips
// on real input; it only exists so an adversarial input fails closed.

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace titan_ptopo {

using Int = std::int64_t;
using Z3 = std::array<Int, 3>;
using Vec3 = std::array<double, 3>;
// Row-major: m[row][col].
using IntMatrix3 = std::array<Z3, 3>;

enum class ErrorCode : std::uint8_t {
  InvalidInput,
  InvalidCell,
  AmbiguousGeometry,
  ResourceLimit,
  IncompatibleGraphs,
  UnresolvedBasis,
  NumericalFailure,
  IntegerOverflow,
};

inline const char *error_code_name(ErrorCode code) noexcept {
  switch (code) {
  case ErrorCode::InvalidInput:
    return "InvalidInput";
  case ErrorCode::InvalidCell:
    return "InvalidCell";
  case ErrorCode::AmbiguousGeometry:
    return "AmbiguousGeometry";
  case ErrorCode::ResourceLimit:
    return "ResourceLimit";
  case ErrorCode::IncompatibleGraphs:
    return "IncompatibleGraphs";
  case ErrorCode::UnresolvedBasis:
    return "UnresolvedBasis";
  case ErrorCode::NumericalFailure:
    return "NumericalFailure";
  case ErrorCode::IntegerOverflow:
    return "IntegerOverflow";
  }
  return "Unknown";
}

struct Error {
  ErrorCode code = ErrorCode::InvalidInput;
  std::string message;
};

template <class T> using Result = std::expected<T, Error>;

inline std::unexpected<Error> make_error(ErrorCode code, std::string message) {
  return std::unexpected(Error{code, std::move(message)});
}

inline std::unexpected<Error> overflow_error(const char *where) {
  return make_error(ErrorCode::IntegerOverflow,
                    std::string("int64 overflow in ") + where +
                        "; no partial result returned");
}

// ---------------------------------------------------------------------------
// Checked int64 arithmetic. Each returns false on overflow and leaves `out`
// unspecified. C++23 has no standard checked arithmetic (<stdckdint.h> is C23
// and C++26), so GCC/Clang builtins are used with a portable fallback.
// ---------------------------------------------------------------------------
namespace checked {

[[nodiscard]] inline bool add(Int a, Int b, Int &out) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  return !__builtin_add_overflow(a, b, &out);
#else
  if ((b > 0 && a > std::numeric_limits<Int>::max() - b) ||
      (b < 0 && a < std::numeric_limits<Int>::min() - b))
    return false;
  out = a + b;
  return true;
#endif
}

[[nodiscard]] inline bool sub(Int a, Int b, Int &out) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  return !__builtin_sub_overflow(a, b, &out);
#else
  if ((b < 0 && a > std::numeric_limits<Int>::max() + b) ||
      (b > 0 && a < std::numeric_limits<Int>::min() + b))
    return false;
  out = a - b;
  return true;
#endif
}

[[nodiscard]] inline bool mul(Int a, Int b, Int &out) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  return !__builtin_mul_overflow(a, b, &out);
#else
  if (a == 0 || b == 0) {
    out = 0;
    return true;
  }
  constexpr Int hi = std::numeric_limits<Int>::max();
  constexpr Int lo = std::numeric_limits<Int>::min();
  if (a > 0 ? (b > 0 ? a > hi / b : b < lo / a)
            : (b > 0 ? a < lo / b : a < hi / b))
    return false;
  out = a * b;
  return true;
#endif
}

[[nodiscard]] inline bool neg(Int a, Int &out) noexcept {
  if (a == std::numeric_limits<Int>::min())
    return false;
  out = -a;
  return true;
}

[[nodiscard]] inline bool abs(Int a, Int &out) noexcept {
  if (a == std::numeric_limits<Int>::min())
    return false;
  out = a < 0 ? -a : a;
  return true;
}

// Truncating division, as C++ '/'. Fails for b == 0 and for min / -1.
[[nodiscard]] inline bool div(Int a, Int b, Int &out) noexcept {
  if (b == 0 || (a == std::numeric_limits<Int>::min() && b == -1))
    return false;
  out = a / b;
  return true;
}

// floor(a / b) for b > 0.
[[nodiscard]] inline bool floordiv_positive(Int a, Int b, Int &out) noexcept {
  if (b <= 0)
    return false;
  Int q = a / b;
  const Int r = a % b;
  // r < 0 needs b >= 2, so q > INT64_MIN and the decrement cannot wrap.
  if (r < 0)
    --q;
  out = q;
  return true;
}

// dst -= q * src, componentwise.
[[nodiscard]] inline bool sub_multiple(Z3 &dst, const Z3 &src,
                                       Int q) noexcept {
  for (std::size_t k = 0; k < 3; ++k) {
    Int product = 0;
    if (!mul(q, src[k], product) || !sub(dst[k], product, dst[k]))
      return false;
  }
  return true;
}

[[nodiscard]] inline bool add3(const Z3 &a, const Z3 &b, Z3 &out) noexcept {
  return add(a[0], b[0], out[0]) && add(a[1], b[1], out[1]) &&
         add(a[2], b[2], out[2]);
}

[[nodiscard]] inline bool sub3(const Z3 &a, const Z3 &b, Z3 &out) noexcept {
  return sub(a[0], b[0], out[0]) && sub(a[1], b[1], out[1]) &&
         sub(a[2], b[2], out[2]);
}

[[nodiscard]] inline bool neg3(const Z3 &a, Z3 &out) noexcept {
  return neg(a[0], out[0]) && neg(a[1], out[1]) && neg(a[2], out[2]);
}

// gcd(|a|, |b|); fails only when an argument is INT64_MIN.
[[nodiscard]] inline bool gcd(Int a, Int b, Int &out) noexcept {
  if (!abs(a, a) || !abs(b, b))
    return false;
  while (b != 0) {
    const Int r = a % b;
    a = b;
    b = r;
  }
  out = a;
  return true;
}

} // namespace checked

inline bool is_zero(const Z3 &a) noexcept {
  return a[0] == 0 && a[1] == 0 && a[2] == 0;
}

// Lexicographic sign of a nonzero vector: +1 if the first nonzero entry is
// positive. Used to keep exactly one of the self-image pair (+a, -a).
inline int lexicographic_sign(const Z3 &a) noexcept {
  for (const Int v : a)
    if (v != 0)
      return v > 0 ? 1 : -1;
  return 0;
}

// ---------------------------------------------------------------------------
// Periodic multigraph (Backbone V.2/V.3). Edge u -> v with shift a has the
// physical displacement B (s_v - s_u + a); a node gauge s'_v = s_v + h_v
// relabels a'_uv = a_uv + h_u - h_v.
// ---------------------------------------------------------------------------
enum class Species : std::uint8_t { Protein, RNA, Other };

enum class EdgeKind : std::uint8_t {
  Hard,
  ProteinProtein,
  ProteinRNA,
  RNAInteraction,
  HydrogenBond,
  SaltBridge,
  Generic,
};

inline const char *species_name(Species species) noexcept {
  switch (species) {
  case Species::Protein:
    return "protein";
  case Species::RNA:
    return "rna";
  case Species::Other:
    return "other";
  }
  return "unknown";
}

inline const char *edge_kind_name(EdgeKind kind) noexcept {
  switch (kind) {
  case EdgeKind::Hard:
    return "hard";
  case EdgeKind::ProteinProtein:
    return "protein-protein";
  case EdgeKind::ProteinRNA:
    return "protein-rna";
  case EdgeKind::RNAInteraction:
    return "rna-rna";
  case EdgeKind::HydrogenBond:
    return "hbond";
  case EdgeKind::SaltBridge:
    return "salt-bridge";
  case EdgeKind::Generic:
    return "generic";
  }
  return "unknown";
}

struct Node {
  std::uint64_t id = 0;          // stable within an identity epoch
  std::uint64_t molecule_id = 0; // never inferred from a contact component
  Species species = Species::Other;
};

struct Edge {
  std::uint64_t id = 0; // unique within this graph; not a temporal identity
  std::uint32_t u = 0;  // indices into Graph::nodes
  std::uint32_t v = 0;
  Z3 shift{};
  EdgeKind kind = EdgeKind::Generic;
  double activation = 0.0; // nonnegative filtration value; hard edges first
};

struct Graph {
  std::vector<Node> nodes;
  // Periodic multigraph: parallel edges and self-image loops are allowed.
  std::vector<Edge> edges;
  // Caller-supplied labels, required nonempty; not verified hashes.
  std::string identity_epoch;
  std::string policy_id;
  std::string reference_basis_id;
};

} // namespace titan_ptopo

#endif // TITAN_PTOPO_TYPES_H
