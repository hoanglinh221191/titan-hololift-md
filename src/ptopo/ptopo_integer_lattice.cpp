#include "ptopo_integer_lattice.h"

#include <utility>

namespace titan_ptopo {
namespace {

// In-place row-HNF of the first m rows of `a` (m <= 4). On success the first
// `m` rows (m updated to the rank) are the canonical HNF and the rest are
// zero. Returns false on int64 overflow.
bool reduce_rows(std::array<Z3, 4> &a, std::size_t &m) noexcept {
  std::size_t kept = 0;
  for (std::size_t i = 0; i < m; ++i)
    if (!is_zero(a[i]))
      a[kept++] = a[i];
  m = kept;
  std::size_t pivot = 0;
  for (std::size_t col = 0; col < 3 && pivot < m; ++col) {
    std::size_t first = pivot;
    while (first < m && a[first][col] == 0)
      ++first;
    if (first == m)
      continue;
    std::swap(a[pivot], a[first]);
    // Euclid on the column: |a[i][col]| strictly decreases.
    for (std::size_t i = pivot + 1; i < m; ++i) {
      while (a[i][col] != 0) {
        Int q = 0;
        if (!checked::div(a[pivot][col], a[i][col], q) ||
            !checked::sub_multiple(a[pivot], a[i], q))
          return false;
        std::swap(a[pivot], a[i]);
      }
    }
    if (a[pivot][col] < 0 && !checked::neg3(a[pivot], a[pivot]))
      return false;
    // Reduce the entries above the pivot into [0, pivot). The pivot row is
    // zero in every earlier pivot column, so earlier reductions survive.
    for (std::size_t i = 0; i < pivot; ++i) {
      Int q = 0;
      if (!checked::floordiv_positive(a[i][col], a[pivot][col], q) ||
          !checked::sub_multiple(a[i], a[pivot], q))
        return false;
    }
    ++pivot;
  }
  m = pivot;
  return true;
}

bool det3(const IntMatrix3 &a, Int &out) noexcept {
  const auto minor2 = [](Int p, Int q, Int r, Int s, Int &m) {
    Int ps = 0, qr = 0;
    return checked::mul(p, s, ps) && checked::mul(q, r, qr) &&
           checked::sub(ps, qr, m);
  };
  Int m0 = 0, m1 = 0, m2 = 0, t0 = 0, t1 = 0, t2 = 0, partial = 0;
  return minor2(a[1][1], a[1][2], a[2][1], a[2][2], m0) &&
         minor2(a[1][0], a[1][2], a[2][0], a[2][2], m1) &&
         minor2(a[1][0], a[1][1], a[2][0], a[2][1], m2) &&
         checked::mul(a[0][0], m0, t0) && checked::mul(a[0][1], m1, t1) &&
         checked::mul(a[0][2], m2, t2) && checked::sub(t0, t1, partial) &&
         checked::add(partial, t2, out);
}

} // namespace

Result<void> hnf_insert(Hnf3 &hnf, const Z3 &generator, bool *changed) {
  if (changed != nullptr)
    *changed = false;
  if (is_zero(generator))
    return {};
  auto present = hnf_contains(hnf, generator);
  if (!present)
    return std::unexpected(present.error());
  if (*present)
    return {};
  std::array<Z3, 4> rows{};
  std::size_t m = hnf.rank;
  for (std::size_t i = 0; i < m; ++i)
    rows[i] = hnf.rows[i];
  rows[m++] = generator;
  if (!reduce_rows(rows, m))
    return overflow_error("hnf_insert");
  hnf.rank = static_cast<std::uint8_t>(m);
  for (std::size_t i = 0; i < 3; ++i)
    hnf.rows[i] = i < m ? rows[i] : Z3{};
  if (changed != nullptr)
    *changed = true;
  return {};
}

Result<void> hnf_merge(Hnf3 &into, const Hnf3 &other, bool *changed) {
  bool any = false;
  for (std::uint8_t i = 0; i < other.rank; ++i) {
    bool grew = false;
    auto step = hnf_insert(into, other.rows[i], &grew);
    if (!step)
      return step;
    any = any || grew;
  }
  if (changed != nullptr)
    *changed = any;
  return {};
}

Result<bool> hnf_contains(const Hnf3 &hnf, Z3 v) {
  for (std::uint8_t r = 0; r < hnf.rank; ++r) {
    const Z3 &row = hnf.rows[r];
    std::size_t col = 0;
    while (col < 3 && row[col] == 0)
      ++col;
    if (col == 3)
      continue;
    // row[col] > 0 in a canonical HNF, so '%' and '/' are well defined.
    if (v[col] % row[col] != 0)
      return false;
    if (!checked::sub_multiple(v, row, v[col] / row[col]))
      return overflow_error("hnf_contains");
  }
  return is_zero(v);
}

Result<Hnf3> hnf_from_rows(std::span<const Z3> generators) {
  Hnf3 hnf;
  for (const Z3 &g : generators) {
    auto step = hnf_insert(hnf, g);
    if (!step)
      return std::unexpected(step.error());
  }
  return hnf;
}

Result<LatticeBasis> row_hnf(std::span<const Z3> generators) {
  auto hnf = hnf_from_rows(generators);
  if (!hnf)
    return std::unexpected(hnf.error());
  return hnf->to_basis();
}

Result<SmithData> smith_invariants(std::span<const Z3> generators) {
  auto canonical = hnf_from_rows(generators);
  if (!canonical)
    return std::unexpected(canonical.error());
  const Hnf3 &h = *canonical;
  SmithData s;
  s.rank = h.rank;
  s.free_rank = 3 - s.rank;
  if (h.rank == 0)
    return s;
  Int g1 = 0;
  for (std::uint8_t i = 0; i < h.rank; ++i)
    for (const Int v : h.rows[i])
      if (!checked::gcd(g1, v, g1))
        return overflow_error("smith_invariants gcd");
  s.factors.push_back(g1);
  if (h.rank < 2)
    return s;
  Int g2 = 0;
  for (std::uint8_t i = 0; i < h.rank; ++i)
    for (std::uint8_t j = i + 1; j < h.rank; ++j)
      for (std::size_t c = 0; c < 3; ++c)
        for (std::size_t d = c + 1; d < 3; ++d) {
          Int p = 0, q = 0, minor = 0;
          if (!checked::mul(h.rows[i][c], h.rows[j][d], p) ||
              !checked::mul(h.rows[i][d], h.rows[j][c], q) ||
              !checked::sub(p, q, minor) || !checked::gcd(g2, minor, g2))
            return overflow_error("smith_invariants 2x2 minors");
        }
  // D1 | D2 and D2 > 0 when rank >= 2.
  s.factors.push_back(g2 / g1);
  if (h.rank == 3) {
    Int det = 0;
    if (!det3(IntMatrix3{h.rows[0], h.rows[1], h.rows[2]}, det) ||
        !checked::abs(det, det))
      return overflow_error("smith_invariants determinant");
    s.factors.push_back(det / g2);
  }
  return s;
}

Result<bool> contains(std::span<const Z3> generators, Z3 v) {
  auto hnf = hnf_from_rows(generators);
  if (!hnf)
    return std::unexpected(hnf.error());
  return hnf_contains(*hnf, v);
}

Result<IntMatrix3> inverse_unimodular(const IntMatrix3 &a) {
  Int d = 0;
  if (!det3(a, d))
    return overflow_error("inverse_unimodular determinant");
  if (d != 1 && d != -1)
    return make_error(ErrorCode::InvalidInput,
                      "basis transform must have determinant +1 or -1");
  IntMatrix3 b{};
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j) {
      // adj(a)[i][j] = (-1)^(i+j) * minor(a; row j, column i).
      std::array<std::size_t, 2> rr{}, cc{};
      std::size_t r = 0, c = 0;
      for (std::size_t k = 0; k < 3; ++k) {
        if (k != j)
          rr[r++] = k;
        if (k != i)
          cc[c++] = k;
      }
      Int p = 0, q = 0, minor = 0;
      if (!checked::mul(a[rr[0]][cc[0]], a[rr[1]][cc[1]], p) ||
          !checked::mul(a[rr[0]][cc[1]], a[rr[1]][cc[0]], q) ||
          !checked::sub(p, q, minor))
        return overflow_error("inverse_unimodular cofactor");
      if ((i + j) % 2 != 0 && !checked::neg(minor, minor))
        return overflow_error("inverse_unimodular cofactor sign");
      // 1/d == d for d = +-1.
      if (!checked::mul(minor, d, b[i][j]))
        return overflow_error("inverse_unimodular scale");
    }
  return b;
}

Result<Z3> multiply(const IntMatrix3 &a, const Z3 &v) {
  Z3 out{};
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j) {
      Int product = 0;
      if (!checked::mul(a[i][j], v[j], product) ||
          !checked::add(out[i], product, out[i]))
        return overflow_error("multiply");
    }
  return out;
}

Result<LatticeBasis> reexpress(std::span<const Z3> old_basis,
                               const IntMatrix3 &U) {
  auto inverse = inverse_unimodular(U);
  if (!inverse)
    return std::unexpected(inverse.error());
  Hnf3 out;
  for (const Z3 &w : old_basis) {
    auto mapped = multiply(*inverse, w);
    if (!mapped)
      return std::unexpected(mapped.error());
    auto step = hnf_insert(out, *mapped);
    if (!step)
      return std::unexpected(step.error());
  }
  return out.to_basis();
}

} // namespace titan_ptopo
