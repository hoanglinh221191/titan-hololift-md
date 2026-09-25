#include "spatial_lift.h"

#include "../titan_sha256.h"
#include "lattice_math.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>

namespace titan_pbctopo {
namespace {

constexpr std::uint64_t kSpatialLiftDomain = 0x5042434c49465431ULL;

class Hash128Builder {
public:
  Hash128Builder() noexcept { hash_.word(kSpatialLiftDomain); }

  void word(std::uint64_t value) noexcept { hash_.word(value); }

  PbctopoHash128 finish() const noexcept {
    const auto digest = hash_.finish128();
    return {digest[0], digest[1]};
  }

private:
  titan_hash::Sha256Builder hash_;
};

bool invert_box_matrix(std::span<const double, 9> source,
                       double matrix[3][3], double inverse[3][3]) {
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      const double value = source[static_cast<std::size_t>(3 * row + col)];
      if (!std::isfinite(value))
        return false;
      matrix[row][col] = value;
    }
  }

  const double determinant =
      matrix[0][0] *
          (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
      matrix[0][1] *
          (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
      matrix[0][2] *
          (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
  if (!std::isfinite(determinant) || std::abs(determinant) <= 1.0e-18)
    return false;

  const double inv_det = 1.0 / determinant;
  inverse[0][0] =
      (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * inv_det;
  inverse[0][1] =
      (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * inv_det;
  inverse[0][2] =
      (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * inv_det;
  inverse[1][0] =
      (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * inv_det;
  inverse[1][1] =
      (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * inv_det;
  inverse[1][2] =
      (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * inv_det;
  inverse[2][0] =
      (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * inv_det;
  inverse[2][1] =
      (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) * inv_det;
  inverse[2][2] =
      (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * inv_det;
  return true;
}

bool source_atom_less(const PbctopoSpatialLiftAtom &lhs,
                      const PbctopoSpatialLiftAtom &rhs) {
  return std::tie(lhs.owner, lhs.source_atom_id) <
         std::tie(rhs.owner, rhs.source_atom_id);
}

bool checked_add_image(const Int3 &lhs, const Int3 &rhs, Int3 &result) {
  return checked_int64_add(lhs.x, rhs.x, result.x) &&
         checked_int64_add(lhs.y, rhs.y, result.y) &&
         checked_int64_add(lhs.z, rhs.z, result.z);
}

bool checked_negate_image(const Int3 &source, Int3 &result) {
  constexpr auto min_value = std::numeric_limits<std::int64_t>::min();
  if (source.x == min_value || source.y == min_value || source.z == min_value)
    return false;
  result = {-source.x, -source.y, -source.z};
  return true;
}

struct LiftEdgeDelta {
  std::uint32_t atom_a = 0;
  std::uint32_t atom_b = 0;
  Int3 delta;
  bool ambiguous = false;
};

struct AdjacencyEntry {
  std::uint32_t atom = 0;
  Int3 delta;
};

} // namespace

PbctopoSpatialLiftResult build_pbctopo_spatial_lift(
    std::span<const double, 9> box_matrix,
    std::span<const PbctopoSpatialLiftAtom> atoms,
    std::span<const PbctopoSpatialLiftEdge> hard_edges) {
  PbctopoSpatialLiftResult result;
  if (atoms.empty()) {
    result.error = "spatial lift requires at least one atom";
    return result;
  }
  if (atoms.size() > std::numeric_limits<std::uint32_t>::max()) {
    result.error = "spatial lift atom count exceeds uint32 range";
    return result;
  }

  double matrix[3][3]{};
  double inverse[3][3]{};
  if (!invert_box_matrix(box_matrix, matrix, inverse)) {
    result.error = "spatial lift box matrix is invalid";
    return result;
  }
  const PbctopoLatticeMetric lattice(matrix, inverse);
  if (!lattice.valid()) {
    result.error = "spatial lift lattice metric is invalid";
    return result;
  }

  std::vector<std::size_t> canonical_to_input(atoms.size());
  std::iota(canonical_to_input.begin(), canonical_to_input.end(), 0);
  std::sort(canonical_to_input.begin(), canonical_to_input.end(),
            [&](std::size_t lhs, std::size_t rhs) {
              return source_atom_less(atoms[lhs], atoms[rhs]);
            });
  for (std::size_t idx = 0; idx < canonical_to_input.size(); ++idx) {
    const auto &atom = atoms[canonical_to_input[idx]];
    for (double value : atom.wrapped_cartesian) {
      if (!std::isfinite(value)) {
        result.error = "spatial lift atom coordinate is non-finite";
        return result;
      }
    }
    if (idx != 0) {
      const auto &previous = atoms[canonical_to_input[idx - 1]];
      if (previous.owner == atom.owner &&
          previous.source_atom_id == atom.source_atom_id) {
        result.error = "spatial lift atom keys are not unique";
        return result;
      }
    }
  }

  std::vector<std::array<double, 3>> scaled(atoms.size());
  for (std::size_t dense_idx = 0; dense_idx < atoms.size(); ++dense_idx) {
    const auto &atom = atoms[canonical_to_input[dense_idx]];
    scaled[dense_idx] = lattice.cartesian_to_scaled(
        atom.wrapped_cartesian[0], atom.wrapped_cartesian[1],
        atom.wrapped_cartesian[2]);
  }

  std::vector<PbctopoSpatialLiftEdge> edges(hard_edges.begin(),
                                             hard_edges.end());
  std::sort(edges.begin(), edges.end(), [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.atom_a, lhs.atom_b) <
           std::tie(rhs.atom_a, rhs.atom_b);
  });
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

  std::vector<LiftEdgeDelta> deltas;
  deltas.reserve(edges.size());
  for (const auto &edge : edges) {
    if (edge.atom_a >= edge.atom_b || edge.atom_b >= atoms.size()) {
      result.error = "spatial lift hard edge is not canonical";
      return result;
    }
    const auto &a = scaled[edge.atom_a];
    const auto &b = scaled[edge.atom_b];
    const auto mic = lattice.nearest_image_scaled(
        b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    if (!mic.valid) {
      result.error = "spatial lift MIC query failed";
      return result;
    }
    deltas.push_back({edge.atom_a, edge.atom_b,
                      {mic.image_x, mic.image_y, mic.image_z},
                      mic.ambiguous});
    if (mic.ambiguous)
      ++result.ambiguous_hard_edges;
  }

  std::vector<std::vector<AdjacencyEntry>> adjacency(atoms.size());
  for (const auto &edge : deltas) {
    Int3 reverse;
    if (!checked_negate_image(edge.delta, reverse)) {
      result.error = "spatial lift image negation overflow";
      return result;
    }
    adjacency[edge.atom_a].push_back({edge.atom_b, edge.delta});
    adjacency[edge.atom_b].push_back({edge.atom_a, reverse});
  }
  for (auto &neighbors : adjacency) {
    std::sort(neighbors.begin(), neighbors.end(), [](const auto &lhs,
                                                      const auto &rhs) {
      return lhs.atom < rhs.atom;
    });
  }

  std::vector<Int3> canonical_images(atoms.size());
  std::vector<unsigned char> seen(atoms.size(), 0);
  std::vector<std::uint32_t> queue;
  queue.reserve(atoms.size());
  for (std::uint32_t root = 0; root < atoms.size(); ++root) {
    if (seen[root] != 0)
      continue;
    seen[root] = 1;
    canonical_images[root] = {};
    queue.clear();
    queue.push_back(root);
    for (std::size_t head = 0; head < queue.size(); ++head) {
      const std::uint32_t current = queue[head];
      for (const auto &neighbor : adjacency[current]) {
        if (seen[neighbor.atom] != 0)
          continue;
        if (!checked_add_image(canonical_images[current], neighbor.delta,
                               canonical_images[neighbor.atom])) {
          result.error = "spatial lift image propagation overflow";
          return result;
        }
        seen[neighbor.atom] = 1;
        queue.push_back(neighbor.atom);
      }
    }
  }

  for (const auto &edge : deltas) {
    Int3 expected;
    if (!checked_add_image(canonical_images[edge.atom_a], edge.delta,
                           expected)) {
      result.error = "spatial lift cycle audit overflow";
      return result;
    }
    if (!(expected == canonical_images[edge.atom_b]))
      ++result.cycle_residuals;
  }

  result.atom_images.resize(atoms.size());
  for (std::size_t dense_idx = 0; dense_idx < atoms.size(); ++dense_idx)
    result.atom_images[canonical_to_input[dense_idx]] =
        canonical_images[dense_idx];

  Hash128Builder hash;
  hash.word(PBCTOPO_SPATIAL_LIFT_POLICY_VERSION);
  hash.word(atoms.size());
  for (std::size_t dense_idx = 0; dense_idx < atoms.size(); ++dense_idx) {
    const auto &atom = atoms[canonical_to_input[dense_idx]];
    const auto &image = canonical_images[dense_idx];
    hash.word(static_cast<std::uint64_t>(static_cast<std::int64_t>(atom.owner)));
    hash.word(atom.source_atom_id);
    hash.word(static_cast<std::uint64_t>(image.x));
    hash.word(static_cast<std::uint64_t>(image.y));
    hash.word(static_cast<std::uint64_t>(image.z));
  }
  hash.word(deltas.size());
  for (const auto &edge : deltas) {
    hash.word(edge.atom_a);
    hash.word(edge.atom_b);
    hash.word(static_cast<std::uint64_t>(edge.delta.x));
    hash.word(static_cast<std::uint64_t>(edge.delta.y));
    hash.word(static_cast<std::uint64_t>(edge.delta.z));
    hash.word(edge.ambiguous ? 1U : 0U);
  }
  hash.word(result.ambiguous_hard_edges);
  hash.word(result.cycle_residuals);
  result.identity = hash.finish();
  result.valid = !result.identity.empty();
  return result;
}

} // namespace titan_pbctopo
