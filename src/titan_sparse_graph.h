#pragma once

#include <algorithm>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace titan_sparse {

class CSRTopology {
public:
  void clear() noexcept {
    node_count_ = 0;
    row_offsets_.clear();
    col_indices_.clear();
  }

  void assign(int node_count, std::vector<int> row_offsets,
              std::vector<int> col_indices) {
    if (node_count < 0) {
      throw std::invalid_argument("CSRTopology: node_count must be >= 0");
    }
    if (row_offsets.size() != static_cast<std::size_t>(node_count + 1)) {
      throw std::invalid_argument("CSRTopology: row_offsets size mismatch");
    }
    if (!row_offsets.empty() && row_offsets.front() != 0) {
      throw std::invalid_argument("CSRTopology: row_offsets must start at 0");
    }
    if (!row_offsets.empty() &&
        row_offsets.back() != static_cast<int>(col_indices.size())) {
      throw std::invalid_argument("CSRTopology: row_offsets end mismatch");
    }

    for (int node = 0; node < node_count; ++node) {
      const int row_begin = row_offsets[static_cast<std::size_t>(node)];
      const int row_end = row_offsets[static_cast<std::size_t>(node + 1)];
      if (row_begin > row_end) {
        throw std::invalid_argument(
            "CSRTopology: row_offsets must be nondecreasing");
      }
      for (int edge_idx = row_begin; edge_idx < row_end; ++edge_idx) {
        const int col = col_indices[static_cast<std::size_t>(edge_idx)];
        if (col < 0 || col >= node_count) {
          throw std::invalid_argument(
              "CSRTopology: column index out of range");
        }
      }
    }

    node_count_ = node_count;
    row_offsets_ = std::move(row_offsets);
    col_indices_ = std::move(col_indices);
  }

  [[nodiscard]] int node_count() const noexcept { return node_count_; }
  [[nodiscard]] bool empty() const noexcept { return node_count_ == 0; }

  [[nodiscard]] const std::vector<int> &row_offsets() const noexcept {
    return row_offsets_;
  }
  [[nodiscard]] const std::vector<int> &col_indices() const noexcept {
    return col_indices_;
  }

  [[nodiscard]] int row_begin(int node) const noexcept {
    return row_offsets_[static_cast<std::size_t>(node)];
  }
  [[nodiscard]] int row_end(int node) const noexcept {
    return row_offsets_[static_cast<std::size_t>(node + 1)];
  }
  [[nodiscard]] int degree(int node) const noexcept {
    return row_end(node) - row_begin(node);
  }
  [[nodiscard]] int col_index(int edge_idx) const noexcept {
    return col_indices_[static_cast<std::size_t>(edge_idx)];
  }
  [[nodiscard]] std::span<const int> neighbors(int node) const noexcept {
    const int begin = row_begin(node);
    return std::span<const int>(col_indices_.data() + begin,
                                static_cast<std::size_t>(degree(node)));
  }

  [[nodiscard]] int find_edge(int from, int to) const noexcept {
    if (from < 0 || from >= node_count_ || to < 0 || to >= node_count_) {
      return -1;
    }
    for (int edge_idx = row_begin(from); edge_idx < row_end(from); ++edge_idx) {
      if (col_indices_[static_cast<std::size_t>(edge_idx)] == to) {
        return edge_idx;
      }
    }
    return -1;
  }

  static CSRTopology from_undirected_edges(
      int node_count, const std::vector<std::pair<int, int>> &edges) {
    if (node_count < 0) {
      throw std::invalid_argument("CSRTopology: node_count must be >= 0");
    }

    std::vector<int> row_offsets(static_cast<std::size_t>(node_count + 1), 0);
    for (const auto &[lhs, rhs] : edges) {
      if (lhs < 0 || rhs < 0 || lhs >= node_count || rhs >= node_count) {
        throw std::invalid_argument(
            "CSRTopology: undirected edge index out of range");
      }
      if (lhs == rhs) {
        continue;
      }
      ++row_offsets[static_cast<std::size_t>(lhs + 1)];
      ++row_offsets[static_cast<std::size_t>(rhs + 1)];
    }

    for (int node = 0; node < node_count; ++node) {
      row_offsets[static_cast<std::size_t>(node + 1)] +=
          row_offsets[static_cast<std::size_t>(node)];
    }

    std::vector<int> col_indices(
        static_cast<std::size_t>(row_offsets[static_cast<std::size_t>(node_count)]),
        -1);
    std::vector<int> cursor = row_offsets;
    for (const auto &[lhs, rhs] : edges) {
      if (lhs == rhs) {
        continue;
      }
      col_indices[static_cast<std::size_t>(cursor[static_cast<std::size_t>(lhs)]++)] =
          rhs;
      col_indices[static_cast<std::size_t>(cursor[static_cast<std::size_t>(rhs)]++)] =
          lhs;
    }

    CSRTopology topology;
    topology.assign(node_count, std::move(row_offsets), std::move(col_indices));
    return topology;
  }

  static CSRTopology from_directed_edges(
      int node_count, const std::vector<std::pair<int, int>> &edges) {
    if (node_count < 0) {
      throw std::invalid_argument("CSRTopology: node_count must be >= 0");
    }

    std::vector<int> row_offsets(static_cast<std::size_t>(node_count + 1), 0);
    for (const auto &[from, to] : edges) {
      if (from < 0 || to < 0 || from >= node_count || to >= node_count) {
        throw std::invalid_argument(
            "CSRTopology: directed edge index out of range");
      }
      if (from == to) {
        continue;
      }
      ++row_offsets[static_cast<std::size_t>(from + 1)];
    }

    for (int node = 0; node < node_count; ++node) {
      row_offsets[static_cast<std::size_t>(node + 1)] +=
          row_offsets[static_cast<std::size_t>(node)];
    }

    std::vector<int> col_indices(
        static_cast<std::size_t>(row_offsets[static_cast<std::size_t>(node_count)]),
        -1);
    std::vector<int> cursor = row_offsets;
    for (const auto &[from, to] : edges) {
      if (from == to) {
        continue;
      }
      col_indices[static_cast<std::size_t>(cursor[static_cast<std::size_t>(from)]++)] =
          to;
    }

    CSRTopology topology;
    topology.assign(node_count, std::move(row_offsets), std::move(col_indices));
    return topology;
  }

private:
  int node_count_{0};
  std::vector<int> row_offsets_;
  std::vector<int> col_indices_;
};

template <typename WeightT> class WeightedCSRGraph {
public:
  void clear() noexcept {
    topology_.clear();
    values_.clear();
  }

  void assign(int node_count, std::vector<int> row_offsets,
              std::vector<int> col_indices, std::vector<WeightT> values) {
    if (col_indices.size() != values.size()) {
      throw std::invalid_argument(
          "WeightedCSRGraph: value count must match edge count");
    }
    topology_.assign(node_count, std::move(row_offsets), std::move(col_indices));
    values_ = std::move(values);
  }

  static WeightedCSRGraph from_undirected_edges(
      int node_count, const std::vector<std::tuple<int, int, WeightT>> &edges) {
    if (node_count < 0) {
      throw std::invalid_argument(
          "WeightedCSRGraph: node_count must be >= 0");
    }

    std::vector<int> row_offsets(static_cast<std::size_t>(node_count + 1), 0);
    for (const auto &[lhs, rhs, weight] : edges) {
      (void)weight;
      if (lhs < 0 || rhs < 0 || lhs >= node_count || rhs >= node_count) {
        throw std::invalid_argument(
            "WeightedCSRGraph: undirected edge index out of range");
      }
      if (lhs == rhs) {
        continue;
      }
      ++row_offsets[static_cast<std::size_t>(lhs + 1)];
      ++row_offsets[static_cast<std::size_t>(rhs + 1)];
    }

    for (int node = 0; node < node_count; ++node) {
      row_offsets[static_cast<std::size_t>(node + 1)] +=
          row_offsets[static_cast<std::size_t>(node)];
    }

    std::vector<int> col_indices(
        static_cast<std::size_t>(row_offsets[static_cast<std::size_t>(node_count)]),
        -1);
    std::vector<WeightT> values(col_indices.size(), WeightT{});
    std::vector<int> cursor = row_offsets;

    for (const auto &[lhs, rhs, weight] : edges) {
      if (lhs == rhs) {
        continue;
      }
      const int lhs_slot = cursor[static_cast<std::size_t>(lhs)]++;
      col_indices[static_cast<std::size_t>(lhs_slot)] = rhs;
      values[static_cast<std::size_t>(lhs_slot)] = weight;

      const int rhs_slot = cursor[static_cast<std::size_t>(rhs)]++;
      col_indices[static_cast<std::size_t>(rhs_slot)] = lhs;
      values[static_cast<std::size_t>(rhs_slot)] = weight;
    }

    WeightedCSRGraph graph;
    graph.assign(node_count, std::move(row_offsets), std::move(col_indices),
                 std::move(values));
    return graph;
  }

  static WeightedCSRGraph from_directed_edges(
      int node_count, const std::vector<std::tuple<int, int, WeightT>> &edges) {
    if (node_count < 0) {
      throw std::invalid_argument(
          "WeightedCSRGraph: node_count must be >= 0");
    }

    std::vector<int> row_offsets(static_cast<std::size_t>(node_count + 1), 0);
    for (const auto &[from, to, weight] : edges) {
      (void)weight;
      if (from < 0 || to < 0 || from >= node_count || to >= node_count) {
        throw std::invalid_argument(
            "WeightedCSRGraph: directed edge index out of range");
      }
      if (from == to) {
        continue;
      }
      ++row_offsets[static_cast<std::size_t>(from + 1)];
    }

    for (int node = 0; node < node_count; ++node) {
      row_offsets[static_cast<std::size_t>(node + 1)] +=
          row_offsets[static_cast<std::size_t>(node)];
    }

    std::vector<int> col_indices(
        static_cast<std::size_t>(row_offsets[static_cast<std::size_t>(node_count)]),
        -1);
    std::vector<WeightT> values(col_indices.size(), WeightT{});
    std::vector<int> cursor = row_offsets;

    for (const auto &[from, to, weight] : edges) {
      if (from == to) {
        continue;
      }
      const int slot = cursor[static_cast<std::size_t>(from)]++;
      col_indices[static_cast<std::size_t>(slot)] = to;
      values[static_cast<std::size_t>(slot)] = weight;
    }

    WeightedCSRGraph graph;
    graph.assign(node_count, std::move(row_offsets), std::move(col_indices),
                 std::move(values));
    return graph;
  }

  [[nodiscard]] const CSRTopology &topology() const noexcept { return topology_; }
  [[nodiscard]] int node_count() const noexcept { return topology_.node_count(); }
  [[nodiscard]] bool empty() const noexcept { return topology_.empty(); }

  [[nodiscard]] const std::vector<int> &row_offsets() const noexcept {
    return topology_.row_offsets();
  }
  [[nodiscard]] const std::vector<int> &col_indices() const noexcept {
    return topology_.col_indices();
  }
  [[nodiscard]] const std::vector<WeightT> &values() const noexcept {
    return values_;
  }

  [[nodiscard]] int row_begin(int node) const noexcept {
    return topology_.row_begin(node);
  }
  [[nodiscard]] int row_end(int node) const noexcept {
    return topology_.row_end(node);
  }
  [[nodiscard]] int degree(int node) const noexcept {
    return topology_.degree(node);
  }
  [[nodiscard]] int col_index(int edge_idx) const noexcept {
    return topology_.col_index(edge_idx);
  }
  [[nodiscard]] const WeightT &value(int edge_idx) const noexcept {
    return values_[static_cast<std::size_t>(edge_idx)];
  }

  [[nodiscard]] WeightT find_value(int from, int to,
                                   WeightT fallback = WeightT{}) const noexcept {
    const int edge_idx = topology_.find_edge(from, to);
    if (edge_idx < 0) {
      return fallback;
    }
    return values_[static_cast<std::size_t>(edge_idx)];
  }

private:
  CSRTopology topology_;
  std::vector<WeightT> values_;
};

} // namespace titan_sparse
