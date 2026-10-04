#include "evidence_connectivity_margin.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace titan_pbctopo {
namespace {

class EvidenceDisjointSet {
public:
  explicit EvidenceDisjointSet(std::size_t count)
      : parent_(count), size_(count, 1), component_count_(count) {
    std::iota(parent_.begin(), parent_.end(), 0);
  }

  std::size_t component_count() const noexcept { return component_count_; }

  bool unite(std::size_t a, std::size_t b) {
    a = find(a);
    b = find(b);
    if (a == b)
      return false;

    if (size_[a] < size_[b] || (size_[a] == size_[b] && a > b))
      std::swap(a, b);
    parent_[b] = a;
    size_[a] += size_[b];
    --component_count_;
    return true;
  }

private:
  std::size_t find(std::size_t value) {
    while (parent_[value] != value) {
      parent_[value] = parent_[parent_[value]];
      value = parent_[value];
    }
    return value;
  }

  std::vector<std::size_t> parent_;
  std::vector<std::size_t> size_;
  std::size_t component_count_ = 0;
};

bool stronger_edge_first(const PbctopoEvidenceSupportEdge &lhs,
                         const PbctopoEvidenceSupportEdge &rhs) {
  if (lhs.support != rhs.support)
    return lhs.support > rhs.support;
  if (lhs.a != rhs.a)
    return lhs.a < rhs.a;
  return lhs.b < rhs.b;
}

bool endpoint_pair_less(const PbctopoEvidenceSupportEdge &lhs,
                        const PbctopoEvidenceSupportEdge &rhs) {
  if (lhs.a != rhs.a)
    return lhs.a < rhs.a;
  return lhs.b < rhs.b;
}

std::vector<PbctopoEvidenceSupportEdge> canonicalize_edges(
    std::size_t node_count,
    std::span<const PbctopoEvidenceSupportEdge> edges) {
  std::map<std::pair<std::size_t, std::size_t>, double> support_by_pair;
  for (const auto &edge : edges) {
    if (!std::isfinite(edge.support))
      throw std::invalid_argument("evidence edge support must be finite");
    if (edge.a >= node_count || edge.b >= node_count)
      throw std::invalid_argument("evidence edge endpoint is out of range");
    if (edge.a == edge.b)
      throw std::invalid_argument("evidence edge must not be a self edge");

    const std::size_t lo = std::min(edge.a, edge.b);
    const std::size_t hi = std::max(edge.a, edge.b);
    const double support = edge.support == 0.0 ? 0.0 : edge.support;
    auto [it, inserted] = support_by_pair.try_emplace({lo, hi}, support);
    if (!inserted && support > it->second)
      it->second = support;
  }

  std::vector<PbctopoEvidenceSupportEdge> canonical;
  canonical.reserve(support_by_pair.size());
  for (const auto &[endpoints, support] : support_by_pair)
    canonical.push_back({endpoints.first, endpoints.second, support});
  return canonical;
}

} // namespace

PbctopoEvidenceConnectivityMargin
compute_pbctopo_evidence_connectivity_margin(
    std::size_t node_count,
    std::span<const PbctopoEvidenceSupportEdge> edges, double threshold) {
  if (node_count == 0)
    throw std::invalid_argument("evidence connectivity node count must be positive");
  if (!std::isfinite(threshold))
    throw std::invalid_argument("evidence connectivity threshold must be finite");

  PbctopoEvidenceConnectivityMargin result;
  result.node_count = node_count;
  result.threshold = threshold == 0.0 ? 0.0 : threshold;

  auto ranked_edges = canonicalize_edges(node_count, edges);
  result.canonical_edge_count = ranked_edges.size();
  std::sort(ranked_edges.begin(), ranked_edges.end(), stronger_edge_first);

  EvidenceDisjointSet thresholded_sets(node_count);
  for (const auto &edge : ranked_edges) {
    if (edge.support < result.threshold)
      continue;
    ++result.thresholded_edge_count;
    thresholded_sets.unite(edge.a, edge.b);
  }
  result.thresholded_component_count = thresholded_sets.component_count();
  result.thresholded_connected =
      node_count > 0 && result.thresholded_component_count == 1;

  if (node_count == 1) {
    result.supplied_relation_spans = true;
    return result;
  }
  EvidenceDisjointSet spanning_sets(node_count);
  std::vector<PbctopoEvidenceSupportEdge> tree;
  tree.reserve(node_count - 1);
  for (const auto &edge : ranked_edges) {
    if (!spanning_sets.unite(edge.a, edge.b))
      continue;
    tree.push_back(edge);
    if (tree.size() == node_count - 1)
      break;
  }

  result.supplied_relation_spans = tree.size() == node_count - 1;
  if (!result.supplied_relation_spans)
    return result;

  result.margin_defined = true;
  result.spanning_tree_witness = std::move(tree);
  result.maximum_bottleneck_support =
      result.spanning_tree_witness.back().support;
  result.signed_margin = result.maximum_bottleneck_support - result.threshold;
  if (!std::isfinite(result.signed_margin))
    throw std::overflow_error(
        "evidence connectivity signed margin is not representable");

  auto bottleneck = result.spanning_tree_witness.end();
  for (auto it = result.spanning_tree_witness.begin();
       it != result.spanning_tree_witness.end(); ++it) {
    if (it->support != result.maximum_bottleneck_support)
      continue;
    if (bottleneck == result.spanning_tree_witness.end() ||
        endpoint_pair_less(*it, *bottleneck)) {
      bottleneck = it;
    }
  }
  result.bottleneck_witness = *bottleneck;
  return result;
}

} // namespace titan_pbctopo
