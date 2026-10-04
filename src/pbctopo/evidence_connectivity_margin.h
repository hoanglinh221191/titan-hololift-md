#ifndef TITAN_PBCTOPO_EVIDENCE_CONNECTIVITY_MARGIN_H
#define TITAN_PBCTOPO_EVIDENCE_CONNECTIVITY_MARGIN_H

#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace titan_pbctopo {

// Provider-agnostic evidence between two component nodes. Larger support means
// stronger evidence. Endpoints are canonicalized by the kernel.
struct PbctopoEvidenceSupportEdge {
  std::size_t a = 0;
  std::size_t b = 0;
  double support = 0.0;
};

struct PbctopoEvidenceConnectivityMargin {
  std::size_t node_count = 0;
  double threshold = std::numeric_limits<double>::quiet_NaN();
  std::size_t canonical_edge_count = 0;
  std::size_t thresholded_edge_count = 0;
  std::size_t thresholded_component_count = 0;
  bool thresholded_connected = false;

  // True when the complete supplied edge relation, without thresholding, can
  // span every node. A singleton spans vacuously, but has no defined margin.
  bool supplied_relation_spans = false;
  bool margin_defined = false;
  double maximum_bottleneck_support =
      std::numeric_limits<double>::quiet_NaN();
  double signed_margin = std::numeric_limits<double>::quiet_NaN();

  PbctopoEvidenceSupportEdge bottleneck_witness{};
  std::vector<PbctopoEvidenceSupportEdge> spanning_tree_witness;
};

// Canonicalizes undirected endpoints, retains the maximum support for duplicate
// endpoint pairs, and includes an edge in the thresholded graph when
// edge.support >= threshold. Throws std::invalid_argument for zero nodes, a
// nonfinite threshold/support, a self edge, or an endpoint outside
// [0, node_count). Throws std::overflow_error when a finite bottleneck and
// threshold cannot be represented as a finite double-precision margin.
PbctopoEvidenceConnectivityMargin
compute_pbctopo_evidence_connectivity_margin(
    std::size_t node_count,
    std::span<const PbctopoEvidenceSupportEdge> edges, double threshold);

} // namespace titan_pbctopo

#endif
