#ifndef TITAN_PTOPO_PERIODIC_GRAPH_H
#define TITAN_PTOPO_PERIODIC_GRAPH_H

// Winding audit of a shift-labelled periodic multigraph (Backbone III.3-4,
// V.3, VI.2).
//
// For each connected component C: beta1 = E - V + 1, W_C = im Phi_C (the
// winding subgroup, row-HNF), Smith factors of coker Phi_C = Z^3 / W_C and
// ambient_null_rank = beta1 - rank W_C. The aggregate subgroup sums the
// components and is NOT the winding of one connected cluster: two disjoint
// components spanning x and y give maximum rank 1 and aggregate rank 2.
// ambient_null_rank is the rank of ker Phi, not a count of loops or cavities.

#include "ptopo_integer_lattice.h"

#include <span>

namespace titan_ptopo {

struct OrientedEdge {
  std::uint64_t edge_id = 0;
  std::int8_t orientation = 1; // +1 traverses u -> v, -1 traverses v -> u
};

// Closed walk: one non-tree edge, then the spanning-tree path back. Not a
// shortest or least-energy loop; a cycle basis is not a physical observable.
struct CycleWitness {
  std::vector<OrientedEdge> walk;
  Z3 winding{};
};

struct ComponentAudit {
  std::uint64_t canonical_node_id = 0; // minimum stable node ID
  std::vector<std::uint32_t> node_indices;
  std::size_t edge_count = 0;
  std::size_t beta1 = 0;
  std::size_t ambient_null_rank = 0;
  LatticeBasis winding_hnf;
  SmithData cokernel;
};

struct GraphAudit {
  // Sorted by canonical_node_id.
  std::vector<ComponentAudit> components;
  // node index -> index into `components`.
  std::vector<std::uint32_t> node_component;
  std::size_t beta0 = 0;
  std::size_t beta1 = 0;
  std::size_t maximum_component_winding_rank = 0;
  // Sum over components of (beta1_C - rank W_C).
  std::size_t componentwise_ambient_null_rank = 0;
  // Aggregate image; NOT the winding rank of one connected cluster.
  LatticeBasis aggregate_winding_hnf;
  // beta1 - rank(aggregate); may include cancellation across components.
  std::size_t global_ambient_null_rank = 0;
  std::vector<CycleWitness> witnesses;
};

struct AuditOptions {
  bool collect_cycle_witnesses = false;
  // Only cycles with a nonzero winding get a witness. A bead-level contact
  // graph has beta1 ~ 1e5 contractible cycles; their walks are rarely wanted.
  bool witnesses_for_nonzero_winding_only = false;
  // Total walk steps over all witnesses; exceeding it fails closed.
  std::size_t max_witness_steps = 1000000;
};

Result<void> validate(const Graph &graph);
// Weighted union-find with Z^3 potentials (iterative find, path compression,
// union by size). Edges are inserted in edge-ID order.
Result<GraphAudit> audit(const Graph &graph, const AuditOptions &options = {});
// Independent spanning-forest replay: BFS potentials, every edge's residual
// recomputed, no union-find. Shares only the HNF/Smith routines.
Result<GraphAudit> audit_reference(const Graph &graph);
Result<void> verify_witness(const Graph &graph, const CycleWitness &witness);
bool equivalent_invariants(const GraphAudit &a, const GraphAudit &b);
// s'_v = s_v + h_v relabels a'_uv = a_uv + h_u - h_v.
Result<Graph> gauge_transform(const Graph &graph,
                              std::span<const Z3> node_images);

} // namespace titan_ptopo

#endif // TITAN_PTOPO_PERIODIC_GRAPH_H
