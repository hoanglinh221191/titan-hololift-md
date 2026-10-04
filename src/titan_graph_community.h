#ifndef TITAN_GRAPH_COMMUNITY_H
#define TITAN_GRAPH_COMMUNITY_H

#include "titan_sparse_graph.h"
#include <span>
#include <vector>

namespace titan_graph_community {

struct PartitionStats {
  std::vector<int> community_sizes;
  std::vector<double> internal_weights;
  std::vector<double> total_incident_weights;
  std::vector<double> modularity_contributions;
  double modularity{0.0};
};

struct LouvainResult {
  std::vector<int> membership;
  PartitionStats stats;
  int level_count{0};
};

[[nodiscard]] LouvainResult
run_louvain(const titan_sparse::WeightedCSRGraph<double> &graph,
            double gain_tolerance = 1e-12);

[[nodiscard]] PartitionStats
summarize_partition(const titan_sparse::WeightedCSRGraph<double> &graph,
                    std::span<const int> membership);

} // namespace titan_graph_community

#endif // TITAN_GRAPH_COMMUNITY_H
