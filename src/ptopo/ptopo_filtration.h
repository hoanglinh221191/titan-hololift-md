#ifndef TITAN_PTOPO_FILTRATION_H
#define TITAN_PTOPO_FILTRATION_H

// Cutoff filtration of a periodic graph (Backbone III.5, V.3).
//
// Hard edges are present at every level; contacts enter in nondecreasing
// activation. Adding edges only merges components and grows subgroups, so
// one union-find pass gives every level exactly: O(E alpha(V)) plus O(1)
// fixed-size HNF work per merge or nonzero residual. Equal activations are
// one batch (exact double equality, no epsilon: geometric tolerance and
// filtration order are separate policies). A graph filtration is a
// 1-complex filtration, not cavity persistence.

#include "ptopo_periodic_graph.h"

#include <optional>

namespace titan_ptopo {

struct FiltrationStep {
  // nullopt = hard-only baseline before every contact.
  std::optional<double> activation;
  std::size_t edges_present = 0;
  std::size_t beta0 = 0;
  std::size_t beta1 = 0;
  std::size_t maximum_component_winding_rank = 0;
  std::size_t componentwise_ambient_null_rank = 0;
  std::size_t global_ambient_null_rank = 0;
  std::size_t largest_component_nodes = 0;
  // Aggregate image; NOT the winding of one connected cluster.
  LatticeBasis aggregate_winding_hnf;

  friend bool operator==(const FiltrationStep &,
                         const FiltrationStep &) = default;
};

struct FiltrationOptions {
  // Includes the hard-only baseline.
  std::size_t max_steps = 1000000;
  // Empty: one step per distinct activation (every breakpoint). Otherwise
  // strictly increasing, finite, >= 0; the step at t holds the hard edges
  // and every contact with activation <= t.
  std::vector<double> report_thresholds;
};

Result<std::vector<FiltrationStep>>
filtration(const Graph &graph, const FiltrationOptions &options = {});

// Reference: a full audit() per reported level, O(K E). For tests and
// fixtures only; it must agree with filtration() step for step.
Result<std::vector<FiltrationStep>>
filtration_reference(const Graph &graph, const FiltrationOptions &options = {});

FiltrationStep summarize(const GraphAudit &audit,
                         std::optional<double> activation,
                         std::size_t edges_present);

} // namespace titan_ptopo

#endif // TITAN_PTOPO_FILTRATION_H
