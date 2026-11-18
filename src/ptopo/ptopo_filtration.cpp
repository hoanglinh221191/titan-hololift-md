#include "ptopo_filtration.h"
#include "ptopo_potential_dsu.h"

#include <algorithm>
#include <cmath>

namespace titan_ptopo {
namespace {

struct EdgePlan {
  std::vector<std::uint32_t> hard;     // edge-ID order
  std::vector<std::uint32_t> contacts; // (activation, edge ID) order
};

Result<EdgePlan> plan_edges(const Graph &g, const FiltrationOptions &options) {
  const auto &thresholds = options.report_thresholds;
  for (std::size_t i = 0; i < thresholds.size(); ++i) {
    if (!std::isfinite(thresholds[i]) || thresholds[i] < 0.0 ||
        (i > 0 && !(thresholds[i - 1] < thresholds[i])))
      return make_error(ErrorCode::InvalidInput,
                        "report thresholds must be finite, >= 0 and strictly "
                        "increasing");
  }
  EdgePlan plan;
  for (std::uint32_t i = 0; i < g.edges.size(); ++i)
    (g.edges[i].kind == EdgeKind::Hard ? plan.hard : plan.contacts)
        .push_back(i);
  std::sort(plan.hard.begin(), plan.hard.end(),
            [&](std::uint32_t a, std::uint32_t b) {
              return g.edges[a].id < g.edges[b].id;
            });
  std::sort(plan.contacts.begin(), plan.contacts.end(),
            [&](std::uint32_t a, std::uint32_t b) {
              const Edge &x = g.edges[a];
              const Edge &y = g.edges[b];
              return x.activation != y.activation ? x.activation < y.activation
                                                  : x.id < y.id;
            });
  std::size_t levels = 1 + thresholds.size();
  if (thresholds.empty()) {
    for (std::size_t i = 0; i < plan.contacts.size(); ++i)
      if (i == 0 || g.edges[plan.contacts[i]].activation !=
                        g.edges[plan.contacts[i - 1]].activation)
        ++levels;
  }
  if (levels > options.max_steps)
    return make_error(ErrorCode::ResourceLimit,
                      "filtration step budget exceeded (includes hard-only "
                      "baseline)");
  return plan;
}

// One component per root: its subgroup lives at the root. Ranks only grow
// in an add-only filtration, so the maximum and the largest size are
// running maxima.
class IncrementalState {
public:
  explicit IncrementalState(std::size_t n)
      : dsu_(n), hnf_(n), beta0_(n), largest_(n > 0 ? 1 : 0) {}

  Result<void> add(const Edge &e) {
    detail::DsuInsert s;
    if (!dsu_.insert(e.u, e.v, e.shift, s))
      return overflow_error("filtration union-find potentials");
    ++edges_;
    if (s.merged) {
      --beta0_;
      // A bridge adds no cycle: W(C1 + C2) = W(C1) + W(C2).
      Hnf3 merged = hnf_[s.root];
      if (auto m = hnf_merge(merged, hnf_[s.absorbed]); !m)
        return m;
      sum_rank_ = sum_rank_ - hnf_[s.root].rank - hnf_[s.absorbed].rank +
                  merged.rank;
      hnf_[s.root] = merged;
      hnf_[s.absorbed] = Hnf3{};
      max_rank_ = std::max<std::size_t>(max_rank_, merged.rank);
      largest_ = std::max<std::size_t>(largest_, dsu_.component_size(s.root));
      return {};
    }
    ++beta1_;
    if (is_zero(s.residual))
      return {};
    Hnf3 &h = hnf_[s.root];
    const std::size_t before = h.rank;
    if (auto ins = hnf_insert(h, s.residual); !ins)
      return ins;
    sum_rank_ = sum_rank_ - before + h.rank;
    max_rank_ = std::max<std::size_t>(max_rank_, h.rank);
    return hnf_insert(aggregate_, s.residual);
  }

  [[nodiscard]] FiltrationStep snapshot(std::optional<double> level) const {
    FiltrationStep step;
    step.activation = level;
    step.edges_present = edges_;
    step.beta0 = beta0_;
    step.beta1 = beta1_;
    step.maximum_component_winding_rank = max_rank_;
    step.componentwise_ambient_null_rank = beta1_ - sum_rank_;
    step.global_ambient_null_rank = beta1_ - aggregate_.rank;
    step.largest_component_nodes = largest_;
    step.aggregate_winding_hnf = aggregate_.to_basis();
    return step;
  }

private:
  detail::PotentialDsu dsu_;
  std::vector<Hnf3> hnf_; // meaningful at roots only
  Hnf3 aggregate_;
  std::size_t beta0_ = 0;
  std::size_t beta1_ = 0;
  std::size_t edges_ = 0;
  std::size_t sum_rank_ = 0; // sum over components of rank W_C
  std::size_t max_rank_ = 0;
  std::size_t largest_ = 0;
};

} // namespace

FiltrationStep summarize(const GraphAudit &a, std::optional<double> activation,
                         std::size_t edges_present) {
  FiltrationStep step;
  step.activation = activation;
  step.edges_present = edges_present;
  step.beta0 = a.beta0;
  step.beta1 = a.beta1;
  step.maximum_component_winding_rank = a.maximum_component_winding_rank;
  step.componentwise_ambient_null_rank = a.componentwise_ambient_null_rank;
  step.global_ambient_null_rank = a.global_ambient_null_rank;
  for (const ComponentAudit &c : a.components)
    step.largest_component_nodes =
        std::max(step.largest_component_nodes, c.node_indices.size());
  step.aggregate_winding_hnf = a.aggregate_winding_hnf;
  return step;
}

Result<std::vector<FiltrationStep>>
filtration(const Graph &g, const FiltrationOptions &options) {
  if (auto ok = validate(g); !ok)
    return std::unexpected(ok.error());
  auto plan = plan_edges(g, options);
  if (!plan)
    return std::unexpected(plan.error());
  IncrementalState state(g.nodes.size());
  for (const std::uint32_t index : plan->hard)
    if (auto ok = state.add(g.edges[index]); !ok)
      return std::unexpected(ok.error());
  std::vector<FiltrationStep> steps;
  steps.push_back(state.snapshot(std::nullopt));
  const auto &contacts = plan->contacts;
  std::size_t next = 0;
  if (options.report_thresholds.empty()) {
    while (next < contacts.size()) {
      const double level = g.edges[contacts[next]].activation;
      while (next < contacts.size() &&
             g.edges[contacts[next]].activation == level) {
        if (auto ok = state.add(g.edges[contacts[next]]); !ok)
          return std::unexpected(ok.error());
        ++next;
      }
      steps.push_back(state.snapshot(level));
    }
    return steps;
  }
  for (const double level : options.report_thresholds) {
    while (next < contacts.size() &&
           g.edges[contacts[next]].activation <= level) {
      if (auto ok = state.add(g.edges[contacts[next]]); !ok)
        return std::unexpected(ok.error());
      ++next;
    }
    steps.push_back(state.snapshot(level));
  }
  return steps;
}

Result<std::vector<FiltrationStep>>
filtration_reference(const Graph &g, const FiltrationOptions &options) {
  if (auto ok = validate(g); !ok)
    return std::unexpected(ok.error());
  auto plan = plan_edges(g, options);
  if (!plan)
    return std::unexpected(plan.error());
  Graph current = g;
  current.edges.clear();
  for (const std::uint32_t index : plan->hard)
    current.edges.push_back(g.edges[index]);
  std::vector<FiltrationStep> steps;
  const auto record = [&](std::optional<double> level) -> Result<void> {
    auto a = audit(current);
    if (!a)
      return std::unexpected(a.error());
    steps.push_back(summarize(*a, level, current.edges.size()));
    return {};
  };
  if (auto ok = record(std::nullopt); !ok)
    return std::unexpected(ok.error());
  const auto &contacts = plan->contacts;
  std::size_t next = 0;
  if (options.report_thresholds.empty()) {
    while (next < contacts.size()) {
      const double level = g.edges[contacts[next]].activation;
      while (next < contacts.size() &&
             g.edges[contacts[next]].activation == level)
        current.edges.push_back(g.edges[contacts[next++]]);
      if (auto ok = record(level); !ok)
        return std::unexpected(ok.error());
    }
    return steps;
  }
  for (const double level : options.report_thresholds) {
    while (next < contacts.size() &&
           g.edges[contacts[next]].activation <= level)
      current.edges.push_back(g.edges[contacts[next++]]);
    if (auto ok = record(level); !ok)
      return std::unexpected(ok.error());
  }
  return steps;
}

} // namespace titan_ptopo
