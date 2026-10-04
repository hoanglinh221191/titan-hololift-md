#include "pbctopo/config.h"

#include "arguments.h"

#include <string>

namespace titan_pbctopo {

RuntimeCutoffs resolve_runtime_cutoffs(const arguments &Arguments) {
  RuntimeCutoffs cutoffs;
  if (Arguments.force_f == "MARTINI") {
    cutoffs.bond_cutoff = 5.5;
    cutoffs.clash_cutoff = 3.5;
    cutoffs.pair_contact_cutoff = 8.0;
    cutoffs.continuity_cutoff = 20.0;
  }
  if (Arguments.pbc.pbctopo_bond_cutoff_set)
    cutoffs.bond_cutoff = Arguments.pbc.pbctopo_bond_cutoff;
  if (Arguments.pbc.pbctopo_clash_cutoff_set)
    cutoffs.clash_cutoff = Arguments.pbc.pbctopo_clash_cutoff;
  if (Arguments.pbc.pbctopo_contact_cutoff_set)
    cutoffs.pair_contact_cutoff = Arguments.pbc.pbctopo_contact_cutoff;
  if (Arguments.pbc.pbctopo_continuity_cutoff_set)
    cutoffs.continuity_cutoff = Arguments.pbc.pbctopo_continuity_cutoff;
  cutoffs.vdw_clash_scale = Arguments.pbc.pbctopo_vdw_clash_scale;
  return cutoffs;
}

AnchorMetricMode resolve_anchor_metric_mode(const arguments &Arguments) {
  const std::string &mode = Arguments.pbc.pbctopo_anchor_metrics;
  if (mode == "mst-only")
    return AnchorMetricMode::MstOnly;
  if (mode == "no-order")
    return AnchorMetricMode::NoOrder;
  if (mode == "no-contact")
    return AnchorMetricMode::NoContact;
  return AnchorMetricMode::Full;
}

const char *anchor_metric_mode_name(AnchorMetricMode mode) {
  switch (mode) {
  case AnchorMetricMode::MstOnly:
    return "mst-only";
  case AnchorMetricMode::NoOrder:
    return "no-order";
  case AnchorMetricMode::NoContact:
    return "no-contact";
  case AnchorMetricMode::Full:
  default:
    return "full";
  }
}

TemporalMode resolve_temporal_mode(const arguments &Arguments) {
  return Arguments.pbc.pbctopo_temporal == "viterbi" ? TemporalMode::Viterbi
                                                       : TemporalMode::Greedy;
}

const char *temporal_mode_name(TemporalMode mode) {
  return mode == TemporalMode::Viterbi ? "viterbi" : "greedy";
}

SolveMode resolve_solve_mode(const arguments &Arguments) {
  if (Arguments.pbc.pbctopo_mode == "llps-local")
    return SolveMode::LlpsLocal;
  return SolveMode::Vibe;
}

const char *solve_mode_name(SolveMode mode) {
  switch (mode) {
  case SolveMode::LlpsLocal:
    return "llps-local";
  case SolveMode::Vibe:
  default:
    return "vibe";
  }
}

SearchPolicy resolve_search_policy(const arguments &Arguments) {
  return Arguments.pbc.pbctopo_search_policy == "audit-exact"
             ? SearchPolicy::AuditExact
             : SearchPolicy::FastCertified;
}

const char *search_policy_name(SearchPolicy policy) {
  return policy == SearchPolicy::AuditExact ? "audit-exact"
                                             : "fast-certified";
}

AnchorBackend resolve_anchor_backend(const arguments &Arguments) {
  const std::string &backend = Arguments.pbc.pbctopo_anchor_backend;
  if (backend == "prim")
    return AnchorBackend::Prim;
  if (backend == "spatial-all")
    return AnchorBackend::SpatialAll;
  return AnchorBackend::Auto;
}

const char *anchor_backend_name(AnchorBackend backend) {
  switch (backend) {
  case AnchorBackend::Prim:
    return "prim";
  case AnchorBackend::SpatialAll:
    return "spatial-all";
  case AnchorBackend::Auto:
  default:
    return "auto";
  }
}

} // namespace titan_pbctopo
