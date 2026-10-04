#ifndef TITAN_PBCTOPO_CONFIG_H
#define TITAN_PBCTOPO_CONFIG_H

struct arguments;

namespace titan_pbctopo {

enum class AnchorBackend {
  Auto,
  Prim,
  SpatialAll,
};

enum class SolveMode {
  Vibe,
  LlpsLocal,
};

enum class SearchPolicy {
  FastCertified,
  AuditExact,
};

enum class AnchorMetricMode {
  Full,
  MstOnly,
  NoOrder,
  NoContact,
};

enum class TemporalMode {
  Greedy,
  Viterbi,
};

constexpr int kDefaultExactComponentLimit = 6;
constexpr int kMaxExactComponentLimit = 16;
constexpr int kDefaultBeamWidth = 12;
constexpr int kMaxBeamWidth = 4096;

struct RuntimeCutoffs {
  double bond_cutoff = 2.15;
  double clash_cutoff = 1.05;
  double pair_contact_cutoff = 4.5;
  double continuity_cutoff = 20.0;
  double vdw_clash_scale = 0.55;
};

RuntimeCutoffs resolve_runtime_cutoffs(const arguments &Arguments);
SolveMode resolve_solve_mode(const arguments &Arguments);
const char *solve_mode_name(SolveMode mode);
SearchPolicy resolve_search_policy(const arguments &Arguments);
const char *search_policy_name(SearchPolicy policy);
AnchorBackend resolve_anchor_backend(const arguments &Arguments);
const char *anchor_backend_name(AnchorBackend backend);
AnchorMetricMode resolve_anchor_metric_mode(const arguments &Arguments);
const char *anchor_metric_mode_name(AnchorMetricMode mode);
TemporalMode resolve_temporal_mode(const arguments &Arguments);
const char *temporal_mode_name(TemporalMode mode);

} // namespace titan_pbctopo

#endif
