#ifndef TITAN_SPARSE_GRAPH_MODULE_H
#define TITAN_SPARSE_GRAPH_MODULE_H

#include "arguments.h"
#include "dccm_network.h"
#include "info.h"
#include "titan_graph_community.h"
#include "titan_sparse_graph.h"
#include "titan_tpt.h"

#include <fstream>
#include <string>
#include <vector>

class TitanExecutor;

namespace titan_sparse_graph_module {

enum class SourceKind { Dccm, Lspmd, AllosteryGc, AllosteryTe, Tpt };

std::string source_kind_name(SourceKind source);

struct NodeMetadata {
  std::string node_id{};
  std::string label{};
  std::string type{};
  std::string resname{};
  std::string chain{};
  int resnr{-1};
  int is_lig{0};
  int state{-1};
  int macrostate{-1};
  char set_label{'-'};
};

struct EdgeMetadata {
  int source{-1};
  int target{-1};
  double weight{0.0};
  double raw_score{0.0};
};

struct SourceGraph {
  SourceKind source{SourceKind::Dccm};
  bool directed{false};
  std::vector<NodeMetadata> nodes{};
  std::vector<EdgeMetadata> edges{};
};

struct TptCapturedEdgeRow {
  int from{-1};
  int to{-1};
  int from_macro{-1};
  int to_macro{-1};
  double gross_flux{0.0};
  double net_flux{0.0};
  double relative_net_flux{0.0};
  double from_q{0.0};
  double to_q{0.0};
  double midpoint_q{0.0};
  double transition_proximity{0.0};
  double bottleneck_score{0.0};
  double dominant_path_support_flux{0.0};
  double dominant_path_support_fraction{0.0};
  std::size_t dominant_path_count{0};
};

struct TptCapturedNodeRow {
  int state{-1};
  char label{'I'};
  int macro{-1};
  double pi{0.0};
  double q_plus{0.0};
  double q_minus{0.0};
  double reactive_density{0.0};
  double net_in_flux{0.0};
  double net_out_flux{0.0};
  double through_flux{0.0};
  double transition_proximity{0.0};
  double bottleneck_score{0.0};
  double dominant_path_support_flux{0.0};
  double dominant_path_support_fraction{0.0};
  std::size_t dominant_path_count{0};
};

struct AnalysisConfig {
  std::string community_method{"auto"};
  int filter_topk{0};
  std::string filter_component{"all"};
};

struct AnalysisResult {
  SourceGraph filtered_graph{};
  std::vector<int> component_ids{};
  int component_count{0};
  std::vector<int> node_degree{};
  std::vector<double> node_strength{};
  std::vector<double> node_strength_norm{};
  std::vector<double> node_betweenness{};
  std::vector<double> node_betweenness_norm{};
  std::vector<double> edge_betweenness{};
  std::vector<double> edge_betweenness_norm{};
  bool community_available{false};
  std::vector<int> community_membership{};
  titan_graph_community::PartitionStats community_stats{};
};

AnalysisResult analyze_graph(const SourceGraph &graph,
                             const AnalysisConfig &cfg,
                             TitanExecutor &executor);

SourceGraph build_dccm_source_graph(const DCCMNetwork &dccm);
SourceGraph build_allostery_source_graph(const DCCMNetwork &dccm,
                                         SourceKind source);
SourceGraph build_lspmd_source_graph(const DCCMNetwork &dccm,
                                     const DCCMNetwork::LSPMDResult &result);
SourceGraph build_tpt_source_graph(const TitanTPT &analyzer,
                                   const std::vector<int> &macro_assignments);

class TitanUnifiedGraphRuntime {
public:
  void configure(const arguments &args);

  bool enabled() const noexcept { return enabled_; }
  bool wants_source(SourceKind source) const noexcept;

  void capture_lspmd(const DCCMNetwork &dccm,
                     const DCCMNetwork::LSPMDResult &result);
  void capture_tpt(const TitanTPT &analyzer,
                   const std::vector<int> &macro_assignments,
                   const std::vector<TptCapturedEdgeRow> &edge_rows,
                   const std::vector<TptCapturedNodeRow> &node_rows);

  void export_all(const arguments &args, DCCMNetwork &dccm,
                  TitanExecutor &executor, info &info_output,
                  std::fstream &log) const;

private:
  bool enabled_{false};
  bool auto_sources_{true};
  std::vector<SourceKind> requested_sources_{};
  AnalysisConfig analysis_cfg_{};

  bool have_lspmd_capture_{false};
  SourceGraph lspmd_graph_{};

  bool have_tpt_capture_{false};
  SourceGraph tpt_graph_{};
  std::vector<TptCapturedEdgeRow> tpt_edge_rows_{};
  std::vector<TptCapturedNodeRow> tpt_node_rows_{};
};

} // namespace titan_sparse_graph_module

#endif // TITAN_SPARSE_GRAPH_MODULE_H
