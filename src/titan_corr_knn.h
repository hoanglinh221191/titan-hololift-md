#ifndef TITAN_CORR_KNN_H
#define TITAN_CORR_KNN_H

#include "arguments.h"
#include "dccm_network.h"
#include "info.h"
#include "titan_corr_knn_core.h"

#include <fstream>
#include <string>

class TitanExecutor;

class TitanCorrKnnAnalyzer {
public:
  struct Config {
    int estimator_k{4};
    int tau_lag{5};
    bool compute_te{true};
    titan_corr_knn::PairPruningConfig pair_pruning{};
  };

  void configure(const Config &cfg) { config_ = cfg; }
  bool run(const DCCMNetwork &dccm_analyzer, TitanExecutor &executor,
           std::fstream &log);

  void export_module_outputs(const info &info_output) const;
  void export_legacy_allostery_outputs(const std::string &mi_path,
                                       const std::string &gc_path,
                                       const std::string &te_path) const;

  const Config &config() const { return config_; }
  const titan_corr_knn::ExactComputationResult &result() const { return result_; }

  static Config from_corr_knn_args(const CorrKnnArgs &args);
  static Config from_allostery_args(const arguments &args);
  static bool equivalent_config(const Config &lhs, const Config &rhs);

private:
  Config config_{};
  titan_corr_knn::ExactComputationResult result_{};
};

#endif // TITAN_CORR_KNN_H
