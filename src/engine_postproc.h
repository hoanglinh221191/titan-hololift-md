#ifndef ENGINE_POSTPROC_H
#define ENGINE_POSTPROC_H

// engine_postproc.h - Post-loop analysis orchestration
// Extracted from impl_engine.cpp for compilation speed and clarity.
// Each function owns its own timing and logs to the shared fstream.

#include "arguments.h"
#include "cas_analysis.h"
#include "dccm_network.h"
#include "info.h"
#include "ml_extractor.h"
#include "nucleic_profiler.h"
#include "titan_kinetics.h"
#include "titan_llps.h"
#include "titan_mbar.h"
#include "titan_msm.h"
#include "titan_pca.h"
#include "titan_rms.h"
#include "titan_qnative.h"
#include "titan_tpt.h"
#include "titan_validation.h"
#include "titan_dssp.h"
#include "titan_umap.h"
#include "water_analysis.h"

#include <chrono>
#include <Eigen/Dense>
#include <fstream>
#include <memory>
#include <vector>

#include "gbsa_intrin.h" // GBSA_Aggregator complete type

class TitanExecutor;
namespace titan_sparse_graph_module {
class TitanUnifiedGraphRuntime;
}

struct PostprocEquilibriumWeights {
  std::string requested_source{"auto"};
  std::string resolved_source{"uniform"};
  std::string source_name{"uniform"};
  std::string mode_label{"uniform"};
  double target_temp_k{300.0};
  const std::vector<double> *global_frame_weights{nullptr};
  const std::vector<double> *tica_global_frame_weights{nullptr};
  const std::vector<double> *pca_global_frame_weights{nullptr};
  bool use_weights{false};
};

// ---- 1. GBSA + ML + CAS + Nucleic Acid post-processing -------------------
void run_gbsa_postproc(const arguments &args, size_t tot_snap_allfiles,
                       MLFeatureExtractor &ml_extractor,
                       GBSA_Aggregator &global_gbsa_agg,
                       CASAnalyzer &cas_analyzer, NucleicProfiler &na_profiler,
                       bool global_has_nucleic_acid, info &info_output,
                       std::fstream &log);

void run_dssp_postproc(const arguments &args, TitanDSSPAnalyzer &mol1_dssp,
                       TitanDSSPAnalyzer &mol2_dssp, info &info_output,
                       std::fstream &log);

// ---- 2. TICA + MSM post-processing ---------------------------------------
void run_tica_msm_postproc(const arguments &args,
                           DCCMNetwork &dccm_analyzer,
                           std::unique_ptr<TitanKinetics> &tica_analyzer,
                           std::unique_ptr<TitanMSM> &msm_analyzer,
                           TitanExecutor &executor, info &info_output,
                           std::fstream &log,
                           TitanValidation::RuntimeCache *validation_cache =
                               nullptr);

void run_tica_fes_postproc(const arguments &args,
                           std::unique_ptr<TitanKinetics> &tica_analyzer,
                           std::unique_ptr<TitanMSM> &msm_analyzer,
                           TitanExecutor &executor, info &info_output,
                           std::fstream &log,
                           const PostprocEquilibriumWeights &weight_selection =
                               PostprocEquilibriumWeights{},
                           TitanPCA::FesPostprocMetrics *metrics = nullptr);

void run_umap_postproc(const arguments &args,
                       std::unique_ptr<TitanMSM> &msm_analyzer,
                       std::unique_ptr<TitanUMAP> &umap_analyzer,
                       TitanExecutor &executor,
                       const info &info_output, std::fstream &log,
                       const PostprocEquilibriumWeights &weight_selection =
                           PostprocEquilibriumWeights{});

// ---- 3. LLPS post-processing ----------------------------------------------
void run_llps_postproc(const arguments &args,
                       std::unique_ptr<TitanLLPS> &llps_analyzer,
                       TitanExecutor &executor, info &info_output,
                       std::fstream &log);

// ---- 4. DCCM + PCA post-processing ----------------------------------------
void run_dccm_pca_postproc(const arguments &args, size_t tot_snap_allfiles,
                           DCCMNetwork &dccm_analyzer, TitanExecutor &executor,
                           std::chrono::milliseconds &time_network,
                           info &info_output, std::fstream &log,
                           const PostprocEquilibriumWeights &weight_selection =
                               PostprocEquilibriumWeights{},
                           const std::vector<std::string> &replica_traj_files = {},
                           TitanPCA::FesPostprocMetrics *metrics = nullptr,
                           titan_sparse_graph_module::TitanUnifiedGraphRuntime
                               *graph_runtime = nullptr);

// ---- 5. LSP-MD post-processing ---------------------------------------------
void run_lspmd_postproc(const arguments &args, DCCMNetwork &dccm_analyzer,
                        TitanExecutor &executor, info &info_output,
                        std::fstream &log,
                        const PostprocEquilibriumWeights &weight_selection =
                            PostprocEquilibriumWeights{},
                        titan_sparse_graph_module::TitanUnifiedGraphRuntime
                            *graph_runtime = nullptr);

// ---- 6. DHAM post-processing -----------------------------------------------
#include "titan_dham.h"
void run_dham_postproc(const arguments &args,
                       std::unique_ptr<TitanMSM> &msm_analyzer,
                       TitanExecutor &executor, info &info_output,
                       std::fstream &log,
                       std::vector<double> &out_frame_weights,
                       Eigen::MatrixXd *out_reweighted_tpm = nullptr,
                       Eigen::VectorXd *out_reweighted_pi = nullptr,
                       bool *out_converged = nullptr,
                       TitanValidation::DhamRebuildContext
                           *out_validation_dham_context = nullptr);

TitanMBAR::CoreResult
run_mbar_postproc(const arguments &args,
                  std::unique_ptr<TitanMSM> &msm_analyzer,
                  DCCMNetwork *dccm_analyzer, TitanExecutor &executor,
                  info &info_output, std::fstream &log,
                  std::vector<double> *out_frame_weights = nullptr,
                  std::vector<double> *out_pca_frame_weights = nullptr);

void run_mbar_projection_exports_postproc(
    const arguments &args, std::unique_ptr<TitanMSM> &msm_analyzer,
    DCCMNetwork *dccm_analyzer, TitanExecutor &executor, info &info_output,
    std::fstream &log, const TitanMBAR::CoreResult &mbar_result);

PostprocEquilibriumWeights resolve_postproc_equilibrium_weights(
    const arguments &args, const std::vector<double> *dham_frame_weights,
    const std::vector<double> *mbar_tica_frame_weights,
    const std::vector<double> *mbar_pca_frame_weights, std::fstream &log);

void run_msm_equilibrium_postproc(
    const arguments &args, std::unique_ptr<TitanMSM> &msm_analyzer,
    info &info_output, std::fstream &log,
    const PostprocEquilibriumWeights &weight_selection);

void run_tpt_postproc(const arguments &args,
                      std::unique_ptr<TitanMSM> &msm_analyzer,
                      TitanExecutor &executor, info &info_output,
                      std::fstream &log,
                      const std::vector<std::string> &lj_feature_files = {},
                      const TitanRMSAnalyzer *rms_analyzer = nullptr,
                      const TitanDSSPAnalyzer *protein_dssp_analyzer = nullptr,
                      const TitanQNativeAnalyzer *qnative_analyzer = nullptr,
                      const Eigen::MatrixXd *dham_tpm = nullptr,
                      const Eigen::VectorXd *dham_pi = nullptr,
                      bool dham_converged = false,
                      const PostprocEquilibriumWeights *weight_selection =
                          nullptr,
                      TitanValidation::RuntimeCache *validation_cache =
                          nullptr,
                      titan_sparse_graph_module::TitanUnifiedGraphRuntime
                          *graph_runtime = nullptr);

void run_validation_postproc(const arguments &args,
                             std::unique_ptr<TitanKinetics> &tica_analyzer,
                             std::unique_ptr<TitanMSM> &msm_analyzer,
                             TitanExecutor &executor, info &info_output,
                             std::fstream &log,
                             TitanValidation::RuntimeCache &validation_cache);

void run_awh_exploratory_postproc(const arguments &args,
                                  std::unique_ptr<TitanMSM> &msm_analyzer,
                                  std::unique_ptr<TitanUMAP> &umap_analyzer,
                                  DCCMNetwork *dccm_analyzer, info &info_output,
                                  std::fstream &log);
#endif // ENGINE_POSTPROC_H
