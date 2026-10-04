#pragma once

#include "arguments.h"
#include "info.h"
#include "titan_executor.h"
#include "titan_kinetics.h"
#include "titan_msm.h"
#include "titan_validation_core.h"

#include <fstream>
#include <memory>

namespace TitanValidation {

struct RuntimeCache {
  bool have_vamp_sweep{false};
  bool auto_pick_applied{false};
  ModelConfig current_effective_model{};
  VampSweepResult vamp_sweep{};
  DhamRebuildContext dham_context{};
  TptValidationContext tpt_context{};
};

bool validation_bootstrap_matches_tpt_legacy(const arguments &args);

void maybe_run_preselect(const arguments &args, const TitanKinetics &tica_analyzer,
                         int current_effective_k,
                         std::unique_ptr<TitanMSM> &msm_analyzer,
                         TitanExecutor &executor, std::fstream &log,
                         RuntimeCache &cache);

void run_postproc(const arguments &args, const TitanKinetics &tica_analyzer,
                  std::unique_ptr<TitanMSM> &msm_analyzer,
                  TitanExecutor &executor, info &info_output,
                  std::fstream &log, RuntimeCache &cache);

} // namespace TitanValidation

