#include "postproc/postproc_common.h"

using namespace postproc_detail;

// ============================================================================
// 4. LLPS post-processing
// ============================================================================
void run_llps_postproc(const arguments &args,
                       std::unique_ptr<TitanLLPS> &llps_analyzer,
                       TitanExecutor &executor, info &info_output,
                       std::fstream &log) {
  if (!(args.llps.flag_llps && llps_analyzer &&
        llps_analyzer->n_frames_processed() > 0))
    return;

  try {
    auto start_llps = std::chrono::steady_clock::now();
    TitanUI::print_module_header("LLPS Analysis");
    llps_analyzer->finalize(executor, args.llps.llps_maxlag);
    llps_analyzer->export_results(info_output, log);
    auto time_llps = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_llps);
    TitanUI::log_time(log, "LLPS Analysis", time_llps.count() / 1000.0);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Exception in LLPS: " << e.what() << "\n";
  }
}
