// HoloLift temporal-value runner: a VIBE package and its trajectory in, the HoloLift records out.  The work is
// titan_hololift::run_hololift_temporal_value (src/hololift/temporal_runner.cpp), which TITAN also runs for
// -pbctopo-temporal=hololift; this tool keeps the command line and the environment settings of the H1-H10 runs:
//   HOLOLIFT_EVIDENCE_ABSENT=strict|carry|restore   HOLOLIFT_RANK_WEIGHT  HOLOLIFT_TRANSITION_WEIGHT
//   HOLOLIFT_RANK_COST=linear|saturated  HOLOLIFT_EVIDENCE_WEIGHT  HOLOLIFT_EVIDENCE_SET_WEIGHT  HOLOLIFT_DP_WORKERS
//   HOLOLIFT_COORDINATES=none|lattice|toroidal  HOLOLIFT_PDB=none|first|all  HOLOLIFT_REFERENCE_PDB=<pdb>
//   HOLOLIFT_TIMING (any value: wall time of each stage on stderr)
//   HOLOLIFT_TRAJECTORY_WORKERS  HOLOLIFT_TRAJECTORY_BLOCK=<frames per block>
// The per-frame trajectory work runs on HOLOLIFT_TRAJECTORY_WORKERS threads, else on HOLOLIFT_DP_WORKERS, else on
// every hardware thread; no worker count or block size changes a result.
#include "hololift/temporal_runner.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace fs = std::filesystem;
using namespace titan_hololift;

namespace {

// Read the environment exactly as the runner did before the move: an unparsable or out-of-range value throws, and
// main reports it as UNHANDLED_ERROR (exit 11).
void read_environment(HoloLiftRunOptions &options) {
  if (const char *p = std::getenv("HOLOLIFT_EVIDENCE_ABSENT"); p != nullptr)
    options.evidence_absent = p;
  if (const char *w = std::getenv("HOLOLIFT_RANK_WEIGHT"); w != nullptr)
    options.rank_weight = std::stod(w);
  if (const char *w = std::getenv("HOLOLIFT_TRANSITION_WEIGHT"); w != nullptr)
    options.transition_weight = std::stod(w);
  if (const char *f = std::getenv("HOLOLIFT_RANK_COST"); f != nullptr)
    options.rank_cost = f;
  const auto weight = [](const char *name) -> std::optional<double> {
    const char *w = std::getenv(name);
    if (w == nullptr)
      return std::nullopt;
    std::size_t used = 0;
    const double value = std::stod(w, &used);
    if (used != std::string_view(w).size() || !std::isfinite(value) || value < 0.0)
      throw std::runtime_error(std::string(name) + " must be finite and nonnegative");
    return value;
  };
  options.unique_evidence_weight = weight("HOLOLIFT_EVIDENCE_WEIGHT");
  options.supported_evidence_weight = weight("HOLOLIFT_EVIDENCE_SET_WEIGHT");
  if (const char *w = std::getenv("HOLOLIFT_DP_WORKERS"); w != nullptr) {
    std::size_t used = 0;
    const auto workers = std::stoull(w, &used);
    if (used != std::string_view(w).size() || workers == 0 || workers > 64)
      throw std::runtime_error("HOLOLIFT_DP_WORKERS must be in [1,64]");
    options.dp_workers = static_cast<std::size_t>(workers);
    options.trajectory_workers = static_cast<std::size_t>(workers);
  } else {
    options.trajectory_workers = std::min<std::size_t>(64, std::max(1U, std::thread::hardware_concurrency()));
  }
  if (const char *c = std::getenv("HOLOLIFT_COORDINATES"); c != nullptr) {
    const auto parsed = parse_hololift_coordinate_output(c);
    if (!parsed)
      throw std::runtime_error("HOLOLIFT_COORDINATES must be none, lattice or toroidal");
    options.coordinates = *parsed;
    options.pdb = HoloLiftPdbOutput::First;
  }
  if (const char *c = std::getenv("HOLOLIFT_PDB"); c != nullptr) {
    const auto parsed = parse_hololift_pdb_output(c);
    if (!parsed)
      throw std::runtime_error("HOLOLIFT_PDB must be none, first or all");
    options.pdb = *parsed;
  }
  if (const char *r = std::getenv("HOLOLIFT_REFERENCE_PDB"); r != nullptr)
    options.reference_pdb = r;
  if (const char *w = std::getenv("HOLOLIFT_TRAJECTORY_WORKERS"); w != nullptr) {
    std::size_t used = 0;
    const auto workers = std::stoull(w, &used);
    if (used != std::string_view(w).size() || workers == 0 || workers > 64)
      throw std::runtime_error("HOLOLIFT_TRAJECTORY_WORKERS must be in [1,64]");
    options.trajectory_workers = static_cast<std::size_t>(workers);
  }
  if (const char *b = std::getenv("HOLOLIFT_TRAJECTORY_BLOCK"); b != nullptr) {
    std::size_t used = 0;
    const auto frames = std::stoull(b, &used);
    if (used != std::string_view(b).size() || frames == 0)
      throw std::runtime_error("HOLOLIFT_TRAJECTORY_BLOCK must be a positive frame count");
    options.trajectory_block_frames = static_cast<std::size_t>(frames);
  }
  options.report_stage_times = std::getenv("HOLOLIFT_TIMING") != nullptr;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 5 && argc != 7 && argc != 8) {
    std::cerr << "usage: hololift_temporal_value_runner <system-label> "
                 "<vibe-package-dir> <trajectory.xtc> <output-dir> "
                 "[--transport-radius <N> "
                 "[--synthetic-rank-inversion-v1|"
                 "--synthetic-domain-miss-v1|"
                 "--synthetic-domain-miss-v2]]\n";
    return 1;
  }
  try {
    std::size_t transport_radius = 0;
    bool synthetic_rank_inversion_v1 = false;
    bool synthetic_domain_miss_v1 = false;
    bool synthetic_domain_miss_v2 = false;
    if (argc >= 7) {
      if (std::string_view(argv[5]) != "--transport-radius") {
        std::cerr << "unknown optional argument " << argv[5] << '\n';
        return 1;
      }
      const unsigned long long parsed = std::stoull(argv[6]);
      if (parsed == 0 ||
          parsed > static_cast<unsigned long long>(
                       std::numeric_limits<std::size_t>::max())) {
        std::cerr << "transport radius must be a positive size_t\n";
        return 1;
      }
      transport_radius = static_cast<std::size_t>(parsed);
    }
    if (argc == 8) {
      const std::string_view control(argv[7]);
      if (control != "--synthetic-rank-inversion-v1" &&
          control != "--synthetic-domain-miss-v1" &&
          control != "--synthetic-domain-miss-v2") {
        std::cerr << "unknown optional argument " << argv[7] << '\n';
        return 1;
      }
      synthetic_rank_inversion_v1 =
          control == "--synthetic-rank-inversion-v1";
      synthetic_domain_miss_v1 =
          control == "--synthetic-domain-miss-v1";
      synthetic_domain_miss_v2 =
          control == "--synthetic-domain-miss-v2";
    }
    HoloLiftRunOptions options;
    options.system_label = argv[1];
    options.package_directory = fs::path(argv[2]);
    options.trajectory_path = fs::path(argv[3]);
    options.output_directory = fs::path(argv[4]);
    options.transport_radius = transport_radius;
    options.synthetic_rank_inversion_v1 = synthetic_rank_inversion_v1;
    options.synthetic_domain_miss_v1 = synthetic_domain_miss_v1;
    options.synthetic_domain_miss_v2 = synthetic_domain_miss_v2;
    read_environment(options);
    return run_hololift_temporal_value(options);
  } catch (const std::exception &error) {
    std::cerr << "UNHANDLED_ERROR " << error.what() << '\n';
    return 11;
  }
}
