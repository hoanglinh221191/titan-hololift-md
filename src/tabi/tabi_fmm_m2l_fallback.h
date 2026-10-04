#ifndef TITAN_TABI_FMM_M2L_FALLBACK_H
#define TITAN_TABI_FMM_M2L_FALLBACK_H

#include "tabi_types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

class TitanExecutor;

namespace titan::tabi::fmm_m2l {

struct PackedFallbackSourceSpan {
  std::size_t first_panel{0};
  std::size_t panel_count{0};
  bool refined_fallback{false};
};

struct PackedFallbackLeafInput {
  std::vector<std::size_t> target_panels;
  std::vector<PackedFallbackSourceSpan> source_spans;
};

struct PackedFallbackPlanMetrics {
  std::uint64_t worklist_hash{0};
  std::size_t build_count{0};
  std::size_t target_batch_size{4};
  std::size_t source_tile_size{32};
  std::size_t target_batches{0};
  std::size_t task_count{0};
  std::size_t metadata_bytes{0};
  std::uint64_t total_pairs{0};
  std::uint64_t base_near_pairs{0};
  std::uint64_t refined_fallback_pairs{0};
  std::uint64_t self_pairs{0};
};

#if defined(TITAN_CARTESIAN_FMM_TESTING)
// P3D0 test-only broad timing regions.  This intentionally has no production
// layout or runtime cost: the enclosing declaration is compiled out entirely.
struct PackedFallbackPhaseTiming {
  bool available{false};
  double total_seconds{0.0};
  double unknown_cache_reorder_seconds{0.0};
  double worker_scratch_prep_seconds{0.0};
  double parallel_region_seconds{0.0};
  double worker_metric_reduction_seconds{0.0};
  double finite_scan_seconds{0.0};
  double wrapper_residual_seconds{0.0};
  double phase_sum_seconds{0.0};
  double closure_seconds{0.0};
};
#endif

struct PackedFallbackApplyMetrics {
  std::string effective_kernel{"scalar"};
  std::size_t vectorized_sources{0};
  std::size_t vectorized_lanes{0};
  std::size_t scalar_self_pairs{0};
  // Number of semantic target/source pairs whose direct contribution was
  // actually evaluated during this apply. This is distinct from the static
  // worklist's total_pairs and includes scalar self corrections once.
  std::uint64_t applied_pairs{0};
  double apply_seconds{0.0};
  bool finite_outputs{true};
#if defined(TITAN_CARTESIAN_FMM_TESTING)
  PackedFallbackPhaseTiming testing_phase_timing{};
#endif
};

struct PackedFallbackApplyResult : PackedFallbackApplyMetrics {
  std::vector<double> phi;
  std::vector<double> phin;
};

#if defined(TITAN_CARTESIAN_FMM_TESTING)
// Test-only structural readout for P3C0.  This is intentionally absent from
// production builds and never affects the packed worklist or apply path.
struct PackedFallbackPlanInspection {
  bool available{false};
  std::size_t target_batches{0};
  std::size_t full_four_lane_batches{0};
  std::size_t tail_batches{0};
  std::size_t full_four_lane_task_count{0};
  std::size_t tail_task_count{0};
  std::size_t target_lanes{0};
  std::size_t task_count{0};
  std::size_t source_span_count{0};
  double source_spans_per_task{0.0};
  double tasks_per_batch_mean{0.0};
  std::size_t tasks_per_batch_min{0};
  std::size_t tasks_per_batch_p50{0};
  std::size_t tasks_per_batch_p95{0};
  std::size_t tasks_per_batch_max{0};
  double source_panels_per_task_mean{0.0};
  std::size_t source_panels_per_task_min{0};
  std::size_t source_panels_per_task_p50{0};
  std::size_t source_panels_per_task_p95{0};
  std::size_t source_panels_per_task_max{0};
  std::size_t metadata_bytes{0};
  std::uint64_t total_pairs{0};
};
#endif

class PackedJcpFallbackPlan {
public:
  PackedJcpFallbackPlan();
  ~PackedJcpFallbackPlan();
  PackedJcpFallbackPlan(PackedJcpFallbackPlan &&) noexcept;
  PackedJcpFallbackPlan &operator=(PackedJcpFallbackPlan &&) noexcept;
  PackedJcpFallbackPlan(const PackedJcpFallbackPlan &) = delete;
  PackedJcpFallbackPlan &operator=(const PackedJcpFallbackPlan &) = delete;

  [[nodiscard]] PackedFallbackPlanMetrics metrics() const noexcept;

#if defined(TITAN_CARTESIAN_FMM_TESTING)
  [[nodiscard]] PackedFallbackPlanInspection inspection_for_testing() const;
#endif

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend PackedJcpFallbackPlan build_packed_jcp_fallback_plan(
      const std::vector<TitanPBE::Tabi::Panel> &,
      const std::vector<std::size_t> &,
      const std::vector<PackedFallbackLeafInput> &, std::size_t,
      std::size_t);
  friend PackedFallbackApplyResult apply_packed_jcp_fallback(
      PackedJcpFallbackPlan &, const std::vector<TitanPBE::Tabi::Panel> &,
      const std::vector<double> &, const std::vector<double> &, double,
      double, bool, const std::string &, const std::string &,
      const std::string &, TitanExecutor &);
  friend PackedFallbackApplyMetrics accumulate_packed_jcp_fallback(
      PackedJcpFallbackPlan &, const std::vector<TitanPBE::Tabi::Panel> &,
      const std::vector<double> &, const std::vector<double> &, double,
      double, bool, const std::string &, const std::string &,
      const std::string &, TitanExecutor &, std::span<double>,
      std::span<double>);
};

PackedJcpFallbackPlan build_packed_jcp_fallback_plan(
    const std::vector<TitanPBE::Tabi::Panel> &panels,
    const std::vector<std::size_t> &source_order,
    const std::vector<PackedFallbackLeafInput> &leaves,
    std::size_t target_batch_size = 4, std::size_t source_tile_size = 32);

PackedFallbackApplyResult apply_packed_jcp_fallback(
    PackedJcpFallbackPlan &plan,
    const std::vector<TitanPBE::Tabi::Panel> &panels,
    const std::vector<double> &phi, const std::vector<double> &phin,
    double eps_ratio, double kappa, bool coupled,
    const std::string &near_integration, const std::string &quad_buffer,
    const std::string &kernel, TitanExecutor &executor);

// Evaluate the packed direct worklist and add it directly to caller-owned
// buffers.  This is the full-FMM path: far-field values are already present in
// the buffers, so avoiding temporary N-panel result vectors also removes a
// second writeback pass.
PackedFallbackApplyMetrics accumulate_packed_jcp_fallback(
    PackedJcpFallbackPlan &plan,
    const std::vector<TitanPBE::Tabi::Panel> &panels,
    const std::vector<double> &phi, const std::vector<double> &phin,
    double eps_ratio, double kappa, bool coupled,
    const std::string &near_integration, const std::string &quad_buffer,
    const std::string &kernel, TitanExecutor &executor,
    std::span<double> output_phi, std::span<double> output_phin);

// Lightweight runtime feature query for callers that can use the packed
// worklist opportunistically without depending on treecode internals.
bool packed_jcp_fallback_avx2_available() noexcept;

} // namespace titan::tabi::fmm_m2l

#endif
