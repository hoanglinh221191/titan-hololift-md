#include "tabi_fmm_m2l_fallback.h"

#include "tabi_near.h"
#include "tabi_treecode_internal.h"
#include "titan_executor.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace titan::tabi::fmm_m2l {
namespace {
using Clock = std::chrono::steady_clock;
using TitanPBE::Tabi::JcpNearPairKernel;
using TitanPBE::Tabi::Panel;
using TitanPBE::Tabi::detail::JcpAccum;

struct PackedTask {
  std::size_t source_first{0};
  std::size_t source_count{0};
  bool refined_fallback{false};
};

struct PackedBatch {
  std::vector<std::size_t> targets;
  std::size_t task_begin{0};
  std::size_t task_count{0};
  std::size_t interaction_work{0};
  std::vector<std::size_t> self_targets;
};

struct PackedWorkerScratch {
  std::vector<JcpAccum> total;
  std::vector<JcpAccum> task_accum;
  std::size_t vectorized_sources{0};
  std::size_t vectorized_lanes{0};
  std::size_t scalar_self_pairs{0};
  std::uint64_t applied_pairs{0};
};

void hash_mix(std::uint64_t &hash, std::uint64_t value) noexcept {
  hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
}

std::size_t checked_add_size(std::size_t left, std::size_t right,
                             const char *context) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error(
        std::string("packed fallback size overflow while ") + context);
  }
  return left + right;
}

std::size_t checked_multiply_size(std::size_t left, std::size_t right,
                                  const char *context) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error(
        std::string("packed fallback size overflow while ") + context);
  }
  return left * right;
}

std::uint64_t checked_add_pairs(std::uint64_t left, std::uint64_t right,
                                const char *context) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw std::overflow_error(
        std::string("packed fallback pair counter overflow while ") + context);
  }
  return left + right;
}

std::uint64_t checked_multiply_pairs(std::size_t left, std::size_t right,
                                     const char *context) {
  const auto left_u64 = static_cast<std::uint64_t>(left);
  const auto right_u64 = static_cast<std::uint64_t>(right);
  if (left_u64 != left || right_u64 != right ||
      (left_u64 != 0 &&
       right_u64 > std::numeric_limits<std::uint64_t>::max() / left_u64)) {
    throw std::overflow_error(
        std::string("packed fallback pair counter overflow while ") + context);
  }
  return left_u64 * right_u64;
}

void add_kernel(JcpAccum &acc, const JcpNearPairKernel &kernel,
                double phi, double phin, bool coupled) noexcept {
  acc.phi += kernel.k2 * phi;
  if (!coupled)
    return;
  acc.phi += kernel.k1 * phin;
  acc.phin += kernel.k3 * phin + kernel.k4 * phi;
}

[[noreturn]] void fail_invalid_direct_kernel(
    const char *path, std::size_t target, std::size_t source,
    const JcpNearPairKernel &kernel, bool coupled,
    const std::string &near_integration, const std::string &quad_buffer,
    bool intentional_zero_self) {
  std::ostringstream message;
  message << "packed fallback " << path << " direct kernel invalid: "
          << "target_panel=" << target << " source_panel=" << source
          << " coupled=" << (coupled ? "true" : "false")
          << " near_integration=" << near_integration
          << " quad_buffer=" << quad_buffer
          << " kernel_self_pair=" << (kernel.self_pair ? "true" : "false")
          << " intentional_zero_self="
          << (intentional_zero_self ? "true" : "false")
          << " kernel_policy=" << kernel.policy;
  throw std::runtime_error(message.str());
}
} // namespace

struct PackedJcpFallbackPlan::Impl {
  std::size_t panel_count{0};
  std::vector<std::size_t> source_order;
  TitanPBE::Tabi::TreePanelCache panel_cache;
  TitanPBE::Tabi::detail::TreeUnknownCache unknown_cache;
  std::vector<PackedBatch> batches;
  std::vector<PackedTask> tasks;
  std::vector<std::size_t> batch_schedule;
  std::vector<PackedWorkerScratch> worker_scratch;
#if defined(TITAN_CARTESIAN_FMM_TESTING)
  // Sidecar only: retain the production PackedBatch layout and metadata
  // accounting while exposing source-span fanout to the P3C0 inspector.
  std::vector<std::size_t> test_source_span_counts;
#endif
  PackedFallbackPlanMetrics metrics;
};

PackedJcpFallbackPlan::PackedJcpFallbackPlan()
    : impl_(std::make_unique<Impl>()) {}
PackedJcpFallbackPlan::~PackedJcpFallbackPlan() = default;
PackedJcpFallbackPlan::PackedJcpFallbackPlan(PackedJcpFallbackPlan &&) noexcept =
    default;
PackedJcpFallbackPlan &PackedJcpFallbackPlan::operator=(
    PackedJcpFallbackPlan &&) noexcept = default;

PackedFallbackPlanMetrics PackedJcpFallbackPlan::metrics() const noexcept {
  return impl_ ? impl_->metrics : PackedFallbackPlanMetrics{};
}

#if defined(TITAN_CARTESIAN_FMM_TESTING)
PackedFallbackPlanInspection
PackedJcpFallbackPlan::inspection_for_testing() const {
  PackedFallbackPlanInspection inspection;
  if (!impl_ || impl_->batches.empty() || impl_->tasks.empty())
    return inspection;

  inspection.available = true;
  inspection.target_batches = impl_->batches.size();
  inspection.task_count = impl_->tasks.size();
  inspection.metadata_bytes = impl_->metrics.metadata_bytes;
  inspection.total_pairs = impl_->metrics.total_pairs;
  if (impl_->test_source_span_counts.size() != impl_->batches.size()) {
    throw std::logic_error(
        "P3C0 test-only packed-plan source-span sidecar mismatch");
  }

  std::vector<std::size_t> task_counts;
  task_counts.reserve(impl_->batches.size());
  std::vector<std::size_t> source_panel_counts;
  source_panel_counts.reserve(impl_->tasks.size());
  std::size_t total_tasks = 0;
  std::size_t total_source_panels = 0;
  for (std::size_t batch_index = 0; batch_index < impl_->batches.size();
       ++batch_index) {
    const PackedBatch &batch = impl_->batches[batch_index];
    const std::size_t lanes = batch.targets.size();
    inspection.target_lanes += lanes;
    if (lanes == 4) {
      ++inspection.full_four_lane_batches;
      inspection.full_four_lane_task_count += batch.task_count;
    } else {
      ++inspection.tail_batches;
      inspection.tail_task_count += batch.task_count;
    }
    inspection.source_span_count += impl_->test_source_span_counts[batch_index];
    task_counts.push_back(batch.task_count);
    total_tasks += batch.task_count;
    for (std::size_t offset = 0; offset < batch.task_count; ++offset) {
      const std::size_t source_count =
          impl_->tasks[batch.task_begin + offset].source_count;
      source_panel_counts.push_back(source_count);
      total_source_panels += source_count;
    }
  }

  auto percentile_nearest_rank = [](std::vector<std::size_t> &values,
                                    double fraction) {
    std::sort(values.begin(), values.end());
    const double rank = std::ceil(fraction * values.size());
    const std::size_t index = static_cast<std::size_t>(
        std::clamp(rank > 0.0 ? rank - 1.0 : 0.0, 0.0,
                   static_cast<double>(values.size() - 1)));
    return values[index];
  };
  inspection.tasks_per_batch_mean =
      static_cast<double>(total_tasks) /
      static_cast<double>(inspection.target_batches);
  inspection.source_spans_per_task =
      static_cast<double>(inspection.source_span_count) /
      static_cast<double>(inspection.task_count);
  inspection.source_panels_per_task_mean =
      static_cast<double>(total_source_panels) /
      static_cast<double>(inspection.task_count);
  inspection.tasks_per_batch_min =
      *std::min_element(task_counts.begin(), task_counts.end());
  inspection.tasks_per_batch_max =
      *std::max_element(task_counts.begin(), task_counts.end());
  inspection.tasks_per_batch_p50 = percentile_nearest_rank(task_counts, 0.50);
  inspection.tasks_per_batch_p95 = percentile_nearest_rank(task_counts, 0.95);
  inspection.source_panels_per_task_min =
      *std::min_element(source_panel_counts.begin(), source_panel_counts.end());
  inspection.source_panels_per_task_max =
      *std::max_element(source_panel_counts.begin(), source_panel_counts.end());
  inspection.source_panels_per_task_p50 =
      percentile_nearest_rank(source_panel_counts, 0.50);
  inspection.source_panels_per_task_p95 =
      percentile_nearest_rank(source_panel_counts, 0.95);
  return inspection;
}
#endif

PackedJcpFallbackPlan build_packed_jcp_fallback_plan(
    const std::vector<Panel> &panels,
    const std::vector<std::size_t> &source_order,
    const std::vector<PackedFallbackLeafInput> &leaves,
    std::size_t target_batch_size, std::size_t source_tile_size) {
  if (panels.empty() || source_order.size() != panels.size())
    throw std::invalid_argument(
        "packed fallback requires a complete source order");
  if (target_batch_size != 4 || source_tile_size != 32)
    throw std::invalid_argument(
        "packed fallback requires target batch 4 and source tile 32");
  std::vector<bool> seen_order(panels.size(), false);
  for (const std::size_t panel : source_order) {
    if (panel >= panels.size() || seen_order[panel])
      throw std::invalid_argument(
          "packed fallback source order must be a panel permutation");
    seen_order[panel] = true;
  }

  PackedJcpFallbackPlan plan;
  auto &impl = *plan.impl_;
  if (source_order.size() > impl.source_order.max_size() ||
      panels.size() > impl.unknown_cache.phi.max_size() ||
      panels.size() > impl.unknown_cache.phin.max_size() ||
      panels.size() > impl.unknown_cache.ordered_phi.max_size() ||
      panels.size() > impl.unknown_cache.ordered_phin.max_size()) {
    throw std::length_error(
        "packed fallback cache allocation exceeds container limit");
  }
  const std::size_t density_cache_bytes = checked_multiply_size(
      panels.size(), sizeof(double), "preflighting packed fallback density cache");
  (void)checked_add_size(
      checked_add_size(density_cache_bytes, density_cache_bytes,
                       "preflighting packed fallback density cache"),
      checked_add_size(density_cache_bytes, density_cache_bytes,
                       "preflighting packed fallback density cache"),
      "preflighting packed fallback density cache");
  impl.panel_count = panels.size();
  impl.source_order = source_order;
  impl.panel_cache = TitanPBE::Tabi::detail::build_tree_panel_cache(
      panels, source_order);
  impl.unknown_cache.phi.assign(panels.size(), 0.0);
  impl.unknown_cache.phin.assign(panels.size(), 0.0);
  impl.unknown_cache.ordered_phi.assign(panels.size(), 0.0);
  impl.unknown_cache.ordered_phin.assign(panels.size(), 0.0);
  impl.metrics.build_count = 1;
  impl.metrics.target_batch_size = target_batch_size;
  impl.metrics.source_tile_size = source_tile_size;

  for (const PackedFallbackLeafInput &leaf : leaves) {
    std::unordered_set<std::size_t> covered_sources;
    for (const PackedFallbackSourceSpan &span : leaf.source_spans) {
      const std::size_t span_end = checked_add_size(
          span.first_panel, span.panel_count, "validating source spans");
      if (span_end > source_order.size())
        throw std::invalid_argument("packed fallback source span is invalid");
      for (std::size_t pos = span.first_panel; pos < span_end; ++pos) {
        if (!covered_sources.insert(source_order[pos]).second)
          throw std::invalid_argument(
              "packed fallback source spans overlap within a target leaf");
      }
      const std::uint64_t pairs = checked_multiply_pairs(
          leaf.target_panels.size(), span.panel_count,
          "counting packed fallback worklist pairs");
      impl.metrics.total_pairs = checked_add_pairs(
          impl.metrics.total_pairs, pairs, "counting packed fallback pairs");
      if (span.refined_fallback)
        impl.metrics.refined_fallback_pairs = checked_add_pairs(
            impl.metrics.refined_fallback_pairs, pairs,
            "counting refined packed fallback pairs");
      else
        impl.metrics.base_near_pairs = checked_add_pairs(
            impl.metrics.base_near_pairs, pairs,
            "counting base packed fallback pairs");
    }
    for (std::size_t target_begin = 0;
         target_begin < leaf.target_panels.size();
         target_begin += target_batch_size) {
      PackedBatch batch;
#if defined(TITAN_CARTESIAN_FMM_TESTING)
      std::size_t test_source_span_count = 0;
#endif
      const std::size_t next_target = checked_add_size(
          target_begin, target_batch_size, "building packed fallback batches");
      const std::size_t target_end = std::min(
          leaf.target_panels.size(), next_target);
      batch.targets.assign(leaf.target_panels.begin() + target_begin,
                           leaf.target_panels.begin() + target_end);
      for (const std::size_t target : batch.targets) {
        if (target >= panels.size())
          throw std::invalid_argument("packed fallback target is invalid");
        if (covered_sources.contains(target))
          batch.self_targets.push_back(target);
      }
      impl.metrics.self_pairs = checked_add_pairs(
          impl.metrics.self_pairs,
          static_cast<std::uint64_t>(batch.self_targets.size()),
          "counting packed fallback self pairs");
      batch.task_begin = impl.tasks.size();
      for (const PackedFallbackSourceSpan &span : leaf.source_spans) {
#if defined(TITAN_CARTESIAN_FMM_TESTING)
        ++test_source_span_count;
#endif
        batch.interaction_work = checked_add_size(
            batch.interaction_work,
            checked_multiply_size(batch.targets.size(), span.panel_count,
                                  "counting packed fallback batch work"),
            "counting packed fallback batch work");
        for (std::size_t offset = 0; offset < span.panel_count;
             offset += source_tile_size) {
          const std::size_t source_first = checked_add_size(
              span.first_panel, offset, "building packed fallback tasks");
          impl.tasks.push_back(
              {source_first,
               std::min(source_tile_size, span.panel_count - offset),
               span.refined_fallback});
        }
      }
      batch.task_count = impl.tasks.size() - batch.task_begin;
#if defined(TITAN_CARTESIAN_FMM_TESTING)
      impl.test_source_span_counts.push_back(test_source_span_count);
#endif
      impl.batches.push_back(std::move(batch));
    }
  }
  impl.metrics.target_batches = impl.batches.size();
  impl.metrics.task_count = impl.tasks.size();
  impl.batch_schedule.resize(impl.batches.size());
  std::iota(impl.batch_schedule.begin(), impl.batch_schedule.end(),
            std::size_t{0});
  std::stable_sort(
      impl.batch_schedule.begin(), impl.batch_schedule.end(),
      [&](std::size_t left, std::size_t right) {
        return impl.batches[left].interaction_work >
               impl.batches[right].interaction_work;
      });
  impl.metrics.metadata_bytes = checked_add_size(
      checked_add_size(
          checked_multiply_size(impl.tasks.size(), sizeof(PackedTask),
                                "accounting packed fallback tasks"),
          checked_multiply_size(impl.batches.size(), sizeof(PackedBatch),
                                "accounting packed fallback batches"),
          "accounting packed fallback metadata"),
      checked_multiply_size(impl.batch_schedule.capacity(), sizeof(std::size_t),
                            "accounting packed fallback schedule"),
      "accounting packed fallback metadata");
  for (const PackedBatch &batch : impl.batches) {
    const std::size_t batch_bytes = checked_add_size(
        checked_multiply_size(batch.targets.capacity(), sizeof(std::size_t),
                              "accounting packed fallback targets"),
        checked_multiply_size(batch.self_targets.capacity(), sizeof(std::size_t),
                              "accounting packed fallback self targets"),
        "accounting packed fallback batch metadata");
    impl.metrics.metadata_bytes = checked_add_size(
        impl.metrics.metadata_bytes, batch_bytes,
        "accounting packed fallback metadata");
  }

  std::uint64_t hash = 0xcbf29ce484222325ULL;
  hash_mix(hash, target_batch_size);
  hash_mix(hash, source_tile_size);
  for (const PackedBatch &batch : impl.batches) {
    for (const std::size_t target : batch.targets)
      hash_mix(hash, target);
    for (std::size_t i = 0; i < batch.task_count; ++i) {
      const PackedTask &task = impl.tasks[batch.task_begin + i];
      hash_mix(hash, task.source_first);
      hash_mix(hash, task.source_count);
      hash_mix(hash, task.refined_fallback ? 1 : 0);
    }
  }
  impl.metrics.worklist_hash = hash;
  return plan;
}

PackedFallbackApplyMetrics accumulate_packed_jcp_fallback(
    PackedJcpFallbackPlan &plan, const std::vector<Panel> &panels,
    const std::vector<double> &phi, const std::vector<double> &phin,
    double eps_ratio, double kappa, bool coupled,
    const std::string &near_integration, const std::string &quad_buffer,
    const std::string &kernel, TitanExecutor &executor,
    std::span<double> output_phi, std::span<double> output_phin) {
#if defined(TITAN_CARTESIAN_FMM_TESTING)
  // Deliberately broad-only P3D0 timing: no clock/counter is introduced into
  // task, source, lane, or worker-hot loops.
  const auto testing_phase_total_begin = Clock::now();
  PackedFallbackPhaseTiming testing_phase;
  testing_phase.available = true;
#endif
  if (!plan.impl_ || panels.size() != plan.impl_->panel_count ||
      phi.size() != panels.size() || phin.size() != panels.size() ||
      output_phi.size() != panels.size() ||
      (coupled && output_phin.size() != panels.size()))
    throw std::invalid_argument("packed fallback apply shape mismatch");
  if (kernel != "scalar" && kernel != "packed-avx2" &&
      kernel != "packed-avx2-vector-exp")
    throw std::invalid_argument(
        "packed fallback kernel must be scalar, packed-avx2, or "
        "packed-avx2-vector-exp");
  const bool packed_avx2 = kernel == "packed-avx2" ||
                           kernel == "packed-avx2-vector-exp";
  const bool vector_exp = kernel == "packed-avx2-vector-exp";
  if (packed_avx2 &&
      (near_integration != "analytic_singular" || quad_buffer != "off"))
    throw std::invalid_argument(
        "packed-avx2 requires analytic_singular and quad-buffer=off");
  if (packed_avx2 &&
      !TitanPBE::Tabi::detail::packed_jcp_avx2_available())
    throw std::runtime_error("packed_avx2_unavailable");

  auto &impl = *plan.impl_;
#if defined(TITAN_CARTESIAN_FMM_TESTING)
  const auto testing_unknown_cache_begin = Clock::now();
#endif
  impl.unknown_cache.phi = phi;
  impl.unknown_cache.phin = phin;
  for (std::size_t pos = 0; pos < impl.source_order.size(); ++pos) {
    const std::size_t source = impl.source_order[pos];
    impl.unknown_cache.ordered_phi[pos] = phi[source];
    impl.unknown_cache.ordered_phin[pos] = phin[source];
  }
#if defined(TITAN_CARTESIAN_FMM_TESTING)
  testing_phase.unknown_cache_reorder_seconds =
      std::chrono::duration<double>(Clock::now() - testing_unknown_cache_begin)
          .count();
#endif

  PackedFallbackApplyMetrics result;
  result.effective_kernel = kernel;
  const auto begin_time = Clock::now();
#if defined(TITAN_CARTESIAN_FMM_TESTING)
  const auto testing_worker_prep_begin = Clock::now();
#endif
  const std::size_t workers = std::max<std::size_t>(1, executor.capacity());
  if (impl.worker_scratch.size() < workers)
    impl.worker_scratch.resize(workers);
  for (std::size_t worker = 0; worker < workers; ++worker) {
    impl.worker_scratch[worker].vectorized_sources = 0;
    impl.worker_scratch[worker].vectorized_lanes = 0;
    impl.worker_scratch[worker].scalar_self_pairs = 0;
    impl.worker_scratch[worker].applied_pairs = 0;
  }
#if defined(TITAN_CARTESIAN_FMM_TESTING)
  testing_phase.worker_scratch_prep_seconds =
      std::chrono::duration<double>(Clock::now() - testing_worker_prep_begin)
          .count();
#endif
  std::atomic<std::size_t> next_batch{0};
#if defined(TITAN_CARTESIAN_FMM_TESTING)
  const auto testing_parallel_begin = Clock::now();
#endif
  executor.parallel_do(
      workers, [&](std::size_t worker) {
        PackedWorkerScratch &scratch = impl.worker_scratch[worker];
        for (;;) {
          const std::size_t scheduled =
              next_batch.fetch_add(1, std::memory_order_relaxed);
          if (scheduled >= impl.batch_schedule.size())
            break;
          const std::size_t batch_index = impl.batch_schedule[scheduled];
          const PackedBatch &batch = impl.batches[batch_index];
          scratch.total.assign(batch.targets.size(), JcpAccum{});
          if (packed_avx2) {
            for (std::size_t task_offset = 0;
                 task_offset < batch.task_count; ++task_offset) {
              const PackedTask &task =
                  impl.tasks[batch.task_begin + task_offset];
              std::size_t near_count = 0;
              TitanPBE::Tabi::detail::direct_jcp_packed_target_source_span_avx2(
                  panels, batch.targets, impl.panel_cache,
                  impl.unknown_cache, eps_ratio, kappa, coupled,
                  task.source_first, task.source_count, scratch.task_accum,
                  near_count, scratch.vectorized_sources,
                  scratch.vectorized_lanes, vector_exp);
              scratch.applied_pairs = checked_add_pairs(
                  scratch.applied_pairs,
                  static_cast<std::uint64_t>(near_count),
                  "counting packed AVX2 direct pairs");
              for (std::size_t lane = 0; lane < scratch.total.size(); ++lane) {
                scratch.total[lane].phi += scratch.task_accum[lane].phi;
                scratch.total[lane].phin += scratch.task_accum[lane].phin;
              }
            }
            for (const std::size_t target : batch.self_targets) {
              const JcpNearPairKernel self =
                  TitanPBE::Tabi::evaluate_jcp_near_pair_kernel(
                      panels[target], target, panels[target], target,
                      eps_ratio, kappa, coupled, near_integration,
                      quad_buffer, nullptr);
              const bool intentional_zero_self =
                  TitanPBE::Tabi::is_intentional_zero_self_kernel(
                      self, coupled, near_integration);
              if (!self.valid && !intentional_zero_self) {
                fail_invalid_direct_kernel(
                    "packed self", target, target, self, coupled,
                    near_integration, quad_buffer, intentional_zero_self);
              }
              const auto position = std::find(batch.targets.begin(),
                                              batch.targets.end(), target);
              const std::size_t lane = static_cast<std::size_t>(
                  std::distance(batch.targets.begin(), position));
              // The zero-self contract represents a processed semantic pair,
              // but no numerical term is added to the accumulation.
              if (self.valid) {
                add_kernel(scratch.total[lane], self, phi[target],
                           phin[target], coupled);
              }
              ++scratch.scalar_self_pairs;
              scratch.applied_pairs = checked_add_pairs(
                  scratch.applied_pairs, std::uint64_t{1},
                  "counting packed scalar self pairs");
            }
          } else {
            for (std::size_t lane = 0; lane < batch.targets.size(); ++lane) {
              const std::size_t target = batch.targets[lane];
              for (std::size_t task_offset = 0;
                   task_offset < batch.task_count; ++task_offset) {
                const PackedTask &task =
                    impl.tasks[batch.task_begin + task_offset];
                const std::size_t task_end = checked_add_size(
                    task.source_first, task.source_count,
                    "evaluating packed fallback scalar task");
                for (std::size_t pos = task.source_first;
                     pos < task_end; ++pos) {
                  const std::size_t source = impl.source_order[pos];
                  const JcpNearPairKernel selected =
                      TitanPBE::Tabi::evaluate_jcp_near_pair_kernel(
                          panels[target], target, panels[source], source,
                          eps_ratio, kappa, coupled, near_integration,
                          quad_buffer, nullptr);
                  const bool intentional_zero_self =
                      TitanPBE::Tabi::is_intentional_zero_self_kernel(
                          selected, coupled, near_integration);
                  if (!selected.valid && !intentional_zero_self) {
                    fail_invalid_direct_kernel(
                        "scalar", target, source, selected, coupled,
                        near_integration, quad_buffer, intentional_zero_self);
                  }
                  if (selected.valid) {
                    add_kernel(scratch.total[lane], selected, phi[source],
                               phin[source], coupled);
                  }
                  scratch.applied_pairs = checked_add_pairs(
                      scratch.applied_pairs, std::uint64_t{1},
                      "counting packed scalar direct pairs");
                  if (selected.self_pair)
                    ++scratch.scalar_self_pairs;
                }
              }
            }
          }
          for (std::size_t lane = 0; lane < batch.targets.size(); ++lane) {
            const std::size_t target = batch.targets[lane];
            output_phi[target] += scratch.total[lane].phi;
            if (coupled)
              output_phin[target] += scratch.total[lane].phin;
          }
        }
      });
#if defined(TITAN_CARTESIAN_FMM_TESTING)
  testing_phase.parallel_region_seconds =
      std::chrono::duration<double>(Clock::now() - testing_parallel_begin)
          .count();
  const auto testing_metric_reduction_begin = Clock::now();
#endif
  for (std::size_t worker = 0; worker < workers; ++worker) {
    result.vectorized_sources +=
        impl.worker_scratch[worker].vectorized_sources;
    result.vectorized_lanes += impl.worker_scratch[worker].vectorized_lanes;
    result.scalar_self_pairs +=
        impl.worker_scratch[worker].scalar_self_pairs;
    result.applied_pairs = checked_add_pairs(
        result.applied_pairs, impl.worker_scratch[worker].applied_pairs,
        "summing packed fallback applied pairs");
  }
#if defined(TITAN_CARTESIAN_FMM_TESTING)
  testing_phase.worker_metric_reduction_seconds =
      std::chrono::duration<double>(Clock::now() -
                                    testing_metric_reduction_begin)
          .count();
#endif
  result.apply_seconds =
      std::chrono::duration<double>(Clock::now() - begin_time).count();
#if defined(TITAN_CARTESIAN_FMM_TESTING)
  const auto testing_finite_scan_begin = Clock::now();
#endif
  result.finite_outputs =
      std::all_of(output_phi.begin(), output_phi.end(),
                  [](double value) { return std::isfinite(value); }) &&
       (!coupled ||
        std::all_of(output_phin.begin(), output_phin.end(),
                    [](double value) { return std::isfinite(value); }));
#if defined(TITAN_CARTESIAN_FMM_TESTING)
  testing_phase.finite_scan_seconds =
      std::chrono::duration<double>(Clock::now() - testing_finite_scan_begin)
          .count();
  testing_phase.total_seconds =
      std::chrono::duration<double>(Clock::now() - testing_phase_total_begin)
          .count();
  const double measured_phase_sum =
      testing_phase.unknown_cache_reorder_seconds +
      testing_phase.worker_scratch_prep_seconds +
      testing_phase.parallel_region_seconds +
      testing_phase.worker_metric_reduction_seconds +
      testing_phase.finite_scan_seconds;
  const double tolerance =
      std::max(1.0e-6, 1.0e-5 * testing_phase.total_seconds);
  if (!std::isfinite(testing_phase.total_seconds) ||
      !std::isfinite(measured_phase_sum) || testing_phase.total_seconds < 0.0 ||
      testing_phase.unknown_cache_reorder_seconds < 0.0 ||
      testing_phase.worker_scratch_prep_seconds < 0.0 ||
      testing_phase.parallel_region_seconds < 0.0 ||
      testing_phase.worker_metric_reduction_seconds < 0.0 ||
      testing_phase.finite_scan_seconds < 0.0 ||
      measured_phase_sum > testing_phase.total_seconds + tolerance) {
    throw std::runtime_error(
        "P3D0 packed fallback phase timing is nonfinite, negative, or exceeds total");
  }
  testing_phase.wrapper_residual_seconds =
      std::max(0.0, testing_phase.total_seconds - measured_phase_sum);
  testing_phase.phase_sum_seconds =
      measured_phase_sum + testing_phase.wrapper_residual_seconds;
  testing_phase.closure_seconds =
      testing_phase.total_seconds - testing_phase.phase_sum_seconds;
  if (!std::isfinite(testing_phase.wrapper_residual_seconds) ||
      !std::isfinite(testing_phase.phase_sum_seconds) ||
      !std::isfinite(testing_phase.closure_seconds) ||
      testing_phase.wrapper_residual_seconds < 0.0 ||
      std::abs(testing_phase.closure_seconds) > tolerance) {
    throw std::runtime_error(
        "P3D0 packed fallback phase timing closure is invalid");
  }
  result.testing_phase_timing = testing_phase;
#endif
  return result;
}

PackedFallbackApplyResult apply_packed_jcp_fallback(
    PackedJcpFallbackPlan &plan, const std::vector<Panel> &panels,
    const std::vector<double> &phi, const std::vector<double> &phin,
    double eps_ratio, double kappa, bool coupled,
    const std::string &near_integration, const std::string &quad_buffer,
    const std::string &kernel, TitanExecutor &executor) {
  PackedFallbackApplyResult result;
  result.phi.assign(panels.size(), 0.0);
  result.phin.assign(panels.size(), 0.0);
  const PackedFallbackApplyMetrics metrics = accumulate_packed_jcp_fallback(
      plan, panels, phi, phin, eps_ratio, kappa, coupled, near_integration,
      quad_buffer, kernel, executor, result.phi, result.phin);
  static_cast<PackedFallbackApplyMetrics &>(result) = metrics;
  return result;
}

bool packed_jcp_fallback_avx2_available() noexcept {
  return TitanPBE::Tabi::detail::packed_jcp_avx2_available();
}

} // namespace titan::tabi::fmm_m2l
