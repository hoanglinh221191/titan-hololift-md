#include "tabi_fmm_kifmm_hierarchy_jcp.h"

#include "tabi_fmm_kifmm_cartesian_bridge.h"
#include "tabi_fmm_kifmm_solve.h"

#include <Eigen/Dense>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <cstdio>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <thread>
#include <unordered_map>
#include <vector>

namespace titan::tabi::fmm_kifmm {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

double elapsed_seconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

struct ProcessMemorySample {
  std::uint64_t rss_bytes{0};
  std::uint64_t peak_working_set_bytes{0};
};

ProcessMemorySample current_process_memory_sample() noexcept {
  ProcessMemorySample sample;
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters{};
  counters.cb = sizeof(counters);
  if (K32GetProcessMemoryInfo(GetCurrentProcess(), &counters,
                              sizeof(counters))) {
    sample.rss_bytes = static_cast<std::uint64_t>(counters.WorkingSetSize);
    sample.peak_working_set_bytes =
        static_cast<std::uint64_t>(counters.PeakWorkingSetSize);
  }
#else
  FILE *file = std::fopen("/proc/self/status", "r");
  if (file != nullptr) {
    char line[256];
    char label[64];
    long value_kb = 0;
    while (std::fgets(line, sizeof(line), file) != nullptr) {
      if (std::sscanf(line, "%63s %ld", label, &value_kb) != 2) {
        continue;
      }
      if (std::string(label) == "VmRSS:") {
        sample.rss_bytes =
            static_cast<std::uint64_t>(std::max<long>(0, value_kb)) * 1024u;
      } else if (std::string(label) == "VmHWM:") {
        sample.peak_working_set_bytes =
            static_cast<std::uint64_t>(std::max<long>(0, value_kb)) * 1024u;
      }
    }
    std::fclose(file);
  }
#endif
  return sample;
}

void update_hierarchy_peak_delta(KifmmHierarchyJcpResult &result) {
  if (result.hierarchy_peak_after_finalize_bytes >=
      result.hierarchy_peak_start_bytes) {
    result.hierarchy_peak_delta_bytes =
        result.hierarchy_peak_after_finalize_bytes -
        result.hierarchy_peak_start_bytes;
  } else {
    result.hierarchy_peak_delta_bytes = 0;
  }
}

void record_hierarchy_memory_start(KifmmHierarchyJcpResult &result) {
  const ProcessMemorySample sample = current_process_memory_sample();
  result.hierarchy_rss_start_bytes = sample.rss_bytes;
  result.hierarchy_peak_start_bytes = sample.peak_working_set_bytes;
  result.hierarchy_rss_after_finalize_bytes = sample.rss_bytes;
  result.hierarchy_peak_after_finalize_bytes = sample.peak_working_set_bytes;
  update_hierarchy_peak_delta(result);
}

void record_hierarchy_memory_after_direct_fallback(
    KifmmHierarchyJcpResult &result) {
  const ProcessMemorySample sample = current_process_memory_sample();
  result.hierarchy_rss_after_direct_fallback_bytes = sample.rss_bytes;
  result.hierarchy_peak_after_direct_fallback_bytes =
      sample.peak_working_set_bytes;
}

void record_hierarchy_memory_after_p2m(KifmmHierarchyJcpResult &result) {
  const ProcessMemorySample sample = current_process_memory_sample();
  result.hierarchy_rss_after_p2m_bytes = sample.rss_bytes;
  result.hierarchy_peak_after_p2m_bytes = sample.peak_working_set_bytes;
}

void record_hierarchy_memory_after_m2m(KifmmHierarchyJcpResult &result) {
  const ProcessMemorySample sample = current_process_memory_sample();
  result.hierarchy_rss_after_m2m_bytes = sample.rss_bytes;
  result.hierarchy_peak_after_m2m_bytes = sample.peak_working_set_bytes;
}

void record_hierarchy_memory_after_m2l(KifmmHierarchyJcpResult &result) {
  const ProcessMemorySample sample = current_process_memory_sample();
  result.hierarchy_rss_after_m2l_bytes = sample.rss_bytes;
  result.hierarchy_peak_after_m2l_bytes = sample.peak_working_set_bytes;
}

void record_hierarchy_memory_after_l2l(KifmmHierarchyJcpResult &result) {
  const ProcessMemorySample sample = current_process_memory_sample();
  result.hierarchy_rss_after_l2l_bytes = sample.rss_bytes;
  result.hierarchy_peak_after_l2l_bytes = sample.peak_working_set_bytes;
}

void record_hierarchy_memory_after_l2p(KifmmHierarchyJcpResult &result) {
  const ProcessMemorySample sample = current_process_memory_sample();
  result.hierarchy_rss_after_l2p_bytes = sample.rss_bytes;
  result.hierarchy_peak_after_l2p_bytes = sample.peak_working_set_bytes;
}

void record_hierarchy_memory_after_dense_oracle(
    KifmmHierarchyJcpResult &result) {
  const ProcessMemorySample sample = current_process_memory_sample();
  result.hierarchy_rss_after_dense_oracle_bytes = sample.rss_bytes;
  result.hierarchy_peak_after_dense_oracle_bytes =
      sample.peak_working_set_bytes;
}

void record_hierarchy_memory_after_finalize(KifmmHierarchyJcpResult &result) {
  const ProcessMemorySample sample = current_process_memory_sample();
  result.hierarchy_rss_after_finalize_bytes = sample.rss_bytes;
  result.hierarchy_peak_after_finalize_bytes = sample.peak_working_set_bytes;
  update_hierarchy_peak_delta(result);
}

void hash_u64(std::uint64_t &hash, std::uint64_t value) noexcept {
  hash ^= value;
  hash *= kFnvPrime;
}

void hash_double(std::uint64_t &hash, double value) noexcept {
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  hash_u64(hash, bits);
}

std::uint64_t hash_channel_rows(
    std::span<const KifmmJcpChannelRow> rows) noexcept {
  std::uint64_t hash = kFnvOffset;
  for (const KifmmJcpChannelRow &row : rows) {
    hash_double(hash, row.k1);
    hash_double(hash, row.k2);
    hash_double(hash, row.k3);
    hash_double(hash, row.k4);
  }
  return hash == 0 ? 1 : hash;
}

bool finite_vec(Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

double dot(Vec3 a, Vec3 b) noexcept {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

bool finite_source(const KifmmPanelSource &source) noexcept {
  return finite_vec(source.position) && finite_vec(source.normal) &&
         std::isfinite(source.area) && std::isfinite(source.phi) &&
         std::isfinite(source.phin) && source.area > 0.0;
}

std::vector<Vec3> source_positions(std::span<const KifmmPanelSource> sources) {
  std::vector<Vec3> positions;
  positions.reserve(sources.size());
  for (const KifmmPanelSource &source : sources) {
    positions.push_back(source.position);
  }
  return positions;
}

KifmmBox box_from_node(const KifmmHierarchyNode &node) noexcept {
  return {node.center, node.half_width, node.level,
          static_cast<std::size_t>(node.id)};
}

std::vector<KifmmPanelSource>
gather_sources(std::span<const KifmmPanelSource> sources,
               const std::vector<int> &indices) {
  std::vector<KifmmPanelSource> gathered;
  gathered.reserve(indices.size());
  for (const int index : indices) {
    gathered.push_back(sources[static_cast<std::size_t>(index)]);
  }
  return gathered;
}

const std::array<KifmmFieldFamily, 4> kFamilies{
    KifmmFieldFamily::PhinCharge, KifmmFieldFamily::PhiNormalX,
    KifmmFieldFamily::PhiNormalY, KifmmFieldFamily::PhiNormalZ};

struct JcpBundle {
  std::array<KifmmEquivalentSolveResult, 4> laplace{};
  std::array<KifmmEquivalentSolveResult, 4> yukawa{};
  bool valid{false};
};

struct JcpBundleRhs {
  Eigen::MatrixXd laplace{};
  Eigen::MatrixXd yukawa{};
};

struct M2lMatrixCacheKey {
  int kernel{0};
  int order{0};
  int source_level{0};
  int target_level{0};
  std::int64_t kappa{0};
  std::int64_t source_half_width{0};
  std::int64_t target_half_width{0};
  std::int64_t source_equiv_scale{0};
  std::int64_t target_check_scale{0};
  std::int64_t target_source_dx{0};
  std::int64_t target_source_dy{0};
  std::int64_t target_source_dz{0};
  std::size_t point_count{0};
  std::size_t equiv_count{0};

  bool operator==(const M2lMatrixCacheKey &other) const noexcept {
    return kernel == other.kernel && order == other.order &&
           source_level == other.source_level &&
           target_level == other.target_level &&
           kappa == other.kappa &&
           source_half_width == other.source_half_width &&
           target_half_width == other.target_half_width &&
           source_equiv_scale == other.source_equiv_scale &&
           target_check_scale == other.target_check_scale &&
           target_source_dx == other.target_source_dx &&
           target_source_dy == other.target_source_dy &&
           target_source_dz == other.target_source_dz &&
           point_count == other.point_count && equiv_count == other.equiv_count;
  }
};

struct M2lMatrixCacheKeyHash {
  std::size_t operator()(const M2lMatrixCacheKey &key) const noexcept {
    std::uint64_t hash = kFnvOffset;
    auto mix = [&hash](std::uint64_t value) { hash_u64(hash, value); };
    mix(static_cast<std::uint64_t>(key.kernel));
    mix(static_cast<std::uint64_t>(key.order));
    mix(static_cast<std::uint64_t>(key.source_level));
    mix(static_cast<std::uint64_t>(key.target_level));
    mix(static_cast<std::uint64_t>(key.kappa));
    mix(static_cast<std::uint64_t>(key.source_half_width));
    mix(static_cast<std::uint64_t>(key.target_half_width));
    mix(static_cast<std::uint64_t>(key.source_equiv_scale));
    mix(static_cast<std::uint64_t>(key.target_check_scale));
    mix(static_cast<std::uint64_t>(key.target_source_dx));
    mix(static_cast<std::uint64_t>(key.target_source_dy));
    mix(static_cast<std::uint64_t>(key.target_source_dz));
    mix(static_cast<std::uint64_t>(key.point_count));
    mix(static_cast<std::uint64_t>(key.equiv_count));
    return static_cast<std::size_t>(hash);
  }
};

struct M2lMatrixCacheEntry {
  M2lMatrixCacheKey key{};
  Eigen::MatrixXd matrix{};
  std::uint64_t bytes{0};
  std::uint64_t last_used_tick{0};
};

struct M2lMatrixCache {
  std::string policy{"off"};
  std::vector<M2lMatrixCacheEntry> entries{};
  std::unordered_map<M2lMatrixCacheKey, std::size_t, M2lMatrixCacheKeyHash>
      m2l_matrix_cache_key_index{};
  std::uint64_t limit_bytes{0};
  std::uint64_t hits{0};
  std::uint64_t misses{0};
  std::uint64_t evictions{0};
  std::uint64_t access_tick{0};
  std::uint64_t peak_entries{0};
  std::uint64_t estimated_bytes{0};
  std::uint64_t peak_estimated_bytes{0};
  std::uint64_t total_built_bytes{0};
  double build_seconds{0.0};
  double gemm_seconds{0.0};
  std::string key_status{"not_used"};
};

bool m2l_matrix_cache_enabled(const std::string &policy) noexcept {
  return policy == "relative-geometry-cache" ||
         policy == "static-translation-table-cache" ||
         policy == "canonical-offset-cache" ||
         policy == "target-node-streaming-cache" ||
         policy == "bounded-lru-cache";
}

bool m2l_matrix_cache_uses_canonical_offsets(
    const std::string &policy) noexcept {
  return policy == "canonical-offset-cache" ||
         policy == "static-translation-table-cache" ||
         policy == "target-node-streaming-cache" ||
         policy == "bounded-lru-cache";
}

bool m2l_matrix_cache_is_streaming(const std::string &policy) noexcept {
  return policy == "target-node-streaming-cache";
}

bool m2l_matrix_cache_is_bounded_lru(const std::string &policy) noexcept {
  return policy == "bounded-lru-cache";
}

bool m2l_matrix_cache_is_static_table(const std::string &policy) noexcept {
  return policy == "static-translation-table-cache";
}

std::uint64_t m2l_matrix_bytes(const Eigen::MatrixXd &matrix) noexcept {
  return
      static_cast<std::uint64_t>(matrix.rows()) *
      static_cast<std::uint64_t>(matrix.cols()) *
      static_cast<std::uint64_t>(sizeof(double));
}

std::uint64_t m2l_matrix_cache_limit_bytes_from_mb(double megabytes) noexcept {
  if (!std::isfinite(megabytes) || megabytes <= 0.0) {
    return 0;
  }
  const long double bytes =
      static_cast<long double>(megabytes) * 1024.0L * 1024.0L;
  const long double max_value =
      static_cast<long double>(std::numeric_limits<std::uint64_t>::max());
  if (bytes >= max_value) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return static_cast<std::uint64_t>(std::llround(bytes));
}

void evict_bounded_lru_m2l_cache(M2lMatrixCache &cache,
                                 std::uint64_t incoming_bytes) {
  if (!m2l_matrix_cache_is_bounded_lru(cache.policy) ||
      cache.limit_bytes == 0 || incoming_bytes > cache.limit_bytes) {
    return;
  }
  while (cache.estimated_bytes + incoming_bytes > cache.limit_bytes &&
         !cache.entries.empty()) {
    auto victim = std::min_element(
        cache.entries.begin(), cache.entries.end(),
        [](const M2lMatrixCacheEntry &a, const M2lMatrixCacheEntry &b) {
          return a.last_used_tick < b.last_used_tick;
        });
    cache.estimated_bytes =
        victim->bytes > cache.estimated_bytes ? 0
                                              : cache.estimated_bytes -
                                                    victim->bytes;
    cache.entries.erase(victim);
    cache.m2l_matrix_cache_key_index.clear();
    for (std::size_t i = 0; i < cache.entries.size(); ++i) {
      cache.m2l_matrix_cache_key_index.emplace(cache.entries[i].key, i);
    }
    ++cache.evictions;
  }
}

const Eigen::MatrixXd *insert_m2l_matrix_cache_entry(
    M2lMatrixCache &cache, const M2lMatrixCacheKey &key,
    Eigen::MatrixXd &local_kernel_matrix) {
  const std::uint64_t bytes = m2l_matrix_bytes(local_kernel_matrix);
  cache.total_built_bytes += bytes;
  if (m2l_matrix_cache_is_bounded_lru(cache.policy)) {
    if (cache.limit_bytes == 0) {
      cache.key_status = "bounded_lru_missing_limit";
      return &local_kernel_matrix;
    }
    if (bytes > cache.limit_bytes) {
      cache.key_status = "bounded_lru_entry_too_large";
      return &local_kernel_matrix;
    }
    evict_bounded_lru_m2l_cache(cache, bytes);
  }
  cache.estimated_bytes += bytes;
  cache.peak_estimated_bytes =
      std::max(cache.peak_estimated_bytes, cache.estimated_bytes);
  cache.peak_entries =
      std::max(cache.peak_entries,
               static_cast<std::uint64_t>(cache.entries.size() + 1));
  cache.entries.push_back(
      {key, std::move(local_kernel_matrix), bytes, ++cache.access_tick});
  if (m2l_matrix_cache_is_static_table(cache.policy) ||
      m2l_matrix_cache_is_bounded_lru(cache.policy)) {
    cache.m2l_matrix_cache_key_index[key] = cache.entries.size() - 1;
  }
  if (m2l_matrix_cache_is_bounded_lru(cache.policy)) {
    cache.key_status =
        cache.evictions > 0 ? "bounded_lru_evicted" : "bounded_lru_pass";
  } else if (m2l_matrix_cache_is_static_table(cache.policy)) {
    cache.key_status = "static_translation_table_pass";
  } else {
    cache.key_status =
        cache.policy == "target-node-streaming-cache"
            ? "target_node_streaming_pass"
            : (cache.policy == "canonical-offset-cache"
                   ? "canonical_offset_pass"
                   : "pass");
  }
  return &cache.entries.back().matrix;
}

void clear_target_node_streaming_cache(M2lMatrixCache &cache) {
  if (!m2l_matrix_cache_is_streaming(cache.policy)) {
    return;
  }
  cache.entries.clear();
  cache.m2l_matrix_cache_key_index.clear();
  cache.estimated_bytes = 0;
  cache.key_status = "target_node_streaming_pass";
}

class M2lMatrixCacheScopeGuard {
public:
  explicit M2lMatrixCacheScopeGuard(M2lMatrixCache &cache) : cache_(&cache) {}
  M2lMatrixCacheScopeGuard(const M2lMatrixCacheScopeGuard &) = delete;
  M2lMatrixCacheScopeGuard &
  operator=(const M2lMatrixCacheScopeGuard &) = delete;
  ~M2lMatrixCacheScopeGuard() {
    if (cache_ != nullptr) {
      clear_target_node_streaming_cache(*cache_);
    }
  }

private:
  M2lMatrixCache *cache_{nullptr};
};

std::uint64_t matrix_storage_bytes(const Eigen::MatrixXd &matrix) noexcept {
  return static_cast<std::uint64_t>(matrix.size()) *
         static_cast<std::uint64_t>(sizeof(double));
}

std::uint64_t solve_result_storage_bytes(
    const KifmmEquivalentSolveResult &result) noexcept {
  return static_cast<std::uint64_t>(result.charges.size()) *
         static_cast<std::uint64_t>(sizeof(double));
}

std::uint64_t bundle_storage_bytes(const JcpBundle &bundle) noexcept {
  std::uint64_t bytes = 0;
  for (const KifmmEquivalentSolveResult &result : bundle.laplace) {
    bytes += solve_result_storage_bytes(result);
  }
  for (const KifmmEquivalentSolveResult &result : bundle.yukawa) {
    bytes += solve_result_storage_bytes(result);
  }
  return bytes;
}

std::uint64_t bundle_vector_storage_bytes(
    const std::vector<JcpBundle> &bundles) noexcept {
  std::uint64_t bytes = static_cast<std::uint64_t>(bundles.capacity()) *
                        static_cast<std::uint64_t>(sizeof(JcpBundle));
  for (const JcpBundle &bundle : bundles) {
    bytes += bundle_storage_bytes(bundle);
  }
  return bytes;
}

std::uint64_t rhs_storage_bytes(const JcpBundleRhs &rhs) noexcept {
  return matrix_storage_bytes(rhs.laplace) + matrix_storage_bytes(rhs.yukawa);
}

std::uint64_t rhs_vector_storage_bytes(
    const std::vector<JcpBundleRhs> &rhs_values) noexcept {
  std::uint64_t bytes = static_cast<std::uint64_t>(rhs_values.capacity()) *
                        static_cast<std::uint64_t>(sizeof(JcpBundleRhs));
  for (const JcpBundleRhs &rhs : rhs_values) {
    bytes += rhs_storage_bytes(rhs);
  }
  return bytes;
}

std::uint64_t row_storage_bytes(
    const std::vector<KifmmJcpChannelRow> &rows) noexcept {
  return static_cast<std::uint64_t>(rows.capacity()) *
         static_cast<std::uint64_t>(sizeof(KifmmJcpChannelRow));
}

std::uint64_t equivalent_system_cache_storage_bytes(
    const KifmmEquivalentSystemCache &cache) noexcept {
  std::uint64_t bytes = static_cast<std::uint64_t>(cache.entries.capacity()) *
                        static_cast<std::uint64_t>(
                            sizeof(KifmmEquivalentSystemCacheEntry));
  for (const KifmmEquivalentSystemCacheEntry &entry : cache.entries) {
    bytes += matrix_storage_bytes(entry.canonical_system.check_to_equiv);
    bytes += static_cast<std::uint64_t>(
                 entry.canonical_system.equiv_points.capacity()) *
             static_cast<std::uint64_t>(sizeof(KifmmSurfacePoint));
    bytes += static_cast<std::uint64_t>(
                 entry.canonical_system.check_points.capacity()) *
             static_cast<std::uint64_t>(sizeof(KifmmSurfacePoint));
  }
  return bytes;
}

void update_k123_retained_memory_estimates(
    KifmmHierarchyJcpResult &result,
    const std::vector<JcpBundle> &upward,
    const std::vector<JcpBundleRhs> &local_rhs,
    const std::vector<JcpBundleRhs> &dense_local_rhs,
    const std::vector<JcpBundle> &local,
    const std::vector<KifmmJcpChannelRow> &approx,
    const std::vector<KifmmJcpChannelRow> &direct,
    const KifmmEquivalentSystemCache &upward_cache,
    const KifmmEquivalentSystemCache &local_cache) {
  result.k123_upward_equivalent_estimated_bytes =
      bundle_vector_storage_bytes(upward);
  result.k123_local_rhs_estimated_bytes = rhs_vector_storage_bytes(local_rhs);
  result.k123_dense_local_rhs_estimated_bytes =
      rhs_vector_storage_bytes(dense_local_rhs);
  result.k123_local_equivalent_estimated_bytes =
      bundle_vector_storage_bytes(local);
  result.k123_rows_estimated_bytes =
      row_storage_bytes(approx) + row_storage_bytes(direct);
  result.k123_equivalent_system_cache_estimated_bytes =
      equivalent_system_cache_storage_bytes(upward_cache) +
      equivalent_system_cache_storage_bytes(local_cache);
  result.k123_total_estimated_bytes =
      result.k123_upward_equivalent_estimated_bytes +
      result.k123_local_rhs_estimated_bytes +
      result.k123_dense_local_rhs_estimated_bytes +
      result.k123_local_equivalent_estimated_bytes +
      result.k123_rows_estimated_bytes +
      result.k123_equivalent_system_cache_estimated_bytes +
      result.m2l_matrix_cache_estimated_bytes;
}

struct RmsAccumulator {
  long double err_sq{0.0L};
  long double ref_sq{0.0L};
  long double abs_sq{0.0L};
  std::uint64_t count{0};
  double max_abs_error{0.0};

  void add(double approx, double ref) noexcept {
    const long double diff = static_cast<long double>(approx) - ref;
    err_sq += diff * diff;
    ref_sq += static_cast<long double>(ref) * ref;
    abs_sq += diff * diff;
    ++count;
    max_abs_error =
        std::max(max_abs_error, std::abs(static_cast<double>(diff)));
  }

  double relative_rms() const noexcept {
    return std::sqrt(static_cast<double>(err_sq /
                                        std::max(ref_sq, 1.0e-300L)));
  }

  double abs_rms() const noexcept {
    if (count == 0) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return std::sqrt(static_cast<double>(abs_sq /
                                        static_cast<long double>(count)));
  }

  double reference_rms() const noexcept {
    if (count == 0) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return std::sqrt(static_cast<double>(ref_sq /
                                        static_cast<long double>(count)));
  }
};

struct K4Parts {
  double laplace{0.0};
  double yukawa{0.0};
  double combined{0.0};
};

KifmmJcpChannelRow channel_row_from_bundle(const KifmmSurfaceRule &rule,
                                           const KifmmHierarchyNode &node,
                                           const JcpBundle &bundle,
                                           double kappa, double eps_ratio,
                                           double inv_eps_ratio, Vec3 target,
                                           Vec3 target_normal);

KifmmJcpChannelRow channel_row_from_bundle_k123_only(
    const KifmmSurfaceRule &rule, const KifmmHierarchyNode &node,
    const JcpBundle &bundle, double kappa, double eps_ratio,
    double inv_eps_ratio, Vec3 target, Vec3 target_normal);

K4Parts k4_parts_from_bundle(const KifmmSurfaceRule &rule,
                             const KifmmHierarchyNode &node,
                             const JcpBundle &bundle, double kappa,
                             Vec3 target, Vec3 target_normal);

void add_direct_sources_to_points(std::span<const KifmmPanelSource> sources,
                                  const KifmmHierarchyNode &source_node,
                                  KifmmKernel kernel, double kappa,
                                  std::span<const KifmmSurfacePoint> points,
                                  Eigen::MatrixXd &rhs);

const char *family_name(KifmmFieldFamily family) noexcept {
  switch (family) {
  case KifmmFieldFamily::PhinCharge:
    return "PhinCharge";
  case KifmmFieldFamily::PhiNormalX:
    return "PhiNormalX";
  case KifmmFieldFamily::PhiNormalY:
    return "PhiNormalY";
  case KifmmFieldFamily::PhiNormalZ:
    return "PhiNormalZ";
  }
  return "unknown";
}

const char *kernel_name(KifmmKernel kernel) noexcept {
  return kernel == KifmmKernel::Laplace ? "Laplace" : "Yukawa";
}

Eigen::MatrixXd build_rhs_matrix(const KifmmEquivalentSystem &system,
                                 std::span<const KifmmPanelSource> sources,
                                 KifmmKernel kernel, double kappa) {
  Eigen::MatrixXd rhs(static_cast<Eigen::Index>(system.check_points.size()),
                      static_cast<Eigen::Index>(kFamilies.size()));
  for (std::size_t col = 0; col < kFamilies.size(); ++col) {
    for (std::size_t row = 0; row < system.check_points.size(); ++row) {
      rhs(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
          direct_field_potential(sources, kFamilies[col], kernel, kappa,
                                 system.check_points[row].x);
    }
  }
  return rhs;
}

bool finite_matrix(const Eigen::MatrixXd &matrix) noexcept {
  for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
    for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
      if (!std::isfinite(matrix(i, j))) {
        return false;
      }
    }
  }
  return true;
}

bool bundle_passed(const JcpBundle &bundle) noexcept {
  if (!bundle.valid) {
    return false;
  }
  for (std::size_t i = 0; i < kFamilies.size(); ++i) {
    if (!bundle.laplace[i].finite || bundle.laplace[i].status != "pass" ||
        !bundle.yukawa[i].finite || bundle.yukawa[i].status != "pass") {
      return false;
    }
  }
  return true;
}

JcpBundle bundle_from_results(
    const std::vector<KifmmEquivalentSolveResult> &laplace,
    const std::vector<KifmmEquivalentSolveResult> &yukawa) {
  JcpBundle bundle;
  if (laplace.size() != kFamilies.size() || yukawa.size() != kFamilies.size()) {
    return bundle;
  }
  for (std::size_t i = 0; i < kFamilies.size(); ++i) {
    bundle.laplace[i] = laplace[i];
    bundle.yukawa[i] = yukawa[i];
  }
  bundle.valid = true;
  bundle.valid = bundle_passed(bundle);
  return bundle;
}

KifmmJcpChannels channels_from_bundle(const JcpBundle &bundle) {
  KifmmJcpChannels channels;
  channels.phin_laplace = bundle.laplace[0];
  channels.phin_yukawa = bundle.yukawa[0];
  for (std::size_t i = 0; i < 3; ++i) {
    channels.phi_normal_laplace[i] = bundle.laplace[i + 1];
    channels.phi_normal_yukawa[i] = bundle.yukawa[i + 1];
  }
  channels.finite = bundle_passed(bundle);
  channels.status = channels.finite ? "pass" : "fail";
  for (std::size_t i = 0; i < kFamilies.size(); ++i) {
    channels.max_relative_residual =
        std::max(channels.max_relative_residual,
                 bundle.laplace[i].relative_residual);
    channels.max_relative_residual =
        std::max(channels.max_relative_residual,
                 bundle.yukawa[i].relative_residual);
    channels.max_condition_estimate =
        std::max(channels.max_condition_estimate,
                 bundle.laplace[i].condition_estimate);
    channels.max_condition_estimate =
        std::max(channels.max_condition_estimate,
                 bundle.yukawa[i].condition_estimate);
  }
  return channels;
}

JcpBundle bundle_from_channels(const KifmmJcpChannels &channels) {
  JcpBundle bundle;
  bundle.laplace[0] = channels.phin_laplace;
  bundle.yukawa[0] = channels.phin_yukawa;
  for (std::size_t i = 0; i < 3; ++i) {
    bundle.laplace[i + 1] = channels.phi_normal_laplace[i];
    bundle.yukawa[i + 1] = channels.phi_normal_yukawa[i];
  }
  bundle.valid = channels.finite && bundle_passed(bundle);
  return bundle;
}

double bundle_max_relative_residual(const JcpBundle &bundle) noexcept {
  double value = 0.0;
  for (std::size_t i = 0; i < kFamilies.size(); ++i) {
    value = std::max(value, bundle.laplace[i].relative_residual);
    value = std::max(value, bundle.yukawa[i].relative_residual);
  }
  return value;
}

KifmmCartesianBridgeConfig
bridge_config_from(const KifmmHierarchyJcpConfig &config) {
  KifmmCartesianBridgeConfig bridge;
  bridge.surface_order = config.hierarchy.order;
  bridge.cartesian_order = std::min(5, std::max(1, config.hierarchy.order));
  bridge.equiv_scale = config.hierarchy.equiv_scale;
  bridge.upward_check_scale = config.hierarchy.upward_check_scale;
  bridge.local_equiv_scale = config.hierarchy.local_equiv_scale;
  bridge.local_check_scale = config.hierarchy.local_check_scale;
  bridge.kappa = config.kappa;
  bridge.eps_ratio = config.eps_ratio;
  bridge.aggregate_relative_rms_gate = config.aggregate_relative_rms_gate;
  bridge.max_channel_relative_rms_gate = config.max_channel_relative_rms_gate;
  bridge.solver = "qr";
  bridge.condition_policy = config.hierarchy.condition_policy;
  return bridge;
}

std::vector<titan::tabi::fmm_m2l::PrototypePanel>
prototype_panels_from_sources(std::span<const KifmmPanelSource> sources) {
  std::vector<titan::tabi::fmm_m2l::PrototypePanel> panels;
  panels.reserve(sources.size());
  for (const KifmmPanelSource &source : sources) {
    titan::tabi::fmm_m2l::PrototypePanel panel;
    panel.x = source.position.x;
    panel.y = source.position.y;
    panel.z = source.position.z;
    panel.nx = source.normal.x;
    panel.ny = source.normal.y;
    panel.nz = source.normal.z;
    panel.area = source.area;
    panel.scalar_density = source.phin;
    panels.push_back(panel);
  }
  return panels;
}

std::vector<double>
phi_values_from_sources(std::span<const KifmmPanelSource> sources) {
  std::vector<double> values;
  values.reserve(sources.size());
  for (const KifmmPanelSource &source : sources) {
    values.push_back(source.phi);
  }
  return values;
}

std::vector<double>
phin_values_from_sources(std::span<const KifmmPanelSource> sources) {
  std::vector<double> values;
  values.reserve(sources.size());
  for (const KifmmPanelSource &source : sources) {
    values.push_back(source.phin);
  }
  return values;
}

std::vector<std::size_t>
source_indices_as_size_t(const KifmmHierarchyNode &node) {
  std::vector<std::size_t> indices;
  indices.reserve(node.source_indices.size());
  for (const int index : node.source_indices) {
    indices.push_back(static_cast<std::size_t>(index));
  }
  return indices;
}

JcpBundle solve_source_bundle_cartesian_m2e(
    const KifmmSurfaceRule &rule, KifmmEquivalentSystemCache &cache,
    const std::vector<titan::tabi::fmm_m2l::PrototypePanel> &panels,
    const std::vector<double> &phi_values,
    const std::vector<double> &phin_values,
    const KifmmHierarchyNode &source_node,
    const KifmmCartesianBridgeConfig &bridge_config, double &rhs_seconds,
    double &solve_seconds) {
  const auto rhs_start = Clock::now();
  const std::vector<std::size_t> indices = source_indices_as_size_t(source_node);
  const titan::tabi::fmm_m2l::CartesianJcpMoments moments =
      titan::tabi::fmm_m2l::build_cartesian_jcp_moments(
          panels, phi_values, phin_values, indices, source_node.center.x,
          source_node.center.y, source_node.center.z,
          bridge_config.cartesian_order);
  rhs_seconds += elapsed_seconds(rhs_start, Clock::now());
  const auto solve_start = Clock::now();
  const KifmmJcpChannels channels = bridge_cartesian_moments_to_equivalents(
      rule, box_from_node(source_node), moments, bridge_config, &cache);
  solve_seconds += elapsed_seconds(solve_start, Clock::now());
  return bundle_from_channels(channels);
}

KifmmJcpChannelRow channel_row_from_cartesian_local(
    const titan::tabi::fmm_m2l::CartesianJcpLocalP4 &local,
    const KifmmHierarchyNode &node, Vec3 target, Vec3 target_normal,
    int target_order) {
  titan::tabi::fmm_m2l::PrototypePanel panel;
  panel.x = target.x;
  panel.y = target.y;
  panel.z = target.z;
  panel.nx = target_normal.x;
  panel.ny = target_normal.y;
  panel.nz = target_normal.z;
  panel.area = 1.0;
  const std::array<double, 4> values =
      titan::tabi::fmm_m2l::evaluate_cartesian_jcp_l2p_order(
          panel, node.center.x, node.center.y, node.center.z, local,
          target_order);
  return {values[0], values[1], values[2], values[3]};
}

JcpBundle solve_source_bundle(
    const KifmmSurfaceRule &rule, KifmmEquivalentSystemCache &cache,
    std::span<const KifmmPanelSource> sources,
    const KifmmHierarchyNode &source_node, double kappa,
    const std::string &solver, const std::string &condition_policy,
    double &rhs_seconds, double &solve_seconds) {
  const KifmmBox box = box_from_node(source_node);
  const KifmmEquivalentSystem laplace_system = build_equivalent_system_cached(
      cache, rule, box, KifmmKernel::Laplace, 0.0, solver, condition_policy);
  const KifmmEquivalentSystem yukawa_system = build_equivalent_system_cached(
      cache, rule, box, KifmmKernel::Yukawa, kappa, solver, condition_policy);
  if (!laplace_system.finite || !yukawa_system.finite) {
    return {};
  }
  const std::vector<KifmmPanelSource> node_sources =
      gather_sources(sources, source_node.source_indices);
  const auto rhs_start = Clock::now();
  const Eigen::MatrixXd laplace_rhs =
      build_rhs_matrix(laplace_system, node_sources, KifmmKernel::Laplace, 0.0);
  const Eigen::MatrixXd yukawa_rhs =
      build_rhs_matrix(yukawa_system, node_sources, KifmmKernel::Yukawa, kappa);
  rhs_seconds += elapsed_seconds(rhs_start, Clock::now());
  const auto solve_start = Clock::now();
  const std::vector<KifmmEquivalentSolveResult> laplace =
      solve_equivalent_rhs_batch(laplace_system, laplace_rhs, solver);
  const std::vector<KifmmEquivalentSolveResult> yukawa =
      solve_equivalent_rhs_batch(yukawa_system, yukawa_rhs, solver);
  solve_seconds += elapsed_seconds(solve_start, Clock::now());
  return bundle_from_results(laplace, yukawa);
}

bool ensure_rhs_for_node(std::vector<JcpBundleRhs> &rhs_by_node,
                         const KifmmEquivalentSystem &laplace_system,
                         const KifmmEquivalentSystem &yukawa_system,
                         int node_id) {
  JcpBundleRhs &rhs = rhs_by_node[static_cast<std::size_t>(node_id)];
  const Eigen::Index laplace_rows =
      static_cast<Eigen::Index>(laplace_system.check_points.size());
  const Eigen::Index yukawa_rows =
      static_cast<Eigen::Index>(yukawa_system.check_points.size());
  const Eigen::Index cols = static_cast<Eigen::Index>(kFamilies.size());
  if (rhs.laplace.size() == 0 && rhs.yukawa.size() == 0) {
    rhs.laplace = Eigen::MatrixXd::Zero(laplace_rows, cols);
    rhs.yukawa = Eigen::MatrixXd::Zero(yukawa_rows, cols);
    return true;
  }
  return rhs.laplace.rows() == laplace_rows && rhs.laplace.cols() == cols &&
         rhs.yukawa.rows() == yukawa_rows && rhs.yukawa.cols() == cols;
}

std::int64_t quantize_m2l_key(double value) noexcept {
  if (!std::isfinite(value)) {
    return std::numeric_limits<std::int64_t>::min();
  }
  return static_cast<std::int64_t>(std::llround(value * 1.0e12));
}

bool valid_m2l_matrix_cache_policy(const std::string &policy) noexcept {
  return policy == "off" || policy == "relative-geometry-cache" ||
         policy == "static-translation-table-cache" ||
         policy == "canonical-offset-cache" ||
         policy == "target-node-streaming-cache" ||
         policy == "bounded-lru-cache";
}

std::int64_t canonical_offset_component(double delta, double scale) noexcept {
  if (!std::isfinite(delta) || !std::isfinite(scale) || !(scale > 0.0)) {
    return std::numeric_limits<std::int64_t>::min();
  }
  return static_cast<std::int64_t>(std::llround(delta / scale));
}

M2lMatrixCacheKey make_m2l_matrix_cache_key(
    const KifmmSurfaceRule &source_rule, const KifmmHierarchyNode &source_node,
    const KifmmHierarchyNode &target_node, KifmmKernel kernel, double kappa,
    double target_check_scale, std::span<const KifmmSurfacePoint> points,
    std::size_t equiv_count, const std::string &policy) {
  M2lMatrixCacheKey key;
  key.kernel = kernel == KifmmKernel::Laplace ? 0 : 1;
  key.order = source_rule.order;
  key.source_level = source_node.level;
  key.target_level = target_node.level;
  key.kappa = quantize_m2l_key(kappa);
  key.source_half_width = quantize_m2l_key(source_node.half_width);
  key.target_half_width = quantize_m2l_key(target_node.half_width);
  key.source_equiv_scale = quantize_m2l_key(source_rule.equiv_scale);
  key.target_check_scale = quantize_m2l_key(target_check_scale);
  const Vec3 delta{target_node.center.x - source_node.center.x,
                   target_node.center.y - source_node.center.y,
                   target_node.center.z - source_node.center.z};
  if (m2l_matrix_cache_uses_canonical_offsets(policy)) {
    const double offset_scale =
        std::min(source_node.half_width, target_node.half_width);
    key.target_source_dx = canonical_offset_component(delta.x, offset_scale);
    key.target_source_dy = canonical_offset_component(delta.y, offset_scale);
    key.target_source_dz = canonical_offset_component(delta.z, offset_scale);
  } else {
    key.target_source_dx = quantize_m2l_key(delta.x);
    key.target_source_dy = quantize_m2l_key(delta.y);
    key.target_source_dz = quantize_m2l_key(delta.z);
  }
  key.point_count = points.size();
  key.equiv_count = equiv_count;
  return key;
}

const Eigen::MatrixXd *find_m2l_matrix_cache_entry(
    M2lMatrixCache &cache, const M2lMatrixCacheKey &key) {
  if (m2l_matrix_cache_is_static_table(cache.policy) ||
      m2l_matrix_cache_is_bounded_lru(cache.policy)) {
    const auto found = cache.m2l_matrix_cache_key_index.find(key);
    if (found != cache.m2l_matrix_cache_key_index.end()) {
      M2lMatrixCacheEntry &entry = cache.entries[found->second];
      ++cache.hits;
      entry.last_used_tick = ++cache.access_tick;
      return &entry.matrix;
    }
    ++cache.misses;
    return nullptr;
  }
  for (M2lMatrixCacheEntry &entry : cache.entries) {
    if (entry.key == key) {
      ++cache.hits;
      entry.last_used_tick = ++cache.access_tick;
      return &entry.matrix;
    }
  }
  ++cache.misses;
  return nullptr;
}

const Eigen::MatrixXd *
find_m2l_matrix_cache_entry_readonly(const M2lMatrixCache &cache,
                                     const M2lMatrixCacheKey &key) {
  if (m2l_matrix_cache_is_static_table(cache.policy) ||
      m2l_matrix_cache_is_bounded_lru(cache.policy)) {
    const auto found = cache.m2l_matrix_cache_key_index.find(key);
    if (found != cache.m2l_matrix_cache_key_index.end()) {
      return &cache.entries[found->second].matrix;
    }
    return nullptr;
  }
  for (const M2lMatrixCacheEntry &entry : cache.entries) {
    if (entry.key == key) {
      return &entry.matrix;
    }
  }
  return nullptr;
}

Eigen::MatrixXd build_m2l_kernel_matrix(
    KifmmKernel kernel, double kappa,
    std::span<const KifmmSurfacePoint> points,
    const std::vector<KifmmSurfacePoint> &equiv_points) {
  const Eigen::Index point_count = static_cast<Eigen::Index>(points.size());
  const Eigen::Index equiv_count =
      static_cast<Eigen::Index>(equiv_points.size());
  Eigen::MatrixXd kernel_matrix(point_count, equiv_count);
  for (Eigen::Index i = 0; i < point_count; ++i) {
    for (Eigen::Index j = 0; j < equiv_count; ++j) {
      kernel_matrix(i, j) = scalar_kernel_value(
          kernel, points[static_cast<std::size_t>(i)].x,
          equiv_points[static_cast<std::size_t>(j)].x, kappa);
    }
  }
  return kernel_matrix;
}

const Eigen::MatrixXd *get_or_build_m2l_kernel_matrix(
    const KifmmSurfaceRule &source_rule, const KifmmHierarchyNode &source_node,
    const KifmmHierarchyNode &target_node, KifmmKernel kernel, double kappa,
    double target_check_scale, std::span<const KifmmSurfacePoint> points,
    const std::vector<KifmmSurfacePoint> &equiv_points,
    M2lMatrixCache &matrix_cache, Eigen::MatrixXd &local_kernel_matrix) {
  if (m2l_matrix_cache_enabled(matrix_cache.policy)) {
    const M2lMatrixCacheKey key =
        make_m2l_matrix_cache_key(source_rule, source_node, target_node, kernel,
                                  kappa, target_check_scale, points,
                                  equiv_points.size(), matrix_cache.policy);
    if (const Eigen::MatrixXd *matrix =
            find_m2l_matrix_cache_entry(matrix_cache, key)) {
      return matrix;
    }
    const auto build_start = Clock::now();
    local_kernel_matrix =
        build_m2l_kernel_matrix(kernel, kappa, points, equiv_points);
    matrix_cache.build_seconds += elapsed_seconds(build_start, Clock::now());
    return insert_m2l_matrix_cache_entry(matrix_cache, key,
                                         local_kernel_matrix);
  }
  const auto build_start = Clock::now();
  local_kernel_matrix =
      build_m2l_kernel_matrix(kernel, kappa, points, equiv_points);
  matrix_cache.build_seconds += elapsed_seconds(build_start, Clock::now());
  matrix_cache.key_status = "off";
  return &local_kernel_matrix;
}

void add_bundle_to_points(const KifmmSurfaceRule &source_rule,
                           const KifmmHierarchyNode &source_node,
                           const JcpBundle &source_bundle, KifmmKernel kernel,
                           double kappa,
                           std::span<const KifmmSurfacePoint> points,
                          Eigen::MatrixXd &rhs) {
  const KifmmBox source_box = box_from_node(source_node);
  const auto &fields = kernel == KifmmKernel::Laplace ? source_bundle.laplace
                                                       : source_bundle.yukawa;
  for (std::size_t i = 0; i < points.size(); ++i) {
    for (std::size_t col = 0; col < kFamilies.size(); ++col) {
      rhs(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(col)) +=
          equivalent_scalar_potential(source_rule, source_box, fields[col],
                                       kernel, kappa, points[i].x);
    }
  }
}

bool add_bundle_to_points_batched(const KifmmSurfaceRule &source_rule,
                                  const KifmmHierarchyNode &source_node,
                                  const JcpBundle &source_bundle,
                                  KifmmKernel kernel, double kappa,
                                  const KifmmHierarchyNode &target_node,
                                  double target_check_scale,
                                  std::span<const KifmmSurfacePoint> points,
                                  Eigen::MatrixXd &rhs,
                                  M2lMatrixCache &matrix_cache) {
  const KifmmBox source_box = box_from_node(source_node);
  const std::vector<KifmmSurfacePoint> equiv_points =
      instantiate_surface(source_rule.unit_equiv, source_box);
  const auto &fields = kernel == KifmmKernel::Laplace ? source_bundle.laplace
                                                       : source_bundle.yukawa;
  const Eigen::Index point_count = static_cast<Eigen::Index>(points.size());
  const Eigen::Index equiv_count =
      static_cast<Eigen::Index>(equiv_points.size());
  const Eigen::Index column_count = static_cast<Eigen::Index>(kFamilies.size());
  if (rhs.rows() != point_count || rhs.cols() != column_count ||
      equiv_count <= 0) {
    return false;
  }
  Eigen::MatrixXd charges(equiv_count, column_count);
  for (Eigen::Index col = 0; col < column_count; ++col) {
    const KifmmEquivalentSolveResult &field =
        fields[static_cast<std::size_t>(col)];
    if (!field.finite ||
        field.charges.size() != static_cast<std::size_t>(equiv_count)) {
      return false;
    }
    for (Eigen::Index j = 0; j < equiv_count; ++j) {
      charges(j, col) = field.charges[static_cast<std::size_t>(j)];
    }
  }
  const Eigen::MatrixXd *kernel_matrix = nullptr;
  Eigen::MatrixXd local_kernel_matrix;
  kernel_matrix = get_or_build_m2l_kernel_matrix(
      source_rule, source_node, target_node, kernel, kappa, target_check_scale,
      points, equiv_points, matrix_cache, local_kernel_matrix);
  const auto gemm_start = Clock::now();
  rhs.noalias() += (*kernel_matrix) * charges;
  matrix_cache.gemm_seconds += elapsed_seconds(gemm_start, Clock::now());
  return finite_matrix(rhs);
}

bool valid_m2l_eval_policy(const std::string &policy) noexcept {
  return policy == "scalar-existing" || policy == "target-check-batched";
}

bool valid_m2l_gemm_batch_policy(const std::string &policy) noexcept {
  return policy == "per-pair" || policy == "group-by-cache-key";
}

bool valid_m2l_parallel_policy(const std::string &policy) noexcept {
  return policy == "serial" || policy == "target-node-key-group";
}

bool valid_jcp_channel_policy(const std::string &policy) noexcept {
  return policy == "all" || policy == "k123-only";
}

bool m2l_local_oracle_mode(const std::string &mode) noexcept {
  return mode == "local-check-oracle" || mode == "l2p-oracle";
}

struct M2lTargetSystems {
  KifmmEquivalentSystem laplace{};
  KifmmEquivalentSystem yukawa{};
};

void account_m2l_rhs_add(KifmmHierarchyJcpResult &result,
                         const KifmmSurfaceRule &upward_rule,
                         const M2lTargetSystems &systems) {
  const std::uint64_t laplace_checks =
      static_cast<std::uint64_t>(systems.laplace.check_points.size());
  const std::uint64_t yukawa_checks =
      static_cast<std::uint64_t>(systems.yukawa.check_points.size());
  const std::uint64_t equiv_points =
      static_cast<std::uint64_t>(upward_rule.unit_equiv.size());
  result.m2l_check_points_total += laplace_checks + yukawa_checks;
  result.m2l_equivalent_points += 2 * equiv_points;
  result.m2l_kernel_evals_estimated +=
      (laplace_checks + yukawa_checks) * equiv_points;
  result.local_rhs_adds += laplace_checks + yukawa_checks;
}

bool prepare_m2l_target_systems(
    KifmmEquivalentSystemCache &local_cache, const KifmmSurfaceRule &local_rule,
    const KifmmHierarchyNode &target_node, const KifmmHierarchyJcpConfig &config,
    const std::string &solver, std::vector<JcpBundleRhs> &local_rhs,
    std::vector<JcpBundleRhs> &dense_local_rhs, bool include_dense_oracle,
    M2lTargetSystems &systems, KifmmHierarchyJcpResult &result) {
  const int target_node_id = target_node.id;
  const KifmmBox target_box = box_from_node(target_node);
  systems.laplace = build_equivalent_system_cached(
      local_cache, local_rule, target_box, KifmmKernel::Laplace, 0.0, solver,
      config.hierarchy.condition_policy);
  systems.yukawa = build_equivalent_system_cached(
      local_cache, local_rule, target_box, KifmmKernel::Yukawa, config.kappa,
      solver, config.hierarchy.condition_policy);
  if (!systems.laplace.finite || !systems.yukawa.finite ||
      !ensure_rhs_for_node(local_rhs, systems.laplace, systems.yukawa,
                           target_node_id)) {
    result.status = "m2l_local_system_gap";
    return false;
  }
  if (include_dense_oracle &&
      !ensure_rhs_for_node(dense_local_rhs, systems.laplace, systems.yukawa,
                           target_node_id)) {
    result.status = "m2l_dense_local_system_gap";
    return false;
  }
  return true;
}

void add_m2l_dense_oracle_rhs(std::span<const KifmmPanelSource> sources,
                              const KifmmHierarchyNode &source_node,
                              double kappa, const M2lTargetSystems &systems,
                              JcpBundleRhs &rhs) {
  add_direct_sources_to_points(sources, source_node, KifmmKernel::Laplace, 0.0,
                               systems.laplace.check_points, rhs.laplace);
  add_direct_sources_to_points(sources, source_node, KifmmKernel::Yukawa, kappa,
                               systems.yukawa.check_points, rhs.yukawa);
}

bool accumulate_direct_target_m2l_pair(
    const KifmmSurfaceRule &upward_rule, const KifmmHierarchyNode &source_node,
    const KifmmHierarchyNode &target_node, const JcpBundle &source_bundle,
    double kappa, double eps_ratio, double inv_eps_ratio,
    std::span<const Vec3> targets, std::span<const Vec3> target_normals,
    std::vector<KifmmJcpChannelRow> &approx,
    std::vector<K4Parts> &approx_k4_parts) {
  if (!bundle_passed(source_bundle)) {
    return false;
  }
  for (const int target_index : target_node.target_indices) {
    const std::size_t ti = static_cast<std::size_t>(target_index);
    const KifmmJcpChannelRow row = channel_row_from_bundle(
        upward_rule, source_node, source_bundle, kappa, eps_ratio,
        inv_eps_ratio, targets[ti], target_normals[ti]);
    approx[ti].k1 += row.k1;
    approx[ti].k2 += row.k2;
    approx[ti].k3 += row.k3;
    approx[ti].k4 += row.k4;
    const K4Parts parts = k4_parts_from_bundle(
        upward_rule, source_node, source_bundle, kappa, targets[ti],
        target_normals[ti]);
    approx_k4_parts[ti].laplace += parts.laplace;
    approx_k4_parts[ti].yukawa += parts.yukawa;
    approx_k4_parts[ti].combined += parts.combined;
  }
  return true;
}

bool accumulate_standard_m2l_pair(
    const KifmmSurfaceRule &upward_rule,
    std::span<const KifmmPanelSource> sources,
    const KifmmHierarchyNode &source_node, const JcpBundle &source_bundle,
    const M2lTargetSystems &systems, double kappa, bool include_dense_oracle,
    JcpBundleRhs &local_rhs, JcpBundleRhs &dense_rhs,
    KifmmHierarchyJcpResult &result) {
  if (!bundle_passed(source_bundle)) {
    result.status = "m2l_missing_upward";
    return false;
  }
  add_bundle_to_points(upward_rule, source_node, source_bundle,
                       KifmmKernel::Laplace, 0.0,
                       systems.laplace.check_points, local_rhs.laplace);
  add_bundle_to_points(upward_rule, source_node, source_bundle,
                       KifmmKernel::Yukawa, kappa,
                       systems.yukawa.check_points, local_rhs.yukawa);
  if (include_dense_oracle) {
    add_m2l_dense_oracle_rhs(sources, source_node, kappa, systems, dense_rhs);
  }
  account_m2l_rhs_add(result, upward_rule, systems);
  ++result.m2l_translations;
  return true;
}

bool accumulate_batched_m2l_pair(
    const KifmmSurfaceRule &upward_rule,
    std::span<const KifmmPanelSource> sources,
    const KifmmHierarchyNode &source_node, const JcpBundle &source_bundle,
    const KifmmHierarchyNode &target_node, const M2lTargetSystems &systems,
    double kappa, bool include_dense_oracle, JcpBundleRhs &local_rhs,
    JcpBundleRhs &dense_rhs, M2lMatrixCache &matrix_cache,
    double target_check_scale, KifmmHierarchyJcpResult &result) {
  if (!bundle_passed(source_bundle)) {
    result.status = "m2l_missing_upward";
    return false;
  }
  if (!add_bundle_to_points_batched(upward_rule, source_node, source_bundle,
                                    KifmmKernel::Laplace, 0.0, target_node,
                                    target_check_scale,
                                    systems.laplace.check_points,
                                    local_rhs.laplace, matrix_cache) ||
      !add_bundle_to_points_batched(upward_rule, source_node, source_bundle,
                                    KifmmKernel::Yukawa, kappa, target_node,
                                    target_check_scale,
                                    systems.yukawa.check_points,
                                    local_rhs.yukawa, matrix_cache)) {
    result.status = "m2l_batched_accumulation_gap";
    return false;
  }
  if (include_dense_oracle) {
    add_m2l_dense_oracle_rhs(sources, source_node, kappa, systems, dense_rhs);
  }
  account_m2l_rhs_add(result, upward_rule, systems);
  ++result.m2l_translations;
  return true;
}

struct M2lGroupedSourceSet {
  M2lMatrixCacheKey key{};
  std::vector<int> source_node_ids{};
};

struct M2lPairRecord {
  int target_node_id{-1};
  int source_node_id{-1};
};

struct M2lGroupedPairSet {
  M2lMatrixCacheKey key{};
  std::vector<M2lPairRecord> records{};
};

void append_grouped_source(std::vector<M2lGroupedSourceSet> &groups,
                           const M2lMatrixCacheKey &key,
                           int source_node_id) {
  for (M2lGroupedSourceSet &group : groups) {
    if (group.key == key) {
      group.source_node_ids.push_back(source_node_id);
      return;
    }
  }
  groups.push_back({key, {source_node_id}});
}

void append_grouped_record(std::vector<M2lGroupedPairSet> &groups,
                           const M2lMatrixCacheKey &key,
                           M2lPairRecord record) {
  for (M2lGroupedPairSet &group : groups) {
    if (group.key == key) {
      group.records.push_back(record);
      return;
    }
  }
  groups.push_back({key, {record}});
}

std::vector<M2lGroupedSourceSet> build_m2l_source_groups(
    const KifmmSurfaceRule &upward_rule, const KifmmHierarchyNode &target_node,
    std::span<const KifmmSurfacePoint> points,
    const std::vector<int> &source_node_ids,
    const std::vector<KifmmHierarchyNode> &nodes, KifmmKernel kernel,
    double kappa, double target_check_scale, std::size_t equiv_count,
    const std::string &cache_policy) {
  std::vector<M2lGroupedSourceSet> groups;
  for (const int source_node_id : source_node_ids) {
    const KifmmHierarchyNode &source_node =
        nodes[static_cast<std::size_t>(source_node_id)];
    append_grouped_source(
        groups,
        make_m2l_matrix_cache_key(upward_rule, source_node, target_node,
                                  kernel, kappa, target_check_scale, points,
                                  equiv_count, cache_policy),
        source_node_id);
  }
  return groups;
}

std::vector<M2lGroupedPairSet> build_m2l_record_groups(
    const KifmmSurfaceRule &upward_rule,
    const std::vector<M2lPairRecord> &records,
    const std::vector<M2lTargetSystems> &systems_by_node,
    const std::vector<KifmmHierarchyNode> &nodes, KifmmKernel kernel,
    double kappa, double target_check_scale, std::size_t equiv_count,
    const std::string &cache_policy) {
  std::vector<M2lGroupedPairSet> groups;
  for (const M2lPairRecord &record : records) {
    const KifmmHierarchyNode &source_node =
        nodes[static_cast<std::size_t>(record.source_node_id)];
    const KifmmHierarchyNode &target_node =
        nodes[static_cast<std::size_t>(record.target_node_id)];
    const auto &points = kernel == KifmmKernel::Laplace
                             ? systems_by_node[static_cast<std::size_t>(
                                   record.target_node_id)]
                                   .laplace.check_points
                             : systems_by_node[static_cast<std::size_t>(
                                   record.target_node_id)]
                                   .yukawa.check_points;
    append_grouped_record(
        groups,
        make_m2l_matrix_cache_key(upward_rule, source_node, target_node,
                                  kernel, kappa, target_check_scale, points,
                                  equiv_count, cache_policy),
        record);
  }
  return groups;
}

bool add_grouped_bundle_to_points_batched(
    const KifmmSurfaceRule &source_rule,
    const std::vector<KifmmHierarchyNode> &nodes,
    const std::vector<JcpBundle> &upward,
    const KifmmHierarchyNode &target_node, const M2lGroupedSourceSet &group,
    KifmmKernel kernel, double kappa, double target_check_scale,
    std::span<const KifmmSurfacePoint> points, Eigen::MatrixXd &rhs,
    M2lMatrixCache &matrix_cache, KifmmHierarchyJcpResult &result) {
  if (group.source_node_ids.empty()) {
    return true;
  }
  const KifmmHierarchyNode &first_source =
      nodes[static_cast<std::size_t>(group.source_node_ids.front())];
  const KifmmBox source_box = box_from_node(first_source);
  const std::vector<KifmmSurfacePoint> equiv_points =
      instantiate_surface(source_rule.unit_equiv, source_box);
  const Eigen::Index point_count = static_cast<Eigen::Index>(points.size());
  const Eigen::Index equiv_count =
      static_cast<Eigen::Index>(equiv_points.size());
  const Eigen::Index family_count = static_cast<Eigen::Index>(kFamilies.size());
  const Eigen::Index group_size =
      static_cast<Eigen::Index>(group.source_node_ids.size());
  if (rhs.rows() != point_count || rhs.cols() != family_count ||
      equiv_count <= 0 || group_size <= 0) {
    return false;
  }
  Eigen::MatrixXd charges(equiv_count, family_count * group_size);
  for (Eigen::Index group_index = 0; group_index < group_size; ++group_index) {
    const int source_node_id =
        group.source_node_ids[static_cast<std::size_t>(group_index)];
    const JcpBundle &source_bundle =
        upward[static_cast<std::size_t>(source_node_id)];
    if (!bundle_passed(source_bundle)) {
      result.status = "m2l_missing_upward";
      return false;
    }
    const auto &fields = kernel == KifmmKernel::Laplace
                             ? source_bundle.laplace
                             : source_bundle.yukawa;
    for (Eigen::Index col = 0; col < family_count; ++col) {
      const KifmmEquivalentSolveResult &field =
          fields[static_cast<std::size_t>(col)];
      if (!field.finite ||
          field.charges.size() != static_cast<std::size_t>(equiv_count)) {
        return false;
      }
      const Eigen::Index out_col = group_index * family_count + col;
      for (Eigen::Index j = 0; j < equiv_count; ++j) {
        charges(j, out_col) = field.charges[static_cast<std::size_t>(j)];
      }
    }
  }
  Eigen::MatrixXd local_kernel_matrix;
  const Eigen::MatrixXd *kernel_matrix = get_or_build_m2l_kernel_matrix(
      source_rule, first_source, target_node, kernel, kappa, target_check_scale,
      points, equiv_points, matrix_cache, local_kernel_matrix);
  const auto gemm_start = Clock::now();
  const Eigen::MatrixXd product = (*kernel_matrix) * charges;
  matrix_cache.gemm_seconds += elapsed_seconds(gemm_start, Clock::now());
  for (Eigen::Index group_index = 0; group_index < group_size; ++group_index) {
    for (Eigen::Index col = 0; col < family_count; ++col) {
      rhs.col(col) += product.col(group_index * family_count + col);
    }
  }
  ++result.m2l_gemm_group_count;
  result.m2l_gemm_grouped_pair_count +=
      static_cast<std::uint64_t>(group.source_node_ids.size());
  result.m2l_gemm_grouped_rhs_columns +=
      static_cast<std::uint64_t>(group.source_node_ids.size()) *
      static_cast<std::uint64_t>(kFamilies.size());
  result.m2l_gemm_max_group_size =
      std::max(result.m2l_gemm_max_group_size,
               static_cast<std::uint64_t>(group.source_node_ids.size()));
  return finite_matrix(rhs);
}

bool add_global_grouped_bundle_to_points_batched(
    const KifmmSurfaceRule &source_rule,
    const std::vector<KifmmHierarchyNode> &nodes,
    const std::vector<JcpBundle> &upward,
    const std::vector<M2lTargetSystems> &systems_by_node,
    const M2lGroupedPairSet &group, KifmmKernel kernel, double kappa,
    double target_check_scale, std::vector<JcpBundleRhs> &local_rhs,
    M2lMatrixCache &matrix_cache, KifmmHierarchyJcpResult &result) {
  if (group.records.empty()) {
    return true;
  }
  if (group.records.size() == 1) {
    const M2lPairRecord &record = group.records.front();
    const KifmmHierarchyNode &source_node =
        nodes[static_cast<std::size_t>(record.source_node_id)];
    const KifmmHierarchyNode &target_node =
        nodes[static_cast<std::size_t>(record.target_node_id)];
    const auto &points =
        kernel == KifmmKernel::Laplace
            ? systems_by_node[static_cast<std::size_t>(record.target_node_id)]
                  .laplace.check_points
            : systems_by_node[static_cast<std::size_t>(record.target_node_id)]
                  .yukawa.check_points;
    Eigen::MatrixXd &rhs =
        kernel == KifmmKernel::Laplace
            ? local_rhs[static_cast<std::size_t>(record.target_node_id)]
                  .laplace
            : local_rhs[static_cast<std::size_t>(record.target_node_id)]
                  .yukawa;
    return add_bundle_to_points_batched(
        source_rule, source_node,
        upward[static_cast<std::size_t>(record.source_node_id)], kernel, kappa,
        target_node, target_check_scale, points, rhs, matrix_cache);
  }
  const M2lPairRecord &first = group.records.front();
  const KifmmHierarchyNode &first_source =
      nodes[static_cast<std::size_t>(first.source_node_id)];
  const KifmmHierarchyNode &first_target =
      nodes[static_cast<std::size_t>(first.target_node_id)];
  const auto &first_points =
      kernel == KifmmKernel::Laplace
          ? systems_by_node[static_cast<std::size_t>(first.target_node_id)]
                .laplace.check_points
          : systems_by_node[static_cast<std::size_t>(first.target_node_id)]
                .yukawa.check_points;
  const KifmmBox source_box = box_from_node(first_source);
  const std::vector<KifmmSurfacePoint> equiv_points =
      instantiate_surface(source_rule.unit_equiv, source_box);
  const Eigen::Index point_count =
      static_cast<Eigen::Index>(first_points.size());
  const Eigen::Index equiv_count =
      static_cast<Eigen::Index>(equiv_points.size());
  const Eigen::Index family_count = static_cast<Eigen::Index>(kFamilies.size());
  const Eigen::Index group_size =
      static_cast<Eigen::Index>(group.records.size());
  if (point_count <= 0 || equiv_count <= 0 || family_count <= 0 ||
      group_size <= 0) {
    return false;
  }
  Eigen::MatrixXd charges(equiv_count, family_count * group_size);
  for (Eigen::Index group_index = 0; group_index < group_size; ++group_index) {
    const M2lPairRecord &record =
        group.records[static_cast<std::size_t>(group_index)];
    const JcpBundle &source_bundle =
        upward[static_cast<std::size_t>(record.source_node_id)];
    if (!bundle_passed(source_bundle)) {
      result.status = "m2l_missing_upward";
      return false;
    }
    const auto &fields = kernel == KifmmKernel::Laplace
                             ? source_bundle.laplace
                             : source_bundle.yukawa;
    for (Eigen::Index col = 0; col < family_count; ++col) {
      const KifmmEquivalentSolveResult &field =
          fields[static_cast<std::size_t>(col)];
      if (!field.finite ||
          field.charges.size() != static_cast<std::size_t>(equiv_count)) {
        return false;
      }
      const Eigen::Index out_col = group_index * family_count + col;
      for (Eigen::Index j = 0; j < equiv_count; ++j) {
        charges(j, out_col) = field.charges[static_cast<std::size_t>(j)];
      }
    }
  }
  Eigen::MatrixXd local_kernel_matrix;
  const Eigen::MatrixXd *kernel_matrix = get_or_build_m2l_kernel_matrix(
      source_rule, first_source, first_target, kernel, kappa,
      target_check_scale, first_points, equiv_points, matrix_cache,
      local_kernel_matrix);
  const auto gemm_start = Clock::now();
  const Eigen::MatrixXd product = (*kernel_matrix) * charges;
  matrix_cache.gemm_seconds += elapsed_seconds(gemm_start, Clock::now());
  for (Eigen::Index group_index = 0; group_index < group_size; ++group_index) {
    const M2lPairRecord &record =
        group.records[static_cast<std::size_t>(group_index)];
    Eigen::MatrixXd &rhs = kernel == KifmmKernel::Laplace
                               ? local_rhs[static_cast<std::size_t>(
                                     record.target_node_id)]
                                     .laplace
                               : local_rhs[static_cast<std::size_t>(
                                     record.target_node_id)]
                                     .yukawa;
    if (rhs.rows() != point_count || rhs.cols() != family_count) {
      return false;
    }
    for (Eigen::Index col = 0; col < family_count; ++col) {
      rhs.col(col) += product.col(group_index * family_count + col);
    }
  }
  ++result.m2l_gemm_group_count;
  result.m2l_gemm_grouped_pair_count +=
      static_cast<std::uint64_t>(group.records.size());
  result.m2l_gemm_grouped_rhs_columns +=
      static_cast<std::uint64_t>(group.records.size()) *
      static_cast<std::uint64_t>(kFamilies.size());
  result.m2l_gemm_max_group_size =
      std::max(result.m2l_gemm_max_group_size,
               static_cast<std::uint64_t>(group.records.size()));
  return true;
}

bool accumulate_grouped_m2l_target(
    const KifmmSurfaceRule &upward_rule,
    std::span<const KifmmPanelSource> sources, const KifmmHierarchyTree &tree,
    const std::vector<int> &source_node_ids,
    const std::vector<JcpBundle> &upward,
    const KifmmHierarchyNode &target_node, const M2lTargetSystems &systems,
    double kappa, bool include_dense_oracle, JcpBundleRhs &local_rhs,
    JcpBundleRhs &dense_rhs, M2lMatrixCache &matrix_cache,
    double target_check_scale, KifmmHierarchyJcpResult &result) {
  const std::size_t equiv_count = upward_rule.unit_equiv.size();
  const std::vector<M2lGroupedSourceSet> laplace_groups =
      build_m2l_source_groups(upward_rule, target_node,
                              systems.laplace.check_points, source_node_ids,
                              tree.nodes, KifmmKernel::Laplace, 0.0,
                              target_check_scale, equiv_count,
                              matrix_cache.policy);
  const std::vector<M2lGroupedSourceSet> yukawa_groups =
      build_m2l_source_groups(upward_rule, target_node,
                              systems.yukawa.check_points, source_node_ids,
                              tree.nodes, KifmmKernel::Yukawa, kappa,
                              target_check_scale, equiv_count,
                              matrix_cache.policy);
  for (const M2lGroupedSourceSet &group : laplace_groups) {
    if (!add_grouped_bundle_to_points_batched(
            upward_rule, tree.nodes, upward, target_node, group,
            KifmmKernel::Laplace, 0.0, target_check_scale,
            systems.laplace.check_points, local_rhs.laplace, matrix_cache,
            result)) {
      if (result.status.empty() || result.status == "not_started") {
        result.status = "m2l_grouped_accumulation_gap";
      }
      return false;
    }
  }
  for (const M2lGroupedSourceSet &group : yukawa_groups) {
    if (!add_grouped_bundle_to_points_batched(
            upward_rule, tree.nodes, upward, target_node, group,
            KifmmKernel::Yukawa, kappa, target_check_scale,
            systems.yukawa.check_points, local_rhs.yukawa, matrix_cache,
            result)) {
      if (result.status.empty() || result.status == "not_started") {
        result.status = "m2l_grouped_accumulation_gap";
      }
      return false;
    }
  }
  for (const int source_node_id : source_node_ids) {
    const KifmmHierarchyNode &source_node =
        tree.nodes[static_cast<std::size_t>(source_node_id)];
    if (!upward[static_cast<std::size_t>(source_node_id)].valid) {
      result.status = "m2l_missing_upward";
      return false;
    }
    if (include_dense_oracle) {
      add_m2l_dense_oracle_rhs(sources, source_node, kappa, systems,
                               dense_rhs);
    }
    account_m2l_rhs_add(result, upward_rule, systems);
    ++result.m2l_translations;
  }
  return true;
}

bool accumulate_global_grouped_m2l_pairs(
    const KifmmSurfaceRule &upward_rule,
    std::span<const KifmmPanelSource> sources, const KifmmHierarchyTree &tree,
    const std::vector<M2lPairRecord> &records,
    const std::vector<JcpBundle> &upward,
    const std::vector<M2lTargetSystems> &systems_by_node, double kappa,
    bool include_dense_oracle, std::vector<JcpBundleRhs> &local_rhs,
    std::vector<JcpBundleRhs> &dense_rhs, M2lMatrixCache &matrix_cache,
    double target_check_scale, KifmmHierarchyJcpResult &result) {
  const std::size_t equiv_count = upward_rule.unit_equiv.size();
  const std::vector<M2lGroupedPairSet> laplace_groups =
      build_m2l_record_groups(upward_rule, records, systems_by_node,
                              tree.nodes, KifmmKernel::Laplace, 0.0,
                              target_check_scale, equiv_count,
                              matrix_cache.policy);
  const std::vector<M2lGroupedPairSet> yukawa_groups =
      build_m2l_record_groups(upward_rule, records, systems_by_node,
                              tree.nodes, KifmmKernel::Yukawa, kappa,
                              target_check_scale, equiv_count,
                              matrix_cache.policy);
  for (const M2lGroupedPairSet &group : laplace_groups) {
    if (!add_global_grouped_bundle_to_points_batched(
            upward_rule, tree.nodes, upward, systems_by_node, group,
            KifmmKernel::Laplace, 0.0, target_check_scale, local_rhs,
            matrix_cache, result)) {
      if (result.status.empty() || result.status == "not_started") {
        result.status = "m2l_global_grouped_accumulation_gap";
      }
      return false;
    }
  }
  for (const M2lGroupedPairSet &group : yukawa_groups) {
    if (!add_global_grouped_bundle_to_points_batched(
            upward_rule, tree.nodes, upward, systems_by_node, group,
            KifmmKernel::Yukawa, kappa, target_check_scale, local_rhs,
            matrix_cache, result)) {
      if (result.status.empty() || result.status == "not_started") {
        result.status = "m2l_global_grouped_accumulation_gap";
      }
      return false;
    }
  }
  for (const M2lPairRecord &record : records) {
    const KifmmHierarchyNode &source_node =
        tree.nodes[static_cast<std::size_t>(record.source_node_id)];
    if (!upward[static_cast<std::size_t>(record.source_node_id)].valid) {
      result.status = "m2l_missing_upward";
      return false;
    }
    if (include_dense_oracle) {
      add_m2l_dense_oracle_rhs(
          sources, source_node, kappa,
          systems_by_node[static_cast<std::size_t>(record.target_node_id)],
          dense_rhs[static_cast<std::size_t>(record.target_node_id)]);
    }
    account_m2l_rhs_add(
        result, upward_rule,
        systems_by_node[static_cast<std::size_t>(record.target_node_id)]);
    ++result.m2l_translations;
  }
  return true;
}

struct M2lParallelTargetTask {
  int target_node_id{-1};
  std::vector<int> source_node_ids{};
};

struct M2lParallelWorkerStats {
  bool ok{true};
  std::string status{"ok"};
  std::uint64_t translations{0};
  std::uint64_t rhs_adds{0};
  std::uint64_t check_points_total{0};
  std::uint64_t equivalent_points{0};
  std::uint64_t kernel_evals_estimated{0};
  std::uint64_t group_count{0};
  std::uint64_t grouped_pair_count{0};
  std::uint64_t grouped_rhs_columns{0};
  std::uint64_t max_group_size{0};
  double gemm_seconds{0.0};
  std::uint64_t cache_hits{0};
  std::uint64_t cache_misses{0};
  std::uint64_t cache_peak_entries{0};
  std::uint64_t cache_peak_estimated_bytes{0};
  std::uint64_t cache_total_built_bytes{0};
  double cache_build_seconds{0.0};
};

std::vector<M2lParallelTargetTask> build_m2l_parallel_target_tasks(
    const std::vector<std::pair<int, int>> &m2l_pairs) {
  std::vector<std::pair<int, int>> sorted_pairs = m2l_pairs;
  std::sort(sorted_pairs.begin(), sorted_pairs.end());
  std::vector<M2lParallelTargetTask> tasks;
  for (const auto &[target_node_id, source_node_id] : sorted_pairs) {
    if (tasks.empty() || tasks.back().target_node_id != target_node_id) {
      tasks.push_back({target_node_id, {}});
    }
    tasks.back().source_node_ids.push_back(source_node_id);
  }
  return tasks;
}

void account_m2l_rhs_add(M2lParallelWorkerStats &stats,
                         const KifmmSurfaceRule &upward_rule,
                         const M2lTargetSystems &systems) {
  const std::uint64_t laplace_checks =
      static_cast<std::uint64_t>(systems.laplace.check_points.size());
  const std::uint64_t yukawa_checks =
      static_cast<std::uint64_t>(systems.yukawa.check_points.size());
  const std::uint64_t equiv_points =
      static_cast<std::uint64_t>(upward_rule.unit_equiv.size());
  stats.check_points_total += laplace_checks + yukawa_checks;
  stats.equivalent_points += 2 * equiv_points;
  stats.kernel_evals_estimated +=
      (laplace_checks + yukawa_checks) * equiv_points;
  stats.rhs_adds += laplace_checks + yukawa_checks;
}

bool prewarm_m2l_matrices_for_task(
    const KifmmSurfaceRule &upward_rule, const KifmmHierarchyTree &tree,
    const M2lParallelTargetTask &task, const M2lTargetSystems &systems,
    double kappa, double target_check_scale, M2lMatrixCache &matrix_cache) {
  if (!m2l_matrix_cache_enabled(matrix_cache.policy) ||
      m2l_matrix_cache_is_streaming(matrix_cache.policy)) {
    return true;
  }
  const std::size_t equiv_count = upward_rule.unit_equiv.size();
  const KifmmHierarchyNode &target_node =
      tree.nodes[static_cast<std::size_t>(task.target_node_id)];
  const std::vector<M2lGroupedSourceSet> laplace_groups =
      build_m2l_source_groups(upward_rule, target_node,
                              systems.laplace.check_points,
                              task.source_node_ids, tree.nodes,
                              KifmmKernel::Laplace, 0.0,
                              target_check_scale, equiv_count,
                              matrix_cache.policy);
  const std::vector<M2lGroupedSourceSet> yukawa_groups =
      build_m2l_source_groups(upward_rule, target_node,
                              systems.yukawa.check_points,
                              task.source_node_ids, tree.nodes,
                              KifmmKernel::Yukawa, kappa,
                              target_check_scale, equiv_count,
                              matrix_cache.policy);
  const auto prewarm_group = [&](const M2lGroupedSourceSet &group,
                                 KifmmKernel kernel, double kernel_kappa,
                                 std::span<const KifmmSurfacePoint> points) {
    if (group.source_node_ids.empty()) {
      return true;
    }
    const KifmmHierarchyNode &first_source =
        tree.nodes[static_cast<std::size_t>(group.source_node_ids.front())];
    const KifmmBox source_box = box_from_node(first_source);
    const std::vector<KifmmSurfacePoint> equiv_points =
        instantiate_surface(upward_rule.unit_equiv, source_box);
    Eigen::MatrixXd local_kernel_matrix;
    return get_or_build_m2l_kernel_matrix(
               upward_rule, first_source, target_node, kernel, kernel_kappa,
               target_check_scale, points, equiv_points, matrix_cache,
               local_kernel_matrix) != nullptr;
  };
  for (const M2lGroupedSourceSet &group : laplace_groups) {
    if (!prewarm_group(group, KifmmKernel::Laplace, 0.0,
                       systems.laplace.check_points)) {
      return false;
    }
  }
  for (const M2lGroupedSourceSet &group : yukawa_groups) {
    if (!prewarm_group(group, KifmmKernel::Yukawa, kappa,
                       systems.yukawa.check_points)) {
      return false;
    }
  }
  return true;
}

bool add_grouped_bundle_to_points_batched_readonly(
    const KifmmSurfaceRule &source_rule,
    const std::vector<KifmmHierarchyNode> &nodes,
    const std::vector<JcpBundle> &upward,
    const KifmmHierarchyNode &target_node, const M2lGroupedSourceSet &group,
    KifmmKernel kernel, double kappa, double target_check_scale,
    std::span<const KifmmSurfacePoint> points, Eigen::MatrixXd &rhs,
    M2lMatrixCache &matrix_cache, M2lParallelWorkerStats &stats) {
  if (group.source_node_ids.empty()) {
    return true;
  }
  const KifmmHierarchyNode &first_source =
      nodes[static_cast<std::size_t>(group.source_node_ids.front())];
  const KifmmBox source_box = box_from_node(first_source);
  const std::vector<KifmmSurfacePoint> equiv_points =
      instantiate_surface(source_rule.unit_equiv, source_box);
  const Eigen::Index point_count = static_cast<Eigen::Index>(points.size());
  const Eigen::Index equiv_count =
      static_cast<Eigen::Index>(equiv_points.size());
  const Eigen::Index family_count = static_cast<Eigen::Index>(kFamilies.size());
  const Eigen::Index group_size =
      static_cast<Eigen::Index>(group.source_node_ids.size());
  if (rhs.rows() != point_count || rhs.cols() != family_count ||
      equiv_count <= 0 || group_size <= 0) {
    stats.ok = false;
    stats.status = "m2l_parallel_shape_gap";
    return false;
  }
  Eigen::MatrixXd charges(equiv_count, family_count * group_size);
  for (Eigen::Index group_index = 0; group_index < group_size; ++group_index) {
    const int source_node_id =
        group.source_node_ids[static_cast<std::size_t>(group_index)];
    const JcpBundle &source_bundle =
        upward[static_cast<std::size_t>(source_node_id)];
    if (!bundle_passed(source_bundle)) {
      stats.ok = false;
      stats.status = "m2l_missing_upward";
      return false;
    }
    const auto &fields = kernel == KifmmKernel::Laplace
                             ? source_bundle.laplace
                             : source_bundle.yukawa;
    for (Eigen::Index col = 0; col < family_count; ++col) {
      const KifmmEquivalentSolveResult &field =
          fields[static_cast<std::size_t>(col)];
      if (!field.finite ||
          field.charges.size() != static_cast<std::size_t>(equiv_count)) {
        stats.ok = false;
        stats.status = "m2l_missing_upward";
        return false;
      }
      const Eigen::Index out_col = group_index * family_count + col;
      for (Eigen::Index j = 0; j < equiv_count; ++j) {
        charges(j, out_col) = field.charges[static_cast<std::size_t>(j)];
      }
    }
  }
  Eigen::MatrixXd local_kernel_matrix;
  const Eigen::MatrixXd *kernel_matrix = nullptr;
  if (m2l_matrix_cache_is_streaming(matrix_cache.policy)) {
    kernel_matrix = get_or_build_m2l_kernel_matrix(
        source_rule, first_source, target_node, kernel, kappa,
        target_check_scale, points, equiv_points, matrix_cache,
        local_kernel_matrix);
  } else if (matrix_cache.policy == "relative-geometry-cache" ||
             matrix_cache.policy == "static-translation-table-cache" ||
             matrix_cache.policy == "canonical-offset-cache" ||
             matrix_cache.policy == "bounded-lru-cache") {
    const M2lMatrixCacheKey key = make_m2l_matrix_cache_key(
        source_rule, first_source, target_node, kernel, kappa,
        target_check_scale, points, equiv_points.size(), matrix_cache.policy);
    kernel_matrix = find_m2l_matrix_cache_entry_readonly(matrix_cache, key);
    if (kernel_matrix == nullptr) {
      if (!m2l_matrix_cache_is_bounded_lru(matrix_cache.policy)) {
        stats.ok = false;
        stats.status = "m2l_parallel_cache_gap";
        return false;
      }
      const auto build_start = Clock::now();
      local_kernel_matrix =
          build_m2l_kernel_matrix(kernel, kappa, points, equiv_points);
      stats.cache_build_seconds += elapsed_seconds(build_start, Clock::now());
      stats.cache_total_built_bytes +=
          m2l_matrix_bytes(local_kernel_matrix);
      kernel_matrix = &local_kernel_matrix;
    }
  } else {
    local_kernel_matrix =
        build_m2l_kernel_matrix(kernel, kappa, points, equiv_points);
    kernel_matrix = &local_kernel_matrix;
  }
  const auto gemm_start = Clock::now();
  const Eigen::MatrixXd product = (*kernel_matrix) * charges;
  stats.gemm_seconds += elapsed_seconds(gemm_start, Clock::now());
  for (Eigen::Index group_index = 0; group_index < group_size; ++group_index) {
    for (Eigen::Index col = 0; col < family_count; ++col) {
      rhs.col(col) += product.col(group_index * family_count + col);
    }
  }
  ++stats.group_count;
  stats.grouped_pair_count +=
      static_cast<std::uint64_t>(group.source_node_ids.size());
  stats.grouped_rhs_columns +=
      static_cast<std::uint64_t>(group.source_node_ids.size()) *
      static_cast<std::uint64_t>(kFamilies.size());
  stats.max_group_size =
      std::max(stats.max_group_size,
               static_cast<std::uint64_t>(group.source_node_ids.size()));
  return finite_matrix(rhs);
}

M2lParallelWorkerStats accumulate_parallel_m2l_target_task(
    const KifmmSurfaceRule &upward_rule,
    std::span<const KifmmPanelSource> sources, const KifmmHierarchyTree &tree,
    const M2lParallelTargetTask &task, const std::vector<JcpBundle> &upward,
    const M2lTargetSystems &systems, double kappa, bool include_dense_oracle,
    JcpBundleRhs &local_rhs, JcpBundleRhs &dense_rhs,
    M2lMatrixCache &matrix_cache, double target_check_scale) {
  M2lParallelWorkerStats stats;
  M2lMatrixCacheScopeGuard streaming_scope(matrix_cache);
  const std::uint64_t start_hits = matrix_cache.hits;
  const std::uint64_t start_misses = matrix_cache.misses;
  const std::uint64_t start_total_built_bytes =
      matrix_cache.total_built_bytes;
  const double start_build_seconds = matrix_cache.build_seconds;
  const std::size_t equiv_count = upward_rule.unit_equiv.size();
  const KifmmHierarchyNode &target_node =
      tree.nodes[static_cast<std::size_t>(task.target_node_id)];
  const std::vector<M2lGroupedSourceSet> laplace_groups =
      build_m2l_source_groups(upward_rule, target_node,
                              systems.laplace.check_points,
                              task.source_node_ids, tree.nodes,
                              KifmmKernel::Laplace, 0.0,
                              target_check_scale, equiv_count,
                              matrix_cache.policy);
  const std::vector<M2lGroupedSourceSet> yukawa_groups =
      build_m2l_source_groups(upward_rule, target_node,
                              systems.yukawa.check_points,
                              task.source_node_ids, tree.nodes,
                              KifmmKernel::Yukawa, kappa,
                              target_check_scale, equiv_count,
                              matrix_cache.policy);
  for (const M2lGroupedSourceSet &group : laplace_groups) {
    if (!add_grouped_bundle_to_points_batched_readonly(
            upward_rule, tree.nodes, upward, target_node, group,
            KifmmKernel::Laplace, 0.0, target_check_scale,
            systems.laplace.check_points, local_rhs.laplace, matrix_cache,
            stats)) {
      if (stats.status == "ok") {
        stats.status = "m2l_parallel_accumulation_gap";
      }
      stats.ok = false;
      return stats;
    }
  }
  for (const M2lGroupedSourceSet &group : yukawa_groups) {
    if (!add_grouped_bundle_to_points_batched_readonly(
            upward_rule, tree.nodes, upward, target_node, group,
            KifmmKernel::Yukawa, kappa, target_check_scale,
            systems.yukawa.check_points, local_rhs.yukawa, matrix_cache,
            stats)) {
      if (stats.status == "ok") {
        stats.status = "m2l_parallel_accumulation_gap";
      }
      stats.ok = false;
      return stats;
    }
  }
  for (const int source_node_id : task.source_node_ids) {
    const KifmmHierarchyNode &source_node =
        tree.nodes[static_cast<std::size_t>(source_node_id)];
    if (!upward[static_cast<std::size_t>(source_node_id)].valid) {
      stats.ok = false;
      stats.status = "m2l_missing_upward";
      return stats;
    }
    if (include_dense_oracle) {
      add_m2l_dense_oracle_rhs(sources, source_node, kappa, systems,
                               dense_rhs);
    }
    account_m2l_rhs_add(stats, upward_rule, systems);
    ++stats.translations;
  }
  stats.cache_hits += matrix_cache.hits - start_hits;
  stats.cache_misses += matrix_cache.misses - start_misses;
  stats.cache_peak_entries =
      std::max(stats.cache_peak_entries, matrix_cache.peak_entries);
  stats.cache_peak_estimated_bytes =
      std::max(stats.cache_peak_estimated_bytes,
               matrix_cache.peak_estimated_bytes);
  stats.cache_total_built_bytes +=
      matrix_cache.total_built_bytes - start_total_built_bytes;
  stats.cache_build_seconds += matrix_cache.build_seconds - start_build_seconds;
  return stats;
}

bool accumulate_parallel_m2l_by_target_node(
    const KifmmSurfaceRule &upward_rule,
    std::span<const KifmmPanelSource> sources, const KifmmHierarchyTree &tree,
    const std::vector<std::pair<int, int>> &m2l_pairs,
    const std::vector<JcpBundle> &upward,
    KifmmEquivalentSystemCache &local_cache, const KifmmSurfaceRule &local_rule,
    const KifmmHierarchyJcpConfig &config, const std::string &solver,
    bool include_dense_oracle, std::vector<JcpBundleRhs> &local_rhs,
    std::vector<JcpBundleRhs> &dense_local_rhs,
    std::vector<M2lTargetSystems> &systems_by_node,
    M2lMatrixCache &matrix_cache, KifmmHierarchyJcpResult &result) {
  const auto prepare_start = Clock::now();
  const std::vector<M2lParallelTargetTask> tasks =
      build_m2l_parallel_target_tasks(m2l_pairs);
  result.m2l_parallel_task_count =
      static_cast<std::uint64_t>(tasks.size());
  for (const M2lParallelTargetTask &task : tasks) {
    const KifmmHierarchyNode &target_node =
        tree.nodes[static_cast<std::size_t>(task.target_node_id)];
    if (!prepare_m2l_target_systems(
            local_cache, local_rule, target_node, config, solver, local_rhs,
            dense_local_rhs, include_dense_oracle,
            systems_by_node[static_cast<std::size_t>(task.target_node_id)],
            result)) {
      return false;
    }
    for (const int source_node_id : task.source_node_ids) {
      if (!upward[static_cast<std::size_t>(source_node_id)].valid) {
        result.status = "m2l_missing_upward";
        return false;
      }
    }
    result.m2l_parallel_max_task_pairs =
        std::max(result.m2l_parallel_max_task_pairs,
                 static_cast<std::uint64_t>(task.source_node_ids.size()));
    if (!prewarm_m2l_matrices_for_task(
            upward_rule, tree, task,
            systems_by_node[static_cast<std::size_t>(task.target_node_id)],
            config.kappa, config.hierarchy.local_check_scale, matrix_cache)) {
      result.status = "m2l_parallel_cache_prepare_gap";
      return false;
    }
  }
  result.m2l_parallel_prepare_seconds =
      elapsed_seconds(prepare_start, Clock::now());

  const int requested_workers =
      std::max(1, config.hierarchy.m2l_parallel_workers);
  const std::size_t worker_count =
      std::min<std::size_t>(tasks.size(), static_cast<std::size_t>(
                                             requested_workers));
  if (worker_count == 0) {
    return true;
  }
  const auto eval_start = Clock::now();
  std::vector<std::future<M2lParallelWorkerStats>> futures;
  futures.reserve(worker_count);
  for (std::size_t worker = 0; worker < worker_count; ++worker) {
    futures.push_back(std::async(std::launch::async, [&, worker]() {
      M2lParallelWorkerStats combined;
      M2lMatrixCache worker_streaming_cache;
      worker_streaming_cache.policy = matrix_cache.policy;
      for (std::size_t task_index = worker; task_index < tasks.size();
           task_index += worker_count) {
        const M2lParallelTargetTask &task = tasks[task_index];
        M2lMatrixCache &task_matrix_cache =
            m2l_matrix_cache_is_streaming(matrix_cache.policy)
                ? worker_streaming_cache
                : matrix_cache;
        M2lParallelWorkerStats stats = accumulate_parallel_m2l_target_task(
            upward_rule, sources, tree, task, upward,
            systems_by_node[static_cast<std::size_t>(task.target_node_id)],
            config.kappa, include_dense_oracle,
            local_rhs[static_cast<std::size_t>(task.target_node_id)],
            dense_local_rhs[static_cast<std::size_t>(task.target_node_id)],
            task_matrix_cache, config.hierarchy.local_check_scale);
        if (!stats.ok) {
          return stats;
        }
        combined.translations += stats.translations;
        combined.rhs_adds += stats.rhs_adds;
        combined.check_points_total += stats.check_points_total;
        combined.equivalent_points += stats.equivalent_points;
        combined.kernel_evals_estimated += stats.kernel_evals_estimated;
        combined.group_count += stats.group_count;
        combined.grouped_pair_count += stats.grouped_pair_count;
        combined.grouped_rhs_columns += stats.grouped_rhs_columns;
        combined.max_group_size =
            std::max(combined.max_group_size, stats.max_group_size);
        combined.gemm_seconds += stats.gemm_seconds;
        combined.cache_hits += stats.cache_hits;
        combined.cache_misses += stats.cache_misses;
        combined.cache_peak_entries =
            std::max(combined.cache_peak_entries, stats.cache_peak_entries);
        combined.cache_peak_estimated_bytes =
            std::max(combined.cache_peak_estimated_bytes,
                     stats.cache_peak_estimated_bytes);
        combined.cache_total_built_bytes += stats.cache_total_built_bytes;
        combined.cache_build_seconds += stats.cache_build_seconds;
      }
      return combined;
    }));
  }
  std::vector<M2lParallelWorkerStats> worker_stats;
  worker_stats.reserve(futures.size());
  for (std::future<M2lParallelWorkerStats> &future : futures) {
    M2lParallelWorkerStats stats = future.get();
    if (!stats.ok) {
      result.status = stats.status;
      return false;
    }
    worker_stats.push_back(std::move(stats));
  }
  result.m2l_parallel_eval_seconds =
      elapsed_seconds(eval_start, Clock::now());
  const auto merge_start = Clock::now();
  for (const M2lParallelWorkerStats &stats : worker_stats) {
    result.m2l_translations += stats.translations;
    result.local_rhs_adds += stats.rhs_adds;
    result.m2l_check_points_total += stats.check_points_total;
    result.m2l_equivalent_points += stats.equivalent_points;
    result.m2l_kernel_evals_estimated += stats.kernel_evals_estimated;
    result.m2l_gemm_group_count += stats.group_count;
    result.m2l_gemm_grouped_pair_count += stats.grouped_pair_count;
    result.m2l_gemm_grouped_rhs_columns += stats.grouped_rhs_columns;
    result.m2l_gemm_max_group_size =
        std::max(result.m2l_gemm_max_group_size, stats.max_group_size);
    matrix_cache.gemm_seconds += stats.gemm_seconds;
    if (m2l_matrix_cache_is_streaming(matrix_cache.policy)) {
      matrix_cache.hits += stats.cache_hits;
      matrix_cache.misses += stats.cache_misses;
      matrix_cache.peak_entries =
          std::max(matrix_cache.peak_entries, stats.cache_peak_entries);
      matrix_cache.peak_estimated_bytes =
          std::max(matrix_cache.peak_estimated_bytes,
                   stats.cache_peak_estimated_bytes);
      matrix_cache.total_built_bytes += stats.cache_total_built_bytes;
      matrix_cache.build_seconds += stats.cache_build_seconds;
      matrix_cache.key_status = "target_node_streaming_pass";
    } else if (m2l_matrix_cache_is_bounded_lru(matrix_cache.policy)) {
      matrix_cache.total_built_bytes += stats.cache_total_built_bytes;
      matrix_cache.build_seconds += stats.cache_build_seconds;
    }
  }
  result.m2l_parallel_group_count = result.m2l_gemm_group_count;
  result.m2l_parallel_merge_seconds =
      elapsed_seconds(merge_start, Clock::now());
  return true;
}

JcpBundle solve_rhs_bundle(const KifmmSurfaceRule &rule,
                           KifmmEquivalentSystemCache &cache,
                           const KifmmHierarchyNode &node, double kappa,
                           const std::string &solver,
                           const std::string &condition_policy,
                           const JcpBundleRhs &rhs, double &solve_seconds) {
  const KifmmBox box = box_from_node(node);
  const KifmmEquivalentSystem laplace_system = build_equivalent_system_cached(
      cache, rule, box, KifmmKernel::Laplace, 0.0, solver, condition_policy);
  const KifmmEquivalentSystem yukawa_system = build_equivalent_system_cached(
      cache, rule, box, KifmmKernel::Yukawa, kappa, solver, condition_policy);
  if (!laplace_system.finite || !yukawa_system.finite ||
      !finite_matrix(rhs.laplace) || !finite_matrix(rhs.yukawa)) {
    return {};
  }
  const auto solve_start = Clock::now();
  const std::vector<KifmmEquivalentSolveResult> laplace =
      solve_equivalent_rhs_batch(laplace_system, rhs.laplace, solver);
  const std::vector<KifmmEquivalentSolveResult> yukawa =
      solve_equivalent_rhs_batch(yukawa_system, rhs.yukawa, solver);
  solve_seconds += elapsed_seconds(solve_start, Clock::now());
  return bundle_from_results(laplace, yukawa);
}

KifmmJcpChannelRow channel_row_from_bundle(const KifmmSurfaceRule &rule,
                                           const KifmmHierarchyNode &node,
                                           const JcpBundle &bundle,
                                           double kappa, double eps_ratio,
                                           double inv_eps_ratio, Vec3 target,
                                           Vec3 target_normal) {
  KifmmJcpChannels channels;
  channels.phin_laplace = bundle.laplace[0];
  channels.phin_yukawa = bundle.yukawa[0];
  for (std::size_t i = 0; i < 3; ++i) {
    channels.phi_normal_laplace[i] = bundle.laplace[i + 1];
    channels.phi_normal_yukawa[i] = bundle.yukawa[i + 1];
  }
  return assemble_kifmm_jcp_channel_row(rule, box_from_node(node), channels,
                                        kappa, eps_ratio, inv_eps_ratio,
                                        target, target_normal);
}

KifmmJcpChannelRow channel_row_from_bundle_k123_only(
    const KifmmSurfaceRule &rule, const KifmmHierarchyNode &node,
    const JcpBundle &bundle, double kappa, double eps_ratio,
    double inv_eps_ratio, Vec3 target, Vec3 target_normal) {
  const KifmmBox box = box_from_node(node);
  const double phin_lap = equivalent_field_potential(
      rule, box, bundle.laplace[0], KifmmFieldFamily::PhinCharge,
      KifmmKernel::Laplace, 0.0, target);
  const double phin_yuk = equivalent_field_potential(
      rule, box, bundle.yukawa[0], KifmmFieldFamily::PhinCharge,
      KifmmKernel::Yukawa, kappa, target);
  const Vec3 phin_lap_grad = equivalent_field_grad_target(
      rule, box, bundle.laplace[0], KifmmFieldFamily::PhinCharge,
      KifmmKernel::Laplace, 0.0, target);
  const Vec3 phin_yuk_grad = equivalent_field_grad_target(
      rule, box, bundle.yukawa[0], KifmmFieldFamily::PhinCharge,
      KifmmKernel::Yukawa, kappa, target);

  std::array<double, 3> normal_lap{};
  std::array<double, 3> normal_yuk{};
  for (std::size_t i = 0; i < 3; ++i) {
    normal_lap[i] = equivalent_field_potential(
        rule, box, bundle.laplace[i + 1], kFamilies[i + 1],
        KifmmKernel::Laplace, 0.0, target);
    normal_yuk[i] = equivalent_field_potential(
        rule, box, bundle.yukawa[i + 1], kFamilies[i + 1],
        KifmmKernel::Yukawa, kappa, target);
  }

  const double k1 = phin_lap - phin_yuk;
  const double k2 = eps_ratio * (normal_yuk[0] + normal_yuk[1] +
                                 normal_yuk[2]) -
                    (normal_lap[0] + normal_lap[1] + normal_lap[2]);
  const double k3 =
      dot(phin_lap_grad, target_normal) -
      inv_eps_ratio * dot(phin_yuk_grad, target_normal);
  return {k1, k2, k3, 0.0};
}

K4Parts k4_parts_from_bundle(const KifmmSurfaceRule &rule,
                             const KifmmHierarchyNode &node,
                             const JcpBundle &bundle, double kappa,
                             Vec3 target, Vec3 target_normal) {
  const KifmmBox box = box_from_node(node);
  std::array<Vec3, 3> normal_lap_grad{};
  std::array<Vec3, 3> normal_yuk_grad{};
  for (std::size_t i = 0; i < 3; ++i) {
    normal_lap_grad[i] = equivalent_field_grad_target(
        rule, box, bundle.laplace[i + 1], kFamilies[i + 1],
        KifmmKernel::Laplace, 0.0, target);
    normal_yuk_grad[i] = equivalent_field_grad_target(
        rule, box, bundle.yukawa[i + 1], kFamilies[i + 1],
        KifmmKernel::Yukawa, kappa, target);
  }
  K4Parts parts;
  for (std::size_t i = 0; i < 3; ++i) {
    parts.laplace += normal_lap_grad[i].x * target_normal.x +
                     normal_lap_grad[i].y * target_normal.y +
                     normal_lap_grad[i].z * target_normal.z;
    parts.yukawa += normal_yuk_grad[i].x * target_normal.x +
                    normal_yuk_grad[i].y * target_normal.y +
                    normal_yuk_grad[i].z * target_normal.z;
  }
  parts.combined = parts.laplace - parts.yukawa;
  return parts;
}

K4Parts direct_k4_parts(std::span<const KifmmPanelSource> sources,
                        double kappa, Vec3 target, Vec3 target_normal) {
  K4Parts parts;
  for (std::size_t i = 1; i < kFamilies.size(); ++i) {
    const Vec3 lap_grad = direct_field_grad_target(
        sources, kFamilies[i], KifmmKernel::Laplace, 0.0, target);
    const Vec3 yuk_grad = direct_field_grad_target(
        sources, kFamilies[i], KifmmKernel::Yukawa, kappa, target);
    parts.laplace += lap_grad.x * target_normal.x +
                     lap_grad.y * target_normal.y +
                     lap_grad.z * target_normal.z;
    parts.yukawa += yuk_grad.x * target_normal.x +
                    yuk_grad.y * target_normal.y +
                    yuk_grad.z * target_normal.z;
  }
  parts.combined = parts.laplace - parts.yukawa;
  return parts;
}

void add_direct_pair_contribution(std::span<const KifmmPanelSource> sources,
                                  std::span<const Vec3> targets,
                                  std::span<const Vec3> target_normals,
                                  const KifmmHierarchyNode &source_node,
                                  const KifmmHierarchyNode &target_node,
                                  double kappa, double eps_ratio,
                                  double inv_eps_ratio,
                                  std::vector<KifmmJcpChannelRow> &values) {
  const std::vector<KifmmPanelSource> node_sources =
      gather_sources(sources, source_node.source_indices);
  for (const int target_index : target_node.target_indices) {
    const std::size_t ti = static_cast<std::size_t>(target_index);
    const KifmmJcpChannelRow row = assemble_direct_jcp_channel_row(
        node_sources, kappa, eps_ratio, inv_eps_ratio, targets[ti],
        target_normals[ti]);
    values[ti].k1 += row.k1;
    values[ti].k2 += row.k2;
    values[ti].k3 += row.k3;
    values[ti].k4 += row.k4;
  }
}

void add_direct_pair_k4_parts(std::span<const KifmmPanelSource> sources,
                              std::span<const Vec3> targets,
                              std::span<const Vec3> target_normals,
                              const KifmmHierarchyNode &source_node,
                              const KifmmHierarchyNode &target_node,
                              double kappa,
                              std::vector<K4Parts> &values) {
  const std::vector<KifmmPanelSource> node_sources =
      gather_sources(sources, source_node.source_indices);
  for (const int target_index : target_node.target_indices) {
    const std::size_t ti = static_cast<std::size_t>(target_index);
    const K4Parts parts =
        direct_k4_parts(node_sources, kappa, targets[ti], target_normals[ti]);
    values[ti].laplace += parts.laplace;
    values[ti].yukawa += parts.yukawa;
    values[ti].combined += parts.combined;
  }
}

void add_direct_sources_to_points(std::span<const KifmmPanelSource> sources,
                                  const KifmmHierarchyNode &source_node,
                                  KifmmKernel kernel, double kappa,
                                  std::span<const KifmmSurfacePoint> points,
                                  Eigen::MatrixXd &rhs) {
  const std::vector<KifmmPanelSource> node_sources =
      gather_sources(sources, source_node.source_indices);
  for (std::size_t i = 0; i < points.size(); ++i) {
    for (std::size_t col = 0; col < kFamilies.size(); ++col) {
      rhs(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(col)) +=
          direct_field_potential(node_sources, kFamilies[col], kernel, kappa,
                                 points[i].x);
    }
  }
}

void add_matrix_attribution(
    const std::string &scope, KifmmKernel kernel, const Eigen::MatrixXd &approx,
    const Eigen::MatrixXd &direct,
    std::vector<KifmmHierarchyFieldAttribution> &rows,
    RmsAccumulator &overall, int node_id, int node_level) {
  if (approx.rows() != direct.rows() || approx.cols() != direct.cols() ||
      approx.cols() != static_cast<Eigen::Index>(kFamilies.size())) {
    return;
  }
  for (Eigen::Index col = 0; col < approx.cols(); ++col) {
    RmsAccumulator acc;
    for (Eigen::Index row = 0; row < approx.rows(); ++row) {
      acc.add(approx(row, col), direct(row, col));
      overall.add(approx(row, col), direct(row, col));
    }
    KifmmHierarchyFieldAttribution item;
    item.scope = scope;
    item.field_name = family_name(kFamilies[static_cast<std::size_t>(col)]);
    item.kernel = kernel_name(kernel);
    item.node_id = node_id;
    item.node_level = node_level;
    item.relative_rms = acc.relative_rms();
    item.abs_rms = acc.abs_rms();
    item.reference_rms = acc.reference_rms();
    item.max_abs_error = acc.max_abs_error;
    rows.push_back(std::move(item));
  }
}

void collect_source_leaf_descendants(const KifmmHierarchyTree &tree, int node_id,
                                     std::vector<int> &leaf_ids) {
  const KifmmHierarchyNode &node = tree.nodes[static_cast<std::size_t>(node_id)];
  if (node.is_leaf) {
    if (!node.source_indices.empty()) {
      leaf_ids.push_back(node_id);
    }
    return;
  }
  bool used_child = false;
  for (const int child_id : node.child) {
    if (child_id >= 0 &&
        !tree.nodes[static_cast<std::size_t>(child_id)].source_indices.empty()) {
      collect_source_leaf_descendants(tree, child_id, leaf_ids);
      used_child = true;
    }
  }
  if (!used_child && !node.source_indices.empty()) {
    leaf_ids.push_back(node_id);
  }
}

void collect_target_leaf_descendants(const KifmmHierarchyTree &tree, int node_id,
                                     std::vector<int> &leaf_ids) {
  const KifmmHierarchyNode &node = tree.nodes[static_cast<std::size_t>(node_id)];
  if (node.is_leaf) {
    if (!node.target_indices.empty()) {
      leaf_ids.push_back(node_id);
    }
    return;
  }
  bool used_child = false;
  for (const int child_id : node.child) {
    if (child_id >= 0 &&
        !tree.nodes[static_cast<std::size_t>(child_id)].target_indices.empty()) {
      collect_target_leaf_descendants(tree, child_id, leaf_ids);
      used_child = true;
    }
  }
  if (!used_child && !node.target_indices.empty()) {
    leaf_ids.push_back(node_id);
  }
}

std::vector<std::pair<int, int>> effective_m2l_pairs(
    const KifmmHierarchyTree &tree, const KifmmHierarchyWorklist &worklist,
    const std::string &mode) {
  if (mode != "no-m2m" && mode != "no-l2l") {
    return worklist.m2l_pairs;
  }
  std::vector<std::pair<int, int>> pairs;
  for (const auto &[target_node_id, source_node_id] : worklist.m2l_pairs) {
    std::vector<int> source_nodes{source_node_id};
    std::vector<int> target_nodes{target_node_id};
    if (mode == "no-m2m") {
      source_nodes.clear();
      collect_source_leaf_descendants(tree, source_node_id, source_nodes);
    }
    if (mode == "no-l2l") {
      target_nodes.clear();
      collect_target_leaf_descendants(tree, target_node_id, target_nodes);
    }
    for (const int target : target_nodes) {
      for (const int source : source_nodes) {
        pairs.emplace_back(target, source);
      }
    }
  }
  return pairs;
}

std::vector<std::vector<int>>
build_nodes_by_level(const KifmmHierarchyTree &tree) {
  int max_level = 0;
  for (const KifmmHierarchyNode &node : tree.nodes) {
    max_level = std::max(max_level, node.level);
  }
  std::vector<std::vector<int>> levels(static_cast<std::size_t>(max_level + 1));
  for (const KifmmHierarchyNode &node : tree.nodes) {
    if (node.level >= 0 && node.level <= max_level) {
      levels[static_cast<std::size_t>(node.level)].push_back(node.id);
    }
  }
  return levels;
}

bool hierarchy_links_are_valid(const KifmmHierarchyTree &tree) noexcept {
  for (const KifmmHierarchyNode &node : tree.nodes) {
    for (const int child_id : node.child) {
      if (child_id < 0) {
        continue;
      }
      if (static_cast<std::size_t>(child_id) >= tree.nodes.size()) {
        return false;
      }
      const KifmmHierarchyNode &child =
          tree.nodes[static_cast<std::size_t>(child_id)];
      if (child.parent != node.id || child.level != node.level + 1) {
        return false;
      }
    }
  }
  return true;
}

double aggregate_relative_rms(std::span<const KifmmJcpChannelRow> approx,
                              std::span<const KifmmJcpChannelRow> direct) {
  long double err_sq = 0.0L;
  long double ref_sq = 0.0L;
  for (std::size_t i = 0; i < approx.size(); ++i) {
    const long double approx_phi = approx[i].k1 + approx[i].k2;
    const long double approx_phin = approx[i].k3 + approx[i].k4;
    const long double direct_phi = direct[i].k1 + direct[i].k2;
    const long double direct_phin = direct[i].k3 + direct[i].k4;
    const long double d_phi = approx_phi - direct_phi;
    const long double d_phin = approx_phin - direct_phin;
    err_sq += d_phi * d_phi + d_phin * d_phin;
    ref_sq += direct_phi * direct_phi + direct_phin * direct_phin;
  }
  return std::sqrt(static_cast<double>(err_sq / std::max(ref_sq, 1.0e-300L)));
}

double channel_relative_rms(std::span<const KifmmJcpChannelRow> approx,
                            std::span<const KifmmJcpChannelRow> direct,
                            int channel) {
  long double err_sq = 0.0L;
  long double ref_sq = 0.0L;
  for (std::size_t i = 0; i < approx.size(); ++i) {
    const double *a = &approx[i].k1;
    const double *d = &direct[i].k1;
    const long double diff = a[channel] - d[channel];
    err_sq += diff * diff;
    ref_sq += static_cast<long double>(d[channel]) * d[channel];
  }
  return std::sqrt(static_cast<double>(err_sq / std::max(ref_sq, 1.0e-300L)));
}

double max_abs_error(std::span<const KifmmJcpChannelRow> approx,
                     std::span<const KifmmJcpChannelRow> direct) {
  double value = 0.0;
  for (std::size_t i = 0; i < approx.size(); ++i) {
    value = std::max(value, std::abs(approx[i].k1 - direct[i].k1));
    value = std::max(value, std::abs(approx[i].k2 - direct[i].k2));
    value = std::max(value, std::abs(approx[i].k3 - direct[i].k3));
    value = std::max(value, std::abs(approx[i].k4 - direct[i].k4));
  }
  return value;
}

bool finite_channels(std::span<const KifmmJcpChannelRow> rows) noexcept {
  for (const KifmmJcpChannelRow &row : rows) {
    if (!std::isfinite(row.k1) || !std::isfinite(row.k2) ||
        !std::isfinite(row.k3) || !std::isfinite(row.k4)) {
      return false;
    }
  }
  return true;
}

} // namespace

KifmmHierarchyJcpResult run_full_hierarchical_jcp_kifmm_impl(
    std::span<const KifmmPanelSource> sources, std::span<const Vec3> targets,
    std::span<const Vec3> target_normals,
    const KifmmHierarchyJcpConfig &config, const KifmmHierarchyTree &tree,
    const KifmmHierarchyWorklist &worklist, Clock::time_point total_start) {
  KifmmHierarchyJcpResult result;
  result.execution_status = "fail";
  record_hierarchy_memory_start(result);
  result.aggregate_relative_rms_gate = config.aggregate_relative_rms_gate;
  result.max_channel_relative_rms_gate =
      config.max_channel_relative_rms_gate;
  if (sources.empty() || targets.empty() || targets.size() != target_normals.size()) {
    result.status = "invalid_input_size";
    return result;
  }
  if (!std::isfinite(config.kappa) || config.kappa < 0.0 ||
      !std::isfinite(config.eps_ratio) || config.eps_ratio == 0.0) {
    result.status = "invalid_physics";
    return result;
  }
  if (config.hierarchy.order > 10) {
    result.status = "unsupported_order_for_hierarchy_jcp";
    result.accuracy_failure = "unsupported_order";
    return result;
  }
  if (config.hierarchy.upward_check_scale > 4.0 ||
      config.hierarchy.local_equiv_scale > 3.0) {
    result.status = "unsupported_surface_scale_for_hierarchy_jcp";
    result.accuracy_failure = "unsupported_surface_scale";
    return result;
  }
  result.m2l_eval_policy = config.hierarchy.m2l_eval_policy.empty()
                               ? std::string("scalar-existing")
                               : config.hierarchy.m2l_eval_policy;
  if (!valid_m2l_eval_policy(result.m2l_eval_policy)) {
    result.status = "invalid_m2l_eval_policy";
    result.accuracy_failure = "invalid_m2l_eval_policy";
    return result;
  }
  result.m2l_matrix_cache_policy =
      config.hierarchy.m2l_matrix_cache_policy.empty()
          ? std::string("off")
          : config.hierarchy.m2l_matrix_cache_policy;
  if (!valid_m2l_matrix_cache_policy(result.m2l_matrix_cache_policy)) {
    result.status = "invalid_m2l_matrix_cache_policy";
    result.accuracy_failure = "invalid_m2l_matrix_cache_policy";
    return result;
  }
  if (m2l_matrix_cache_is_bounded_lru(result.m2l_matrix_cache_policy) &&
      (!std::isfinite(config.hierarchy.m2l_matrix_cache_limit_mb) ||
       config.hierarchy.m2l_matrix_cache_limit_mb <= 0.0)) {
    result.status = "invalid_m2l_matrix_cache_limit";
    result.accuracy_failure = "invalid_m2l_matrix_cache_limit";
    return result;
  }
  result.m2l_gemm_batch_policy =
      config.hierarchy.m2l_gemm_batch_policy.empty()
          ? std::string("per-pair")
          : config.hierarchy.m2l_gemm_batch_policy;
  if (!valid_m2l_gemm_batch_policy(result.m2l_gemm_batch_policy)) {
    result.status = "invalid_m2l_gemm_batch_policy";
    result.accuracy_failure = "invalid_m2l_gemm_batch_policy";
    return result;
  }
  result.m2l_parallel_policy =
      config.hierarchy.m2l_parallel_policy.empty()
          ? std::string("serial")
          : config.hierarchy.m2l_parallel_policy;
  result.m2l_parallel_workers =
      std::max(1, config.hierarchy.m2l_parallel_workers);
  if (!valid_m2l_parallel_policy(result.m2l_parallel_policy)) {
    result.status = "invalid_m2l_parallel_policy";
    result.accuracy_failure = "invalid_m2l_parallel_policy";
    return result;
  }
  if (result.m2l_parallel_policy == "target-node-key-group" &&
      result.m2l_eval_policy != "target-check-batched") {
    result.status = "invalid_m2l_parallel_policy";
    result.accuracy_failure = "parallel_requires_target_check_batched";
    return result;
  }
  result.jcp_channel_policy =
      config.hierarchy.jcp_channel_policy.empty()
          ? std::string("all")
          : config.hierarchy.jcp_channel_policy;
  if (!valid_jcp_channel_policy(result.jcp_channel_policy)) {
    result.status = "invalid_jcp_channel_policy";
    result.accuracy_failure = "invalid_jcp_channel_policy";
    return result;
  }
  const bool k123_only = result.jcp_channel_policy == "k123-only";
  result.hierarchy_k4_policy = k123_only ? "pruned" : "computed";
  const std::string mode = config.hier_debug_mode.empty()
                               ? std::string("full")
                               : config.hier_debug_mode;
  result.hier_debug_mode = mode;
  if (mode != "full" && mode != "m2l-direct-to-targets" &&
      mode != "no-m2m" && mode != "no-l2l" &&
      mode != "local-check-oracle" && mode != "l2p-oracle" &&
      mode != "hybrid-local-bridge" &&
      mode != "hybrid-local-split-bridge" &&
      mode != "hybrid-source-local-bridge") {
    result.status = "invalid_hier_debug_mode";
    result.accuracy_failure = "invalid_hier_debug_mode";
    return result;
  }
  const bool hybrid_local_bridge = mode == "hybrid-local-bridge";
  const bool hybrid_local_split_bridge =
      mode == "hybrid-local-split-bridge";
  const bool hybrid_source_local_bridge =
      mode == "hybrid-source-local-bridge";
  const bool hybrid_bridge_mode =
      hybrid_local_bridge || hybrid_local_split_bridge ||
      hybrid_source_local_bridge;
  const KifmmCartesianBridgeConfig bridge_config = bridge_config_from(config);
  if (hybrid_bridge_mode) {
    result.hybrid_mode = mode;
    result.m2e_max_relative_residual = 0.0;
    result.local_equiv_max_relative_residual = 0.0;
    result.e2l_coefficient_status = "not_computed";
    result.k4_decomposition_status =
        hybrid_local_split_bridge ? "laplace_yukawa_split" : "not_computed";
    result.l2p_gradient_method = "cartesian_l2p_order";
  } else {
    result.e2l_coefficient_status = "not_applicable";
    result.k4_decomposition_status = "laplace_yukawa_split";
  }
  for (const KifmmPanelSource &source : sources) {
    if (!finite_source(source)) {
      result.status = "invalid_source";
      return result;
    }
  }
  for (const Vec3 normal : target_normals) {
    if (!finite_vec(normal)) {
      result.status = "invalid_target_normal";
      return result;
    }
  }

  result.stats = tree.stats;
  result.tree_hash = tree.tree_hash;
  if (tree.status != "ok") {
    result.status = tree.status;
    return result;
  }
  if (!hierarchy_links_are_valid(tree)) {
    result.status = "invalid_tree_links";
    return result;
  }
  result.worklist_hash = worklist.worklist_hash;
  result.policy_hash = worklist.policy_hash;
  result.stats.far_pair_count = static_cast<int>(worklist.m2l_pairs.size());
  result.stats.direct_pair_count =
      static_cast<int>(worklist.direct_leaf_pairs.size());
  result.stats.mac_accepted_pairs = worklist.mac_accepted_pairs;
  result.stats.check_guard_rejected_pairs =
      worklist.check_guard_rejected_pairs;
  result.stats.leaf_direct_pairs = worklist.leaf_direct_pairs;
  if (worklist.worklist_hash == 0) {
    result.status = "worklist_gap";
    return result;
  }
  const std::size_t node_count = tree.nodes.size();
  const std::vector<std::pair<int, int>> m2l_pairs =
      effective_m2l_pairs(tree, worklist, mode);
  result.stats.far_pair_count = static_cast<int>(m2l_pairs.size());
  std::vector<bool> m2l_target_seen(node_count, false);
  std::vector<bool> m2l_source_seen(node_count, false);

  for (const auto &[target_node_id, source_node_id] : m2l_pairs) {
    const KifmmHierarchyNode &source_node =
        tree.nodes[static_cast<std::size_t>(source_node_id)];
    const KifmmHierarchyNode &target_node =
        tree.nodes[static_cast<std::size_t>(target_node_id)];
    if (target_node_id >= 0 &&
        static_cast<std::size_t>(target_node_id) < m2l_target_seen.size() &&
        !m2l_target_seen[static_cast<std::size_t>(target_node_id)]) {
      m2l_target_seen[static_cast<std::size_t>(target_node_id)] = true;
      ++result.m2l_target_nodes;
    }
    if (source_node_id >= 0 &&
        static_cast<std::size_t>(source_node_id) < m2l_source_seen.size() &&
        !m2l_source_seen[static_cast<std::size_t>(source_node_id)]) {
      m2l_source_seen[static_cast<std::size_t>(source_node_id)] = true;
      ++result.m2l_source_nodes;
    }
    result.stats.far_panel_pairs +=
        static_cast<std::uint64_t>(source_node.source_indices.size()) *
        static_cast<std::uint64_t>(target_node.target_indices.size());
  }
  for (const auto &[target_node_id, source_node_id] :
       worklist.direct_leaf_pairs) {
    const KifmmHierarchyNode &source_node =
        tree.nodes[static_cast<std::size_t>(source_node_id)];
    const KifmmHierarchyNode &target_node =
        tree.nodes[static_cast<std::size_t>(target_node_id)];
    result.stats.direct_panel_pairs +=
        static_cast<std::uint64_t>(source_node.source_indices.size()) *
        static_cast<std::uint64_t>(target_node.target_indices.size());
  }
  result.stats.exact_pair_coverage =
      result.stats.far_panel_pairs + result.stats.direct_panel_pairs;
  const std::uint64_t expected_pairs =
      static_cast<std::uint64_t>(sources.size()) *
      static_cast<std::uint64_t>(targets.size());
  if (result.stats.exact_pair_coverage != expected_pairs) {
    result.status = "coverage_gap";
    return result;
  }
  result.stats.far_fraction =
      expected_pairs == 0
          ? 0.0
          : static_cast<double>(result.stats.far_panel_pairs) /
                static_cast<double>(expected_pairs);

  const double inv_eps_ratio = 1.0 / config.eps_ratio;
  std::vector<KifmmJcpChannelRow> approx(targets.size());
  std::vector<KifmmJcpChannelRow> direct(targets.size());
  std::vector<K4Parts> approx_k4_parts(targets.size());
  std::vector<K4Parts> direct_k4_parts_by_target(targets.size());

  const auto direct_fallback_start = Clock::now();
  if (config.include_direct_fallback) {
    for (const auto &[target_node_id, source_node_id] :
         worklist.direct_leaf_pairs) {
      add_direct_pair_contribution(
          sources, targets, target_normals,
          tree.nodes[static_cast<std::size_t>(source_node_id)],
          tree.nodes[static_cast<std::size_t>(target_node_id)], config.kappa,
          config.eps_ratio, inv_eps_ratio, approx);
      add_direct_pair_k4_parts(
          sources, targets, target_normals,
          tree.nodes[static_cast<std::size_t>(source_node_id)],
          tree.nodes[static_cast<std::size_t>(target_node_id)], config.kappa,
          approx_k4_parts);
    }
  }
  result.direct_fallback_seconds =
      elapsed_seconds(direct_fallback_start, Clock::now());
  record_hierarchy_memory_after_direct_fallback(result);

  const auto upward_rule =
      make_cube_surface_rule(config.hierarchy.order,
                             config.hierarchy.equiv_scale,
                             config.hierarchy.upward_check_scale);
  const auto local_rule = make_cube_local_surface_rule(
      config.hierarchy.order, config.hierarchy.local_equiv_scale,
      config.hierarchy.local_check_scale);
  const std::string solver = "qr";
  KifmmEquivalentSystemCache upward_cache;
  KifmmEquivalentSystemCache local_cache;
  std::vector<JcpBundle> upward(node_count);
  std::vector<bool> upward_valid(node_count, false);
  std::vector<JcpBundleRhs> local_rhs(node_count);
  std::vector<JcpBundle> local(node_count);
  std::vector<bool> local_valid(node_count, false);
  const std::vector<std::vector<int>> nodes_by_level =
      build_nodes_by_level(tree);
  std::vector<titan::tabi::fmm_m2l::PrototypePanel> cartesian_panels;
  std::vector<double> cartesian_phi_values;
  std::vector<double> cartesian_phin_values;
  if (hybrid_source_local_bridge) {
    cartesian_panels = prototype_panels_from_sources(sources);
    cartesian_phi_values = phi_values_from_sources(sources);
    cartesian_phin_values = phin_values_from_sources(sources);
  }

  const auto p2m_start = Clock::now();
  for (const KifmmHierarchyNode &node : tree.nodes) {
    if (!node.is_leaf || node.source_indices.empty()) {
      continue;
    }
    double rhs_seconds = 0.0;
    double solve_seconds = 0.0;
    JcpBundle bundle =
        hybrid_source_local_bridge
            ? solve_source_bundle_cartesian_m2e(
                  upward_rule, upward_cache, cartesian_panels,
                  cartesian_phi_values, cartesian_phin_values, node,
                  bridge_config, rhs_seconds, solve_seconds)
            : solve_source_bundle(upward_rule, upward_cache, sources, node,
                                  config.kappa, solver,
                                  config.hierarchy.condition_policy,
                                  rhs_seconds, solve_seconds);
    result.p2m_rhs_seconds += rhs_seconds;
    result.p2m_solve_seconds += solve_seconds;
    if (!bundle_passed(bundle)) {
      result.status = "p2m_field_solve_failed";
      return result;
    }
    result.m2e_max_relative_residual = std::max(
        result.m2e_max_relative_residual, bundle_max_relative_residual(bundle));
    upward[static_cast<std::size_t>(node.id)] = std::move(bundle);
    upward_valid[static_cast<std::size_t>(node.id)] = true;
    ++result.p2m_leaf_solves;
  }
  result.p2m_seconds = elapsed_seconds(p2m_start, Clock::now());
  record_hierarchy_memory_after_p2m(result);

  const auto m2m_start = Clock::now();
  if (mode != "no-m2m") {
    for (int level = static_cast<int>(nodes_by_level.size()) - 2; level >= 0;
         --level) {
      for (const int node_id :
           nodes_by_level[static_cast<std::size_t>(level)]) {
      const KifmmHierarchyNode &node =
          tree.nodes[static_cast<std::size_t>(node_id)];
      if (node.source_indices.empty()) {
        continue;
      }
      const KifmmBox box = box_from_node(node);
      const auto laplace_system = build_equivalent_system_cached(
          upward_cache, upward_rule, box, KifmmKernel::Laplace, 0.0, solver,
          config.hierarchy.condition_policy);
      const auto yukawa_system = build_equivalent_system_cached(
          upward_cache, upward_rule, box, KifmmKernel::Yukawa, config.kappa,
          solver, config.hierarchy.condition_policy);
      if (!laplace_system.finite || !yukawa_system.finite) {
        result.status = "m2m_system_gap";
        return result;
      }
      JcpBundleRhs rhs;
      rhs.laplace =
          Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(
                                    laplace_system.check_points.size()),
                                static_cast<Eigen::Index>(kFamilies.size()));
      rhs.yukawa =
          Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(
                                    yukawa_system.check_points.size()),
                                static_cast<Eigen::Index>(kFamilies.size()));
      bool used_child = false;
      for (const int child_id : node.child) {
        if (child_id < 0 ||
            !upward_valid[static_cast<std::size_t>(child_id)]) {
          continue;
        }
        add_bundle_to_points(
            upward_rule, tree.nodes[static_cast<std::size_t>(child_id)],
            upward[static_cast<std::size_t>(child_id)],
            KifmmKernel::Laplace, 0.0, laplace_system.check_points,
            rhs.laplace);
        add_bundle_to_points(
            upward_rule, tree.nodes[static_cast<std::size_t>(child_id)],
            upward[static_cast<std::size_t>(child_id)], KifmmKernel::Yukawa,
            config.kappa, yukawa_system.check_points, rhs.yukawa);
        result.upward_rhs_adds +=
            static_cast<std::uint64_t>(laplace_system.check_points.size() +
                                       yukawa_system.check_points.size());
        ++result.m2m_translations;
        used_child = true;
      }
      if (!used_child || !finite_matrix(rhs.laplace) ||
          !finite_matrix(rhs.yukawa)) {
        continue;
      }
      double solve_seconds = 0.0;
      JcpBundle bundle = solve_rhs_bundle(
          upward_rule, upward_cache, node, config.kappa, solver,
          config.hierarchy.condition_policy, rhs, solve_seconds);
      if (!bundle_passed(bundle)) {
        result.status = "m2m_field_solve_failed";
        return result;
      }
      if (hybrid_bridge_mode) {
        result.m2e_max_relative_residual =
            std::max(result.m2e_max_relative_residual,
                     bundle_max_relative_residual(bundle));
      }
      upward[static_cast<std::size_t>(node.id)] = std::move(bundle);
      upward_valid[static_cast<std::size_t>(node.id)] = true;
      ++result.m2m_parent_solves;
      }
    }
  }
  result.m2m_seconds = elapsed_seconds(m2m_start, Clock::now());
  record_hierarchy_memory_after_m2m(result);

  M2lMatrixCache matrix_cache;
  matrix_cache.policy = result.m2l_matrix_cache_policy;
  matrix_cache.limit_bytes = m2l_matrix_cache_limit_bytes_from_mb(
      config.hierarchy.m2l_matrix_cache_limit_mb);
  std::vector<JcpBundleRhs> dense_local_rhs(node_count);
  const auto m2l_start = Clock::now();
  const bool local_oracle_mode = m2l_local_oracle_mode(mode);
  const bool use_batched_m2l =
      result.m2l_eval_policy == "target-check-batched" &&
      mode != "m2l-direct-to-targets";
  const bool use_grouped_gemm =
      use_batched_m2l &&
      result.m2l_gemm_batch_policy == "group-by-cache-key";
  const bool use_parallel_m2l =
      use_batched_m2l &&
      result.m2l_parallel_policy == "target-node-key-group";
  if (mode == "m2l-direct-to-targets") {
    for (const auto &[target_node_id, source_node_id] : m2l_pairs) {
      if (!upward_valid[static_cast<std::size_t>(source_node_id)]) {
        result.status = "m2l_missing_upward";
        return result;
      }
      const KifmmHierarchyNode &source_node =
          tree.nodes[static_cast<std::size_t>(source_node_id)];
      const KifmmHierarchyNode &target_node =
          tree.nodes[static_cast<std::size_t>(target_node_id)];
      if (!accumulate_direct_target_m2l_pair(
              upward_rule, source_node, target_node,
              upward[static_cast<std::size_t>(source_node_id)], config.kappa,
              config.eps_ratio, inv_eps_ratio, targets, target_normals, approx,
              approx_k4_parts)) {
        result.status = "m2l_missing_upward";
        return result;
      }
      ++result.m2l_translations;
    }
  } else if (use_batched_m2l && use_grouped_gemm) {
    if (m2l_matrix_cache_is_streaming(matrix_cache.policy)) {
      result.status = "m2l_streaming_cache_incompatible_with_global_grouping";
      result.accuracy_failure =
          "m2l_streaming_cache_incompatible_with_global_grouping";
      return result;
    }
    std::vector<M2lTargetSystems> systems_by_node(node_count);
    std::vector<unsigned char> target_prepared(node_count, 0);
    std::vector<M2lPairRecord> records;
    records.reserve(m2l_pairs.size());
    for (const auto &[target_node_id, source_node_id] : m2l_pairs) {
      if (!upward_valid[static_cast<std::size_t>(source_node_id)]) {
        result.status = "m2l_missing_upward";
        return result;
      }
      const KifmmHierarchyNode &target_node =
          tree.nodes[static_cast<std::size_t>(target_node_id)];
      if (target_prepared[static_cast<std::size_t>(target_node_id)] == 0) {
        if (!prepare_m2l_target_systems(
                local_cache, local_rule, target_node, config, solver,
                local_rhs, dense_local_rhs, local_oracle_mode,
                systems_by_node[static_cast<std::size_t>(target_node_id)],
                result)) {
          return result;
        }
        target_prepared[static_cast<std::size_t>(target_node_id)] = 1;
      }
      records.push_back({target_node_id, source_node_id});
    }
    if (!accumulate_global_grouped_m2l_pairs(
            upward_rule, sources, tree, records, upward, systems_by_node,
            config.kappa, local_oracle_mode, local_rhs, dense_local_rhs,
            matrix_cache, config.hierarchy.local_check_scale, result)) {
      return result;
    }
  } else if (use_parallel_m2l) {
    std::vector<M2lTargetSystems> systems_by_node(node_count);
    if (!accumulate_parallel_m2l_by_target_node(
            upward_rule, sources, tree, m2l_pairs, upward, local_cache,
            local_rule, config, solver, local_oracle_mode, local_rhs,
            dense_local_rhs, systems_by_node, matrix_cache, result)) {
      return result;
    }
  } else if (use_batched_m2l) {
    std::size_t pair_index = 0;
    while (pair_index < m2l_pairs.size()) {
      const int target_node_id = m2l_pairs[pair_index].first;
      M2lMatrixCacheScopeGuard streaming_scope(matrix_cache);
      const KifmmHierarchyNode &target_node =
          tree.nodes[static_cast<std::size_t>(target_node_id)];
      M2lTargetSystems systems;
      if (!prepare_m2l_target_systems(local_cache, local_rule, target_node,
                                      config, solver, local_rhs,
                                      dense_local_rhs, local_oracle_mode,
                                      systems, result)) {
        return result;
      }
      std::vector<int> grouped_source_node_ids;
      while (pair_index < m2l_pairs.size() &&
             m2l_pairs[pair_index].first == target_node_id) {
        const int source_node_id = m2l_pairs[pair_index].second;
        if (!upward_valid[static_cast<std::size_t>(source_node_id)]) {
          result.status = "m2l_missing_upward";
          return result;
        }
        const KifmmHierarchyNode &source_node =
            tree.nodes[static_cast<std::size_t>(source_node_id)];
        if (use_grouped_gemm) {
          (void)source_node;
          grouped_source_node_ids.push_back(source_node_id);
        } else {
          if (!accumulate_batched_m2l_pair(
                  upward_rule, sources, source_node,
                  upward[static_cast<std::size_t>(source_node_id)],
                  target_node, systems, config.kappa, local_oracle_mode,
                  local_rhs[static_cast<std::size_t>(target_node_id)],
                  dense_local_rhs[static_cast<std::size_t>(target_node_id)],
                  matrix_cache, config.hierarchy.local_check_scale, result)) {
            return result;
          }
        }
        ++pair_index;
      }
      if (use_grouped_gemm &&
          !accumulate_grouped_m2l_target(
              upward_rule, sources, tree, grouped_source_node_ids, upward,
              target_node, systems, config.kappa, local_oracle_mode,
              local_rhs[static_cast<std::size_t>(target_node_id)],
              dense_local_rhs[static_cast<std::size_t>(target_node_id)],
              matrix_cache, config.hierarchy.local_check_scale, result)) {
        return result;
      }
    }
  } else {
    for (const auto &[target_node_id, source_node_id] : m2l_pairs) {
      if (!upward_valid[static_cast<std::size_t>(source_node_id)]) {
        result.status = "m2l_missing_upward";
        return result;
      }
      const KifmmHierarchyNode &source_node =
          tree.nodes[static_cast<std::size_t>(source_node_id)];
      const KifmmHierarchyNode &target_node =
          tree.nodes[static_cast<std::size_t>(target_node_id)];
      M2lTargetSystems systems;
      if (!prepare_m2l_target_systems(local_cache, local_rule, target_node,
                                      config, solver, local_rhs,
                                      dense_local_rhs, local_oracle_mode,
                                      systems, result)) {
        return result;
      }
      if (!accumulate_standard_m2l_pair(
              upward_rule, sources, source_node,
              upward[static_cast<std::size_t>(source_node_id)], systems,
              config.kappa, local_oracle_mode,
              local_rhs[static_cast<std::size_t>(target_node_id)],
              dense_local_rhs[static_cast<std::size_t>(target_node_id)],
              result)) {
        return result;
      }
    }
  }
  result.m2l_seconds = elapsed_seconds(m2l_start, Clock::now());
  result.m2l_matrix_cache_entries =
      static_cast<std::uint64_t>(matrix_cache.entries.size());
  result.m2l_matrix_cache_peak_entries = matrix_cache.peak_entries;
  result.m2l_matrix_cache_hits = matrix_cache.hits;
  result.m2l_matrix_cache_misses = matrix_cache.misses;
  const std::uint64_t cache_lookups =
      matrix_cache.hits + matrix_cache.misses;
  result.m2l_matrix_cache_hit_rate =
      cache_lookups == 0
          ? 0.0
          : static_cast<double>(matrix_cache.hits) /
                static_cast<double>(cache_lookups);
  result.m2l_matrix_cache_limit_bytes = matrix_cache.limit_bytes;
  result.m2l_matrix_cache_evictions = matrix_cache.evictions;
  result.m2l_matrix_cache_estimated_bytes = matrix_cache.estimated_bytes;
  result.m2l_matrix_cache_peak_estimated_bytes =
      matrix_cache.peak_estimated_bytes;
  result.m2l_matrix_cache_total_built_bytes =
      matrix_cache.total_built_bytes;
  result.m2l_matrix_cache_build_seconds = matrix_cache.build_seconds;
  result.m2l_matrix_gemm_seconds = matrix_cache.gemm_seconds;
  result.m2l_matrix_cache_key_status = matrix_cache.key_status;
  if (result.m2l_translations > 0) {
    result.m2l_seconds_per_translation =
        result.m2l_seconds / static_cast<double>(result.m2l_translations);
  }
  if (result.m2l_check_points_total > 0) {
    result.m2l_seconds_per_check_point =
        result.m2l_seconds /
        static_cast<double>(result.m2l_check_points_total);
  }
  record_hierarchy_memory_after_m2l(result);

  if (mode == "local-check-oracle" || mode == "l2p-oracle") {
    RmsAccumulator rhs_acc;
    for (std::size_t node_id = 0; node_id < node_count; ++node_id) {
      if (local_rhs[node_id].laplace.size() == 0 &&
          local_rhs[node_id].yukawa.size() == 0) {
        continue;
      }
      add_matrix_attribution("local-check-rhs", KifmmKernel::Laplace,
                             local_rhs[node_id].laplace,
                             dense_local_rhs[node_id].laplace,
                             result.field_attribution, rhs_acc,
                             static_cast<int>(node_id),
                             tree.nodes[node_id].level);
      add_matrix_attribution("local-check-rhs", KifmmKernel::Yukawa,
                             local_rhs[node_id].yukawa,
                             dense_local_rhs[node_id].yukawa,
                             result.field_attribution, rhs_acc,
                             static_cast<int>(node_id),
                             tree.nodes[node_id].level);
      if (mode == "l2p-oracle") {
        local_rhs[node_id] = dense_local_rhs[node_id];
      }
    }
    result.local_check_rhs_relative_rms = rhs_acc.relative_rms();
    result.local_check_rhs_max_abs_error = rhs_acc.max_abs_error;
  }

  double local_solve_seconds = 0.0;
  const auto l2l_start = Clock::now();
  for (std::size_t level = 0; level < nodes_by_level.size(); ++level) {
    for (const int node_id : nodes_by_level[level]) {
      const KifmmHierarchyNode &node =
          tree.nodes[static_cast<std::size_t>(node_id)];
      JcpBundleRhs &rhs = local_rhs[static_cast<std::size_t>(node_id)];
      if (rhs.laplace.size() > 0 || rhs.yukawa.size() > 0) {
        double solve_seconds = 0.0;
        JcpBundle bundle = solve_rhs_bundle(
            local_rule, local_cache, node, config.kappa, solver,
            config.hierarchy.condition_policy, rhs, solve_seconds);
        local_solve_seconds += solve_seconds;
        if (!bundle_passed(bundle)) {
          result.status = "local_field_solve_failed";
          return result;
        }
        if (hybrid_bridge_mode) {
          result.local_equiv_max_relative_residual =
              std::max(result.local_equiv_max_relative_residual,
                       bundle_max_relative_residual(bundle));
        }
        local[static_cast<std::size_t>(node.id)] = std::move(bundle);
        local_valid[static_cast<std::size_t>(node.id)] = true;
        ++result.local_node_solves;
      }
      if (!local_valid[static_cast<std::size_t>(node.id)]) {
        continue;
      }
      for (const int child_id : node.child) {
        if (child_id < 0) {
          continue;
        }
        const KifmmHierarchyNode &child =
            tree.nodes[static_cast<std::size_t>(child_id)];
        if (child.target_indices.empty()) {
          continue;
        }
        const KifmmBox child_box = box_from_node(child);
        const auto laplace_system = build_equivalent_system_cached(
            local_cache, local_rule, child_box, KifmmKernel::Laplace, 0.0,
            solver, config.hierarchy.condition_policy);
        const auto yukawa_system = build_equivalent_system_cached(
            local_cache, local_rule, child_box, KifmmKernel::Yukawa,
            config.kappa, solver, config.hierarchy.condition_policy);
        if (!laplace_system.finite || !yukawa_system.finite ||
            !ensure_rhs_for_node(local_rhs, laplace_system, yukawa_system,
                                 child_id)) {
          result.status = "l2l_local_system_gap";
          return result;
        }
        add_bundle_to_points(
            local_rule, node, local[static_cast<std::size_t>(node.id)],
            KifmmKernel::Laplace, 0.0, laplace_system.check_points,
            local_rhs[static_cast<std::size_t>(child_id)].laplace);
        add_bundle_to_points(
            local_rule, node, local[static_cast<std::size_t>(node.id)],
            KifmmKernel::Yukawa, config.kappa, yukawa_system.check_points,
            local_rhs[static_cast<std::size_t>(child_id)].yukawa);
        result.local_rhs_adds +=
            static_cast<std::uint64_t>(laplace_system.check_points.size() +
                                       yukawa_system.check_points.size());
        ++result.l2l_translations;
      }
    }
  }
  result.l2l_seconds = elapsed_seconds(l2l_start, Clock::now());
  result.local_solve_seconds = local_solve_seconds;
  record_hierarchy_memory_after_l2l(result);

  const auto l2p_start = Clock::now();
  for (const KifmmHierarchyNode &node : tree.nodes) {
    if (!node.is_leaf || node.target_indices.empty() ||
        !local_valid[static_cast<std::size_t>(node.id)]) {
      continue;
    }
    titan::tabi::fmm_m2l::CartesianJcpLocalP4 cartesian_local;
    KifmmCartesianSplitLocalBridgeResult split_local;
    if (hybrid_bridge_mode) {
      const KifmmJcpChannels local_channels =
          channels_from_bundle(local[static_cast<std::size_t>(node.id)]);
      if (hybrid_local_split_bridge) {
        split_local = bridge_equivalent_channels_to_cartesian_local_split_k4(
            local_rule, box_from_node(node), local_channels, bridge_config);
        if (!split_local.finite_outputs) {
          result.status = "hybrid_split_e2l_failed";
          result.e2l_coefficient_status = split_local.status;
          return result;
        }
        cartesian_local = split_local.combined_local;
      } else {
        cartesian_local = bridge_equivalent_channels_to_cartesian_local(
            local_rule, box_from_node(node), local_channels, bridge_config);
        if (!cartesian_local.finite_outputs) {
          result.status = "hybrid_e2l_failed";
          result.e2l_coefficient_status = "failed";
          return result;
        }
      }
    }
    for (const int target_index : node.target_indices) {
      const std::size_t ti = static_cast<std::size_t>(target_index);
      KifmmJcpChannelRow row;
      if (hybrid_bridge_mode) {
        row = channel_row_from_cartesian_local(
            cartesian_local, node, targets[ti], target_normals[ti],
            bridge_config.cartesian_order);
      } else if (k123_only) {
        row = channel_row_from_bundle_k123_only(
            local_rule, node, local[static_cast<std::size_t>(node.id)],
            config.kappa, config.eps_ratio, inv_eps_ratio, targets[ti],
            target_normals[ti]);
      } else {
        row = channel_row_from_bundle(
            local_rule, node, local[static_cast<std::size_t>(node.id)],
            config.kappa, config.eps_ratio, inv_eps_ratio, targets[ti],
            target_normals[ti]);
        ++result.hierarchy_k4_evaluation_count;
      }
      approx[ti].k1 += row.k1;
      approx[ti].k2 += row.k2;
      approx[ti].k3 += row.k3;
      approx[ti].k4 += row.k4;
      K4Parts parts;
      if (hybrid_local_split_bridge) {
        const KifmmJcpChannelRow laplace_row =
            channel_row_from_cartesian_local(
                split_local.k4_laplace_local, node, targets[ti],
                target_normals[ti], bridge_config.cartesian_order);
        const KifmmJcpChannelRow yukawa_row =
            channel_row_from_cartesian_local(
                split_local.k4_yukawa_local, node, targets[ti],
                target_normals[ti], bridge_config.cartesian_order);
        parts.laplace = laplace_row.k4;
        parts.yukawa = yukawa_row.k4;
        parts.combined = parts.laplace - parts.yukawa;
        approx[ti].k4 += parts.combined - row.k4;
        result.k4_decomposition_status = "laplace_yukawa_split";
      } else if (hybrid_bridge_mode) {
        parts.combined = row.k4;
        result.k4_decomposition_status = "combined_only";
      } else if (k123_only) {
        result.k4_decomposition_status = "k4_pruned";
      } else {
        parts = k4_parts_from_bundle(
            local_rule, node, local[static_cast<std::size_t>(node.id)],
            config.kappa, targets[ti], target_normals[ti]);
      }
      approx_k4_parts[ti].laplace += parts.laplace;
      approx_k4_parts[ti].yukawa += parts.yukawa;
      approx_k4_parts[ti].combined += parts.combined;
      ++result.l2p_target_count;
    }
  }
  result.l2p_seconds = elapsed_seconds(l2p_start, Clock::now());
  record_hierarchy_memory_after_l2p(result);

  const auto oracle_start = Clock::now();
  if (config.build_dense_oracle) {
    for (std::size_t i = 0; i < targets.size(); ++i) {
      direct[i] = assemble_direct_jcp_channel_row(
          sources, config.kappa, config.eps_ratio, inv_eps_ratio, targets[i],
          target_normals[i]);
      direct_k4_parts_by_target[i] =
          direct_k4_parts(sources, config.kappa, targets[i], target_normals[i]);
    }
  }
  result.dense_oracle_seconds = elapsed_seconds(oracle_start, Clock::now());
  record_hierarchy_memory_after_dense_oracle(result);

  if (!finite_channels(approx) || !finite_channels(direct)) {
    result.status = "nonfinite_output";
    return result;
  }
  result.aggregate_relative_rms = aggregate_relative_rms(approx, direct);
  result.k1_relative_rms = channel_relative_rms(approx, direct, 0);
  result.k2_relative_rms = channel_relative_rms(approx, direct, 1);
  result.k3_relative_rms = channel_relative_rms(approx, direct, 2);
  result.k4_relative_rms = channel_relative_rms(approx, direct, 3);
  RmsAccumulator k4_laplace_acc;
  RmsAccumulator k4_yukawa_acc;
  RmsAccumulator k4_combined_acc;
  for (std::size_t i = 0; i < targets.size(); ++i) {
    k4_laplace_acc.add(approx_k4_parts[i].laplace,
                       direct_k4_parts_by_target[i].laplace);
    k4_yukawa_acc.add(approx_k4_parts[i].yukawa,
                      direct_k4_parts_by_target[i].yukawa);
    k4_combined_acc.add(approx_k4_parts[i].combined,
                        direct_k4_parts_by_target[i].combined);
  }
  result.k4_laplace_grad_relative_rms = k4_laplace_acc.relative_rms();
  result.k4_yukawa_grad_relative_rms = k4_yukawa_acc.relative_rms();
  result.k4_combined_relative_rms = k4_combined_acc.relative_rms();
  if (hybrid_local_split_bridge) {
    result.split_k4_laplace_relative_rms =
        result.k4_laplace_grad_relative_rms;
    result.split_k4_yukawa_relative_rms =
        result.k4_yukawa_grad_relative_rms;
    result.split_k4_combined_relative_rms =
        result.k4_combined_relative_rms;
    result.split_vs_dense_k4_relative_rms = result.k4_relative_rms;
  }
  result.k4_abs_rms = k4_combined_acc.abs_rms();
  result.k4_reference_rms = k4_combined_acc.reference_rms();
  if (mode == "l2p-oracle") {
    result.l2p_oracle_aggregate_relative_rms = result.aggregate_relative_rms;
    result.l2p_oracle_k4_relative_rms = result.k4_relative_rms;
  }
  result.max_channel_relative_rms = result.k1_relative_rms;
  result.max_channel_name = "K1";
  if (result.k2_relative_rms > result.max_channel_relative_rms) {
    result.max_channel_relative_rms = result.k2_relative_rms;
    result.max_channel_name = "K2";
  }
  if (result.k3_relative_rms > result.max_channel_relative_rms) {
    result.max_channel_relative_rms = result.k3_relative_rms;
    result.max_channel_name = "K3";
  }
  if (result.k4_relative_rms > result.max_channel_relative_rms) {
    result.max_channel_relative_rms = result.k4_relative_rms;
    result.max_channel_name = "K4";
  }
  result.max_abs_error = max_abs_error(approx, direct);
  result.output_hash = hash_channel_rows(approx);
  result.approximate_rows = approx;
  result.direct_rows = direct;
  result.equivalent_factor_cache_entries =
      static_cast<int>(upward_cache.entries.size() + local_cache.entries.size());
  result.equivalent_factor_cache_hits =
      static_cast<int>(upward_cache.stats.hits + local_cache.stats.hits);
  result.equivalent_factor_cache_misses =
      static_cast<int>(upward_cache.stats.misses + local_cache.stats.misses);
  update_k123_retained_memory_estimates(result, upward, local_rhs,
                                        dense_local_rhs, local, approx, direct,
                                        upward_cache, local_cache);
  result.total_seconds = elapsed_seconds(total_start, Clock::now());
  record_hierarchy_memory_after_finalize(result);
  result.hierarchy_runtime_memory_attribution_status = "pass";
  result.status =
      std::isfinite(result.aggregate_relative_rms) ? "ok" : "nonfinite_rms";
  result.execution_status = result.status == "ok" ? "pass" : "fail";
  if (result.execution_status == "pass") {
    const bool aggregate_pass =
        result.aggregate_relative_rms <= result.aggregate_relative_rms_gate;
    const bool channel_pass = result.max_channel_relative_rms <=
                              result.max_channel_relative_rms_gate;
    if (mode == "local-check-oracle") {
      const bool rhs_pass = std::isfinite(result.local_check_rhs_relative_rms) &&
                            result.local_check_rhs_relative_rms <=
                                result.aggregate_relative_rms_gate;
      result.accuracy_status = rhs_pass ? "pass" : "fail";
      if (!rhs_pass) {
        result.accuracy_failure = "local_check_rhs";
      }
    } else {
      result.accuracy_status =
          aggregate_pass && channel_pass ? "pass" : "fail";
      if (!aggregate_pass && !channel_pass) {
        result.accuracy_failure = "aggregate_and_channel";
      } else if (!aggregate_pass) {
        result.accuracy_failure = "aggregate";
      } else if (!channel_pass) {
        result.accuracy_failure = "channel";
      }
    }
  }
  return result;
}

KifmmHierarchyJcpResult run_full_hierarchical_jcp_kifmm(
    std::span<const KifmmPanelSource> sources, std::span<const Vec3> targets,
    std::span<const Vec3> target_normals,
    const KifmmHierarchyJcpConfig &config) {
  const auto total_start = Clock::now();
  const std::vector<Vec3> positions = source_positions(sources);
  const KifmmHierarchyTree tree =
      build_strict_octree(positions, targets, config.hierarchy);
  const KifmmHierarchyWorklist worklist =
      build_hierarchy_worklist(tree, config.hierarchy);
  return run_full_hierarchical_jcp_kifmm_impl(
      sources, targets, target_normals, config, tree, worklist, total_start);
}

KifmmHierarchyJcpResult run_full_hierarchical_jcp_kifmm_with_worklist(
    std::span<const KifmmPanelSource> sources, std::span<const Vec3> targets,
    std::span<const Vec3> target_normals,
    const KifmmHierarchyJcpConfig &config, const KifmmHierarchyTree &tree,
    const KifmmHierarchyWorklist &worklist) {
  return run_full_hierarchical_jcp_kifmm_impl(
      sources, targets, target_normals, config, tree, worklist, Clock::now());
}

KifmmHierarchyJcpResult run_flat_jcp_kifmm_with_worklist(
    std::span<const KifmmPanelSource> sources, std::span<const Vec3> targets,
    std::span<const Vec3> target_normals,
    const KifmmHierarchyJcpConfig &config, const KifmmHierarchyTree &tree,
    const KifmmHierarchyWorklist &worklist) {
  const auto total_start = Clock::now();
  KifmmHierarchyJcpResult result;
  result.execution_status = "fail";
  result.aggregate_relative_rms_gate = config.aggregate_relative_rms_gate;
  result.max_channel_relative_rms_gate =
      config.max_channel_relative_rms_gate;
  if (sources.empty() || targets.empty() ||
      targets.size() != target_normals.size()) {
    result.status = "invalid_input_size";
    return result;
  }
  if (!std::isfinite(config.kappa) || config.kappa < 0.0 ||
      !std::isfinite(config.eps_ratio) || config.eps_ratio == 0.0) {
    result.status = "invalid_physics";
    return result;
  }
  if (config.hierarchy.order > 10) {
    result.status = "unsupported_order_for_hierarchy_jcp";
    result.accuracy_failure = "unsupported_order";
    return result;
  }
  for (const KifmmPanelSource &source : sources) {
    if (!finite_source(source)) {
      result.status = "invalid_source";
      return result;
    }
  }
  for (const Vec3 normal : target_normals) {
    if (!finite_vec(normal)) {
      result.status = "invalid_target_normal";
      return result;
    }
  }

  result.stats = tree.stats;
  result.tree_hash = tree.tree_hash;
  if (tree.status != "ok") {
    result.status = tree.status;
    return result;
  }
  if (!hierarchy_links_are_valid(tree)) {
    result.status = "invalid_tree_links";
    return result;
  }
  result.worklist_hash = worklist.worklist_hash;
  result.policy_hash = worklist.policy_hash;
  result.stats.far_pair_count = static_cast<int>(worklist.m2l_pairs.size());
  result.stats.direct_pair_count =
      static_cast<int>(worklist.direct_leaf_pairs.size());
  result.stats.mac_accepted_pairs = worklist.mac_accepted_pairs;
  result.stats.check_guard_rejected_pairs =
      worklist.check_guard_rejected_pairs;
  result.stats.leaf_direct_pairs = worklist.leaf_direct_pairs;
  if (worklist.worklist_hash == 0) {
    result.status = "worklist_gap";
    return result;
  }
  for (const auto &[target_node_id, source_node_id] : worklist.m2l_pairs) {
    const KifmmHierarchyNode &source_node =
        tree.nodes[static_cast<std::size_t>(source_node_id)];
    const KifmmHierarchyNode &target_node =
        tree.nodes[static_cast<std::size_t>(target_node_id)];
    result.stats.far_panel_pairs +=
        static_cast<std::uint64_t>(source_node.source_indices.size()) *
        static_cast<std::uint64_t>(target_node.target_indices.size());
  }
  for (const auto &[target_node_id, source_node_id] :
       worklist.direct_leaf_pairs) {
    const KifmmHierarchyNode &source_node =
        tree.nodes[static_cast<std::size_t>(source_node_id)];
    const KifmmHierarchyNode &target_node =
        tree.nodes[static_cast<std::size_t>(target_node_id)];
    result.stats.direct_panel_pairs +=
        static_cast<std::uint64_t>(source_node.source_indices.size()) *
        static_cast<std::uint64_t>(target_node.target_indices.size());
  }
  result.stats.exact_pair_coverage =
      result.stats.far_panel_pairs + result.stats.direct_panel_pairs;
  const std::uint64_t expected_pairs =
      static_cast<std::uint64_t>(sources.size()) *
      static_cast<std::uint64_t>(targets.size());
  if (result.stats.exact_pair_coverage != expected_pairs) {
    result.status = "coverage_gap";
    return result;
  }
  result.stats.far_fraction =
      expected_pairs == 0
          ? 0.0
          : static_cast<double>(result.stats.far_panel_pairs) /
                static_cast<double>(expected_pairs);

  const double inv_eps_ratio = 1.0 / config.eps_ratio;
  std::vector<KifmmJcpChannelRow> approx(targets.size());
  std::vector<KifmmJcpChannelRow> direct(targets.size());

  const auto direct_fallback_start = Clock::now();
  if (config.include_direct_fallback) {
    for (const auto &[target_node_id, source_node_id] :
         worklist.direct_leaf_pairs) {
      add_direct_pair_contribution(
          sources, targets, target_normals,
          tree.nodes[static_cast<std::size_t>(source_node_id)],
          tree.nodes[static_cast<std::size_t>(target_node_id)], config.kappa,
          config.eps_ratio, inv_eps_ratio, approx);
    }
  }
  result.direct_fallback_seconds =
      elapsed_seconds(direct_fallback_start, Clock::now());

  const auto upward_rule =
      make_cube_surface_rule(config.hierarchy.order,
                             config.hierarchy.equiv_scale,
                             config.hierarchy.upward_check_scale);
  const std::string solver = "qr";
  KifmmEquivalentSystemCache upward_cache;
  std::vector<JcpBundle> source_bundle(tree.nodes.size());
  std::vector<bool> source_bundle_valid(tree.nodes.size(), false);
  result.jcp_channel_policy =
      config.hierarchy.jcp_channel_policy.empty()
          ? std::string("all")
          : config.hierarchy.jcp_channel_policy;
  if (!valid_jcp_channel_policy(result.jcp_channel_policy)) {
    result.status = "invalid_jcp_channel_policy";
    result.accuracy_failure = "invalid_jcp_channel_policy";
    return result;
  }
  const bool k123_only = result.jcp_channel_policy == "k123-only";
  result.hierarchy_k4_policy = k123_only ? "pruned" : "computed";

  const auto p2m_start = Clock::now();
  for (const auto &[target_node_id, source_node_id] : worklist.m2l_pairs) {
    const KifmmHierarchyNode &source_node =
        tree.nodes[static_cast<std::size_t>(source_node_id)];
    if (!source_bundle_valid[static_cast<std::size_t>(source_node_id)]) {
      double rhs_seconds = 0.0;
      double solve_seconds = 0.0;
      JcpBundle bundle = solve_source_bundle(
          upward_rule, upward_cache, sources, source_node, config.kappa,
          solver, config.hierarchy.condition_policy, rhs_seconds,
          solve_seconds);
      result.p2m_rhs_seconds += rhs_seconds;
      result.p2m_solve_seconds += solve_seconds;
      if (!bundle_passed(bundle)) {
        result.status = "flat_source_field_solve_failed";
        return result;
      }
      source_bundle[static_cast<std::size_t>(source_node_id)] =
          std::move(bundle);
      source_bundle_valid[static_cast<std::size_t>(source_node_id)] = true;
      ++result.p2m_leaf_solves;
    }
    const KifmmHierarchyNode &target_node =
        tree.nodes[static_cast<std::size_t>(target_node_id)];
    for (const int target_index : target_node.target_indices) {
      const std::size_t ti = static_cast<std::size_t>(target_index);
      const KifmmJcpChannelRow row =
          k123_only
              ? channel_row_from_bundle_k123_only(
                    upward_rule, source_node,
                    source_bundle[static_cast<std::size_t>(source_node_id)],
                    config.kappa, config.eps_ratio, inv_eps_ratio,
                    targets[ti], target_normals[ti])
              : channel_row_from_bundle(
                    upward_rule, source_node,
                    source_bundle[static_cast<std::size_t>(source_node_id)],
                    config.kappa, config.eps_ratio, inv_eps_ratio,
                    targets[ti], target_normals[ti]);
      approx[ti].k1 += row.k1;
      approx[ti].k2 += row.k2;
      approx[ti].k3 += row.k3;
      approx[ti].k4 += row.k4;
      if (!k123_only) {
        ++result.hierarchy_k4_evaluation_count;
      }
      ++result.l2p_target_count;
    }
    ++result.m2l_translations;
  }
  result.p2m_seconds = elapsed_seconds(p2m_start, Clock::now());

  const auto oracle_start = Clock::now();
  if (config.build_dense_oracle) {
    for (std::size_t i = 0; i < targets.size(); ++i) {
      direct[i] = assemble_direct_jcp_channel_row(
          sources, config.kappa, config.eps_ratio, inv_eps_ratio, targets[i],
          target_normals[i]);
    }
  }
  result.dense_oracle_seconds = elapsed_seconds(oracle_start, Clock::now());

  if (!finite_channels(approx) ||
      (config.build_dense_oracle && !finite_channels(direct))) {
    result.status = "nonfinite_output";
    return result;
  }
  if (config.build_dense_oracle) {
    result.aggregate_relative_rms = aggregate_relative_rms(approx, direct);
    result.k1_relative_rms = channel_relative_rms(approx, direct, 0);
    result.k2_relative_rms = channel_relative_rms(approx, direct, 1);
    result.k3_relative_rms = channel_relative_rms(approx, direct, 2);
    result.k4_relative_rms = channel_relative_rms(approx, direct, 3);
    result.max_channel_relative_rms = result.k1_relative_rms;
    result.max_channel_name = "K1";
    if (result.k2_relative_rms > result.max_channel_relative_rms) {
      result.max_channel_relative_rms = result.k2_relative_rms;
      result.max_channel_name = "K2";
    }
    if (result.k3_relative_rms > result.max_channel_relative_rms) {
      result.max_channel_relative_rms = result.k3_relative_rms;
      result.max_channel_name = "K3";
    }
    if (result.k4_relative_rms > result.max_channel_relative_rms) {
      result.max_channel_relative_rms = result.k4_relative_rms;
      result.max_channel_name = "K4";
    }
    result.max_abs_error = max_abs_error(approx, direct);
  }
  result.output_hash = hash_channel_rows(approx);
  result.approximate_rows = approx;
  result.direct_rows = direct;
  result.equivalent_factor_cache_entries =
      static_cast<int>(upward_cache.entries.size());
  result.equivalent_factor_cache_hits =
      static_cast<int>(upward_cache.stats.hits);
  result.equivalent_factor_cache_misses =
      static_cast<int>(upward_cache.stats.misses);
  result.total_seconds = elapsed_seconds(total_start, Clock::now());
  result.status = "ok";
  result.execution_status = "pass";
  if (config.build_dense_oracle) {
    const bool aggregate_pass =
        result.aggregate_relative_rms <= result.aggregate_relative_rms_gate;
    const bool channel_pass = result.max_channel_relative_rms <=
                              result.max_channel_relative_rms_gate;
    result.accuracy_status =
        aggregate_pass && channel_pass ? "pass" : "fail";
    if (!aggregate_pass && !channel_pass) {
      result.accuracy_failure = "aggregate_and_channel";
    } else if (!aggregate_pass) {
      result.accuracy_failure = "aggregate";
    } else if (!channel_pass) {
      result.accuracy_failure = "channel";
    }
  }
  return result;
}

} // namespace titan::tabi::fmm_kifmm
