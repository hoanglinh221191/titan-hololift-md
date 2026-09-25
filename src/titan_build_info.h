#ifndef TITAN_BUILD_INFO_H
#define TITAN_BUILD_INFO_H

#if defined(__cpp_lib_jthread) && (__cpp_lib_jthread >= 201911L)
#define TITAN_HAS_STD_JTHREAD 1
#elif defined(__has_include)
#if __has_include(<thread>) && __has_include(<stop_token>) &&                  \
    (__cplusplus >= 202002L)
#define TITAN_HAS_STD_JTHREAD 1
#else
#define TITAN_HAS_STD_JTHREAD 0
#endif
#else
#define TITAN_HAS_STD_JTHREAD 0
#endif

#if defined(__AVX__)
#define TITAN_COMPILED_WITH_AVX 1
#else
#define TITAN_COMPILED_WITH_AVX 0
#endif

#if defined(__AVX2__)
#define TITAN_COMPILED_WITH_AVX2 1
#else
#define TITAN_COMPILED_WITH_AVX2 0
#endif

#if defined(__FMA__)
#define TITAN_COMPILED_WITH_FMA 1
#else
#define TITAN_COMPILED_WITH_FMA 0
#endif

#define TITAN_STRINGIZE_IMPL(x) #x
#define TITAN_STRINGIZE(x) TITAN_STRINGIZE_IMPL(x)

namespace titan_build {

inline constexpr const char *compiler_name() noexcept {
#if defined(__clang__)
  return "Clang " __clang_version__;
#elif defined(__GNUC__)
  return "GCC " TITAN_STRINGIZE(__GNUC__) "." TITAN_STRINGIZE(
      __GNUC_MINOR__) "." TITAN_STRINGIZE(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
  return "MSVC " TITAN_STRINGIZE(_MSC_VER);
#else
  return "unknown";
#endif
}

inline constexpr const char *target_os_name() noexcept {
#if defined(_WIN32)
  return "windows";
#elif defined(__linux__)
  return "linux";
#elif defined(__APPLE__)
  return "macos";
#else
  return "unknown";
#endif
}

inline constexpr const char *target_arch_name() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  return "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
  return "x86";
#elif defined(__aarch64__) || defined(_M_ARM64)
  return "aarch64";
#elif defined(__arm__) || defined(_M_ARM)
  return "arm";
#else
  return "unknown";
#endif
}

enum class SimdLevel {
  scalar = 0,
  avx = 1,
  avx2 = 2,
  avx2_fma = 3,
};

struct RuntimeCpuFeatures {
  bool is_x86 = false;
  bool avx = false;
  bool avx2 = false;
  bool fma = false;
};

inline constexpr const char *build_profile_name() noexcept {
#ifdef TITAN_BUILD_PROFILE_TOKEN
  return TITAN_STRINGIZE(TITAN_BUILD_PROFILE_TOKEN);
#else
  return "unknown";
#endif
}

inline constexpr const char *cxx23_floor_name() noexcept {
#ifdef TITAN_CXX23_FLOOR_CLANG20_GXX13
  return "Clang 20+/GCC 13+ C++23 subset";
#else
  return "unspecified";
#endif
}

inline constexpr bool runtime_dispatch_hotspots_enabled() noexcept {
#ifdef TITAN_RUNTIME_DISPATCH_HOTSPOTS
  return TITAN_RUNTIME_DISPATCH_HOTSPOTS != 0;
#else
  return false;
#endif
}

inline constexpr const char *runtime_dispatch_mode_name() noexcept {
  return runtime_dispatch_hotspots_enabled() ? "AVX2/FMA hotspots" : "none";
}

inline constexpr SimdLevel compiled_simd_level() noexcept {
#if TITAN_COMPILED_WITH_AVX2 && TITAN_COMPILED_WITH_FMA
  return SimdLevel::avx2_fma;
#elif TITAN_COMPILED_WITH_AVX2
  return SimdLevel::avx2;
#elif TITAN_COMPILED_WITH_AVX
  return SimdLevel::avx;
#else
  return SimdLevel::scalar;
#endif
}

inline constexpr const char *simd_level_name(SimdLevel level) noexcept {
  switch (level) {
  case SimdLevel::scalar:
    return "scalar";
  case SimdLevel::avx:
    return "AVX";
  case SimdLevel::avx2:
    return "AVX2";
  case SimdLevel::avx2_fma:
    return "AVX2+FMA";
  }
  return "unknown";
}

inline RuntimeCpuFeatures runtime_cpu_features() noexcept {
  RuntimeCpuFeatures features{};
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) ||           \
    defined(_M_X64)
  features.is_x86 = true;
#if defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init();
  features.avx = __builtin_cpu_supports("avx");
  features.avx2 = __builtin_cpu_supports("avx2");
  features.fma = __builtin_cpu_supports("fma");
#else
  features.avx = TITAN_COMPILED_WITH_AVX != 0;
  features.avx2 = TITAN_COMPILED_WITH_AVX2 != 0;
  features.fma = TITAN_COMPILED_WITH_FMA != 0;
#endif
#else
  features.avx = TITAN_COMPILED_WITH_AVX != 0;
  features.avx2 = TITAN_COMPILED_WITH_AVX2 != 0;
  features.fma = TITAN_COMPILED_WITH_FMA != 0;
#endif
  return features;
}

inline const RuntimeCpuFeatures &runtime_cpu_features_cached() noexcept {
  static const RuntimeCpuFeatures features = runtime_cpu_features();
  return features;
}

inline SimdLevel runtime_simd_level() noexcept {
  const RuntimeCpuFeatures &features = runtime_cpu_features_cached();
  if (features.avx2 && features.fma)
    return SimdLevel::avx2_fma;
  if (features.avx2)
    return SimdLevel::avx2;
  if (features.avx)
    return SimdLevel::avx;
  return SimdLevel::scalar;
}

inline bool runtime_supports_compiled_simd() noexcept {
  const RuntimeCpuFeatures &features = runtime_cpu_features_cached();
  switch (compiled_simd_level()) {
  case SimdLevel::scalar:
    return true;
  case SimdLevel::avx:
    return features.avx;
  case SimdLevel::avx2:
    return features.avx2;
  case SimdLevel::avx2_fma:
    return features.avx2 && features.fma;
  }
  return false;
}

} // namespace titan_build

#endif
