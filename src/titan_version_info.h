#ifndef TITAN_VERSION_INFO_H
#define TITAN_VERSION_INFO_H

#if defined(__has_include)
#if __has_include("titan_version_generated.h")
#include "titan_version_generated.h"
#endif
#endif

#ifndef TITAN_BUILD_METADATA_GENERATED
#define TITAN_BUILD_METADATA_GENERATED 0
#define TITAN_PRODUCT_VERSION "0.0.0"
#define TITAN_FULL_VERSION "0.0.0-dev.0+source.unknown"
#define TITAN_BUILD_NUMBER 0ULL
#define TITAN_BUILD_NUMBER_STRING "0"
#define TITAN_VERSION_MAJOR 0
#define TITAN_VERSION_MINOR 0
#define TITAN_VERSION_PATCH 0
#define TITAN_GIT_COMMIT "unknown"
#define TITAN_GIT_COMMIT_SHORT "unknown"
#define TITAN_GIT_DIRTY 0
#define TITAN_GIT_TAG ""
#define TITAN_RELEASE_BUILD 0
#define TITAN_SOURCE_DATE_EPOCH 0ULL
#define TITAN_SOURCE_DATE_UTC "unknown"
#define TITAN_CI_PROVIDER "unknown"
#define TITAN_CI_BUILD_NUMBER ""
#define TITAN_CI_RUN_ID ""
#define TITAN_GENERATED_BUILD_PROFILE "unknown"
#endif

namespace titan_build {

inline constexpr const char *product_version() noexcept {
  return TITAN_PRODUCT_VERSION;
}

inline constexpr const char *full_version() noexcept {
  return TITAN_FULL_VERSION;
}

inline constexpr unsigned long long build_number() noexcept {
  return TITAN_BUILD_NUMBER;
}

inline constexpr const char *build_number_string() noexcept {
  return TITAN_BUILD_NUMBER_STRING;
}

inline constexpr const char *git_commit() noexcept { return TITAN_GIT_COMMIT; }

inline constexpr const char *git_commit_short() noexcept {
  return TITAN_GIT_COMMIT_SHORT;
}

inline constexpr bool git_dirty() noexcept { return TITAN_GIT_DIRTY != 0; }

inline constexpr const char *git_state_name() noexcept {
  return git_dirty() ? "dirty" : "clean";
}

inline constexpr const char *git_tag() noexcept { return TITAN_GIT_TAG; }

inline constexpr bool release_build() noexcept {
  return TITAN_RELEASE_BUILD != 0;
}

inline constexpr unsigned long long source_date_epoch() noexcept {
  return TITAN_SOURCE_DATE_EPOCH;
}

inline constexpr const char *source_date_utc() noexcept {
  return TITAN_SOURCE_DATE_UTC;
}

inline constexpr const char *ci_provider() noexcept { return TITAN_CI_PROVIDER; }

inline constexpr const char *ci_build_number() noexcept {
  return TITAN_CI_BUILD_NUMBER;
}

inline constexpr const char *ci_run_id() noexcept { return TITAN_CI_RUN_ID; }

inline constexpr bool generated_metadata_available() noexcept {
  return TITAN_BUILD_METADATA_GENERATED != 0;
}

} // namespace titan_build

#endif
