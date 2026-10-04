#include "pbctopo/logging.h"

#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <system_error>

namespace titan_pbctopo {
namespace {

constexpr const char *kDebugLogPath = "pbctopo_debug.log";

std::mutex &log_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::ofstream &log_stream() {
  static std::ofstream stream;
  return stream;
}

struct LogResetState {
  bool initialized = false;
  std::string working_directory;
};

LogResetState &log_reset_state() {
  static LogResetState state;
  return state;
}

} // namespace

void reset_log_file() {
  std::lock_guard<std::mutex> lock(log_mutex());
  std::error_code ec;
  const std::string cwd = std::filesystem::current_path(ec).string();
  auto &state = log_reset_state();
  if (state.initialized && !ec && state.working_directory == cwd)
    return;

  auto &log_file = log_stream();
  if (log_file.is_open())
    log_file.close();
  log_file.open(kDebugLogPath, std::ios::out | std::ios::trunc);
  state.initialized = true;
  state.working_directory = ec ? std::string{} : cwd;
}

void append_log_line(const std::string &message) {
  std::lock_guard<std::mutex> lock(log_mutex());
  auto &log_file = log_stream();
  if (!log_file.is_open())
    log_file.open(kDebugLogPath, std::ios::out | std::ios::app);
  if (!log_file)
    return;
  log_file << message << '\n';
  log_file.flush();
}

} // namespace titan_pbctopo
