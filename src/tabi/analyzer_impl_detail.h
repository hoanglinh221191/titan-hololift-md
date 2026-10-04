#pragma once

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

#include "titan_executor.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace TitanPBE::analyzer_impl_detail {

inline constexpr double kPi = 3.141592653589793238462643383279502884;
inline constexpr double kCoulombKcalPerMol = 332.063713299;
inline constexpr double kKcalToKJ = 4.184;
inline constexpr double kKJToKcal = 1.0 / kKcalToKJ;
inline constexpr double kKBKJPerMolK = 0.00831446261815324;
inline constexpr double kKBKcalPerMolK = kKBKJPerMolK * kKJToKcal;

struct ProcessMemorySample {
  std::size_t working_set_bytes{0};
  std::size_t peak_working_set_bytes{0};
};

inline ProcessMemorySample sample_process_memory() noexcept {
  ProcessMemorySample sample;
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters{};
  counters.cb = sizeof(counters);
  if (K32GetProcessMemoryInfo(GetCurrentProcess(), &counters,
                              sizeof(counters))) {
    sample.working_set_bytes =
        static_cast<std::size_t>(counters.WorkingSetSize);
    sample.peak_working_set_bytes =
        static_cast<std::size_t>(counters.PeakWorkingSetSize);
  }
#else
  FILE *file = std::fopen("/proc/self/status", "r");
  if (file != nullptr) {
    char line[256];
    char label[64];
    long value_kb = 0;
    while (std::fgets(line, sizeof(line), file) != nullptr) {
      if (std::sscanf(line, "%63s %ld", label, &value_kb) != 2)
        continue;
      if (std::string(label) == "VmRSS:")
        sample.working_set_bytes =
            static_cast<std::size_t>(std::max<long>(0, value_kb)) * 1024u;
      else if (std::string(label) == "VmHWM:")
        sample.peak_working_set_bytes =
            static_cast<std::size_t>(std::max<long>(0, value_kb)) * 1024u;
    }
    std::fclose(file);
  }
#endif
  return sample;
}

inline std::size_t idx3(std::size_t i, std::size_t j, std::size_t k, std::size_t nx,
                 std::size_t ny) noexcept {
  return (k * ny + j) * nx + i;
}

inline std::size_t effective_workers(const TitanExecutor &executor,
                              std::size_t work_items) noexcept {
  if (work_items == 0)
    return 1;
  return std::max<std::size_t>(
      1, std::min<std::size_t>(executor.capacity(), work_items));
}

inline std::string format_fixed(double value, int digits = 6) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(digits) << value;
  return oss.str();
}

inline double elapsed_ms(std::chrono::steady_clock::time_point begin,
                  std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

inline std::string csv_escape(const std::string &value) {
  if (value.find_first_of(",\"\n\r") == std::string::npos)
    return value;
  std::string out = "\"";
  for (char ch : value) {
    if (ch == '"')
      out += "\"\"";
    else
      out += ch;
  }
  out += '"';
  return out;
}

inline bool parse_double_token(const std::string &token, double &value) {
  try {
    std::size_t used = 0;
    value = std::stod(token, &used);
    return used == token.size() && std::isfinite(value);
  } catch (...) {
    return false;
  }
}

inline std::string trim_copy(const std::string &value) {
  std::size_t first = 0;
  while (first < value.size() &&
         std::isspace(static_cast<unsigned char>(value[first])))
    ++first;
  std::size_t last = value.size();
  while (last > first &&
         std::isspace(static_cast<unsigned char>(value[last - 1])))
    --last;
  std::string out = value.substr(first, last - first);
  if (out.size() >= 2 && out.front() == '"' && out.back() == '"')
    out = out.substr(1, out.size() - 2);
  return out;
}

inline std::string lowercase_copy(std::string value) {
  for (char &ch : value)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return value;
}

inline std::vector<std::string> split_csv_simple(const std::string &line) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char ch = line[i];
    if (ch == '"') {
      if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
        field.push_back('"');
        ++i;
      } else {
        quoted = !quoted;
      }
    } else if (ch == ',' && !quoted) {
      fields.push_back(trim_copy(field));
      field.clear();
    } else {
      field.push_back(ch);
    }
  }
  fields.push_back(trim_copy(field));
  return fields;
}

inline std::unordered_map<std::string, std::size_t>
csv_header_map(const std::vector<std::string> &header) {
  std::unordered_map<std::string, std::size_t> map;
  for (std::size_t i = 0; i < header.size(); ++i)
    map[lowercase_copy(header[i])] = i;
  return map;
}

inline bool csv_double(const std::vector<std::string> &fields,
                const std::unordered_map<std::string, std::size_t> &header,
                const std::vector<std::string> &names, double &value) {
  for (const std::string &name : names) {
    const auto it = header.find(lowercase_copy(name));
    if (it != header.end() && it->second < fields.size() &&
        parse_double_token(fields[it->second], value))
      return true;
  }
  return false;
}

inline bool csv_index(const std::vector<std::string> &fields,
               const std::unordered_map<std::string, std::size_t> &header,
               const std::string &name, std::size_t &value) {
  double parsed = 0.0;
  if (!csv_double(fields, header, {name}, parsed) || parsed < 0.0)
    return false;
  value = static_cast<std::size_t>(std::llround(parsed));
  return true;
}

inline bool surface_uses_face_refinement(const std::string &surface) noexcept {
  return surface == "delphi-ses" || surface == "delphi-vdw" ||
         surface == "gaussian" || surface == "gaussian-cutoff";
}

} // namespace TitanPBE::analyzer_impl_detail
