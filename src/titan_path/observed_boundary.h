#pragma once
#include "titan_path/continuity.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace titan_path {
// A measured current-time boundary is not an earlier native selector status.
// The wrapper authenticates its hash chain; this parser enforces semantic binding.
inline PreviousRouteState read_observed_boundary(
    std::istream &input, const std::string &route_id,
    const std::string &route_sha, const std::string &reference_sha,
    double total, double time) {
  std::map<std::string, std::string> f;
  std::string line;
  while (std::getline(input, line)) {
    if (line.size() > 4096 || f.size() >= 12)
      throw std::invalid_argument("OBSERVED_BOUNDARY_OVERSIZED");
    if (!line.empty() && line.back() == '\r') line.pop_back();
    auto eq = line.find('=');
    if (eq == std::string::npos || !f.emplace(line.substr(0, eq), line.substr(eq+1)).second)
      throw std::invalid_argument("OBSERVED_BOUNDARY_FIELDS");
  }
  if (input.bad() || f.size() != 10 || f["SCHEMA"] != "TITAN_OBSERVED_BOUNDARY_V1" ||
      f["STATE"] != "OBSERVATION_VALID" || f["ROUTE_ID"] != route_id ||
      f["ROUTE_FILE_SHA256"] != route_sha || f["REFERENCE_SHA256"] != reference_sha)
    throw std::invalid_argument("OBSERVED_BOUNDARY_IDENTITY");
  const auto hash = f["OBSERVATION_SHA256"];
  if (hash.size() != 64 || !std::all_of(hash.begin(), hash.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }))
    throw std::invalid_argument("OBSERVED_BOUNDARY_HASH");
  auto number = [&](const char *key) {
    std::size_t end = 0;
    const double value = std::stod(f.at(key), &end);
    if (end != f.at(key).size() || !std::isfinite(value))
      throw std::invalid_argument("OBSERVED_BOUNDARY_NUMBER");
    return value;
  };
  PreviousRouteState result;
  if (f["LAST_APPLIED_MODE"] == "Advance") result.mode = RouteMode::Advance;
  else if (f["LAST_APPLIED_MODE"] == "Recovery") result.mode = RouteMode::Recovery;
  else throw std::invalid_argument("OBSERVED_BOUNDARY_MODE");
  result.progress_angstrom = number("OBSERVED_PROGRESS_A");
  result.snapshot_time_ps = number("TIME_PS");
  if (!std::isfinite(total) || total <= 0 || !std::isfinite(time) || time < 0 ||
      std::abs(number("PATH_TOTAL_A")-total) > 1e-6 ||
      std::abs(result.snapshot_time_ps-time) > 1e-8 ||
      result.progress_angstrom < 0 || result.progress_angstrom > total)
    throw std::invalid_argument("OBSERVED_BOUNDARY_RANGE_TIME");
  return result;
}
} // namespace titan_path
