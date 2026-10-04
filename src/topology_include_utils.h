#ifndef TOPOLOGY_INCLUDE_UTILS_H
#define TOPOLOGY_INCLUDE_UTILS_H

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace TitanTopo {

inline std::string trim_copy(std::string_view text) {
  const size_t first = text.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos)
    return {};
  const size_t last = text.find_last_not_of(" \t\r\n");
  return std::string(text.substr(first, last - first + 1));
}

inline std::optional<std::string> extract_include_target(std::string_view line) {
  const size_t include_pos = line.find("#include");
  if (include_pos == std::string_view::npos)
    return std::nullopt;

  const size_t quote_open = line.find('"', include_pos);
  if (quote_open != std::string_view::npos) {
    const size_t quote_close = line.find('"', quote_open + 1);
    if (quote_close != std::string_view::npos && quote_close > quote_open + 1) {
      return trim_copy(line.substr(quote_open + 1,
                                   quote_close - quote_open - 1));
    }
  }

  const size_t angle_open = line.find('<', include_pos);
  if (angle_open != std::string_view::npos) {
    const size_t angle_close = line.find('>', angle_open + 1);
    if (angle_close != std::string_view::npos && angle_close > angle_open + 1) {
      return trim_copy(line.substr(angle_open + 1,
                                   angle_close - angle_open - 1));
    }
  }

  return std::nullopt;
}

inline std::vector<std::filesystem::path> gromacs_library_include_roots() {
  namespace fs = std::filesystem;

  std::vector<fs::path> roots;
  const char *raw_gmxlib = std::getenv("GMXLIB");
  if (raw_gmxlib == nullptr || raw_gmxlib[0] == '\0')
    return roots;

#if defined(_WIN32)
  constexpr char path_list_separator = ';';
#else
  constexpr char path_list_separator = ':';
#endif

  const std::string_view value(raw_gmxlib);
  std::size_t begin = 0;
  while (begin <= value.size()) {
    const std::size_t end = value.find(path_list_separator, begin);
    const std::string root = trim_copy(value.substr(
        begin,
        end == std::string_view::npos ? std::string_view::npos : end - begin));
    if (!root.empty()) {
      const fs::path normalized = fs::path(root).lexically_normal();
      if (std::find(roots.begin(), roots.end(), normalized) == roots.end())
        roots.emplace_back(normalized);
    }
    if (end == std::string_view::npos)
      break;
    begin = end + 1;
  }
  return roots;
}

inline std::vector<std::filesystem::path>
build_include_candidates(std::string_view current_file,
                         std::string_view include_target) {
  namespace fs = std::filesystem;

  std::vector<fs::path> candidates;
  const std::string trimmed_target = trim_copy(include_target);
  if (trimmed_target.empty())
    return candidates;

  const fs::path include_path(trimmed_target);
  auto append_unique = [&](const fs::path &candidate) {
    if (candidate.empty())
      return;
    const fs::path normalized = candidate.lexically_normal();
    if (std::find(candidates.begin(), candidates.end(), normalized) ==
        candidates.end()) {
      candidates.emplace_back(normalized);
    }
  };

  if (include_path.is_absolute()) {
    append_unique(include_path);
    return candidates;
  }

  const fs::path current_path(trim_copy(current_file));
  if (!current_path.empty() && current_path.has_parent_path()) {
    append_unique(current_path.parent_path() / include_path);
  }
  for (const fs::path &library_root : gromacs_library_include_roots())
    append_unique(library_root / include_path);
  append_unique(include_path);
  return candidates;
}

} // namespace TitanTopo

#endif
