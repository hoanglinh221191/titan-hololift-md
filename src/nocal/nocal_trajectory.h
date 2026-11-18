#ifndef TITAN_NOCAL_TRAJECTORY_H
#define TITAN_NOCAL_TRAJECTORY_H

#include <cstddef>
#include <string>

namespace titan_nocal {

// The input being analysed, as the analyses name their files.
struct Trajectory {
  std::size_t index{};        // 0-based input number
  std::string folder{"."};    // its own folder with -opdbsep/-split, else "."
  std::string stem;           // input file name without folder and extension
  bool first_in_folder{true}; // no earlier input of the run wrote here

  [[nodiscard]] std::string file(const std::string &suffix) const {
    return folder + "/" + stem + suffix;
  }
};

} // namespace titan_nocal

#endif
