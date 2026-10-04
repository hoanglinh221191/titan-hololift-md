// ============================================================================
// parse_xvg.h — High-performance GROMACS .xvg column parser
//
// Uses std::from_chars (zero-allocation) following the same pattern as
// fast_parse_double() in parse_pdb.cpp.
// ============================================================================
#pragma once

#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace TitanIO {

struct XvgTimeTwoValues {
  double time_ps{0.0};
  double value1{0.0};
  double value2{0.0};
};

// Extracts a specific column (0-indexed) from a GROMACS .xvg file.
// Lines starting with '#' or '@' are skipped.
// For DHAM, pass target_column = 1 (Potential Energy).
std::expected<std::vector<double>, std::string>
parse_xvg_column(const std::string& filename, size_t target_column = 1);

// Extracts two columns (time + value) from a .xvg file.
// Returns vector of {time_ps, value} pairs.
std::expected<std::vector<std::pair<double, double>>, std::string>
parse_xvg_time_value(const std::string& filename,
                     size_t time_column = 0, size_t value_column = 1);

// Extracts three columns (time + two values) from a .xvg file.
// Returns vector of {time_ps, value1, value2} rows.
std::expected<std::vector<XvgTimeTwoValues>, std::string>
parse_xvg_time_two_values(const std::string& filename,
                          size_t time_column,
                          size_t value1_column,
                          size_t value2_column);

} // namespace TitanIO
