// ============================================================================
// parse_xvg.cpp — High-performance GROMACS .xvg column parser
//
// Uses 1 MB buffered reads + std::from_chars pointer arithmetic.
// Skips comment lines ('#' / '@') immediately.
// Zero heap allocation per data line (no istringstream, no stod).
// ============================================================================
#include "parse_xvg.h"
#include "titan_charconv_compat.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <utility>

namespace {

// Skip whitespace, return pointer past spaces/tabs
inline const char* skip_ws(const char* p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t'))
        ++p;
    return p;
}

// Skip non-whitespace (advance past one token)
inline const char* skip_token(const char* p, const char* end) {
    while (p < end && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r')
        ++p;
    return p;
}

// Parse a double at the current position, advance pointer past it.
// Returns false if parsing fails.
inline bool parse_double_advance(const char*& p, const char* end, double& out) {
    p = skip_ws(p, end);
    if (p >= end) return false;
    auto [ptr, ec] = titan::from_chars_compat(p, end, out);
    if (ec != std::errc()) return false;
    p = ptr;
    return true;
}

} // anonymous namespace

namespace TitanIO {

std::expected<std::vector<double>, std::string>
parse_xvg_column(const std::string& filename, size_t target_column) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
        return std::unexpected("Cannot open .xvg file: " + filename);

    // Pre-allocate for typical trajectory size
    std::vector<double> result;
    result.reserve(100000);

    constexpr size_t BUF_SIZE = 1024 * 1024;  // 1 MB
    std::vector<char> buffer(BUF_SIZE);
    std::string remainder;
    remainder.reserve(512);

    while (file) {
        file.read(buffer.data(), static_cast<std::streamsize>(BUF_SIZE));
        std::streamsize bytes_read = file.gcount();
        if (bytes_read <= 0) break;

        const char* chunk = buffer.data();
        size_t start = 0;

        // Find newlines and process lines
        for (size_t pos = 0; pos < static_cast<size_t>(bytes_read); ++pos) {
            if (chunk[pos] != '\n') continue;

            // We have a line: [start, pos)
            std::string_view line_sv;
            std::string merged_line;

            if (!remainder.empty()) {
                merged_line = remainder;
                merged_line.append(chunk + start, pos - start);
                remainder.clear();
                line_sv = merged_line;
            } else {
                line_sv = std::string_view(chunk + start, pos - start);
            }

            start = pos + 1;

            // Strip trailing CR
            if (!line_sv.empty() && line_sv.back() == '\r')
                line_sv.remove_suffix(1);
            if (line_sv.empty()) continue;

            // Skip comment/metadata lines
            char first = line_sv.front();
            if (first == '#' || first == '@') continue;

            // Parse target column using pointer arithmetic
            const char* p = line_sv.data();
            const char* end = p + line_sv.size();

            // Skip columns before target
            for (size_t col = 0; col < target_column; ++col) {
                p = skip_ws(p, end);
                if (p >= end) goto next_line;
                p = skip_token(p, end);
            }

            // Parse target column value
            {
                double val = 0.0;
                if (parse_double_advance(p, end, val))
                    result.push_back(val);
            }

            next_line:;
        }

        // Save remainder (incomplete last line)
        if (start < static_cast<size_t>(bytes_read)) {
            remainder.assign(chunk + start,
                             static_cast<size_t>(bytes_read) - start);
        }
    }

    // Process final remainder
    if (!remainder.empty()) {
        // Strip trailing CR/LF
        while (!remainder.empty() &&
               (remainder.back() == '\r' || remainder.back() == '\n'))
            remainder.pop_back();

        if (!remainder.empty() && remainder.front() != '#' &&
            remainder.front() != '@') {
            const char* p = remainder.data();
            const char* end = p + remainder.size();

            for (size_t col = 0; col < target_column; ++col) {
                p = skip_ws(p, end);
                if (p >= end) goto done;
                p = skip_token(p, end);
            }
            {
                double val = 0.0;
                if (parse_double_advance(p, end, val))
                    result.push_back(val);
            }
        }
    }

done:
    return result;
}

std::expected<std::vector<std::pair<double, double>>, std::string>
parse_xvg_time_value(const std::string& filename,
                     size_t time_column, size_t value_column) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
        return std::unexpected("Cannot open .xvg file: " + filename);

    if (time_column > 15 || value_column > 15)
        return std::unexpected(
            "parse_xvg_time_value: column index > 15 not supported. "
            "Use parse_xvg_column() for single-column extraction instead.");

    size_t max_col = std::max(time_column, value_column);

    std::vector<std::pair<double, double>> result;
    result.reserve(100000);

    constexpr size_t BUF_SIZE = 1024 * 1024;
    std::vector<char> buffer(BUF_SIZE);
    std::string remainder;
    remainder.reserve(512);

    while (file) {
        file.read(buffer.data(), static_cast<std::streamsize>(BUF_SIZE));
        std::streamsize bytes_read = file.gcount();
        if (bytes_read <= 0) break;

        const char* chunk = buffer.data();
        size_t start = 0;

        for (size_t pos = 0; pos < static_cast<size_t>(bytes_read); ++pos) {
            if (chunk[pos] != '\n') continue;

            std::string_view line_sv;
            std::string merged_line;

            if (!remainder.empty()) {
                merged_line = remainder;
                merged_line.append(chunk + start, pos - start);
                remainder.clear();
                line_sv = merged_line;
            } else {
                line_sv = std::string_view(chunk + start, pos - start);
            }
            start = pos + 1;

            if (!line_sv.empty() && line_sv.back() == '\r')
                line_sv.remove_suffix(1);
            if (line_sv.empty()) continue;

            char first = line_sv.front();
            if (first == '#' || first == '@') continue;

            // Parse all columns up to max_col
            const char* p = line_sv.data();
            const char* end = p + line_sv.size();

            double cols[16] = {};  // stack buffer for up to 16 columns
            bool ok = true;
            for (size_t col = 0; col <= max_col && col < 16; ++col) {
                if (!parse_double_advance(p, end, cols[col])) {
                    ok = false;
                    break;
                }
            }
            if (ok && max_col < 16)
                result.emplace_back(cols[time_column], cols[value_column]);
        }

        if (start < static_cast<size_t>(bytes_read)) {
            remainder.assign(chunk + start,
                             static_cast<size_t>(bytes_read) - start);
        }
    }

    // Process final remainder
    if (!remainder.empty()) {
        while (!remainder.empty() &&
               (remainder.back() == '\r' || remainder.back() == '\n'))
            remainder.pop_back();

        if (!remainder.empty() && remainder.front() != '#' &&
            remainder.front() != '@') {
            const char* p = remainder.data();
            const char* end = p + remainder.size();

            double cols[16] = {};
            bool ok = true;
            for (size_t col = 0; col <= max_col && col < 16; ++col) {
                if (!parse_double_advance(p, end, cols[col])) {
                    ok = false;
                    break;
                }
            }
            if (ok && max_col < 16)
                result.emplace_back(cols[time_column], cols[value_column]);
        }
    }

    return result;
}

std::expected<std::vector<XvgTimeTwoValues>, std::string>
parse_xvg_time_two_values(const std::string& filename,
                          size_t time_column,
                          size_t value1_column,
                          size_t value2_column) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
        return std::unexpected("Cannot open .xvg file: " + filename);

    const size_t max_col =
        std::max(time_column, std::max(value1_column, value2_column));
    if (max_col > 15) {
        return std::unexpected(
            "parse_xvg_time_two_values: column index > 15 not supported.");
    }

    std::vector<XvgTimeTwoValues> result;
    result.reserve(100000);

    constexpr size_t BUF_SIZE = 1024 * 1024;
    std::vector<char> buffer(BUF_SIZE);
    std::string remainder;
    remainder.reserve(512);

    auto parse_line = [&](std::string_view line_sv) {
        if (!line_sv.empty() && line_sv.back() == '\r')
            line_sv.remove_suffix(1);
        if (line_sv.empty())
            return;
        const char first = line_sv.front();
        if (first == '#' || first == '@')
            return;

        const char* p = line_sv.data();
        const char* end = p + line_sv.size();
        double cols[16] = {};
        bool ok = true;
        for (size_t col = 0; col <= max_col; ++col) {
            if (!parse_double_advance(p, end, cols[col])) {
                ok = false;
                break;
            }
        }
        if (!ok)
            return;

        XvgTimeTwoValues row;
        row.time_ps = cols[time_column];
        row.value1 = cols[value1_column];
        row.value2 = cols[value2_column];
        result.push_back(row);
    };

    while (file) {
        file.read(buffer.data(), static_cast<std::streamsize>(BUF_SIZE));
        std::streamsize bytes_read = file.gcount();
        if (bytes_read <= 0)
            break;

        const char* chunk = buffer.data();
        size_t start = 0;
        for (size_t pos = 0; pos < static_cast<size_t>(bytes_read); ++pos) {
            if (chunk[pos] != '\n')
                continue;

            std::string_view line_sv;
            std::string merged_line;
            if (!remainder.empty()) {
                merged_line = remainder;
                merged_line.append(chunk + start, pos - start);
                remainder.clear();
                line_sv = merged_line;
            } else {
                line_sv = std::string_view(chunk + start, pos - start);
            }
            start = pos + 1;
            parse_line(line_sv);
        }

        if (start < static_cast<size_t>(bytes_read)) {
            remainder.assign(chunk + start,
                             static_cast<size_t>(bytes_read) - start);
        }
    }

    if (!remainder.empty()) {
        while (!remainder.empty() &&
               (remainder.back() == '\r' || remainder.back() == '\n')) {
            remainder.pop_back();
        }
        if (!remainder.empty())
            parse_line(remainder);
    }

    return result;
}

} // namespace TitanIO
