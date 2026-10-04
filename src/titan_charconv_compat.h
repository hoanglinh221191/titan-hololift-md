#ifndef TITAN_CHARCONV_COMPAT_H
#define TITAN_CHARCONV_COMPAT_H

#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <string>
#include <system_error>
#include <type_traits>

namespace titan {

template <typename T, typename std::enable_if_t<std::is_integral_v<T>, int> = 0>
inline std::from_chars_result from_chars_compat(const char *first,
                                                const char *last,
                                                T &value) {
  return std::from_chars(first, last, value);
}

template <typename T,
          typename std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
inline std::from_chars_result from_chars_compat(const char *first,
                                                const char *last,
                                                T &value) {
#if defined(_LIBCPP_VERSION)
  std::string buffer(first, last);
  char *parse_end = nullptr;
  errno = 0;

  if constexpr (std::is_same_v<T, float>) {
    value = std::strtof(buffer.c_str(), &parse_end);
  } else if constexpr (std::is_same_v<T, double>) {
    value = std::strtod(buffer.c_str(), &parse_end);
  } else {
    value = std::strtold(buffer.c_str(), &parse_end);
  }

  if (parse_end == buffer.c_str())
    return {first, std::errc::invalid_argument};

  const auto advanced = static_cast<std::ptrdiff_t>(parse_end - buffer.c_str());
  if (errno == ERANGE)
    return {first + advanced, std::errc::result_out_of_range};
  return {first + advanced, std::errc{}};
#else
  return std::from_chars(first, last, value);
#endif
}

} // namespace titan

#endif // TITAN_CHARCONV_COMPAT_H
