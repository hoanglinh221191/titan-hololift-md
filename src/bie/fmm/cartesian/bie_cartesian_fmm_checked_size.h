#ifndef TITAN_BIE_CARTESIAN_FMM_CHECKED_SIZE_H
#define TITAN_BIE_CARTESIAN_FMM_CHECKED_SIZE_H

#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <string>
#include <string_view>

namespace titan::bie::fmm::cartesian::detail {

template <typename UInt>
[[nodiscard]] constexpr bool checked_add(UInt left, UInt right,
                                         UInt &result) noexcept {
  static_assert(std::numeric_limits<UInt>::is_integer);
  if (right > std::numeric_limits<UInt>::max() - left)
    return false;
  result = left + right;
  return true;
}

template <typename UInt>
[[nodiscard]] constexpr bool checked_multiply(UInt left, UInt right,
                                              UInt &result) noexcept {
  static_assert(std::numeric_limits<UInt>::is_integer);
  if (left != 0 && right > std::numeric_limits<UInt>::max() / left)
    return false;
  result = left * right;
  return true;
}

template <typename UInt>
[[nodiscard]] std::expected<UInt, std::string>
checked_add_or_error(UInt left, UInt right, std::string_view context) {
  UInt result{};
  if (!checked_add(left, right, result)) {
    return std::unexpected("BIE Cartesian full FMM size overflow while " +
                           std::string(context));
  }
  return result;
}

template <typename UInt>
[[nodiscard]] std::expected<UInt, std::string>
checked_multiply_or_error(UInt left, UInt right, std::string_view context) {
  UInt result{};
  if (!checked_multiply(left, right, result)) {
    return std::unexpected("BIE Cartesian full FMM size overflow while " +
                           std::string(context));
  }
  return result;
}

template <typename Container>
[[nodiscard]] std::expected<std::size_t, std::string>
checked_container_allocation(std::size_t elements, const Container &container,
                             std::string_view context) {
  if (elements > container.max_size()) {
    return std::unexpected(
        "BIE Cartesian full FMM allocation exceeds container limit while " +
        std::string(context));
  }
  return checked_multiply_or_error<std::size_t>(
      elements, sizeof(typename Container::value_type), context);
}

template <typename Container>
[[nodiscard]] std::expected<std::size_t, std::string>
checked_container_capacity_bytes(const Container &container,
                                 std::string_view context) {
  return checked_multiply_or_error<std::size_t>(
      container.capacity(), sizeof(typename Container::value_type), context);
}

} // namespace titan::bie::fmm::cartesian::detail

#endif // TITAN_BIE_CARTESIAN_FMM_CHECKED_SIZE_H
