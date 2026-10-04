#ifndef ALIGNED_ALLOCATOR_H
#define ALIGNED_ALLOCATOR_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#include <malloc.h>
#endif

inline constexpr std::size_t kTitanSimdAlignment = 32;

template <typename T, std::size_t Alignment = kTitanSimdAlignment>
struct AlignedAllocator {
  static_assert(Alignment >= alignof(T),
                "AlignedAllocator alignment must satisfy alignof(T).");
  static_assert((Alignment & (Alignment - 1)) == 0,
                "AlignedAllocator alignment must be a power of two.");
  static_assert(Alignment % alignof(T) == 0,
                "AlignedAllocator alignment must be a multiple of alignof(T).");

  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using is_always_equal = std::true_type;
  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;

  template <class U> struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };

  AlignedAllocator() noexcept = default;

  template <typename U>
  constexpr AlignedAllocator(const AlignedAllocator<U, Alignment> &) noexcept {}

  [[nodiscard]] T *allocate(std::size_t n) {
    if (n == 0)
      return nullptr;
    if (n > max_size())
      throw std::bad_alloc();

    const std::size_t bytes = n * sizeof(T);
    void *ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(bytes, Alignment);
#else
    const std::size_t padded_bytes =
        ((bytes + Alignment - 1) / Alignment) * Alignment;
    ptr = std::aligned_alloc(Alignment, padded_bytes);
#endif
    if (!ptr)
      throw std::bad_alloc();
    return static_cast<T *>(ptr);
  }

  void deallocate(T *p, std::size_t) noexcept {
    if (!p)
      return;
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
  }

  [[nodiscard]] constexpr std::size_t max_size() const noexcept {
    return std::numeric_limits<std::size_t>::max() / sizeof(T);
  }

  template <typename U>
  bool operator==(const AlignedAllocator<U, Alignment> &) const noexcept {
    return true;
  }

  template <typename U>
  bool operator!=(const AlignedAllocator<U, Alignment> &) const noexcept {
    return false;
  }
};

template <typename T, std::size_t Alignment = kTitanSimdAlignment>
using AlignedVector = std::vector<T, AlignedAllocator<T, Alignment>>;

using AlignedDoubleVector = AlignedVector<double, kTitanSimdAlignment>;
using AlignedIntVector = AlignedVector<int32_t, kTitanSimdAlignment>;

#endif
