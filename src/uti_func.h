#ifndef UTI_FUNCTION_H
#define UTI_FUNCTION_H
#include <cmath>
#include <utility>

template <typename T>
inline double square(T &&base) {
  long double t{base * base};
  return t;
}

template <typename T>
inline double cube(T &&base) {
  return square(std::forward<T>(base)) * base;
}

template <typename T>
inline double c6(T &&base) {
  long double temp{cube(base * base)};
  return temp;
}

template <typename T>
inline double c12(T &&base) {
  long double temp{c6(std::forward<T>(base))};
  return temp * temp;
}

template <typename T1, typename T2, typename T3>
inline double norm3(T1 &&x, T2 &&y, T3 &&z) {
  return std::sqrt(square(std::forward<T1>(x)) +
                   square(std::forward<T2>(y)) +
                   square(std::forward<T3>(z)));
}
#endif
