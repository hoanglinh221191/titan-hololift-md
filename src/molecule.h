#ifndef PROTEIN_H
#define PROTEIN_H
#include "uti_func.h"
#include <cmath>
#include <string>
#include <utility>

using namespace std;
class molecule {
public:
  molecule() : x{}, y{}, z{} {}
  template <typename T2>
  molecule(T2 &&_x, T2 &&_y, T2 &&_z)
      : x{std::forward<T2>(_x)}, y{std::forward<T2>(_y)},
        z{std::forward<T2>(_z)} {}
  molecule(const molecule &RHS) : x{RHS.x}, y{RHS.y}, z{RHS.z} {}
  molecule(molecule &&RHS) noexcept
      : x{std::move(RHS.x)}, y{std::move(RHS.y)}, z{std::move(RHS.z)} {}
  molecule &operator=(const molecule &rhs) noexcept {
    x = rhs.x;
    y = rhs.y;
    z = rhs.z;
    return *this;
  }
  molecule &operator=(molecule &&rhs) noexcept {
    x = std::move(rhs.x);
    y = std::move(rhs.y);
    z = std::move(rhs.z);
    return *this;
  }
  ~molecule() = default;
  template <typename T> void set_x(T &&X) { x = std::forward<T>(X); }
  const double get_x() const { return x; }
  template <typename T> void set_y(T &&Y) { y = std::forward<T>(Y); }
  const double get_y() const { return y; }
  template <typename T> void set_z(T &&Z) { z = std::forward<T>(Z); }
  const double get_z() const { return z; }
  template <typename T1, typename T2, typename T3>
  void set_coor(T1 &&_x, T2 &&_y, T3 &&_z) {
    x = std::forward<T1>(_x);
    y = std::forward<T2>(_y);
    z = std::forward<T3>(_z);
  }
  inline const double dis_2_atoms(const molecule &rhs) const {
    return sqrt(square(x - rhs.x) + square(y - rhs.y) + square(z - rhs.z));
  }

private:
  double x{}, y{}, z{};
};
#endif