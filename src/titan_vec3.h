#ifndef TITAN_VEC3_H
#define TITAN_VEC3_H

namespace titan_geom {

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

inline Vec3 operator+(const Vec3 &lhs, const Vec3 &rhs) noexcept {
  return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

inline Vec3 operator-(const Vec3 &lhs, const Vec3 &rhs) noexcept {
  return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

inline Vec3 operator*(const Vec3 &lhs, double rhs) noexcept {
  return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
}

inline double dot(const Vec3 &lhs, const Vec3 &rhs) noexcept {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

inline Vec3 cross(const Vec3 &lhs, const Vec3 &rhs) noexcept {
  return {lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z,
          lhs.x * rhs.y - lhs.y * rhs.x};
}

inline double norm2(const Vec3 &value) noexcept { return dot(value, value); }

} // namespace titan_geom

#endif // TITAN_VEC3_H
