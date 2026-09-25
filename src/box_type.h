#ifndef BOX_TYPE_H
#define BOX_TYPE_H
#include "uti_func.h"
#include <cmath>
#include <string>

const double pi = 3.14159265358979;
class box {
private:
  double a{}, b{}, c{};
  double alpha{}, beta{}, gamma{};
  double v1[3]{}, v2[3]{}, v3[3]{};
  std::string box_type;

public:
  box()
      : a{}, b{}, c{}, alpha{90.0}, beta{90.0}, gamma{90.0},
        box_type{"unknown"} {};
  template <typename T>
  box(T &&_a, T &&_b, T &&_c, T &&_alpha, T &&_beta, T &&_gamma)
      : a{std::forward<T>(_a)}, b{std::forward<T>(_b)}, c{std::forward<T>(_c)},
        alpha{std::forward<T>(_alpha)}, beta{std::forward<T>(_beta)},
        gamma{std::forward<T>(_gamma)} {
    setv();
  }
  box(const box &rhs)
      : a{rhs.a}, b{rhs.b}, c{rhs.c}, alpha{rhs.alpha}, beta{rhs.beta},
        gamma{rhs.gamma} {
    setv();
  }
  box(box &&rhs)
      : a{std::move(rhs.a)}, b{std::move(rhs.b)}, c{std::move(rhs.c)},
        alpha{std::move(rhs.alpha)}, beta{std::move(rhs.beta)},
        gamma{std::move(rhs.gamma)} {
    setv();
  }
  box &operator=(const box &box_) {
    this->a = box_.a;
    this->b = box_.b;
    this->c = box_.c;
    this->alpha = box_.alpha;
    this->beta = box_.beta;
    this->gamma = box_.gamma;
    this->setv();
    return *this;
  }
  box &operator=(box &&box_) {
    this->a = std::move(box_.a);
    this->b = std::move(box_.b);
    this->c = std::move(box_.c);
    this->alpha = std::move(box_.alpha);
    this->beta = std::move(box_.beta);
    this->gamma = std::move(box_.gamma);
    this->setv();
    return *this;
  }
  ~box() = default;
  void set_a(double _x) {
    a = _x;
    setv();
  }
  double get_a() const { return a; }
  void set_b(double _y) {
    b = _y;
    setv();
  }
  double get_b() const { return b; }
  void set_c(double _z) {
    c = _z;
    setv();
  }
  double get_c() const { return c; }
  void set_alpha(double _alpha) {
    alpha = _alpha;
    setv();
  }
  double get_alpha() const { return alpha; }
  void set_beta(double _beta) {
    beta = _beta;
    setv();
  }
  double get_beta() const { return beta; }
  void set_gamma(double _gamma) {
    gamma = _gamma;
    setv();
  }
  double get_gamma() const { return gamma; }
  void set_box_type(std::string _box) { box_type = std::move(_box); }
  std::string get_box_type() { return box_type; }
  void get_v1(double *rhs) {
    rhs[0] = v1[0];
    rhs[1] = v1[1];
    rhs[2] = v1[2];
  }
  void get_v2(double *rhs) {
    rhs[0] = v2[0];
    rhs[1] = v2[1];
    rhs[2] = v2[2];
  }
  void get_v3(double *rhs) {
    rhs[0] = v3[0];
    rhs[1] = v3[1];
    rhs[2] = v3[2];
  }
  bool is_orthorhombic() const noexcept {
    constexpr double angle_tolerance = 1e-8;
    return std::abs(alpha - 90.0) <= angle_tolerance &&
           std::abs(beta - 90.0) <= angle_tolerance &&
           std::abs(gamma - 90.0) <= angle_tolerance;
  }
  void setv() {
    for (int i = 0; i < 3; ++i) v1[i] = v2[i] = v3[i] = 0.0;
    box_type = "unknown";
    if (!(std::isfinite(a) && std::isfinite(b) && std::isfinite(c) &&
          a > 0.0 && b > 0.0 && c > 0.0 && std::isfinite(alpha) &&
          std::isfinite(beta) && std::isfinite(gamma) &&
          alpha > 0.0 && alpha < 180.0 && beta > 0.0 && beta < 180.0 &&
          gamma > 0.0 && gamma < 180.0)) return;
    if (is_orthorhombic()) {
      box_type = a == b && b == c ? "cubic" : "rectangular";
      v1[0] = a; v2[1] = b; v3[2] = c;
      return;
    }
    const double ca = std::cos(alpha * pi / 180.0);
    const double cb = std::cos(beta * pi / 180.0);
    const double cg = std::cos(gamma * pi / 180.0);
    const double sg = std::sin(gamma * pi / 180.0);
    const double gram = 1.0 + 2.0 * ca * cb * cg - ca*ca - cb*cb - cg*cg;
    if (!(gram > 0.0) || !(sg > 0.0)) return;
    // Any valid non-orthorhombic cell needs general lattice vectors.
    box_type = "triclinic";
    if (std::abs(alpha-60.0) <= 1e-8 && std::abs(beta-60.0) <= 1e-8 &&
        std::abs(gamma-90.0) <= 1e-8) box_type = "dodecahedron";
    v1[0] = a;
    v2[0] = b * cg; v2[1] = b * sg;
    v3[0] = c * cb; v3[1] = c * (ca - cb*cg) / sg;
    v3[2] = c * std::sqrt(gram) / sg;
  }
  double volume() const {
    return v1[0] * v2[1] * v3[2];
  }

  int convert_to_v(double _a, double _b, double _c, double *_v1, double *_v2,
                   double *_v3) {
    box converted(_a, _b, _c, alpha, beta, gamma);
    converted.get_v1(_v1); converted.get_v2(_v2); converted.get_v3(_v3);
    return converted.get_box_type() == "unknown" ? 1 : 0;
  }
  // FixBox (Baptista et al. 2022, Eq. 2-3): Cartesian → scaled coordinates
  // s = B⁻¹ · r  where B = [v1|v2|v3] is upper-triangular.
  // For orthorhombic: s[i] = r[i]/L[i], identical to original comparisons.
  void to_scaled(const double *r, double *s) const {
    double bx = v2[0], by = v2[1];
    double cx = v3[0], cy = v3[1], cz = v3[2];
    if (cz == 0.0 || by == 0.0 || v1[0] == 0.0) {
      s[0] = s[1] = s[2] = 0.0; // degenerate box — no-op
      return;
    }
    s[2] = r[2] / cz;
    s[1] = (r[1] - cy * s[2]) / by;
    s[0] = (r[0] - bx * s[1] - cx * s[2]) / v1[0];
  }
};
#endif
