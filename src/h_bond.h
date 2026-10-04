#ifndef H_BONDS_H
#define H_BONDS_H

#include <cstddef>
#include <vector>

// ---------------------------------------------------------------------------
// h_bonds — single hydrogen bond record
// ---------------------------------------------------------------------------
class h_bonds {
  std::size_t index_donor{};
  std::size_t index_acceptor{};
  std::size_t index_h{};
  double angle{};
  double dis{};

public:
  // --- Rule of Zero: compiler-generated copy/move/dtor are sufficient ---

  h_bonds() = default;

  h_bonds(std::size_t _index_donor, std::size_t _index_acceptor,
          std::size_t _index_h, double _angle, double _dis)
      : index_donor{_index_donor}, index_acceptor{_index_acceptor},
        index_h{_index_h}, angle{_angle}, dis{_dis} {}

  // --- Getters (nodiscard, const) ---

  [[nodiscard]] std::size_t get_index_donor() const { return index_donor; }
  [[nodiscard]] std::size_t get_index_acceptor() const { return index_acceptor; }
  [[nodiscard]] std::size_t get_index_h() const { return index_h; }
  [[nodiscard]] double get_angle() const { return angle; }
  [[nodiscard]] double get_dis() const { return dis; }

  // --- Setters (by value) ---

  void set_index_donor(std::size_t i) { index_donor = i; }
  void set_index_acceptor(std::size_t i) { index_acceptor = i; }
  void set_index_h(std::size_t i) { index_h = i; }
  void set_angle(double a) { angle = a; }
  void set_dis(double d) { dis = d; }
};

// ---------------------------------------------------------------------------
// h_bonds_wrapper — collection of hydrogen bonds
// ---------------------------------------------------------------------------
struct h_bonds_wrapper {
  std::vector<h_bonds> dat;

  h_bonds_wrapper() { dat.reserve(200); }

  // Rule of Zero: compiler-generated copy/move/dtor handle std::vector correctly
};

#endif // H_BONDS_H