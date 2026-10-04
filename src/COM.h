#ifndef COM_H
#define COM_H

#include <cstddef>
#include <string>
#include <utility>

// ---------------------------------------------------------------------------
// com — Center of Mass record for a residue / atom group
// ---------------------------------------------------------------------------
class com {
  double x{0.0}, y{0.0}, z{0.0};
  std::size_t res_num{0}, atom_num{}, chainid{}, respos{};
  std::string chain{" "}, res_name{"UND"};

public:
  // --- Rule of Zero: compiler-generated copy/move/dtor are sufficient ---

  com() = default;

  com(double _x, double _y, double _z, std::string _chain,
      std::string _res_name, std::size_t _res_num, std::size_t _atom_num,
      std::size_t _chainid, std::size_t _respos)
      : x{_x}, y{_y}, z{_z}, res_num{_res_num}, atom_num{_atom_num},
        chainid{_chainid}, respos{_respos}, chain{std::move(_chain)},
        res_name{std::move(_res_name)} {}

  // --- Setters ---

  void set_x(double x_temp) { x = x_temp; }
  void set_y(double y_temp) { y = y_temp; }
  void set_z(double z_temp) { z = z_temp; }
  void set_chain(std::string c) { chain = std::move(c); }
  void set_res_name(std::string r) { res_name = std::move(r); }
  void set_res_num(std::size_t num) { res_num = num; }
  void set_atom_num(std::size_t num) { atom_num = num; }
  void set_chainid(std::size_t id) { chainid = id; }
  void set_respos(std::size_t pos) { respos = pos; }

  // --- Getters (nodiscard, const) ---

  [[nodiscard]] double get_x() const { return x; }
  [[nodiscard]] double get_y() const { return y; }
  [[nodiscard]] double get_z() const { return z; }
  [[nodiscard]] const std::string &get_chain() const { return chain; }
  [[nodiscard]] const std::string &get_res_name() const { return res_name; }
  [[nodiscard]] std::size_t get_res_num() const { return res_num; }
  [[nodiscard]] std::size_t get_atom_num() const { return atom_num; }
  [[nodiscard]] std::size_t get_chainid() const { return chainid; }
  [[nodiscard]] std::size_t get_respos() const { return respos; }
};

#endif // COM_H