#ifndef INDEX_H
#define INDEX_H

#include <cassert>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

inline constexpr std::size_t N_const = 500;

// ---------------------------------------------------------------------------
// index_base — per-atom record
// ---------------------------------------------------------------------------
class index_base {
  std::string res_name, type;
  double x, y, z;
  std::size_t res_pos, res_num;

public:
  // --- Rule of Zero: compiler-generated copy/move/dtor are sufficient ---

  index_base()
      : res_name{}, type{"unknown"}, x{}, y{}, z{}, res_pos{}, res_num{} {}

  explicit index_base(std::size_t _res_pos)
      : res_name{}, type{}, x{}, y{}, z{}, res_pos{_res_pos}, res_num{} {}

  index_base(std::size_t _res_pos, std::size_t _res_num, double _x, double _y,
             double _z, std::string _res_name)
      : res_name{std::move(_res_name)}, type{}, x{_x}, y{_y}, z{_z},
        res_pos{_res_pos}, res_num{_res_num} {}

  // --- Setters ---

  void set_res_pos(std::size_t _res_pos) { res_pos = _res_pos; }
  void set_res_num(std::size_t _res_num) { res_num = _res_num; }

  void set_coor(double _x, double _y, double _z) {
    x = _x;
    y = _y;
    z = _z;
  }

  void set_res_name(std::string _res_name) { res_name = std::move(_res_name); }
  void set_type(std::string _type) { type = std::move(_type); }

  // --- Getters (nodiscard, const, return by const-ref for strings) ---

  [[nodiscard]] std::size_t get_res_pos() const { return res_pos; }
  [[nodiscard]] std::size_t get_res_num() const { return res_num; }
  [[nodiscard]] double get_x() const { return x; }
  [[nodiscard]] double get_y() const { return y; }
  [[nodiscard]] double get_z() const { return z; }
  [[nodiscard]] const std::string &get_res_name() const { return res_name; }
  [[nodiscard]] const std::string &get_type() const { return type; }
};

// ---------------------------------------------------------------------------
// index — per-residue container of index_base atoms
// ---------------------------------------------------------------------------
class index {
  std::vector<index_base> base;
  std::string chain{" "}, type{"protein"};
  std::size_t chainid{};

public:
  index() : chain{"NULL"}, type{"protein"} { base.reserve(N_const); }

  // Construct with chain + atoms
  index(std::string _chain, std::size_t _res_pos, std::size_t _res_num,
        double _x, double _y, double _z, std::string _res_name)
      : chain{std::move(_chain)}, type{"protein"} {
    base.emplace_back(_res_pos, _res_num, _x, _y, _z, std::move(_res_name));
  }

  index(std::string _chain, std::size_t _chainid, std::size_t _res_pos,
        std::size_t _res_num, double _x, double _y, double _z,
        std::string _res_name)
      : chain{std::move(_chain)}, type{"protein"}, chainid{_chainid} {
    base.emplace_back(_res_pos, _res_num, _x, _y, _z, std::move(_res_name));
  }

  void add_base(std::size_t _res_pos, std::size_t _res_num, double _x,
                double _y, double _z, std::string _res_name) {
    base.emplace_back(_res_pos, _res_num, _x, _y, _z, std::move(_res_name));
  }

  // Construct from string_view chain
  explicit index(std::string_view _chain) : chain{_chain}, type{"protein"} {}

  index(std::string_view _chain, std::size_t _chainid)
      : chain{_chain}, type{"protein"}, chainid{_chainid} {}

  index(std::string _chain, std::size_t _res_pos)
      : chain{std::move(_chain)}, type{"protein"} {
    base.emplace_back(_res_pos);
  }

  index(std::string _chain, std::size_t _chainid, std::size_t _res_pos)
      : chain{std::move(_chain)}, type{"protein"}, chainid{_chainid} {
    base.emplace_back(_res_pos);
  }

  // --- Getters (bounds-checked with .at()) ---

  [[nodiscard]] std::size_t get_res_pos_e(std::size_t i) const {
    return base.at(i).get_res_pos();
  }
  [[nodiscard]] std::size_t get_res_num_e(std::size_t i) const {
    return base.at(i).get_res_num();
  }
  [[nodiscard]] const std::string &get_chain() const { return chain; }
  [[nodiscard]] std::size_t get_chainid() const { return chainid; }
  [[nodiscard]] const std::string &get_res_name_e(std::size_t i) const {
    return base.at(i).get_res_name();
  }
  [[nodiscard]] double get_res_x_e(std::size_t i) const {
    return base.at(i).get_x();
  }
  [[nodiscard]] double get_res_y_e(std::size_t i) const {
    return base.at(i).get_y();
  }
  [[nodiscard]] double get_res_z_e(std::size_t i) const {
    return base.at(i).get_z();
  }
  [[nodiscard]] std::size_t get_size_v() const { return base.size(); }

  // --- Setters (bounds-checked with .at()) ---

  void set_res_pos_e(std::size_t i, std::size_t _res_pos) {
    base.at(i).set_res_pos(_res_pos);
  }
  void set_res_pos_e(std::size_t _res_pos) {
    base.emplace_back(_res_pos, 0u, 0.0, 0.0, 0.0, "NULL");
  }
  void set_res_num_e(std::size_t i, std::size_t _res_num) {
    base.at(i).set_res_num(_res_num);
  }
  void set_chain(std::string_view _chain) { chain = _chain; }
  void set_chainid(std::size_t _chainid) { chainid = _chainid; }
  void set_type(std::string_view _type) { type = _type; }

  void set_type_base(std::size_t i, std::string _type) {
    base.at(i).set_type(std::move(_type));
  }

  [[nodiscard]] const std::string &get_type() const { return type; }

  [[nodiscard]] int get_type_enum() const {
    if (type == "dna")
      return 0;
    else if (type == "rna")
      return 1;
    else if (type == "protein")
      return 2;
    else if (type == "complex")
      return 3;
    else if (type == "water")
      return 4;
    else if (type == "lipid")
      return 5;
    else
      return 6;
  }

  [[nodiscard]] const std::string &get_type_base(std::size_t i) const {
    static const std::string unknown_str{"unknown"};
    if (i < base.size())
      return base[i].get_type();
    else
      return unknown_str;
  }

  void set_res_name_e(std::size_t i, std::string _res_name) {
    base.at(i).set_res_name(std::move(_res_name));
  }

  void set_res_coor_e(std::size_t i, double _x, double _y, double _z) {
    base.at(i).set_coor(_x, _y, _z);
  }
};

struct index_molecule {
  std::vector<class index> index_mol;
};

#endif
