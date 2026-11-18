#ifndef MOL_INFO_H
#define MOL_INFO_H

#include <cctype>
#include <string>
#include <string_view>
#include <utility>

// KHÔNG BAO GIỜ dùng "using namespace std;" trong file header (.h)
// để tránh hiện tượng Namespace Pollution (ô nhiễm không gian tên) ở các file
// include nó.

// Backbone atoms of an amino-acid residue in the naming schemes TITAN reads:
// the amide H as H (AMBER, GROMOS, OPLS) or HN (CHARMM), the N-terminal
// H1-H3 or HT1-HT3, and the C-terminal oxygens OXT (PDB, AMBER), OT1/OT2
// (CHARMM) or OC1/OC2 (GROMACS ports). Every other atom is side chain.
inline bool is_amino_acid_backbone_atom(std::string_view atom) {
  return atom == "N" || atom == "CA" || atom == "C" || atom == "O" ||
         atom == "H" || atom == "HN" || atom == "HA" || atom == "H1" ||
         atom == "H2" || atom == "H3" || atom == "HT1" || atom == "HT2" ||
         atom == "HT3" || atom == "OXT" || atom == "OT1" || atom == "OT2" ||
         atom == "OC1" || atom == "OC2";
}

// Glycine has no side chain, so its whole residue stands in for one. AMBER
// names the terminal forms NGLY and CGLY.
inline bool is_glycine_residue_name(std::string_view res_name) {
  return res_name.starts_with("GLY") || res_name == "NGLY" ||
         res_name == "CGLY";
}

class mol_info {
private: // OPT: Explicit private access modifier
  // OPT: Memory Alignment & Padding - Reordered members from largest size to
  // smallest string -> size_t -> bool
  std::string atom_type;
  std::string res_name;
  std::string original_res_name;
  std::string atom;
  std::string chain;
  size_t resnr;
  size_t chainnum;
  size_t respos;
  size_t internal_resnr;
  size_t original_atom_id;
  bool original_atom_id_set;
  bool side_chain;

public:
  // OPT: Member initializer list ordered to match declaration order avoiding
  // padding
  mol_info()
      : atom_type{"NULL"}, res_name{"UND"}, original_res_name{"UND"},
        atom{"NULL"}, chain{" "}, resnr{0}, chainnum{0}, respos{0},
        internal_resnr{0}, original_atom_id{0}, original_atom_id_set{false},
        side_chain{false} {}

  // OPT: Constructor using std::string_view for zero-copy string references
  mol_info(size_t _resnr, std::string_view _atom_type, std::string_view _atom,
           std::string_view _chain, std::string_view _res_name)
      : atom_type{_atom_type}, res_name{_res_name},
        original_res_name{_res_name}, atom{_atom}, chain{_chain}, resnr{_resnr},
        chainnum{0}, respos{0}, internal_resnr{0}, original_atom_id{0},
        original_atom_id_set{false}, side_chain{true} {}

  // OPT: Rule of Zero - Deleted manual destructor since compiler-generated
  // default is optimal
  ~mol_info() = default;

  // --- Getters & Setters ---

  void set_resnr(size_t res) noexcept {
    resnr = res;
  } // OPT: Added noexcept for basic setters
  size_t get_resnr() const noexcept { return resnr; } // OPT: Added noexcept

  void set_respos(size_t res) noexcept { respos = res; }
  size_t get_respos() const noexcept { return respos; }

  void set_internal_resnr(size_t res) noexcept { internal_resnr = res; }
  size_t get_internal_resnr() const noexcept { return internal_resnr; }

  void set_original_atom_id(size_t id) noexcept {
    original_atom_id = id;
    original_atom_id_set = true;
  }
  size_t get_original_atom_id() const noexcept { return original_atom_id; }
  bool has_original_atom_id() const noexcept { return original_atom_id_set; }

  void set_chainnum(size_t num) noexcept { chainnum = num; }
  size_t get_chainnum() const noexcept { return chainnum; }

  void set_atom_type(std::string_view type) { atom_type = type; }
  void set_chain(std::string_view cha) { chain = cha; }
  void set_res_name(std::string_view res_Name) { res_name = res_Name; }
  void set_atom(std::string_view ato) { atom = ato; }

  // OPT: Return const std::string& to prevent unintended string copies
  const std::string &get_atom_type() const noexcept { return atom_type; }
  const std::string &get_chain() const noexcept { return chain; }
  const std::string &get_res_name() const noexcept { return res_name; }

  std::string get_original_res_name() const {
    if (original_res_name != "HISE")
      return original_res_name;
    return "HIS";
  }

  std::string get_atom() const {
    if (atom.length() >= 4 && std::isdigit(atom[0]))
      return atom.substr(1) + atom.substr(0, 1);
    return atom;
  }

  bool is_side_chain() const noexcept {
    return side_chain;
  } // OPT: Const correctness and noexcept

  void set_side_chain(bool val) noexcept { side_chain = val; }

  // For an atom of an amino-acid residue.
  bool check_side_chain() {
    side_chain = is_glycine_residue_name(original_res_name) ||
                 !is_amino_acid_backbone_atom(atom);
    return side_chain;
  }
};

#endif
