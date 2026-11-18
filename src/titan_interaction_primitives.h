#ifndef TITAN_INTERACTION_PRIMITIVES_H
#define TITAN_INTERACTION_PRIMITIVES_H

#include "BoxMatrix.h"
#include "MOL.h"
#include "MOL_info.h"
#include "index_v2.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace titan_interaction {

// Minimum image for the H-bond, salt-bridge and water-bridge searches: a
// rectangular box folds each axis by rounding, any other valid box goes
// through BoxMatrix::apply_mic.
class PeriodicImage {
public:
  explicit PeriodicImage(const box &frame_box);
  void apply(double &dx, double &dy, double &dz) const noexcept;

private:
  bool rectangular_{false};
  double length_[3]{};
  double inverse_[3]{};
  BoxMatrix matrix_;
};

// Hydrogen when the atom name, after leading digits, starts with H (HN, HA,
// 1HB, HH11), as GROMACS decides it; NH1, NH2, OH, CH3 are heavy atoms. The
// force-field type decides only for an atom without a name.
bool is_hydrogen_atom(const mol_info &info);
bool same_residue(const std::vector<mol_info> &info, std::size_t lhs,
                  std::size_t rhs);

std::pair<std::size_t, std::size_t>
find_residue_full_range(const std::vector<class index> &Index,
                        std::size_t chainid, std::size_t target_resnr,
                        std::size_t mol_total_atoms);

std::pair<std::size_t, std::size_t>
find_residue_span_by_atom(const std::vector<mol_info> &info,
                          std::size_t atom_idx);

bool is_hbond_donor(const std::string &atom_name,
                    const std::string &res_name);
bool is_hbond_acceptor(const std::string &atom_name,
                       const std::string &res_name);
bool is_acidic(const std::string &atom_name, const std::string &res_name);
bool is_basic(const std::string &atom_name, const std::string &res_name);

// First hydrogen of the residue within 1.2 A of the donor (never the donor).
std::size_t find_bonded_hydrogen(const MOL &mol,
                                 const std::vector<mol_info> &info,
                                 std::size_t donor_idx,
                                 std::size_t res_start,
                                 std::size_t res_end);

// Every hydrogen of the residue within 1.2 A of the donor (minimum image
// when `image` is given), in atom order.
void find_bonded_hydrogens(const MOL &mol, const std::vector<mol_info> &info,
                           std::size_t donor_idx, std::size_t res_start,
                           std::size_t res_end, const PeriodicImage *image,
                           std::vector<std::size_t> &hydrogens);

bool try_calculate_angle_deg(const MOL &donor_mol, std::size_t donor_idx,
                             std::size_t hydrogen_idx,
                             const MOL &acceptor_mol,
                             std::size_t acceptor_idx, double cos_cutoff,
                             double &angle_deg) noexcept;

// The same D-H...A angle with the D-H and A-H vectors minimum-imaged.
bool try_calculate_angle_deg(const MOL &donor_mol, std::size_t donor_idx,
                             std::size_t hydrogen_idx,
                             const MOL &acceptor_mol,
                             std::size_t acceptor_idx, double cos_cutoff,
                             const PeriodicImage &image,
                             double &angle_deg) noexcept;

} // namespace titan_interaction

#endif // TITAN_INTERACTION_PRIMITIVES_H
