#include "gbsa_intrin.h"
#include "gbsa/gb_intramolecular_delta.h"
#include "gbsa/gb_topology_pairs.h"
#include "charmm_nonbonded.h"
#include "gromos_nonbonded.h"
#include "titan_lj_pair_override.h"
#include "titan_electrostatics.h"
#include "titan_gb_model.h"
#include "titan_gbn2.h"
#include "titan_gbnsr6.h"
#include "titan_interaction_entropy.h"
#include "titan_executor.h"
#include "titan_simd_dispatch.h"
#include "titan_avx2_math.h"
#include "titan_sasa.h"
#include "titan_ui.h"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <span>
#include <tuple>
#include <sstream>

#ifdef __AVX__
#include <immintrin.h>

static inline __m256d charmm_lj_energy_gbsa_avx(__m256d sigma_vec,
                                                __m256d epsilon_vec,
                                                __m256d distance_vec) {
  const __m256d avx_zero = _mm256_setzero_pd();
  const __m256d avx_four = _mm256_set1_pd(4.0);
  const __m256d sigma_valid = _mm256_cmp_pd(sigma_vec, avx_zero, _CMP_GT_OQ);
  const __m256d epsilon_valid =
      _mm256_cmp_pd(epsilon_vec, avx_zero, _CMP_NEQ_OQ);
  const __m256d valid_mask = _mm256_and_pd(sigma_valid, epsilon_valid);

  const __m256d sr = _mm256_div_pd(sigma_vec, distance_vec);
  const __m256d sr2 = _mm256_mul_pd(sr, sr);
  const __m256d sr6 = _mm256_mul_pd(_mm256_mul_pd(sr2, sr2), sr2);
  const __m256d sr12 = _mm256_mul_pd(sr6, sr6);
  __m256d energy = _mm256_mul_pd(
      _mm256_mul_pd(avx_four, epsilon_vec), _mm256_sub_pd(sr12, sr6));
  return _mm256_and_pd(energy, valid_mask);
}

namespace {

char first_alpha_upper(const std::string &value) {
  for (const unsigned char c : value) {
    if (std::isalpha(c) != 0)
      return static_cast<char>(std::toupper(c));
  }
  return '\0';
}

char gb_element(const mol_info &atom) {
  std::string name = atom.get_atom();
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::toupper(c));
                 });
  name.erase(std::remove_if(name.begin(), name.end(), [](unsigned char c) {
               return std::isalnum(c) == 0;
             }),
             name.end());

  if (name.starts_with("CL"))
    return 'L';
  if (name.starts_with("BR"))
    return 'B';
  return first_alpha_upper(name.empty() ? atom.get_atom_type() : name);
}

constexpr double kMaximumHydrogenBondLengthA = 1.55;

// Per-atom element symbols plus a uniform grid over the heavy atoms of one
// molecule.
//
// The previous implementation recomputed gb_element() - which builds and
// normalises a std::string - for every candidate of every hydrogen while
// scanning all atoms, i.e. O(N_H * N) string operations, and
// build_gb_atom_parameters() runs once per endpoint per frame. The element
// table plus a grid whose cell edge is at least the 1.55 A bond cutoff return
// the identical partner in O(N + N_H * k).
//
// Result equivalence is deliberate, not incidental: the original ascending
// scan compared with a strict "<" against a running best seeded at the cutoff,
// so an atom exactly at 1.55 A never qualifies and the lowest atom index wins
// an exact distance tie. Both rules are reproduced explicitly below because a
// grid visits candidates in an arbitrary order.
class GbGeometryIndex {
public:
  GbGeometryIndex(const MOL &mol, const std::vector<mol_info> &atom_info) {
    count_ = std::min(mol.size(), atom_info.size());
    element_.resize(count_);
    for (size_t atom = 0; atom < count_; ++atom)
      element_[atom] = gb_element(atom_info[atom]);
    // ENE-17(d) (22/10/2026): residues carrying an OXT, found once; igb 8
    // used to rescan every atom for each backbone O (O(N_res * N)).
    std::set<std::tuple<size_t, size_t, std::string>> with_oxt;
    for (const auto &other : atom_info)
      if (other.get_atom() == "OXT")
        with_oxt.emplace(other.get_resnr(), other.get_chainnum(), other.get_chain());
    residue_has_oxt_.resize(atom_info.size(), 0);
    if (!with_oxt.empty())
      for (size_t atom = 0; atom < atom_info.size(); ++atom)
        residue_has_oxt_[atom] = with_oxt.contains({atom_info[atom].get_resnr(),
                                                    atom_info[atom].get_chainnum(),
                                                    atom_info[atom].get_chain()});
    build_grid(mol);
  }

  char element(size_t atom) const {
    return atom < count_ ? element_[atom] : '\0';
  }

  bool residue_has_oxt(size_t atom) const {
    return atom < residue_has_oxt_.size() && residue_has_oxt_[atom] != 0;
  }

  char nearest_heavy_element(const MOL &mol, size_t atom_index) const {
    if (atom_index >= count_ || heavy_.empty())
      return '\0';
    const double qx = mol.pos.x[atom_index];
    const double qy = mol.pos.y[atom_index];
    const double qz = mol.pos.z[atom_index];
    if (!std::isfinite(qx) || !std::isfinite(qy) || !std::isfinite(qz))
      return '\0';

    constexpr double kCutoff2 =
        kMaximumHydrogenBondLengthA * kMaximumHydrogenBondLengthA;
    double best2 = kCutoff2;
    size_t best = std::numeric_limits<size_t>::max();
    const auto consider = [&](size_t other) {
      if (other == atom_index)
        return;
      const double dx = qx - mol.pos.x[other];
      const double dy = qy - mol.pos.y[other];
      const double dz = qz - mol.pos.z[other];
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (!std::isfinite(r2) || !(r2 < kCutoff2))
        return;
      if (r2 < best2 || (r2 == best2 && other < best)) {
        best2 = r2;
        best = other;
      }
    };

    if (cells_valid_) {
      const size_t cx = cell_of(qx, min_x_, nx_);
      const size_t cy = cell_of(qy, min_y_, ny_);
      const size_t cz = cell_of(qz, min_z_, nz_);
      const size_t x0 = cx > 0 ? cx - 1 : 0;
      const size_t y0 = cy > 0 ? cy - 1 : 0;
      const size_t z0 = cz > 0 ? cz - 1 : 0;
      const size_t x1 = std::min(cx + 1, nx_ - 1);
      const size_t y1 = std::min(cy + 1, ny_ - 1);
      const size_t z1 = std::min(cz + 1, nz_ - 1);
      for (size_t ix = x0; ix <= x1; ++ix) {
        for (size_t iy = y0; iy <= y1; ++iy) {
          for (size_t iz = z0; iz <= z1; ++iz) {
            const size_t cell = (ix * ny_ + iy) * nz_ + iz;
            for (size_t slot = cell_start_[cell]; slot < cell_start_[cell + 1];
                 ++slot)
              consider(cell_items_[slot]);
          }
        }
      }
    } else {
      for (const size_t other : heavy_)
        consider(other);
    }

    return best == std::numeric_limits<size_t>::max() ? '\0' : element_[best];
  }

private:
  size_t cell_of(double value, double origin, size_t extent) const {
    const double scaled = std::floor((value - origin) / cell_edge_);
    if (!(scaled > 0.0))
      return 0;
    if (scaled >= static_cast<double>(extent))
      return extent - 1;
    return static_cast<size_t>(scaled);
  }

  static size_t axis_extent(double span, double cell_edge) {
    if (!std::isfinite(span) || span <= 0.0)
      return 1;
    const double cells = std::floor(span / cell_edge) + 1.0;
    return cells >= 1.0 ? static_cast<size_t>(cells) : 1;
  }

  void build_grid(const MOL &mol) {
    heavy_.reserve(count_);
    for (size_t atom = 0; atom < count_; ++atom) {
      if (element_[atom] == 'H')
        continue;
      if (!std::isfinite(mol.pos.x[atom]) || !std::isfinite(mol.pos.y[atom]) ||
          !std::isfinite(mol.pos.z[atom]))
        continue;
      heavy_.push_back(atom);
    }
    if (heavy_.empty())
      return;

    min_x_ = max_x_ = mol.pos.x[heavy_.front()];
    min_y_ = max_y_ = mol.pos.y[heavy_.front()];
    min_z_ = max_z_ = mol.pos.z[heavy_.front()];
    for (const size_t atom : heavy_) {
      min_x_ = std::min(min_x_, mol.pos.x[atom]);
      max_x_ = std::max(max_x_, mol.pos.x[atom]);
      min_y_ = std::min(min_y_, mol.pos.y[atom]);
      max_y_ = std::max(max_y_, mol.pos.y[atom]);
      min_z_ = std::min(min_z_, mol.pos.z[atom]);
      max_z_ = std::max(max_z_, mol.pos.z[atom]);
    }

    // Grow the cell edge instead of clamping the cell count: a cell edge below
    // the cutoff would break the 27-cell neighbourhood, an edge above it stays
    // correct and only costs a few extra candidates.
    constexpr double kMaxCells = 4.0e6;
    cell_edge_ = kMaximumHydrogenBondLengthA;
    for (int attempt = 0; attempt < 64; ++attempt) {
      nx_ = axis_extent(max_x_ - min_x_, cell_edge_);
      ny_ = axis_extent(max_y_ - min_y_, cell_edge_);
      nz_ = axis_extent(max_z_ - min_z_, cell_edge_);
      const double total = static_cast<double>(nx_) *
                           static_cast<double>(ny_) * static_cast<double>(nz_);
      if (total <= kMaxCells) {
        cells_valid_ = true;
        break;
      }
      cell_edge_ *= 2.0;
    }
    if (!cells_valid_)
      return; // degenerate extent: keep the linear fallback over heavy_

    const size_t cells = nx_ * ny_ * nz_;
    cell_start_.assign(cells + 1, 0);
    for (const size_t atom : heavy_)
      ++cell_start_[cell_for(mol, atom) + 1];
    for (size_t cell = 0; cell < cells; ++cell)
      cell_start_[cell + 1] += cell_start_[cell];
    cell_items_.resize(heavy_.size());
    std::vector<size_t> cursor(cell_start_.begin(), cell_start_.end() - 1);
    // heavy_ is ascending, so each bucket keeps ascending atom indices.
    for (const size_t atom : heavy_)
      cell_items_[cursor[cell_for(mol, atom)]++] = atom;
  }

  size_t cell_for(const MOL &mol, size_t atom) const {
    const size_t ix = cell_of(mol.pos.x[atom], min_x_, nx_);
    const size_t iy = cell_of(mol.pos.y[atom], min_y_, ny_);
    const size_t iz = cell_of(mol.pos.z[atom], min_z_, nz_);
    return (ix * ny_ + iy) * nz_ + iz;
  }

  size_t count_{0};
  std::vector<char> element_;
  std::vector<char> residue_has_oxt_;
  std::vector<size_t> heavy_;
  std::vector<size_t> cell_start_;
  std::vector<size_t> cell_items_;
  double min_x_{0.0}, max_x_{0.0};
  double min_y_{0.0}, max_y_{0.0};
  double min_z_{0.0}, max_z_{0.0};
  double cell_edge_{kMaximumHydrogenBondLengthA};
  size_t nx_{1}, ny_{1}, nz_{1};
  bool cells_valid_{false};
};

char hydrogen_partner_from_atom_type(const mol_info &atom) {
  std::string type = atom.get_atom_type();
  std::transform(type.begin(), type.end(), type.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::toupper(c));
                 });
  type.erase(std::remove_if(type.begin(), type.end(), [](unsigned char c) {
               return std::isalnum(c) == 0;
             }),
             type.end());
  if (type == "H" || type.starts_with("HN"))
    return 'N';
  if (type.starts_with("HO") || type.starts_with("HW"))
    return 'O';
  if (type.starts_with("HS"))
    return 'S';
  if (!type.empty() && type.front() == 'H')
    return 'C';
  return '\0';
}

double gb_intrinsic_radius_angstrom(const MOL &mol,
                                    const std::vector<mol_info> &atom_info,
                                    const GbGeometryIndex &index,
                                    size_t atom_index, int igb) {
  if (atom_index >= atom_info.size())
    return 1.50;
  const char element = index.element(atom_index);
  if (igb == 66 && element == 'H') return 1.20; // Bondi surface radii
  if (igb == 8) {
    const auto &a = atom_info[atom_index];
    const std::string name = a.get_atom();
    const auto &res = a.get_res_name();
    // mbondi3 changes relative to mbondi2. O2 is Amber carboxylate oxygen.
    if (element == 'O' &&
        ((res == "ASP" && name.starts_with("OD")) ||
         (res == "GLU" && name.starts_with("OE")) ||
         name == "OXT" || name == "OT1" || name == "OT2" ||
         a.get_atom_type() == "O2")) return 1.4;
    if (element == 'O' && name == "O" && index.residue_has_oxt(atom_index))
      return 1.4;
    if (element == 'H' && res == "ARG" &&
        (name.starts_with("HH") || name.starts_with("HE"))) return 1.17;
  }
  if (element != 'H') {
    if (element == 'L')
      return 1.70; // chlorine
    if (element == 'B')
      return 1.85; // bromine
    return titan_gb::standard_heavy_atom_radius_angstrom(element);
  }

  char partner = index.nearest_heavy_element(mol, atom_index);
  if (partner == '\0')
    partner = hydrogen_partner_from_atom_type(atom_info[atom_index]);
  if (igb == 1) {
    if (partner == 'C' || partner == 'N')
      return 1.30;
    if (partner == 'O' || partner == 'S')
      return 0.80;
    return 1.20;
  }
  return partner == 'N' ? 1.30 : 1.20;
}

// CAS marks deleted side-chain atoms with this exact radius (A). They keep a
// slot in the arrays but take no part in descreening or surface area, so the
// mutant behaves as if they were absent. (The ALPB size weights atoms by r^3,
// where the sentinel is already negligible.)
bool is_cas_deleted_radius(double radius) {
  return std::abs(radius - 0.001) <= 1.0e-9;
}

struct GbAtomParameters {
  std::vector<double> intrinsic_radius_a;
  std::vector<double> offset_radius_a;
  std::vector<double> scaled_radius_a;
  std::vector<titan_gb::Gbn2Parameters> gbn2;
  std::vector<bool> active;
  std::vector<double> active_mask; // 1 for active, 0 for CAS-deleted atoms
  // GBn2 neck table index of the intrinsic radius, lround((r - 1) * 20), or
  // -1 when r is outside [1, 2] A (titan_gb::gbn2_neck_integral gives NaN).
  std::vector<long long> neck_index;
};

GbAtomParameters build_gb_atom_parameters(
    const MOL &mol, const AlignedDoubleVector &input_vdw_radii,
    const std::vector<mol_info> &atom_info, int igb) {
  GbAtomParameters result;
  result.intrinsic_radius_a.resize(mol.size());
  result.offset_radius_a.resize(mol.size());
  result.scaled_radius_a.resize(mol.size());
  result.gbn2.resize(mol.size());
  result.active.resize(mol.size());
  result.active_mask.resize(mol.size());
  result.neck_index.resize(mol.size());
  const GbGeometryIndex index(mol, atom_info);
  for (size_t atom = 0; atom < mol.size(); ++atom) {
    const double intrinsic =
        gb_intrinsic_radius_angstrom(mol, atom_info, index, atom, igb);
    const double offset = intrinsic - (igb == 8 ? titan_gb::kGbn2OffsetA
                                               : titan_gb::kDielectricOffsetAngstrom);
    result.intrinsic_radius_a[atom] = intrinsic;
    result.offset_radius_a[atom] = offset;
    result.neck_index[atom] =
        std::isfinite(intrinsic) && intrinsic >= 1.0 && intrinsic <= 2.0
            ? std::lround((intrinsic - 1.0) * 20.0)
            : -1;

    // CAS-deleted atoms retain a finite target Born radius but must not
    // descreen neighbours.
    const bool inactive = atom < input_vdw_radii.size() &&
                          is_cas_deleted_radius(input_vdw_radii[atom]);
    const char element = index.element(atom);
    result.active[atom] = !inactive;
    result.active_mask[atom] = inactive ? 0.0 : 1.0;
    result.gbn2[atom] = titan_gb::gbn2_parameters(element,
        atom < atom_info.size() && titan_gb::gbn2_nucleic_residue(atom_info[atom].get_res_name()));
    result.scaled_radius_a[atom] =
        inactive ? 0.0
                 : offset * (igb == 8 ? result.gbn2[atom].screen
                                      : titan_gb::screening_factor_for_element(element));
  }
  return result;
}

#if TITAN_SIMD_X86
// The shell integral of titan_gb::hct_pair_integral_inverse_angstrom for the
// lanes in `act`, whose r and s are valid (other lanes hold 1). Sets
// `contributes` to the lanes with a non-empty shell; the result means
// something only there (the scalar function returns 0 for the others).
TITAN_TARGET_AVX2
inline __m256d hct_shell_avx2(__m256d r, __m256d s_safe, __m256d act,
                              __m256d rho_v, __m256d inv_rho,
                              __m256d rgbmax_v, bool cutoff_active,
                              __m256d &contributes) noexcept {
  const __m256d one = _mm256_set1_pd(1.0);
  const __m256d quarter = _mm256_set1_pd(0.25);
  const __m256d half = _mm256_set1_pd(0.5);
  const __m256d sign = _mm256_set1_pd(-0.0);
  __m256d upper = _mm256_add_pd(r, s_safe);
  if (cutoff_active)
    upper = _mm256_blendv_pd(
        upper, rgbmax_v, _mm256_cmp_pd(upper, rgbmax_v, _CMP_GT_OQ));
  const __m256d lower = _mm256_max_pd(
      rho_v, _mm256_andnot_pd(sign, _mm256_sub_pd(r, s_safe)));
  contributes = _mm256_and_pd(
      act, _mm256_and_pd(_mm256_cmp_pd(rho_v, upper, _CMP_LT_OQ),
                         _mm256_cmp_pd(lower, upper, _CMP_LT_OQ)));
  if (_mm256_movemask_pd(contributes) == 0)
    return _mm256_setzero_pd();
  const __m256d L = _mm256_blendv_pd(one, lower, contributes);
  const __m256d U = _mm256_blendv_pd(one, upper, contributes);
  const __m256d inv_r = _mm256_div_pd(one, r);
  const __m256d lower_inv = _mm256_div_pd(one, L);
  const __m256d upper_inv = _mm256_div_pd(one, U);
  const __m256d lower_inv2 = _mm256_mul_pd(lower_inv, lower_inv);
  const __m256d upper_inv2 = _mm256_mul_pd(upper_inv, upper_inv);
  const __m256d log_term =
      titan_avx2::log_pd(_mm256_mul_pd(upper_inv, L));
  __m256d term = _mm256_sub_pd(lower_inv, upper_inv);
  term = _mm256_add_pd(
      term, _mm256_mul_pd(_mm256_mul_pd(quarter, r),
                          _mm256_sub_pd(upper_inv2, lower_inv2)));
  term = _mm256_add_pd(term,
                       _mm256_mul_pd(_mm256_mul_pd(half, inv_r), log_term));
  term = _mm256_add_pd(
      term,
      _mm256_mul_pd(
          _mm256_mul_pd(_mm256_mul_pd(quarter, _mm256_mul_pd(s_safe, s_safe)),
                        inv_r),
          _mm256_sub_pd(lower_inv2, upper_inv2)));
  // Target atom completely inside the source: Ponder's correction.
  const __m256d inside =
      _mm256_cmp_pd(rho_v, _mm256_sub_pd(s_safe, r), _CMP_LT_OQ);
  term = _mm256_add_pd(
      term, _mm256_and_pd(inside, _mm256_mul_pd(_mm256_set1_pd(2.0),
                                                _mm256_sub_pd(inv_rho,
                                                              lower_inv))));
  return _mm256_mul_pd(half, term);
}

// HCT descreening of one target atom by sources [0, count) of one molecule,
// four at a time, added into four lane sums. Each lane follows
// titan_gb::hct_pair_integral_inverse_angstrom: masks stand for its
// branches (self pair, CAS-deleted source, rgbmax reach, U <= rho, L >= U,
// target inside the source) and the accurate AVX2 log replaces std::log.
// Returns the first source index left to the scalar loop (a multiple of 4),
// or SIZE_MAX when a lane meets a pair the scalar code rejects, in which case
// the atom's radius is NaN.
TITAN_TARGET_AVX2
size_t hct_sources_avx2(double xi, double yi, double zi, double rho,
                        const double *x, const double *y, const double *z,
                        const double *scaled, const double *active,
                        size_t count, size_t self, double rgbmax,
                        bool cutoff_active, double *lanes) noexcept {
  const __m256d one = _mm256_set1_pd(1.0);
  const __m256d zero = _mm256_setzero_pd();
  const __m256d infinity =
      _mm256_set1_pd(std::numeric_limits<double>::infinity());
  const __m256d xi_v = _mm256_set1_pd(xi);
  const __m256d yi_v = _mm256_set1_pd(yi);
  const __m256d zi_v = _mm256_set1_pd(zi);
  const __m256d rho_v = _mm256_set1_pd(rho);
  const __m256d inv_rho = _mm256_set1_pd(1.0 / rho);
  const __m256d rgbmax_v = _mm256_set1_pd(rgbmax);
  const __m256d sign = _mm256_set1_pd(-0.0);
  const __m256i lane_index = _mm256_set_epi64x(3, 2, 1, 0);
  const __m256i self_v = _mm256_set1_epi64x(static_cast<long long>(self));
  __m256d sum = _mm256_load_pd(lanes);
  const size_t vector_end = count & ~static_cast<size_t>(3);
  for (size_t j = 0; j < vector_end; j += 4) {
    const __m256d dx = _mm256_sub_pd(xi_v, _mm256_loadu_pd(x + j));
    const __m256d dy = _mm256_sub_pd(yi_v, _mm256_loadu_pd(y + j));
    const __m256d dz = _mm256_sub_pd(zi_v, _mm256_loadu_pd(z + j));
    const __m256d d2 = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
        _mm256_mul_pd(dz, dz));
    const __m256i index = _mm256_add_epi64(
        _mm256_set1_epi64x(static_cast<long long>(j)), lane_index);
    const __m256d not_self = _mm256_castsi256_pd(_mm256_xor_si256(
        _mm256_cmpeq_epi64(index, self_v), _mm256_set1_epi64x(-1)));
    __m256d act = _mm256_and_pd(
        not_self, _mm256_cmp_pd(_mm256_loadu_pd(active + j), zero,
                                _CMP_NEQ_UQ));
    if (_mm256_movemask_pd(act) == 0)
      continue;
    const __m256d s = _mm256_loadu_pd(scaled + j);
    // As in the scalar loop: a non-finite or zero distance to any active
    // source is rejected before the cutoff, a non-positive or non-finite
    // descreening radius only for a pair the integral actually evaluates.
    const __m256d good_distance =
        _mm256_and_pd(_mm256_cmp_pd(d2, zero, _CMP_GT_OQ),
                      _mm256_cmp_pd(d2, infinity, _CMP_LT_OQ));
    if (_mm256_movemask_pd(_mm256_andnot_pd(good_distance, act)) != 0)
      return static_cast<size_t>(-1);
    if (cutoff_active) {
      const __m256d reach = _mm256_add_pd(rgbmax_v, s);
      act = _mm256_and_pd(
          act, _mm256_cmp_pd(d2, _mm256_mul_pd(reach, reach), _CMP_LT_OQ));
    }
    const __m256d good_radius =
        _mm256_and_pd(_mm256_cmp_pd(s, zero, _CMP_GT_OQ),
                      _mm256_cmp_pd(s, infinity, _CMP_LT_OQ));
    if (_mm256_movemask_pd(_mm256_andnot_pd(good_radius, act)) != 0)
      return static_cast<size_t>(-1);
    const __m256d r = _mm256_sqrt_pd(_mm256_blendv_pd(one, d2, act));
    const __m256d s_safe = _mm256_blendv_pd(one, s, act);
    __m256d contributes;
    const __m256d contribution = hct_shell_avx2(
        r, s_safe, act, rho_v, inv_rho, rgbmax_v, cutoff_active, contributes);
    if (_mm256_movemask_pd(contributes) == 0)
      continue;
    const __m256d finite = _mm256_cmp_pd(_mm256_andnot_pd(sign, contribution),
                                         infinity, _CMP_LT_OQ);
    if (_mm256_movemask_pd(_mm256_andnot_pd(finite, contributes)) != 0)
      return static_cast<size_t>(-1);
    sum = _mm256_add_pd(sum, _mm256_and_pd(contributes, contribution));
  }
  _mm256_store_pd(lanes, sum);
  return vector_end;
}

// GBn2 (igb 8) descreening of one target atom, as hct_sources_avx2 and with
// the scalar loop's choices per lane: Amber's far-field series where
// r > 4 s and the shell stays inside the cutoff (this also covers sulfur's
// negative s), the HCT shell integral otherwise, and the neck term of
// titan_gb::gbn2_neck_integral on every pair inside the cutoff. ri and
// target_neck are the target's intrinsic radius and neck index, rj and
// neck_index the sources' (-1 = outside [1, 2] A, a NaN neck term).
TITAN_TARGET_AVX2
size_t gbn2_sources_avx2(double xi, double yi, double zi, double rho,
                         double ri, long long target_neck, const double *x,
                         const double *y, const double *z,
                         const double *scaled, const double *active,
                         const double *rj, const long long *neck_index,
                         size_t count, size_t self, double rgbmax,
                         bool cutoff_active, double *lanes) noexcept {
  const __m256d one = _mm256_set1_pd(1.0);
  const __m256d zero = _mm256_setzero_pd();
  const __m256d infinity =
      _mm256_set1_pd(std::numeric_limits<double>::infinity());
  const __m256d xi_v = _mm256_set1_pd(xi);
  const __m256d yi_v = _mm256_set1_pd(yi);
  const __m256d zi_v = _mm256_set1_pd(zi);
  const __m256d rho_v = _mm256_set1_pd(rho);
  const __m256d inv_rho = _mm256_set1_pd(1.0 / rho);
  const __m256d rgbmax_v = _mm256_set1_pd(rgbmax);
  const __m256d ri_v = _mm256_set1_pd(ri);
  const __m256d sign = _mm256_set1_pd(-0.0);
  const __m256i lane_index = _mm256_set_epi64x(3, 2, 1, 0);
  const __m256i self_v = _mm256_set1_epi64x(static_cast<long long>(self));
  const double *d0 = titan_gb::neck_d0.data();
  const double *m0 = titan_gb::neck_m0.data();
  __m256d sum = _mm256_load_pd(lanes);
  const size_t vector_end = count & ~static_cast<size_t>(3);
  for (size_t j = 0; j < vector_end; j += 4) {
    const __m256d dx = _mm256_sub_pd(xi_v, _mm256_loadu_pd(x + j));
    const __m256d dy = _mm256_sub_pd(yi_v, _mm256_loadu_pd(y + j));
    const __m256d dz = _mm256_sub_pd(zi_v, _mm256_loadu_pd(z + j));
    const __m256d d2 = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
        _mm256_mul_pd(dz, dz));
    const __m256i index = _mm256_add_epi64(
        _mm256_set1_epi64x(static_cast<long long>(j)), lane_index);
    const __m256d not_self = _mm256_castsi256_pd(_mm256_xor_si256(
        _mm256_cmpeq_epi64(index, self_v), _mm256_set1_epi64x(-1)));
    __m256d act = _mm256_and_pd(
        not_self, _mm256_cmp_pd(_mm256_loadu_pd(active + j), zero,
                                _CMP_NEQ_UQ));
    if (_mm256_movemask_pd(act) == 0)
      continue;
    const __m256d s = _mm256_loadu_pd(scaled + j);
    const __m256d good_distance =
        _mm256_and_pd(_mm256_cmp_pd(d2, zero, _CMP_GT_OQ),
                      _mm256_cmp_pd(d2, infinity, _CMP_LT_OQ));
    if (_mm256_movemask_pd(_mm256_andnot_pd(good_distance, act)) != 0)
      return static_cast<size_t>(-1);
    if (cutoff_active) {
      const __m256d reach = _mm256_add_pd(rgbmax_v, s);
      act = _mm256_and_pd(
          act, _mm256_cmp_pd(d2, _mm256_mul_pd(reach, reach), _CMP_LT_OQ));
    }
    const int act_bits = _mm256_movemask_pd(act);
    if (act_bits == 0)
      continue;
    // Every pair inside the cutoff takes a neck term: an out-of-table
    // radius on either side makes it NaN, and the atom's radius with it.
    const long long neck[4] = {neck_index[j], neck_index[j + 1],
                               neck_index[j + 2], neck_index[j + 3]};
    bool neck_ok = target_neck >= 0;
    for (int lane = 0; lane < 4; ++lane)
      if ((act_bits >> lane) & 1)
        neck_ok = neck_ok && neck[lane] >= 0;
    if (!neck_ok)
      return static_cast<size_t>(-1);

    const __m256d r = _mm256_sqrt_pd(_mm256_blendv_pd(one, d2, act));
    const __m256d s_safe = _mm256_blendv_pd(one, s, act);
    __m256d far = _mm256_and_pd(
        act, _mm256_cmp_pd(r, _mm256_mul_pd(_mm256_set1_pd(4.0), s_safe),
                           _CMP_GT_OQ));
    if (cutoff_active)
      far = _mm256_and_pd(far, _mm256_cmp_pd(r, _mm256_sub_pd(rgbmax_v, s_safe),
                                             _CMP_LE_OQ));
    const __m256d shell = _mm256_andnot_pd(far, act);
    __m256d base = zero;
    if (_mm256_movemask_pd(shell) != 0) {
      // The shell integral rejects a non-positive or non-finite radius.
      const __m256d good_radius =
          _mm256_and_pd(_mm256_cmp_pd(s, zero, _CMP_GT_OQ),
                        _mm256_cmp_pd(s, infinity, _CMP_LT_OQ));
      if (_mm256_movemask_pd(_mm256_andnot_pd(good_radius, shell)) != 0)
        return static_cast<size_t>(-1);
      __m256d contributes;
      const __m256d value =
          hct_shell_avx2(r, _mm256_blendv_pd(one, s, shell), shell, rho_v,
                         inv_rho, rgbmax_v, cutoff_active, contributes);
      base = _mm256_and_pd(contributes, value);
    }
    if (_mm256_movemask_pd(far) != 0) {
      // titan_gb::gbn2_far_integral, in its operation order.
      const __m256d q = _mm256_div_pd(s_safe, r);
      const __m256d u = _mm256_mul_pd(q, q);
      __m256d series = _mm256_div_pd(_mm256_mul_pd(u, _mm256_set1_pd(5.0)),
                                     _mm256_set1_pd(11.0));
      series = _mm256_mul_pd(u, _mm256_add_pd(_mm256_set1_pd(4. / 9.), series));
      series = _mm256_mul_pd(u, _mm256_add_pd(_mm256_set1_pd(3. / 7.), series));
      series = _mm256_mul_pd(u, _mm256_add_pd(_mm256_set1_pd(2. / 5.), series));
      series = _mm256_add_pd(_mm256_set1_pd(1. / 3.), series);
      const __m256d value = _mm256_mul_pd(
          _mm256_div_pd(_mm256_mul_pd(u, s_safe), _mm256_mul_pd(r, r)),
          series);
      base = _mm256_blendv_pd(base, value, far);
    }
    // titan_gb::gbn2_neck_integral, nearest table entry, zero beyond
    // ri + rj + 6.8.
    const auto entry = [&](const double *table, int lane) {
      return ((act_bits >> lane) & 1) ? table[neck[lane] * 21 + target_neck]
                                      : 1.0;
    };
    const __m256d d0_v = _mm256_set_pd(entry(d0, 3), entry(d0, 2),
                                       entry(d0, 1), entry(d0, 0));
    const __m256d m0_v = _mm256_set_pd(entry(m0, 3), entry(m0, 2),
                                       entry(m0, 1), entry(m0, 0));
    const __m256d reach = _mm256_add_pd(
        _mm256_add_pd(ri_v, _mm256_loadu_pd(rj + j)), _mm256_set1_pd(6.8));
    const __m256d d = _mm256_sub_pd(r, d0_v);
    const __m256d dd = _mm256_mul_pd(d, d);
    const __m256d neck_value = _mm256_div_pd(
        _mm256_mul_pd(_mm256_set1_pd(.826836), m0_v),
        _mm256_add_pd(
            _mm256_add_pd(one, dd),
            _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(.3), dd),
                                        dd),
                          dd)));
    const __m256d contribution = _mm256_add_pd(
        base,
        _mm256_and_pd(_mm256_cmp_pd(r, reach, _CMP_LT_OQ), neck_value));
    const __m256d finite = _mm256_cmp_pd(_mm256_andnot_pd(sign, contribution),
                                         infinity, _CMP_LT_OQ);
    if (_mm256_movemask_pd(_mm256_andnot_pd(finite, act)) != 0)
      return static_cast<size_t>(-1);
    sum = _mm256_add_pd(sum, _mm256_and_pd(act, contribution));
  }
  _mm256_store_pd(lanes, sum);
  return vector_end;
}
#endif

// free_radii, when given, receives each atom's radius from its own molecule
// alone: that integral is exactly the complex one before the environment's
// sources are added, so one pass over the molecule's own pairs yields both.
void calculate_born_radius_range(
    const MOL &mol, const GbAtomParameters &primary,
    AlignedDoubleVector &radii, int igb, size_t begin, size_t end,
    const MOL *env_mol, const GbAtomParameters *environment,
    double rgbmax_angstrom, AlignedDoubleVector *free_radii = nullptr) {
  // A pair farther than rgbmax + s_j contributes exactly zero, so it can be
  // rejected on squared distances before paying for the sqrt and the log in
  // the pair integral. Atoms are still visited in ascending index order, which
  // keeps the summation order (and therefore the result) identical to the
  // uncut loop for every pair that does contribute.
  const bool cutoff_active = rgbmax_angstrom > 0.0 &&
                             std::isfinite(rgbmax_angstrom);
  const auto radius_nm = [&](size_t i, bool valid, double integral) {
    const double born_a =
        valid ? (igb == 8 ? titan_gb::gbn2_born_radius(
                    primary.intrinsic_radius_a[i], integral, primary.gbn2[i])
                         : titan_gb::born_radius_angstrom(
                    primary.intrinsic_radius_a[i], integral, igb))
              : std::numeric_limits<double>::quiet_NaN();
    return born_a * 0.1;
  };
  end = std::min(end, mol.size());
#if TITAN_SIMD_X86
  // The AVX2 pair loops: HCT for igb 1/2/5, HCT/far/neck for GBn2. The four
  // lane sums carry on from the molecule's own sources into the
  // environment's, so a fused free/complex pass still equals separate calls
  // bit for bit.
  const bool vector_pairs = titan_simd_dispatch::runtime_has_avx2();
#endif
  for (size_t i = begin; i < end; ++i) {
    double integral = 0.0;
    alignas(32) double lanes[4] = {0.0, 0.0, 0.0, 0.0};
    bool valid = true;
    const auto accumulate_sources = [&](const MOL &sources,
                                        const GbAtomParameters &parameters,
                                        bool same_molecule) {
      size_t first_scalar = 0;
#if TITAN_SIMD_X86
      const double rho = primary.offset_radius_a[i];
      if (vector_pairs && rho > 0.0 && std::isfinite(rho)) {
        first_scalar =
            igb == 8
                ? gbn2_sources_avx2(
                      mol.pos.x[i], mol.pos.y[i], mol.pos.z[i], rho,
                      primary.intrinsic_radius_a[i], primary.neck_index[i],
                      sources.pos.x.data(), sources.pos.y.data(),
                      sources.pos.z.data(), parameters.scaled_radius_a.data(),
                      parameters.active_mask.data(),
                      parameters.intrinsic_radius_a.data(),
                      parameters.neck_index.data(), sources.size(),
                      same_molecule ? i : static_cast<size_t>(-1),
                      rgbmax_angstrom, cutoff_active, lanes)
                : hct_sources_avx2(
                      mol.pos.x[i], mol.pos.y[i], mol.pos.z[i], rho,
                      sources.pos.x.data(), sources.pos.y.data(),
                      sources.pos.z.data(), parameters.scaled_radius_a.data(),
                      parameters.active_mask.data(), sources.size(),
                      same_molecule ? i : static_cast<size_t>(-1),
                      rgbmax_angstrom, cutoff_active, lanes);
        if (first_scalar == static_cast<size_t>(-1)) {
          valid = false;
          return;
        }
      }
#endif
      for (size_t j = first_scalar; j < sources.size(); ++j) {
        if ((same_molecule && i == j) ||
            !parameters.active[j])
          continue;
        const double dx = mol.pos.x[i] - sources.pos.x[j];
        const double dy = mol.pos.y[i] - sources.pos.y[j];
        const double dz = mol.pos.z[i] - sources.pos.z[j];
        const double distance2 = dx * dx + dy * dy + dz * dz;
        if (!std::isfinite(distance2) || distance2 <= 0.) {
          valid = false;
          return;
        }
        if (cutoff_active) {
          const double reach = rgbmax_angstrom + parameters.scaled_radius_a[j];
          if (!(distance2 < reach * reach))
            continue;
        }
        const double distance = std::sqrt(distance2);
        const double scaled = parameters.scaled_radius_a[j];
        double contribution = 0.;
        if (igb == 8 && distance > 4.*scaled &&
            (!cutoff_active || distance <= rgbmax_angstrom-scaled)) {
          contribution = titan_gb::gbn2_far_integral(scaled, distance);
        } else {
          contribution = titan_gb::hct_pair_integral_inverse_angstrom(
                primary.offset_radius_a[i], parameters.scaled_radius_a[j],
                distance, rgbmax_angstrom);
        }
        if (igb == 8)
          contribution += titan_gb::gbn2_neck_integral(
              primary.intrinsic_radius_a[i], parameters.intrinsic_radius_a[j], distance);
        if (!std::isfinite(contribution)) {
          valid = false;
          return;
        }
        integral += contribution;
      }
    };

    const auto total = [&] {
      return integral + ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]));
    };
    accumulate_sources(mol, primary, true);
    if (free_radii != nullptr)
      (*free_radii)[i] = radius_nm(i, valid, total());
    if (valid && env_mol != nullptr && environment != nullptr)
      accumulate_sources(*env_mol, *environment, false);
    radii[i] = radius_nm(i, valid, total());
  }
}

} // namespace

static void calculate_nsr6_radii(const MOL &mol,
    const AlignedDoubleVector &vdw, const std::vector<mol_info> &info,
    AlignedDoubleVector &radii, size_t begin, size_t end, const MOL *env,
    const AlignedDoubleVector *env_vdw, const std::vector<mol_info> *env_info,
    double spacing, TitanExecutor *executor) {
  std::vector<TitanPBE::Tabi::Atom> atoms;
  const auto append = [&](const MOL &m, const AlignedDoubleVector &v,
                           const std::vector<mol_info> &ai) {
    if (ai.size() != m.size() || v.size() != m.size())
      throw std::invalid_argument("GBNSR6: incomplete atom metadata");
    const GbGeometryIndex index(m, ai);
    for (size_t i=0; i<m.size(); ++i) {
      if (is_cas_deleted_radius(v[i]))
        throw std::invalid_argument("GBNSR6: CAS deletion sentinels are unsupported");
      atoms.push_back({m.pos.x[i],m.pos.y[i],m.pos.z[i],0.,
                       gb_intrinsic_radius_angstrom(m,ai,index,i,66)});
    }
  };
  append(mol,vdw,info);
  if (env) {
    if (!env_vdw || !env_info) throw std::invalid_argument("GBNSR6: incomplete environment");
    append(*env,*env_vdw,*env_info);
  }
  const auto result=titan_gb::gbnsr6_born_radii_nm(atoms,spacing,executor);
  radii.resize(mol.size());
  for (size_t i=begin; i<std::min(end,mol.size()); ++i) radii[i]=result[i];
}

void cal_effective_born_radii_avx2(const MOL &mol,
                                   const AlignedDoubleVector &vdw_radii,
                                   const std::vector<mol_info> &atom_info,
                                   AlignedDoubleVector &radii, int igb,
                                   size_t begin, size_t end, const MOL *env_mol,
                                   const AlignedDoubleVector *env_vdw_radii,
                                   const std::vector<mol_info> *env_atom_info,
                                   double rgbmax_angstrom, double nsr6_spacing) {
  if (igb == 66) {
    calculate_nsr6_radii(mol,vdw_radii,atom_info,radii,begin,end,env_mol,
                         env_vdw_radii,env_atom_info,nsr6_spacing,nullptr);
    return;
  }
  if (radii.size() < mol.size())
    radii.resize(mol.size());
  const GbAtomParameters primary =
      build_gb_atom_parameters(mol, vdw_radii, atom_info, igb);
  GbAtomParameters environment;
  const bool has_environment = env_mol != nullptr && env_vdw_radii != nullptr &&
                               env_atom_info != nullptr;
  if (has_environment) {
    environment = build_gb_atom_parameters(*env_mol, *env_vdw_radii,
                                           *env_atom_info, igb);
  }
  calculate_born_radius_range(mol, primary, radii, igb, begin, end, env_mol,
                              has_environment ? &environment : nullptr,
                              rgbmax_angstrom);
}

void cal_effective_born_radii_parallel_avx2(
    const MOL &mol, const AlignedDoubleVector &vdw_radii,
    const std::vector<mol_info> &atom_info, AlignedDoubleVector &radii,
    int igb, TitanExecutor &executor, const MOL *env_mol,
    const AlignedDoubleVector *env_vdw_radii,
    const std::vector<mol_info> *env_atom_info, double rgbmax_angstrom,
    double nsr6_spacing) {
  if (igb == 66) {
    calculate_nsr6_radii(mol,vdw_radii,atom_info,radii,0,mol.size(),env_mol,
                         env_vdw_radii,env_atom_info,nsr6_spacing,&executor);
    return;
  }
  radii.resize(mol.size());
  const GbAtomParameters primary =
      build_gb_atom_parameters(mol, vdw_radii, atom_info, igb);
  GbAtomParameters environment;
  const bool has_environment = env_mol != nullptr && env_vdw_radii != nullptr &&
                               env_atom_info != nullptr;
  if (has_environment) {
    environment = build_gb_atom_parameters(*env_mol, *env_vdw_radii,
                                           *env_atom_info, igb);
  }
  executor.parallel_for(
      mol.size(), executor.capacity(), 32,
      [&](size_t, size_t begin, size_t end) {
        calculate_born_radius_range(
            mol, primary, radii, igb, begin, end, env_mol,
            has_environment ? &environment : nullptr, rgbmax_angstrom);
      });
}

void cal_free_and_complex_born_radii_avx2(
    const MOL &mol, const AlignedDoubleVector &vdw_radii,
    const std::vector<mol_info> &atom_info, AlignedDoubleVector &free_radii,
    AlignedDoubleVector &complex_radii, int igb, size_t begin, size_t end,
    const MOL &env_mol, const AlignedDoubleVector &env_vdw_radii,
    const std::vector<mol_info> &env_atom_info, double rgbmax_angstrom,
    double nsr6_spacing) {
  if (igb == 66) {
    cal_effective_born_radii_avx2(mol, vdw_radii, atom_info, free_radii, igb,
                                  begin, end, nullptr, nullptr, nullptr,
                                  rgbmax_angstrom, nsr6_spacing);
    cal_effective_born_radii_avx2(mol, vdw_radii, atom_info, complex_radii,
                                  igb, begin, end, &env_mol, &env_vdw_radii,
                                  &env_atom_info, rgbmax_angstrom,
                                  nsr6_spacing);
    return;
  }
  if (free_radii.size() < mol.size())
    free_radii.resize(mol.size());
  if (complex_radii.size() < mol.size())
    complex_radii.resize(mol.size());
  const GbAtomParameters primary =
      build_gb_atom_parameters(mol, vdw_radii, atom_info, igb);
  const GbAtomParameters environment =
      build_gb_atom_parameters(env_mol, env_vdw_radii, env_atom_info, igb);
  calculate_born_radius_range(mol, primary, complex_radii, igb, begin, end,
                              &env_mol, &environment, rgbmax_angstrom,
                              &free_radii);
}

void cal_binding_born_radii_parallel_avx2(
    const MOL &rec, const AlignedDoubleVector &rec_vdw_radii,
    const std::vector<mol_info> &rec_info, const MOL &lg,
    const AlignedDoubleVector &lg_vdw_radii,
    const std::vector<mol_info> &lg_info, int igb, TitanExecutor &executor,
    double rgbmax_angstrom, AlignedDoubleVector &rec_free_radii,
    AlignedDoubleVector &lg_free_radii, AlignedDoubleVector &rec_complex_radii,
    AlignedDoubleVector &lg_complex_radii) {
  if (igb == 66)
    throw std::invalid_argument(
        "cal_binding_born_radii_parallel_avx2: GBNSR6 radii come from "
        "calculate_gbnsr6_binding_radii");
  const size_t rec_size = rec.size();
  const size_t lg_size = lg.size();
  rec_free_radii.resize(rec_size);
  rec_complex_radii.resize(rec_size);
  lg_free_radii.resize(lg_size);
  lg_complex_radii.resize(lg_size);
  const GbAtomParameters rec_parameters =
      build_gb_atom_parameters(rec, rec_vdw_radii, rec_info, igb);
  const GbAtomParameters lg_parameters =
      build_gb_atom_parameters(lg, lg_vdw_radii, lg_info, igb);
  // Every atom of either molecule integrates over all rec + lg atoms, so one
  // index range over both keeps the static partition balanced.
  executor.parallel_for(
      rec_size + lg_size, executor.capacity(), 32,
      [&](size_t, size_t begin, size_t end) {
        if (begin < rec_size)
          calculate_born_radius_range(rec, rec_parameters, rec_complex_radii,
                                      igb, begin, std::min(end, rec_size),
                                      &lg, &lg_parameters, rgbmax_angstrom,
                                      &rec_free_radii);
        if (end > rec_size)
          calculate_born_radius_range(
              lg, lg_parameters, lg_complex_radii, igb,
              std::max(begin, rec_size) - rec_size, end - rec_size, &rec,
              &rec_parameters, rgbmax_angstrom, &lg_free_radii);
      });
}

static void build_complex_geometry(const MOL &rec, const MOL &lg,
                                   const Multi_MOL_parameters &rec_parameters,
                                   const Multi_MOL_parameters &lg_parameters,
                                   MOL &complex_mol,
                                   std::vector<double> &complex_radii) {
  const size_t rec_size = rec.size();
  const size_t lg_size = lg.size();
  complex_mol.pos.resize(rec_size + lg_size);
  complex_radii.resize(rec_size + lg_size);

  for (size_t i = 0; i < rec_size; ++i) {
    complex_mol.pos.x[i] = rec.pos.x[i];
    complex_mol.pos.y[i] = rec.pos.y[i];
    complex_mol.pos.z[i] = rec.pos.z[i];
    complex_radii[i] = rec_parameters.radius[i];
  }

  for (size_t j = 0; j < lg_size; ++j) {
    complex_mol.pos.x[rec_size + j] = lg.pos.x[j];
    complex_mol.pos.y[rec_size + j] = lg.pos.y[j];
    complex_mol.pos.z[rec_size + j] = lg.pos.z[j];
    complex_radii[rec_size + j] = lg_parameters.radius[j];
  }
}

static std::vector<TitanSASA::Atom>
make_sasa_atoms(const MOL &mol, std::span<const double> radii) {
  const size_t n = mol.size();
  std::vector<TitanSASA::Atom> atoms;
  atoms.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const double radius = (i < radii.size()) ? radii[i] : 0.0;
    atoms.push_back({mol.pos.x[i], mol.pos.y[i], mol.pos.z[i], radius});
  }
  return atoms;
}

static std::vector<double>
compute_gbsa_sasa(const MOL &mol, std::span<const double> radii,
                  std::span<const size_t> targets, int point_count,
                  TitanExecutor *executor = nullptr) {
  std::vector<TitanSASA::Atom> atoms = make_sasa_atoms(mol, radii);
  TitanSASA::Options options;
  options.probe_radius = TitanSASA::kDefaultProbeRadiusA;
  options.min_cell_size = 3.0;
  options.point_count = point_count;

  // A CAS-deleted atom would otherwise enter as a probe-sized sphere: it
  // would carry area and occlude its neighbours. Drop such atoms, keeping
  // the order of the rest, so the surface is that of the structure without
  // them; their entries stay zero.
  std::vector<size_t> kept;
  kept.reserve(atoms.size());
  for (size_t i = 0; i < atoms.size(); ++i)
    if (!is_cas_deleted_radius(atoms[i].radius))
      kept.push_back(i);
  if (kept.size() == atoms.size())
    return TitanSASA::compute_per_atom_sasa(
        std::span<const TitanSASA::Atom>(atoms.data(), atoms.size()), options,
        targets, executor);

  constexpr size_t dropped = std::numeric_limits<size_t>::max();
  std::vector<size_t> compact_of(atoms.size(), dropped);
  std::vector<TitanSASA::Atom> compact;
  compact.reserve(kept.size());
  for (size_t c = 0; c < kept.size(); ++c) {
    compact_of[kept[c]] = c;
    compact.push_back(atoms[kept[c]]);
  }
  std::vector<double> area(atoms.size(), 0.0);
  std::vector<size_t> compact_targets;
  if (!targets.empty()) {
    compact_targets.reserve(targets.size());
    for (const size_t t : targets)
      if (t < atoms.size() && compact_of[t] != dropped)
        compact_targets.push_back(compact_of[t]);
    // An empty target list means "every atom", not "none".
    if (compact_targets.empty())
      return area;
  }
  const std::vector<double> compact_area = TitanSASA::compute_per_atom_sasa(
      std::span<const TitanSASA::Atom>(compact.data(), compact.size()),
      options, compact_targets, executor);
  for (size_t c = 0; c < kept.size(); ++c)
    area[kept[c]] = compact_area[c];
  return area;
}

static double alpb_electrostatic_size_nm(const MOL &mol,
                                         std::span<const double> radii) {
  const size_t atom_count = mol.size();
  if (radii.size() != atom_count)
    return std::numeric_limits<double>::quiet_NaN();
  return titan_gb::electrostatic_size_adet_nm(
      std::span<const double>(mol.pos.x.data(), atom_count),
      std::span<const double>(mol.pos.y.data(), atom_count),
      std::span<const double>(mol.pos.z.data(), atom_count), radii);
}

void set_gbnsr6_electrostatic_sizes(const MOL &rec, const MOL &lg,
    const Multi_MOL_parameters &rp, const Multi_MOL_parameters &lp,
    const std::vector<mol_info> &ri, const std::vector<mol_info> &li,
    GBSASasaCache &cache) {
  const auto r=build_gb_atom_parameters(rec,rp.radius,ri,66);
  const auto l=build_gb_atom_parameters(lg,lp.radius,li,66);
  cache.alpb_rec_electrostatic_size_nm=alpb_electrostatic_size_nm(rec,r.intrinsic_radius_a);
  cache.alpb_lig_electrostatic_size_nm=alpb_electrostatic_size_nm(lg,l.intrinsic_radius_a);
  MOL complex;
  std::vector<double> radii;
  build_complex_geometry(rec,lg,rp,lp,complex,radii);
  radii=r.intrinsic_radius_a;
  radii.insert(radii.end(),l.intrinsic_radius_a.begin(),l.intrinsic_radius_a.end());
  cache.alpb_complex_electrostatic_size_nm=alpb_electrostatic_size_nm(complex,radii);
  cache.gbnsr6_sizes=true;
}

void calculate_gbnsr6_binding_radii(const MOL &rec, const MOL &lg,
    const AlignedDoubleVector &rvdw, const AlignedDoubleVector &lvdw,
    const std::vector<mol_info> &ri, const std::vector<mol_info> &li,
    AlignedDoubleVector &rf, AlignedDoubleVector &lf,
    AlignedDoubleVector &rc, AlignedDoubleVector &lc,
    double spacing, TitanExecutor *executor) {
  calculate_nsr6_radii(rec,rvdw,ri,rf,0,rec.size(),nullptr,nullptr,nullptr,spacing,executor);
  calculate_nsr6_radii(lg,lvdw,li,lf,0,lg.size(),nullptr,nullptr,nullptr,spacing,executor);
  MOL complex;
  complex.pos.resize(rec.size()+lg.size());
  for (size_t i=0; i<complex.size(); ++i) {
    const auto &m=i<rec.size()?rec:lg;
    const auto j=i<rec.size()?i:i-rec.size();
    complex.pos.x[i]=m.pos.x[j]; complex.pos.y[i]=m.pos.y[j]; complex.pos.z[i]=m.pos.z[j];
  }
  AlignedDoubleVector vdw=rvdw, all;
  vdw.insert(vdw.end(),lvdw.begin(),lvdw.end());
  auto info=ri; info.insert(info.end(),li.begin(),li.end());
  // Both complex endpoints must use the SAME surface and quadrature.
  calculate_nsr6_radii(complex,vdw,info,all,0,complex.size(),nullptr,nullptr,nullptr,spacing,executor);
  rc.assign(all.begin(),all.begin()+rec.size());
  lc.assign(all.begin()+rec.size(),all.end());
}

void build_gbsa_sasa_cache(const MOL &rec, const MOL &lg,
                           const Multi_MOL_parameters &rec_parameters,
                           const Multi_MOL_parameters &lg_parameters,
                           GBSASasaCache &cache_out, int point_count,
                           bool compute_alpb_sizes,
                           TitanExecutor *executor) {
  cache_out.gbnsr6_sizes=false;
  const size_t rec_size = rec.size();
  const size_t lg_size = lg.size();

  std::vector<size_t> rec_targets(rec_size);
  std::iota(rec_targets.begin(), rec_targets.end(), size_t{0});

  std::vector<size_t> lig_targets(lg_size);
  std::iota(lig_targets.begin(), lig_targets.end(), size_t{0});

  std::vector<size_t> com_targets(rec_size + lg_size);
  std::iota(com_targets.begin(), com_targets.end(), size_t{0});

  MOL complex_mol;
  std::vector<double> complex_radii;
  build_complex_geometry(rec, lg, rec_parameters, lg_parameters, complex_mol,
                         complex_radii);

  cache_out.sasa_rec_free = compute_gbsa_sasa(
      rec, std::span<const double>(rec_parameters.radius.data(), rec_size),
      std::span<const size_t>(rec_targets.data(), rec_targets.size()),
      point_count, executor);

  cache_out.sasa_lig_free = compute_gbsa_sasa(
      lg, std::span<const double>(lg_parameters.radius.data(), lg_size),
      std::span<const size_t>(lig_targets.data(), lig_targets.size()),
      point_count, executor);

  cache_out.sasa_complex = compute_gbsa_sasa(
      complex_mol,
      std::span<const double>(complex_radii.data(), rec_size + lg_size),
      std::span<const size_t>(com_targets.data(), com_targets.size()),
      point_count, executor);

  if (compute_alpb_sizes) {
    cache_out.alpb_rec_electrostatic_size_nm = alpb_electrostatic_size_nm(
        rec, std::span<const double>(rec_parameters.radius.data(), rec_size));
    cache_out.alpb_lig_electrostatic_size_nm = alpb_electrostatic_size_nm(
        lg, std::span<const double>(lg_parameters.radius.data(), lg_size));
    cache_out.alpb_complex_electrostatic_size_nm =
        alpb_electrostatic_size_nm(
            complex_mol,
            std::span<const double>(complex_radii.data(), rec_size + lg_size));
  } else {
    cache_out.alpb_rec_electrostatic_size_nm = 0.0;
    cache_out.alpb_lig_electrostatic_size_nm = 0.0;
    cache_out.alpb_complex_electrostatic_size_nm = 0.0;
  }
}

namespace {
// Lennard-Jones (kJ/mol) of receptor atom i and ligand atom j at r_nm, with
// the expressions of the scalar remainder loop of cal_energy_gbsa_avx2.
// ENE-22 uses it again to take the plain term out of excluded pairs.
double gbsa_pair_lj(cache_sigma_epsilon &CachedData,
                    const Multi_MOL_parameters &rec_parameters,
                    const Multi_MOL_parameters &lg_parameters, size_t i,
                    size_t j, double r_nm, double r6_nm, bool is_charmm,
                    bool is_gromos, bool is_oplsaa, bool has_overrides) {
  double vdw;
  if (is_charmm) {
    double sigma_ij{}, epsilon_ij{};
    charmm::pair_params(CachedData, rec_parameters, lg_parameters, i, j,
                        sigma_ij, epsilon_ij);
    vdw = charmm::lj_energy(sigma_ij, epsilon_ij, r_nm);
  } else if (is_gromos) {
    double c6_ij{}, c12_ij{};
    gromos::pair_params(CachedData, rec_parameters, lg_parameters, i, j,
                        c6_ij, c12_ij);
    const double inv_r6 = 1.0 / r6_nm;
    vdw = inv_r6 * (c12_ij * inv_r6 - c6_ij);
  } else {
    double eps_ij =
        std::sqrt(lg_parameters.epsilon[j] * rec_parameters.epsilon[i]);
    if (is_oplsaa) {
      double sig_ij =
          std::sqrt(lg_parameters.sigma[j] * rec_parameters.sigma[i]);
      double sig2 = sig_ij * sig_ij;
      double sig6 = sig2 * sig2 * sig2;
      double sr6 = sig6 / r6_nm;
      vdw = 4.0 * eps_ij * (sr6 * sr6 - sr6);
    } else {
      double sig_ij =
          0.5 * (lg_parameters.sigma[j] + rec_parameters.sigma[i]);
      double sig2 = sig_ij * sig_ij;
      double sig6 = sig2 * sig2 * sig2;
      double sr6 = sig6 / r6_nm;
      vdw = 4.0 * eps_ij * (sr6 * sr6 - sr6);
    }
    if (has_overrides)
      titan_lj::override_lj_energy(CachedData, rec_parameters,
                                   lg_parameters, i, j, r_nm, vdw);
  }
  return vdw;
}
} // namespace

int cal_energy_gbsa_avx2(
    const MOL &rec, const MOL &lg, const std::vector<mol_info> &info_rec,
    const std::vector<mol_info> &info_lg, cache_sigma_epsilon &CachedData,
    size_t rec_begin, size_t rec_end, size_t lg_begin, size_t lg_end,
    GBSA_energy_snap &Energy, size_t num_snap, const arguments &Arguments,
    const Multi_MOL_parameters &rec_parameters,
    const Multi_MOL_parameters &lg_parameters,
    const std::vector<double> &rec_lcpo_p1,
    const std::vector<double> &rec_lcpo_p2,
    const std::vector<double> &lg_lcpo_p1,
    const std::vector<double> &lg_lcpo_p2,
    const AlignedDoubleVector &rec_free_radii,
    const AlignedDoubleVector &lg_free_radii,
    const AlignedDoubleVector &rec_complex_radii,
    const AlignedDoubleVector &lg_complex_radii, GBSA_Aggregator *aggregator,
    const GBSASasaCache *sasa_cache,
    bool write_pair_cache, bool include_rec_onebody,
    bool include_lig_onebody, const titan_gbsa::TopologyPairs *topology_pairs) {
  (void)num_snap;
  (void)rec_lcpo_p1;
  (void)rec_lcpo_p2;
  (void)lg_lcpo_p1;
  (void)lg_lcpo_p2;

  bool is_lig_rec =
      (Arguments.mol_1_type == "ligand" || Arguments.mol_1_type == "lig") &&
      !Arguments.flag_use_common_topo;
  bool is_lig_lg =
      (Arguments.mol_2_type == "ligand" || Arguments.mol_2_type == "lig") &&
      !Arguments.flag_use_common_topo;
  __m256d avx_one = _mm256_set1_pd(1.0);
  __m256d avx_four = _mm256_set1_pd(4.0);
  double ext_diel = Arguments.ext_diel;
  double int_diel = Arguments.int_diel;
  const double kappa_val = Arguments.igb == 66
      ? titan_gb::nsr6_kappa_nm_inverse(Arguments.saltcon,ext_diel,Arguments.gbsa_temperature)
      : titan_electrostatics::amber_gb_kappa_nm_inverse(
          Arguments.saltcon, ext_diel, Arguments.gbsa_temperature);
  __m256d kappa = _mm256_set1_pd(kappa_val);
  __m256d avx_half = _mm256_set1_pd(0.5);
  bool is_oplsaa = (Arguments.force_f == "OPLSAA");

  const bool use_alpb = Arguments.flag_alpb || Arguments.igb == 66;
  __m256d avx_int_diel = _mm256_set1_pd(int_diel);
  __m256d avx_ext_diel = _mm256_set1_pd(ext_diel);
  // Loop-invariant broadcasts, hoisted so the pair loop does not re-materialise
  // them. Same values, so results are unchanged.
  const __m256d avx_minus_one = _mm256_set1_pd(-1.0);
  const __m256d avx_coulomb_constant =
      _mm256_set1_pd(titan_gb::kCoulombKJMolNmPerE2);
  const __m256d avx_neg_coulomb_constant =
      _mm256_set1_pd(-titan_gb::kCoulombKJMolNmPerE2);

  double endpoint_gpol_sum = 0.0;
  const bool is_charmm =
      Arguments.force_f.find("CHARMM") != std::string::npos;
  // ENE-11: [ nonbond_params ] for the AMBER/OPLS combination-rule paths
  // (the CHARMM path reads them through charmm::pair_params_block).
  // ENE-19: GROMOS keeps C6/C12 in the sigma/epsilon slots; its branch mixes
  // them geometrically, E = C12/r^12 - C6/r^6, and takes the pair overrides
  // (C6/C12 as well) through gromos::pair_params, like gromos_nonbonded.h.
  const bool is_gromos =
      !is_charmm && Arguments.force_f.find("GROMOS") != std::string::npos;
  const bool has_overrides =
      !is_charmm && !is_gromos && CachedData.has_charmm_pair_overrides();
  const bool has_gromos_overrides =
      is_gromos && CachedData.has_charmm_pair_overrides();
  const bool can_write_pair_cache =
      write_pair_cache && !CachedData.elec_ener.empty();

  const size_t rec_size = rec.size();
  const size_t lg_size = lg.size();

  double alpb_rec_size_nm = std::numeric_limits<double>::quiet_NaN();
  double alpb_lig_size_nm = std::numeric_limits<double>::quiet_NaN();
  double alpb_complex_size_nm = std::numeric_limits<double>::quiet_NaN();
  if (Arguments.igb == 66 && (!sasa_cache || !sasa_cache->gbnsr6_sizes)) {
    GBSASasaCache sizes;
    set_gbnsr6_electrostatic_sizes(rec,lg,rec_parameters,lg_parameters,info_rec,info_lg,sizes);
    alpb_rec_size_nm = sizes.alpb_rec_electrostatic_size_nm;
    alpb_lig_size_nm = sizes.alpb_lig_electrostatic_size_nm;
    alpb_complex_size_nm = sizes.alpb_complex_electrostatic_size_nm;
  } else if (use_alpb && sasa_cache != nullptr) {
    alpb_rec_size_nm = sasa_cache->alpb_rec_electrostatic_size_nm;
    alpb_lig_size_nm = sasa_cache->alpb_lig_electrostatic_size_nm;
    alpb_complex_size_nm = sasa_cache->alpb_complex_electrostatic_size_nm;
  } else if (use_alpb) {
    alpb_rec_size_nm = alpb_electrostatic_size_nm(
        rec, std::span<const double>(rec_parameters.radius.data(), rec_size));
    alpb_lig_size_nm = alpb_electrostatic_size_nm(
        lg, std::span<const double>(lg_parameters.radius.data(), lg_size));
    MOL complex_mol;
    std::vector<double> complex_radii;
    build_complex_geometry(rec, lg, rec_parameters, lg_parameters, complex_mol,
                           complex_radii);
    alpb_complex_size_nm = alpb_electrostatic_size_nm(
        complex_mol,
        std::span<const double>(complex_radii.data(), complex_radii.size()));
  }
  if (use_alpb &&
      (!(alpb_rec_size_nm > 0.0) || !std::isfinite(alpb_rec_size_nm) ||
       !(alpb_lig_size_nm > 0.0) || !std::isfinite(alpb_lig_size_nm) ||
       !(alpb_complex_size_nm > 0.0) ||
       !std::isfinite(alpb_complex_size_nm)))
    throw std::invalid_argument("ALPB requires a finite positive molecular electrostatic size; check coordinates and nonnegative atomic size radii");

  rec_begin = std::min(rec_begin, rec_size);
  rec_end = std::clamp(rec_end, rec_begin, rec_size);
  lg_begin = std::min(lg_begin, lg_size);
  lg_end = std::clamp(lg_end, lg_begin, lg_size);
  const bool collect_decomposition = (aggregator != nullptr);
  struct LocalDecompEntry {
    bool is_ligand;
    int chainnum;
    int resnr;
    bool detail_atom;
    const mol_info *m_info;
    double ev = 0.0;
    double ee = 0.0;
    double gpol = 0.0;
    double gnp = 0.0;
  };

  auto make_decomp_key = [](int chainnum, int resnr) -> std::uint64_t {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(chainnum))
            << 32) |
           static_cast<std::uint32_t>(resnr);
  };

  std::vector<LocalDecompEntry> local_decomp_entries;
  if (collect_decomposition) {
    local_decomp_entries.reserve(rec_size + (lg_end - lg_begin));
  }

  auto register_decomp_slot =
      [&](std::unordered_map<std::uint64_t, size_t> &slot_map, bool is_ligand,
          int chainnum, int resnr, bool detail_atom,
          const mol_info *m_info) -> size_t {
    const std::uint64_t key = make_decomp_key(chainnum, resnr);
    const auto it = slot_map.find(key);
    if (it != slot_map.end()) {
      return it->second;
    }

    const size_t slot = local_decomp_entries.size();
    slot_map.emplace(key, slot);
    local_decomp_entries.push_back(
        {is_ligand, chainnum, resnr, detail_atom, m_info});
    return slot;
  };

  std::unordered_map<std::uint64_t, size_t> rec_decomp_slots_by_key;
  const size_t invalid_decomp_slot = std::numeric_limits<size_t>::max();
  std::vector<size_t> rec_decomp_slot(rec_size, invalid_decomp_slot);
  const auto rec_slot_for_atom = [&](size_t atom) -> size_t {
    if (rec_decomp_slot[atom] != invalid_decomp_slot)
      return rec_decomp_slot[atom];
    const int rec_id = is_lig_rec ? static_cast<int>(atom + 1)
                                  : info_rec[atom].get_resnr();
    rec_decomp_slot[atom] = register_decomp_slot(
        rec_decomp_slots_by_key, false, info_rec[atom].get_chainnum(), rec_id,
        is_lig_rec, &info_rec[atom]);
    return rec_decomp_slot[atom];
  };
  if (collect_decomposition) {
    rec_decomp_slots_by_key.reserve(rec_end - rec_begin);
    for (size_t i = rec_begin; i < rec_end; ++i)
      (void)rec_slot_for_atom(i);
  }

  std::unordered_map<std::uint64_t, size_t> lig_decomp_slots_by_key;
  std::vector<size_t> lig_decomp_slot(lg_size, invalid_decomp_slot);
  const auto lig_slot_for_atom = [&](size_t atom) -> size_t {
    if (lig_decomp_slot[atom] != invalid_decomp_slot)
      return lig_decomp_slot[atom];
    const int lig_id = is_lig_lg ? static_cast<int>(atom + 1)
                                 : info_lg[atom].get_resnr();
    lig_decomp_slot[atom] = register_decomp_slot(
        lig_decomp_slots_by_key, true, info_lg[atom].get_chainnum(), lig_id,
        is_lig_lg, &info_lg[atom]);
    return lig_decomp_slot[atom];
  };
  if (collect_decomposition) {
    lig_decomp_slots_by_key.reserve(lg_end - lg_begin);
    for (size_t j = lg_begin; j < lg_end; ++j)
      (void)lig_slot_for_atom(j);
  }

  std::vector<double> local_sasa_rec_free;
  std::vector<double> local_sasa_lig_free;
  std::vector<double> local_sasa_complex;
  const std::vector<double> *sasa_rec_free_ptr = nullptr;
  const std::vector<double> *sasa_lig_free_ptr = nullptr;
  const std::vector<double> *sasa_complex_ptr = nullptr;


  const bool use_precomputed_sasa =
      sasa_cache != nullptr &&
      sasa_cache->sasa_rec_free.size() == rec_size &&
      sasa_cache->sasa_lig_free.size() == lg_size &&
      sasa_cache->sasa_complex.size() == rec_size + lg_size;

  if (use_precomputed_sasa) {
    sasa_rec_free_ptr = &sasa_cache->sasa_rec_free;
    sasa_lig_free_ptr = &sasa_cache->sasa_lig_free;
    sasa_complex_ptr = &sasa_cache->sasa_complex;
  } else {
    std::vector<size_t> rec_targets;
    rec_targets.reserve(rec_end - rec_begin);
    for (size_t i = rec_begin; i < rec_end; ++i)
      rec_targets.push_back(i);

    std::vector<size_t> lig_targets;
    lig_targets.reserve(lg_end - lg_begin);
    for (size_t j = lg_begin; j < lg_end; ++j)
      lig_targets.push_back(j);

    std::vector<size_t> com_targets;
    com_targets.reserve((rec_end - rec_begin) + (lg_end - lg_begin));
    for (size_t i = rec_begin; i < rec_end; ++i)
      com_targets.push_back(i);
    for (size_t j = lg_begin; j < lg_end; ++j)
      com_targets.push_back(rec_size + j);

    MOL complex_mol;
    std::vector<double> complex_radii;
    build_complex_geometry(rec, lg, rec_parameters, lg_parameters, complex_mol,
                           complex_radii);

    local_sasa_rec_free = compute_gbsa_sasa(
        rec, std::span<const double>(rec_parameters.radius.data(), rec_size),
        std::span<const size_t>(rec_targets.data(), rec_targets.size()),
        Arguments.gbsa_sasa_points);

    local_sasa_lig_free = compute_gbsa_sasa(
        lg, std::span<const double>(lg_parameters.radius.data(), lg_size),
        std::span<const size_t>(lig_targets.data(), lig_targets.size()),
        Arguments.gbsa_sasa_points);

    local_sasa_complex = compute_gbsa_sasa(
        complex_mol,
        std::span<const double>(complex_radii.data(), rec_size + lg_size),
        std::span<const size_t>(com_targets.data(), com_targets.size()),
        Arguments.gbsa_sasa_points);

    sasa_rec_free_ptr = &local_sasa_rec_free;
    sasa_lig_free_ptr = &local_sasa_lig_free;
    sasa_complex_ptr = &local_sasa_complex;
  }

  // Nonpolar surface tension. The default 0.0072 kcal/mol/A^2 is MMPBSA.py's
  // GB default; sander's own gbsa=1 default is 0.005, so the value has to be
  // selectable and has to appear in the summary provenance rather than being
  // compiled in.
  double gamma_kj = Arguments.gbsa_surften * TitanSASA::kKcalToKJ;
  std::vector<double> dG_np_rec(rec_size, 0.0);
  std::vector<double> dG_np_lig(lg_size, 0.0);

  for (size_t i = rec_begin; include_rec_onebody && i < rec_end; ++i) {
    dG_np_rec[i] =
        gamma_kj * ((*sasa_complex_ptr)[i] - (*sasa_rec_free_ptr)[i]);
    Energy.G_nonpolar += dG_np_rec[i];
  }
  for (size_t j = lg_begin; include_lig_onebody && j < lg_end; ++j) {
    dG_np_lig[j] = gamma_kj *
                   ((*sasa_complex_ptr)[rec_size + j] - (*sasa_lig_free_ptr)[j]);
    Energy.G_nonpolar += dG_np_lig[j];
  }

  for (size_t i = rec_begin; include_rec_onebody && i < rec_end; ++i) {
    const double q = rec_parameters.charge[i];
    const double dG_self =
        0.5 * (titan_gb::polar_pair_energy_kj_mol(
                   q, q, 0.0, rec_complex_radii[i], rec_complex_radii[i],
                   kappa_val, int_diel, ext_diel, use_alpb,
                   alpb_complex_size_nm) -
               titan_gb::polar_pair_energy_kj_mol(
                   q, q, 0.0, rec_free_radii[i], rec_free_radii[i],
                   kappa_val, int_diel, ext_diel, use_alpb,
                   alpb_rec_size_nm));
    endpoint_gpol_sum += dG_self;
    if (collect_decomposition)
      local_decomp_entries[rec_slot_for_atom(i)].gpol += dG_self;
  }
  for (size_t j = lg_begin; include_lig_onebody && j < lg_end; ++j) {
    const double q = lg_parameters.charge[j];
    const double dG_self =
        0.5 * (titan_gb::polar_pair_energy_kj_mol(
                   q, q, 0.0, lg_complex_radii[j], lg_complex_radii[j],
                   kappa_val, int_diel, ext_diel, use_alpb,
                   alpb_complex_size_nm) -
               titan_gb::polar_pair_energy_kj_mol(
                   q, q, 0.0, lg_free_radii[j], lg_free_radii[j], kappa_val,
                   int_diel, ext_diel, use_alpb, alpb_lig_size_nm));
    endpoint_gpol_sum += dG_self;
    if (collect_decomposition)
      local_decomp_entries[lig_slot_for_atom(j)].gpol += dG_self;
  }

  // Complete the endpoint cycle.  Intramolecular GB pair terms do not cancel
  // when ligand descreening changes Born radii between free and complex states.
  // ENE-15: the pairs (i in this call's rows, k > i) run four partners per
  // AVX2 step (gbsa/gb_intramolecular_delta.cpp, equal to the scalar
  // polar_pair_energy_kj_mol to rounding); half of each pair delta goes to
  // both atoms of the per-atom decomposition.
  std::vector<double> intra_half_delta;
  auto intramolecular_delta = [&](const MOL &mol,
                                  const Multi_MOL_parameters &parameters,
                                  const AlignedDoubleVector &complex_radii,
                                  const AlignedDoubleVector &free_radii,
                                  size_t n, size_t row_begin, size_t row_end,
                                  double free_size_nm, auto &&slot_for_atom) {
    if (row_begin >= row_end)
      return;
    titan_gbsa::GbPairConstants constants;
    constants.kappa_nm_inverse = kappa_val;
    constants.internal_dielectric = int_diel;
    constants.external_dielectric = ext_diel;
    constants.use_alpb = use_alpb;
    constants.alpb_complex_size_nm = alpb_complex_size_nm;
    constants.alpb_free_size_nm = free_size_nm;
    if (collect_decomposition)
      intra_half_delta.assign(n, 0.0);
    endpoint_gpol_sum += titan_gbsa::intramolecular_delta_gb(
        mol.pos.x.data(), mol.pos.y.data(), mol.pos.z.data(),
        parameters.charge.data(), complex_radii.data(), free_radii.data(), n,
        row_begin, row_end, constants,
        collect_decomposition ? intra_half_delta.data() : nullptr);
    if (collect_decomposition)
      for (size_t atom = row_begin; atom < n; ++atom)
        if (intra_half_delta[atom] != 0.0)
          local_decomp_entries[slot_for_atom(atom)].gpol +=
              intra_half_delta[atom];
  };
  if (include_rec_onebody)
    intramolecular_delta(rec, rec_parameters, rec_complex_radii,
                         rec_free_radii, rec_size, rec_begin, rec_end,
                         alpb_rec_size_nm, rec_slot_for_atom);
  if (include_lig_onebody)
    intramolecular_delta(lg, lg_parameters, lg_complex_radii, lg_free_radii,
                         lg_size, lg_begin, lg_end, alpb_lig_size_nm,
                         lig_slot_for_atom);
  // FIX: DO NOT divide by scalar_eu_par here. Keep it in kJ/mol.
  Energy.G_polar += endpoint_gpol_sum;

  __m256d v_scale = _mm256_set1_pd(0.01);

  // ENE-04: the four-lane pair energies stay in registers. The totals are
  // vector sums (one horizontal sum at the end), the receptor half of the
  // decomposition goes to per-atom vectors added four lanes at a time and
  // the ligand half to per-row vector sums; both reach the decomposition
  // slots once. Only the summation grouping differs from adding lane by lane.
  __m256d vdw_total = _mm256_setzero_pd();
  __m256d elec_total = _mm256_setzero_pd();
  __m256d gpol_total = _mm256_setzero_pd();
  AlignedDoubleVector rec_half_ev, rec_half_ee, rec_half_gpol;
  if (collect_decomposition && lg_begin < lg_end && rec_begin < rec_end) {
    rec_half_ev.assign(rec_size, 0.0);
    rec_half_ee.assign(rec_size, 0.0);
    rec_half_gpol.assign(rec_size, 0.0);
  }

  for (size_t j = lg_begin; j < lg_end; ++j) {
    double pos_x = lg.pos.x[j], pos_y = lg.pos.y[j], pos_z = lg.pos.z[j];
    double R_j = lg_complex_radii[j];
    __m256d v_pos_x = _mm256_set1_pd(pos_x);
    __m256d v_pos_y = _mm256_set1_pd(pos_y);
    __m256d v_pos_z = _mm256_set1_pd(pos_z);
    __m256d v_R_j = _mm256_set1_pd(R_j);
    __m256d v_q_j = _mm256_set1_pd(lg_parameters.charge[j]);
    const __m256d avx_temp_epsilon_lg = _mm256_set1_pd(lg_parameters.epsilon[j]);
    const __m256d avx_temp_sigma_lg = _mm256_set1_pd(lg_parameters.sigma[j]);
    // The ligand decomposition slot is fixed for this j, so resolve it once
    // instead of once per lane of every block. The index is kept rather than a
    // reference because registering a receptor slot inside the loop could
    // reallocate local_decomp_entries.
    const size_t lig_decomp_slot =
        collect_decomposition ? lig_slot_for_atom(j) : 0;
    __m256d lig_ev_row = _mm256_setzero_pd();
    __m256d lig_ee_row = _mm256_setzero_pd();
    __m256d lig_gpol_row = _mm256_setzero_pd();

    const size_t vector_rec_end =
        rec_begin + ((rec_end - rec_begin) / 4) * 4;
    for (size_t i = rec_begin; i < vector_rec_end; i += 4) {
      __m256d v_x = _mm256_loadu_pd(&rec.pos.x[i]);
      __m256d v_y = _mm256_loadu_pd(&rec.pos.y[i]);
      __m256d v_z = _mm256_loadu_pd(&rec.pos.z[i]);
      __m256d v_q_i = _mm256_loadu_pd(&rec_parameters.charge[i]);
      __m256d v_R_i = _mm256_loadu_pd(&rec_complex_radii[i]);

      __m256d dx = _mm256_sub_pd(v_x, v_pos_x);
      __m256d dy = _mm256_sub_pd(v_y, v_pos_y);
      __m256d dz = _mm256_sub_pd(v_z, v_pos_z);
      __m256d r2_A = _mm256_add_pd(
          _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
          _mm256_mul_pd(dz, dz));
      __m256d r2_nm = _mm256_mul_pd(r2_A, v_scale);
      __m256d r_nm = _mm256_sqrt_pd(r2_nm);

      // VDW Mixing Rules
      __m256d avx_rec_epsilon = _mm256_loadu_pd(&rec_parameters.epsilon[i]);
      __m256d avx_rec_sigma = _mm256_loadu_pd(&rec_parameters.sigma[i]);

      __m256d eps_ij = _mm256_sqrt_pd(_mm256_mul_pd(
          avx_temp_epsilon_lg, avx_rec_epsilon)); // Geometric mean
      // The frame-level close-contact gate rejects r < 0.05 nm before this
      // kernel. Never silently zero or cap an unphysical interaction.
      __m256d r6_nm_raw = _mm256_mul_pd(_mm256_mul_pd(r2_nm, r2_nm), r2_nm);
      __m256d r6_nm = r6_nm_raw;

      __m256d vdw_res;
      if (is_charmm) {
        alignas(32) double sigma_lane[4]{};
        alignas(32) double epsilon_lane[4]{};
        charmm::pair_params_block(CachedData, rec_parameters, lg_parameters, i,
                                  4, j, sigma_lane, epsilon_lane);
        const __m256d sigma_vec = _mm256_loadu_pd(sigma_lane);
        const __m256d epsilon_vec = _mm256_loadu_pd(epsilon_lane);
        vdw_res = charmm_lj_energy_gbsa_avx(sigma_vec, epsilon_vec, r_nm);
      } else if (is_gromos) {
        __m256d c6_ij, c12_ij;
        if (has_gromos_overrides) {
          alignas(32) double c6_lane[4]{};
          alignas(32) double c12_lane[4]{};
          for (size_t lane = 0; lane < 4; ++lane)
            gromos::pair_params(CachedData, rec_parameters, lg_parameters,
                                i + lane, j, c6_lane[lane], c12_lane[lane]);
          c6_ij = _mm256_load_pd(c6_lane);
          c12_ij = _mm256_load_pd(c12_lane);
        } else {
          c6_ij = _mm256_sqrt_pd(
              _mm256_mul_pd(avx_rec_sigma, avx_temp_sigma_lg));
          c12_ij = eps_ij;
        }
        const __m256d inv_r6 = _mm256_div_pd(avx_one, r6_nm);
        vdw_res = _mm256_mul_pd(
            inv_r6, _mm256_sub_pd(_mm256_mul_pd(c12_ij, inv_r6), c6_ij));
      } else {
        const __m256d sig_ij = is_oplsaa
                                   ? _mm256_sqrt_pd(_mm256_mul_pd(
                                         avx_temp_sigma_lg, avx_rec_sigma))
                                   : _mm256_mul_pd(
                                         _mm256_add_pd(avx_temp_sigma_lg,
                                                       avx_rec_sigma),
                                         avx_half);
        // ENE-04: the same operations as
        // titan_simd_dispatch::gbsa_lj_mixed_sigma_epsilon_r6_fourlanes, in
        // registers (no store, indirect call and reload per block)
        const __m256d sig2 = _mm256_mul_pd(sig_ij, sig_ij);
        const __m256d sig6 = _mm256_mul_pd(_mm256_mul_pd(sig2, sig2), sig2);
        const __m256d sr6 = _mm256_div_pd(sig6, r6_nm);
        const __m256d sr12 = _mm256_mul_pd(sr6, sr6);
        const __m256d lj_valid = _mm256_and_pd(
            _mm256_cmp_pd(sig_ij, _mm256_setzero_pd(), _CMP_GT_OQ),
            _mm256_cmp_pd(eps_ij, _mm256_setzero_pd(), _CMP_NEQ_OQ));
        vdw_res = _mm256_and_pd(
            _mm256_mul_pd(_mm256_mul_pd(avx_four, eps_ij),
                          _mm256_sub_pd(sr12, sr6)),
            lj_valid);
        if (has_overrides) {
          alignas(32) double vdw_lane[4];
          alignas(32) double r_lane[4];
          _mm256_store_pd(vdw_lane, vdw_res);
          _mm256_store_pd(r_lane, r_nm);
          titan_lj::override_lj_lanes(CachedData, rec_parameters,
                                      lg_parameters, i, 4, j, r_lane,
                                      vdw_lane);
          vdw_res = _mm256_load_pd(vdw_lane);
        }
      }
      // Charge product feeds both the Coulomb and the GB term; compute it once.
      __m256d v_qij = _mm256_mul_pd(v_q_i, v_q_j);

      // Elec (Vacuum Coulomb MUST NOT be divided by int_diel)
      __m256d elec_res = _mm256_div_pd(
          _mm256_mul_pd(avx_coulomb_constant, v_qij), r_nm);

      // G_polar. ENE-14: both exponentials through titan_avx2::exp_pd (about
      // 1 ulp; f_exp_avx had a relative error up to 3.2e-6). When every lane
      // has r^2/(4 RiRj) >= 36, RiRj exp(-r^2/(4 RiRj)) is below half an ulp
      // of r^2 and f_GB = sqrt(r^2) whatever the exponential returns, so it
      // is skipped there: the result is the same to the bit.
      __m256d RiRj = _mm256_mul_pd(v_R_i, v_R_j);
      const __m256d gb_argument = _mm256_mul_pd(
          avx_minus_one, _mm256_div_pd(r2_nm, _mm256_mul_pd(avx_four, RiRj)));
      __m256d exp_term =
          titan_gbsa::gb_exponential_negligible(gb_argument)
              ? _mm256_setzero_pd()
              : titan_avx2::exp_pd(gb_argument);
      __m256d fGB =
          _mm256_sqrt_pd(_mm256_add_pd(r2_nm, _mm256_mul_pd(RiRj, exp_term)));

      // Proper DH Screening: exp(-kappa * fGB) applied ONLY to external
      // dielectric
      __m256d screen_exp = titan_avx2::exp_pd(
          _mm256_mul_pd(avx_minus_one, _mm256_mul_pd(kappa, fGB)));
      __m256d gpol_res;

      if (use_alpb) {
        gpol_res = titan_gb::alpb_polar_pair_energy_from_fgb_avx(
            v_qij, fGB, screen_exp, int_diel, ext_diel,
            alpb_complex_size_nm);
      } else {
        __m256d term_com_in =
            _mm256_div_pd(avx_one, _mm256_mul_pd(avx_int_diel, fGB));
        __m256d term_com_out =
            _mm256_div_pd(screen_exp, _mm256_mul_pd(avx_ext_diel, fGB));
        gpol_res = _mm256_mul_pd(
            avx_neg_coulomb_constant,
            _mm256_mul_pd(v_qij, _mm256_sub_pd(term_com_in, term_com_out)));
      }

      vdw_total = _mm256_add_pd(vdw_total, vdw_res);
      elec_total = _mm256_add_pd(elec_total, elec_res);
      gpol_total = _mm256_add_pd(gpol_total, gpol_res);

      if (can_write_pair_cache) {
        alignas(32) double vdw_a[4], elec_a[4];
        _mm256_store_pd(vdw_a, vdw_res);
        _mm256_store_pd(elec_a, elec_res);
        for (size_t k = 0; k < 4; ++k)
          CachedData.set_elec_vdw(rec_size * j + (i + k), elec_a[k], vdw_a[k]);
      }
      if (collect_decomposition) {
        const __m256d half_vdw = _mm256_mul_pd(avx_half, vdw_res);
        const __m256d half_elec = _mm256_mul_pd(avx_half, elec_res);
        const __m256d half_gpol = _mm256_mul_pd(avx_half, gpol_res);
        _mm256_storeu_pd(&rec_half_ev[i], _mm256_add_pd(
                                              _mm256_loadu_pd(&rec_half_ev[i]),
                                              half_vdw));
        _mm256_storeu_pd(&rec_half_ee[i], _mm256_add_pd(
                                              _mm256_loadu_pd(&rec_half_ee[i]),
                                              half_elec));
        _mm256_storeu_pd(&rec_half_gpol[i],
                         _mm256_add_pd(_mm256_loadu_pd(&rec_half_gpol[i]),
                                       half_gpol));
        lig_ev_row = _mm256_add_pd(lig_ev_row, half_vdw);
        lig_ee_row = _mm256_add_pd(lig_ee_row, half_elec);
        lig_gpol_row = _mm256_add_pd(lig_gpol_row, half_gpol);
      }
    }
    if (collect_decomposition && vector_rec_end > rec_begin) {
      auto &lig_decomp = local_decomp_entries[lig_decomp_slot];
      lig_decomp.ev += titan_avx2::hsum_pd(lig_ev_row);
      lig_decomp.ee += titan_avx2::hsum_pd(lig_ee_row);
      lig_decomp.gpol += titan_avx2::hsum_pd(lig_gpol_row);
    }

    // Scalar Remainder Loop
    for (size_t i = vector_rec_end; i < rec_end; ++i) {
      double dx = rec.pos.x[i] - pos_x;
      double dy = rec.pos.y[i] - pos_y;
      double dz = rec.pos.z[i] - pos_z;
      double r2_A = dx * dx + dy * dy + dz * dz;
      double r2_nm = r2_A * 0.01;
      double r_nm = std::sqrt(r2_nm);
      double r6_nm = r2_nm * r2_nm * r2_nm;

      const double vdw =
          gbsa_pair_lj(CachedData, rec_parameters, lg_parameters, i, j, r_nm,
                       r6_nm, is_charmm, is_gromos, is_oplsaa, has_overrides);

      // Elec (Remove int_diel division)
      double elec =
          titan_gb::kCoulombKJMolNmPerE2 * lg_parameters.charge[j] *
          rec_parameters.charge[i] / r_nm;

      // G_polar
      const double gpol = titan_gb::polar_pair_energy_kj_mol(
          rec_parameters.charge[i], lg_parameters.charge[j], r2_nm,
          rec_complex_radii[i], R_j, kappa_val, int_diel, ext_diel,
          use_alpb, alpb_complex_size_nm);

      // CRITICAL: Remove all / scalar_eu_par divisions below this point.

      if (can_write_pair_cache) {
        size_t pos = rec.size() * j + i;
        CachedData.set_elec_vdw(pos, elec, vdw);
      }

      Energy.E_vdw += vdw;
      Energy.E_elec += elec;
      Energy.G_polar += gpol;

      if (collect_decomposition) {
        constexpr double half = 0.5;
        rec_half_ev[i] += half * vdw;
        rec_half_ee[i] += half * elec;
        rec_half_gpol[i] += half * gpol;

        auto &lig_decomp = local_decomp_entries[lig_decomp_slot];
        lig_decomp.ev += half * vdw;
        lig_decomp.ee += half * elec;
        lig_decomp.gpol += half * gpol;
      }
    }
  }
  Energy.E_vdw += titan_avx2::hsum_pd(vdw_total);
  Energy.E_elec += titan_avx2::hsum_pd(elec_total);
  Energy.G_polar += titan_avx2::hsum_pd(gpol_total);
  if (!rec_half_ev.empty()) {
    for (size_t i = rec_begin; i < rec_end; ++i) {
      auto &rec_decomp = local_decomp_entries[rec_slot_for_atom(i)];
      rec_decomp.ev += rec_half_ev[i];
      rec_decomp.ee += rec_half_ee[i];
      rec_decomp.gpol += rec_half_gpol[i];
    }
  }

  // CRITICAL: Final injection of SASA nonpolar energy into decomposition
  // structure Because Gnonpol does not distribute purely pairwise, we assign it
  // post-loop directly into the relevant aggregates
  for (size_t i = rec_begin; include_rec_onebody && i < rec_end; ++i) {
    if (collect_decomposition && std::abs(dG_np_rec[i]) > 1e-8) {
      auto &rec_decomp =
          local_decomp_entries[rec_slot_for_atom(i)];
      rec_decomp.gnp += dG_np_rec[i];
    }
  }
  for (size_t j = lg_begin; include_lig_onebody && j < lg_end; ++j) {
    if (collect_decomposition && std::abs(dG_np_lig[j]) > 1e-8) {
      auto &lig_decomp = local_decomp_entries[lig_slot_for_atom(j)];
      lig_decomp.gnp += dG_np_lig[j];
    }
  }

  // ENE-22 (22/10/2026): topology exclusions and listed pairs between the
  // groups (gbsa/gb_topology_pairs.h). The loops above added the plain
  // Coulomb/LJ term of every pair; an excluded pair loses it and a listed
  // pair gains its listed term (plain Coulomb with the listed charge
  // product, as GROMACS evaluates 1-4 pairs). The GB polar term keeps every
  // pair. Each pair is handled by the call whose ranges contain it, and
  // before the ALPB scaling below, which then covers it like the others.
  if (topology_pairs != nullptr && topology_pairs->rec_size == rec_size &&
      topology_pairs->lig_size == lg_size) {
    for (const auto &pair : topology_pairs->pairs) {
      const size_t i = pair.rec;
      const size_t j = pair.lig;
      if (i < rec_begin || i >= rec_end || j < lg_begin || j >= lg_end)
        continue;
      const double dx = rec.pos.x[i] - lg.pos.x[j];
      const double dy = rec.pos.y[i] - lg.pos.y[j];
      const double dz = rec.pos.z[i] - lg.pos.z[j];
      const double r2_nm = (dx * dx + dy * dy + dz * dz) * 0.01;
      const double r_nm = std::sqrt(r2_nm);
      const double r6_nm = r2_nm * r2_nm * r2_nm;
      const double plain_vdw =
          gbsa_pair_lj(CachedData, rec_parameters, lg_parameters, i, j, r_nm,
                       r6_nm, is_charmm, is_gromos, is_oplsaa, has_overrides);
      const double plain_elec = titan_gb::kCoulombKJMolNmPerE2 *
                                lg_parameters.charge[j] *
                                rec_parameters.charge[i] / r_nm;
      const double inv_r6 = 1.0 / r6_nm;
      const double listed_vdw =
          inv_r6 * (pair.listed_c12 * inv_r6 - pair.listed_c6);
      const double listed_elec =
          titan_gb::kCoulombKJMolNmPerE2 * pair.listed_qq / r_nm;
      const double vdw = (pair.excluded ? 0.0 : plain_vdw) + listed_vdw;
      const double elec = (pair.excluded ? 0.0 : plain_elec) + listed_elec;
      Energy.E_vdw += vdw - plain_vdw;
      Energy.E_elec += elec - plain_elec;
      if (can_write_pair_cache)
        CachedData.set_elec_vdw(rec_size * j + i, elec, vdw);
      if (collect_decomposition) {
        auto &rec_decomp = local_decomp_entries[rec_slot_for_atom(i)];
        rec_decomp.ev += 0.5 * (vdw - plain_vdw);
        rec_decomp.ee += 0.5 * (elec - plain_elec);
        auto &lig_decomp = local_decomp_entries[lig_slot_for_atom(j)];
        lig_decomp.ev += 0.5 * (vdw - plain_vdw);
        lig_decomp.ee += 0.5 * (elec - plain_elec);
      }
    }
  }

  // ALPB rescales the direct Coulomb term, not only the GB term. Sander applies
  // the same 1/(1+alpha*beta) factor to EEL: measured against the sander Python
  // API the ratio EEL(alpb=1)/EEL(alpb=0) is 0.992773469127 for eps_in=1,
  // eps_out=78.5, reproduced to 1e-15 across two systems, salt 0 and 0.15 M and
  // igb 1 and 5. Without this, TITAN's raw ALPB total cannot be compared with
  // sander's raw total even though both report the same EGB.
  //
  // Applied once here, after every pair contribution has been accumulated, so
  // the total and the per-residue decomposition stay consistent. The factor is
  // a constant, so a binding delta is scaled by it as well.
  if (use_alpb && Arguments.igb != 66) {
    const double alpb_beta = int_diel / ext_diel;
    const double alpb_coulomb_scale =
        1.0 / (1.0 + titan_gb::kAlpbAlpha * alpb_beta);
    Energy.E_elec *= alpb_coulomb_scale;
    for (auto &entry : local_decomp_entries)
      entry.ee *= alpb_coulomb_scale;
  }

  if (collect_decomposition) {
    for (const auto &entry : local_decomp_entries) {
      if (entry.ev == 0.0 && entry.ee == 0.0 && entry.gpol == 0.0 &&
          entry.gnp == 0.0) {
        continue;
      }
      aggregator->add_decomposition(entry.is_ligand, entry.chainnum,
                                    entry.resnr, entry.detail_atom,
                                    entry.m_info, entry.ev, entry.ee,
                                    entry.gpol, entry.gnp);
    }
  }

  Energy.total_energy =
      Energy.E_vdw + Energy.E_elec + Energy.G_polar + Energy.G_nonpolar;
  return 0;
}
#endif // __AVX__

void GBSA_Aggregator::initialize_trajectory(size_t traj_index) {
  if (trajectories.size() <= traj_index) {
    trajectories.resize(traj_index + 1);
  }
}

void GBSA_Aggregator::add_snapshot(size_t traj_index, GBSA_energy_snap snap) {
  initialize_trajectory(traj_index);
  trajectories[traj_index].emplace_back(std::move(snap));
}

void GBSA_Aggregator::add_decomposition(bool is_ligand, int chainnum, int resnr,
                                        bool detail_atom,
                                        const mol_info *m_info, double ev,
                                        double ee, double gpol, double gnp) {
  DecompKey key{is_ligand, chainnum, resnr};
  auto it = decomp_index.find(key);
  if (it != decomp_index.end()) {
    auto &d = decomp[it->second];
    d.E_vdw += ev;
    d.E_elec += ee;
    d.G_polar += gpol;
    d.G_nonpolar += gnp;
    d.total += (ev + ee + gpol + gnp);
    d.count++;
  } else {
    GBSA_decomposition d;
    d.is_ligand = is_ligand;
    d.chainnum = chainnum;
    d.resnr = resnr;
    d.unit_type_str = is_ligand ? "lig" : "protein";
    d.chain_str = m_info->get_chain();
    d.name_str = detail_atom ? m_info->get_atom() : m_info->get_res_name();
    d.E_vdw = ev;
    d.E_elec = ee;
    d.G_polar = gpol;
    d.G_nonpolar = gnp;
    d.total = ev + ee + gpol + gnp;
    d.count = 1;
    decomp_index[key] = decomp.size();
    decomp.emplace_back(std::move(d));
  }
}

void GBSA_Aggregator::merge(const GBSA_Aggregator &other) {
  for (const auto &od : other.decomp) {
    if (od.count == 0)
      continue;
    DecompKey key{od.is_ligand, od.chainnum, od.resnr};
    auto it = decomp_index.find(key);
    if (it != decomp_index.end()) {
      auto &md = decomp[it->second];
      md.E_vdw += od.E_vdw;
      md.E_elec += od.E_elec;
      md.G_polar += od.G_polar;
      md.G_nonpolar += od.G_nonpolar;
      md.total += od.total;
      md.count += od.count;
    } else {
      decomp_index[key] = decomp.size();
      decomp.emplace_back(od);
    }
  }
}

static TitanEntropy::InteractionEntropyResult
calc_gbsa_interaction_entropy(const std::vector<double> &energies,
                              const arguments &Arguments) {
  constexpr double kGasConstantKJPerMolK = 0.00831446261815324;
  const std::size_t block_size =
      Arguments.gbsa_ie_block == "auto"
          ? std::max<std::size_t>(
                1, static_cast<std::size_t>(
                       std::floor(std::sqrt(energies.size()))))
          : static_cast<std::size_t>(std::stoi(Arguments.gbsa_ie_block));
  TitanEntropy::InteractionEntropyOptions options;
  options.rt_energy =
      kGasConstantKJPerMolK * Arguments.gbsa_temperature;
  options.filter_sigma = Arguments.gbsa_ie_filter_sigma;
  options.min_samples =
      static_cast<std::size_t>(Arguments.gbsa_ie_min_samples);
  options.block_size = block_size;
  return TitanEntropy::compute_interaction_entropy(
      std::span<const double>(energies.data(), energies.size()), options);
}

void GBSA_Aggregator::commit_frame_decompositions(
    const GBSA_Aggregator &frame_data, size_t current_frame_idx) {
  // 1. Ensure all newly discovered residues from frame_data exist in global
  // decomp
  for (const auto &fd : frame_data.decomp) {
    DecompKey key{fd.is_ligand, fd.chainnum, fd.resnr};
    if (decomp_index.find(key) == decomp_index.end()) {
      GBSA_decomposition new_d = fd;
      new_d.total = 0;
      new_d.count = 0;
      new_d.E_vdw = 0;
      new_d.E_elec = 0;
      new_d.G_polar = 0;
      new_d.G_nonpolar = 0;
      new_d.frame_mm_energies.resize(current_frame_idx,
                                     0.0); // Pad previous frames with 0.0
      decomp_index[key] = decomp.size();
      decomp.emplace_back(std::move(new_d));
    }
  }

  // 2. Append current frame energy to ALL global residues
  for (auto &gd : decomp) {
    double current_energy = 0.0;
    DecompKey key{gd.is_ligand, gd.chainnum, gd.resnr};
    auto it = frame_data.decomp_index.find(key);
    if (it != frame_data.decomp_index.end()) {
      const auto &fd = frame_data.decomp[it->second];
      current_energy = fd.E_vdw + fd.E_elec;
      gd.E_vdw += fd.E_vdw;
      gd.E_elec += fd.E_elec;
      gd.G_polar += fd.G_polar;
      gd.G_nonpolar += fd.G_nonpolar;
      gd.total += fd.total;
      gd.count++;
    }
    gd.frame_mm_energies.push_back(current_energy);
  }
}

void GBSA_Aggregator::print_all_stats(const std::string &dir_prefix,
                                      std::fstream *log) {
  const bool capture_performance = Arguments.gbsa_perf != "off";
  const auto entropy_output_begin =
      capture_performance ? std::chrono::steady_clock::now()
                           : std::chrono::steady_clock::time_point{};
  std::string eu_str = "(" + Arguments.eu + ")";

  std::size_t alpb_size_frame_count = 0;
  long double alpb_receptor_size_sum = 0.0L;
  long double alpb_ligand_size_sum = 0.0L;
  long double alpb_complex_size_sum = 0.0L;
  double alpb_receptor_size_min = std::numeric_limits<double>::infinity();
  double alpb_ligand_size_min = std::numeric_limits<double>::infinity();
  double alpb_complex_size_min = std::numeric_limits<double>::infinity();
  double alpb_receptor_size_max = 0.0;
  double alpb_ligand_size_max = 0.0;
  double alpb_complex_size_max = 0.0;
  bool alpb_sizes_file_written = false;
  if (Arguments.flag_alpb) {
    for (const auto &trajectory : trajectories) {
      for (const auto &snapshot : trajectory) {
        ++alpb_size_frame_count;
        alpb_receptor_size_sum += snapshot.alpb_receptor_size_nm;
        alpb_ligand_size_sum += snapshot.alpb_ligand_size_nm;
        alpb_complex_size_sum += snapshot.alpb_complex_size_nm;
        alpb_receptor_size_min =
            std::min(alpb_receptor_size_min, snapshot.alpb_receptor_size_nm);
        alpb_ligand_size_min =
            std::min(alpb_ligand_size_min, snapshot.alpb_ligand_size_nm);
        alpb_complex_size_min =
            std::min(alpb_complex_size_min, snapshot.alpb_complex_size_nm);
        alpb_receptor_size_max =
            std::max(alpb_receptor_size_max, snapshot.alpb_receptor_size_nm);
        alpb_ligand_size_max =
            std::max(alpb_ligand_size_max, snapshot.alpb_ligand_size_nm);
        alpb_complex_size_max =
            std::max(alpb_complex_size_max, snapshot.alpb_complex_size_nm);
      }
    }
  }
  // 1. Raw Snapshots
  for (size_t t = 0; t < trajectories.size(); ++t) {
    std::string fname_snap = dir_prefix + "/traj_" + std::to_string(t) + "_" +
                             info_output.get_gbsa_snapshots_filename();
    std::fstream fsnap(fname_snap, std::ios::out);
    if (!fsnap.is_open()) {
      std::cerr << "[GBSA] ERROR: Cannot open output file: " << fname_snap
                << "\n";
      return;
    }
    fsnap << "Time(ps) | E_vdw" << eu_str << " | E_elec" << eu_str
          << " | G_polar" << eu_str << " | G_nonpolar" << eu_str << " | Total"
          << eu_str << "\n";
    for (const auto &s : trajectories[t]) {
      fsnap << std::fixed << std::setprecision(4) << s.time << " | "
            << s.E_vdw / Arguments.eu_par << " | "
            << s.E_elec / Arguments.eu_par << " | "
            << s.G_polar / Arguments.eu_par << " | "
            << s.G_nonpolar / Arguments.eu_par << " | "
            << s.total_energy / Arguments.eu_par << "\n";
    }
    fsnap.close();
  }

  if (Arguments.flag_alpb) {
    std::ofstream alpb_sizes(dir_prefix + "/gbsa_alpb_sizes.csv",
                             std::ios::out | std::ios::trunc);
    if (alpb_sizes.is_open()) {
      alpb_sizes
          << "trajectory_id,frame_index,time_ps,alpb_alpha_dimensionless,"
             "size_method,A_receptor_nm,A_ligand_nm,A_complex_nm\n";
      alpb_sizes << std::setprecision(12);
      for (std::size_t trajectory_id = 0;
           trajectory_id < trajectories.size(); ++trajectory_id) {
        for (std::size_t frame_index = 0;
             frame_index < trajectories[trajectory_id].size(); ++frame_index) {
          const auto &snapshot = trajectories[trajectory_id][frame_index];
          alpb_sizes << trajectory_id << ',' << frame_index << ','
                     << snapshot.time << ',' << titan_gb::kAlpbAlpha
                     << ",ADet_Sigalov_Eq23,"
                     << snapshot.alpb_receptor_size_nm << ','
                     << snapshot.alpb_ligand_size_nm << ','
                     << snapshot.alpb_complex_size_nm << '\n';
        }
      }
      alpb_sizes.flush();
      alpb_sizes_file_written = alpb_sizes.good();
      if (!alpb_sizes_file_written) {
        TitanUI::print_warning(
            "GBSA: incomplete write of gbsa_alpb_sizes.csv.");
      }
    } else {
      TitanUI::print_warning("GBSA: failed to write gbsa_alpb_sizes.csv.");
    }
  }

  // 2. Time-series Average
  std::string fname_ts =
      dir_prefix + "/" + info_output.get_gbsa_timeseries_avg_filename();
  std::fstream fts(fname_ts, std::ios::out);
  if (!fts.is_open()) {
    std::cerr << "[GBSA] ERROR: Cannot open output file: " << fname_ts << "\n";
    return;
  }
  fts << "Time(ps) | Avg_Total" << eu_str << " | SD_Total" << eu_str
      << " | Avg_VDW" << eu_str << " | Avg_ELEC" << eu_str << " | Avg_GPOL"
      << eu_str << " | Avg_GNONPOL" << eu_str << "\n";

  if (!trajectories.empty()) {
    size_t num_frames = 0;
    for (const auto &traj : trajectories) {
      num_frames = std::max(num_frames, traj.size());
    }
    for (size_t f = 0; f < num_frames; ++f) {
      double avg_tot = 0., avg_vdw = 0., avg_elec = 0., avg_gpol = 0.,
             avg_gnpol = 0.;
      double avg_time_ps = 0.0;
      int count = 0;
      for (size_t t = 0; t < trajectories.size(); ++t) {
        if (f < trajectories[t].size()) {
          avg_time_ps += trajectories[t][f].time;
          avg_tot += trajectories[t][f].total_energy;
          avg_vdw += trajectories[t][f].E_vdw;
          avg_elec += trajectories[t][f].E_elec;
          avg_gpol += trajectories[t][f].G_polar;
          avg_gnpol += trajectories[t][f].G_nonpolar;
          count++;
        }
      }
      if (count > 0) {
        avg_time_ps /= count;
        avg_tot /= count;
        avg_vdw /= count;
        avg_elec /= count;
        avg_gpol /= count;
        avg_gnpol /= count;
        double var_tot = 0.;
        for (size_t t = 0; t < trajectories.size(); ++t) {
          if (f < trajectories[t].size()) {
            var_tot += (trajectories[t][f].total_energy - avg_tot) *
                       (trajectories[t][f].total_energy - avg_tot);
          }
        }
        double sd_tot = std::sqrt(var_tot / count);
        fts << std::fixed << std::setprecision(4) << avg_time_ps << " | "
            << avg_tot / Arguments.eu_par << " | " << sd_tot / Arguments.eu_par
            << " | " << avg_vdw / Arguments.eu_par << " | "
            << avg_elec / Arguments.eu_par << " | "
            << avg_gpol / Arguments.eu_par << " | "
            << avg_gnpol / Arguments.eu_par << "\n";
      }
    }
  }
  fts.close();

  // 3. Summary File & Calculate Total Frames
  std::string fname_sum =
      dir_prefix + "/" + info_output.get_gbsa_summary_filename();
  std::fstream fsum(fname_sum, std::ios::out);
  if (!fsum.is_open()) {
    std::cerr << "[GBSA] ERROR: Cannot open output file: " << fname_sum << "\n";
    return;
  }
  fsum << "energy_distance_policy preprocessed_raw\n";
  fsum << "gbsa_igb " << igb << "\n";
  if (igb == 8) {
    fsum << "gbsa_model GBn2\ngbsa_radii mbondi3\ngbsa_offset_angstrom 0.195141\n";
    fsum << "gbn2_neck_index nearest_0.05A_Amber\ngbn2_parameters Amber_protein_and_nucleic\n";
    fsum << "gbn2_nucleic_aliases ADE:A,CYT:C,GUA:G,URA:U,THY:DT\n";
  } else if (igb == 66) {
    fsum << "gbsa_model GBNSR6\ngbsa_model_status experimental_SES_quadrature\n";
    fsum << "gbsa_radii Bondi\ngbnsr6_surface analytic_SES_patch_ray_experimental_external_only\n";
    fsum << "gbnsr6_spacing_angstrom " << Arguments.gbnsr6_spacing << "\n";
    fsum << "gbnsr6_probe_angstrom 1.4\ngbnsr6_B_inverse_angstrom 0.028\n";
    fsum << "gbnsr6_salt_convention physical_PB_kappa\n";
  }
  fsum << "gbsa_sasa_point_count " << Arguments.gbsa_sasa_points << "\n";
  // The nonpolar model constant belongs in provenance: it differs between
  // Amber contexts (MMPBSA.py GB 0.0072, sander gbsa=1 0.005).
  fsum << "gbsa_surften_kcal_per_mol_a2 " << Arguments.gbsa_surften << "\n";
  fsum << "gbsa_nonpolar_model gamma_times_delta_sasa\n";
  // Born-radius descreening cutoff is a model choice, so it belongs in the
  // provenance block next to the SASA discretisation.
  fsum << "gbsa_rgbmax_angstrom ";
  if (std::isfinite(Arguments.gbsa_rgbmax))
    fsum << Arguments.gbsa_rgbmax << "\n";
  else
    fsum << "off\n";
  if (Arguments.flag_alpb) {
    const std::streamsize previous_precision = fsum.precision();
    const double size_count = static_cast<double>(alpb_size_frame_count);
    fsum << "alpb_enabled true\n";
    fsum << std::setprecision(12);
    fsum << "alpb_alpha_dimensionless " << titan_gb::kAlpbAlpha << "\n";
    fsum << "alpb_formula Sigalov_Eq25_salt_screened\n";
    fsum << "alpb_size_method ADet_Sigalov_Eq23\n";
    fsum << "alpb_size_units nm\n";
    // The direct Coulomb term carries the same 1/(1+alpha*beta) factor as the
    // GB term, matching sander's raw ALPB convention. Recorded because it
    // changes E_elec and the total relative to a vacuum-Coulomb convention.
    fsum << "alpb_coulomb_scaling " << (igb == 66 ? "none_GBNSR6" : "one_over_one_plus_alpha_beta") << "\n";
    fsum << "alpb_coulomb_scale_factor "
         << (igb == 66 ? 1.0 : 1.0 / (1.0 + titan_gb::kAlpbAlpha *
                             (Arguments.int_diel / Arguments.ext_diel)))
         << "\n";
    fsum << "alpb_sizes_export_status "
         << (alpb_sizes_file_written ? "ok" : "failed") << "\n";
    fsum << "alpb_sizes_file "
         << (alpb_sizes_file_written ? "gbsa_alpb_sizes.csv" : "unavailable")
         << "\n";
    if (alpb_size_frame_count > 0) {
      fsum << "alpb_A_receptor_nm_mean "
           << static_cast<double>(alpb_receptor_size_sum / size_count) << "\n";
      fsum << "alpb_A_receptor_nm_range " << alpb_receptor_size_min << ' '
           << alpb_receptor_size_max << "\n";
      fsum << "alpb_A_ligand_nm_mean "
           << static_cast<double>(alpb_ligand_size_sum / size_count) << "\n";
      fsum << "alpb_A_ligand_nm_range " << alpb_ligand_size_min << ' '
           << alpb_ligand_size_max << "\n";
      fsum << "alpb_A_complex_nm_mean "
           << static_cast<double>(alpb_complex_size_sum / size_count) << "\n";
      fsum << "alpb_A_complex_nm_range " << alpb_complex_size_min << ' '
           << alpb_complex_size_max << "\n";
    }
    fsum.precision(previous_precision);
  }
  fsum << "interaction_entropy_source E_MM_elec_plus_vdw\n";
  fsum << "interaction_entropy_temperature_K " << Arguments.gbsa_temperature
       << "\n";
  fsum << "interaction_entropy_min_samples "
       << Arguments.gbsa_ie_min_samples << "\n";
  fsum << "interaction_entropy_filter_sigma "
       << Arguments.gbsa_ie_filter_sigma
       << " # 0 means no hard filtering\n";
  fsum << "interaction_entropy_block_config " << Arguments.gbsa_ie_block
       << "\n";
  fsum << "GRAND AVERAGE OVERALL TRAJECTORIES AND FRAMES " << eu_str << ":\n";
  double g_tot = 0., g_vdw = 0., g_elec = 0., g_pol = 0., g_nonpol = 0.;

  int total_frames = 0;
  std::vector<double> global_mm_energies;

  for (size_t t = 0; t < trajectories.size(); ++t) {
    total_frames += trajectories[t].size();
    for (size_t f = 0; f < trajectories[t].size(); ++f) {
      g_tot += trajectories[t][f].total_energy;
      g_vdw += trajectories[t][f].E_vdw;
      g_elec += trajectories[t][f].E_elec;
      g_pol += trajectories[t][f].G_polar;
      g_nonpol += trajectories[t][f].G_nonpolar;

      global_mm_energies.push_back(trajectories[t][f].E_vdw +
                                   trajectories[t][f].E_elec);
    }
  }

  if (total_frames > 0) {
    double avg_tot = (g_tot / total_frames) / Arguments.eu_par;
    double avg_vdw = (g_vdw / total_frames) / Arguments.eu_par;
    double avg_elec = (g_elec / total_frames) / Arguments.eu_par;
    double avg_gpol = (g_pol / total_frames) / Arguments.eu_par;
    double avg_gnonpol = (g_nonpol / total_frames) / Arguments.eu_par;

    const TitanEntropy::InteractionEntropyResult global_entropy =
        calc_gbsa_interaction_entropy(global_mm_energies, Arguments);
    const bool entropy_ok = global_entropy.status == "ok";
    const double global_minus_TdS =
        entropy_ok ? global_entropy.minus_tds_energy / Arguments.eu_par
                   : std::numeric_limits<double>::quiet_NaN();
    const double global_dG_binding =
        entropy_ok ? avg_tot + global_minus_TdS
                   : std::numeric_limits<double>::quiet_NaN();

    fsum << "Total(dH): " << avg_tot << " | VDW: " << avg_vdw
         << " | Elec: " << avg_elec << " | Gpol: " << avg_gpol
         << " | Gnonpol: " << avg_gnonpol << "\n";
    fsum << "IE_status: " << global_entropy.status
         << " | samples: " << global_entropy.samples
         << " | filtered_samples: " << global_entropy.filtered_samples
         << " | ESS_lag1: " << global_entropy.ess
         << " | block_size: " << global_entropy.block_size
         << " | block_count: " << global_entropy.block_count
         << " | block_std: "
         << global_entropy.block_std_energy / Arguments.eu_par
         << " | block_drift: "
         << global_entropy.block_drift_energy / Arguments.eu_par << "\n";
    fsum << "-TdS_IE  : " << global_minus_TdS << "\n";
    fsum << "Final_dG : " << global_dG_binding
         << " # nan unless IE_status=ok\n\n";
  }

  fsum << "AVERAGE PER TRAJECTORY " << eu_str << ":\n";
  for (size_t t = 0; t < trajectories.size(); ++t) {
    double t_tot = 0.;
    std::vector<double> traj_mm_energies;
    for (const auto &s : trajectories[t]) {
      t_tot += s.total_energy;
      traj_mm_energies.push_back(s.E_vdw + s.E_elec);
    }
    if (!trajectories[t].empty()) {
      double t_avg = (t_tot / trajectories[t].size()) / Arguments.eu_par;
      const TitanEntropy::InteractionEntropyResult trajectory_entropy =
          calc_gbsa_interaction_entropy(traj_mm_energies, Arguments);
      const bool entropy_ok = trajectory_entropy.status == "ok";
      const double t_minus_TdS =
          entropy_ok
              ? trajectory_entropy.minus_tds_energy / Arguments.eu_par
              : std::numeric_limits<double>::quiet_NaN();
      const double t_dG =
          entropy_ok ? t_avg + t_minus_TdS
                     : std::numeric_limits<double>::quiet_NaN();

      fsum << "Trajectory " << t << " Avg Total(dH) = " << t_avg
           << " | IE_status = " << trajectory_entropy.status
           << " | samples = " << trajectory_entropy.samples
           << " | ESS_lag1 = " << trajectory_entropy.ess
           << " | -TdS_IE = " << t_minus_TdS << " | Final dG = " << t_dG
           << "\n";
    }
  }
  fsum.close();

  if (Arguments.flag_alpb && log != nullptr && log->is_open()) {
    std::ostringstream alpb_alpha_text;
    alpb_alpha_text << std::setprecision(12) << titan_gb::kAlpbAlpha
                    << " (dimensionless)";
    TitanUI::log_section(*log, "MM/GBSA ALPB PROVENANCE");
    TitanUI::log_kv(*log, "ALPB Enabled", "true");
    TitanUI::log_kv(*log, "ALPB Alpha", alpb_alpha_text.str());
    TitanUI::log_kv(*log, "ALPB Formula", "Sigalov Eq.25 salt-screened");
    TitanUI::log_kv(*log, "Electrostatic Size", "ADet, Sigalov Eq.23");
    TitanUI::log_kv(*log, "Per-frame Size Artifact",
                    alpb_sizes_file_written ? "gbsa_alpb_sizes.csv (nm)"
                                            : "export failed");
    if (alpb_size_frame_count > 0) {
      const long double count =
          static_cast<long double>(alpb_size_frame_count);
      TitanUI::log_kv(
          *log, "A Receptor Mean (nm)",
          std::to_string(static_cast<double>(alpb_receptor_size_sum / count)));
      TitanUI::log_kv(
          *log, "A Ligand Mean (nm)",
          std::to_string(static_cast<double>(alpb_ligand_size_sum / count)));
      TitanUI::log_kv(
          *log, "A Complex Mean (nm)",
          std::to_string(static_cast<double>(alpb_complex_size_sum / count)));
    }
  }

  std::ofstream ie_diag(dir_prefix + "/gbsa_ie_diagnostics.csv",
                        std::ios::out | std::ios::trunc);
  if (ie_diag.is_open()) {
    ie_diag << "scope,trajectory_id,samples,filtered_samples,ess,status,"
               "block_size,block_count,minus_TDeltaS_IE_kJmol,"
               "block_mean_kJmol,block_std_kJmol,block_drift_kJmol\n";
    const auto write_ie = [&](const char *scope, size_t trajectory_id,
                              const std::vector<double> &values) {
      const TitanEntropy::InteractionEntropyResult entropy =
          calc_gbsa_interaction_entropy(values, Arguments);
      const double minus_tds =
          entropy.status == "ok"
              ? entropy.minus_tds_energy
              : std::numeric_limits<double>::quiet_NaN();
      ie_diag << scope << ',' << trajectory_id << ',' << entropy.samples << ','
              << entropy.filtered_samples << ',' << entropy.ess << ','
              << entropy.status << ',' << entropy.block_size << ','
              << entropy.block_count << ',' << minus_tds << ','
              << entropy.block_mean_energy << ',' << entropy.block_std_energy
              << ',' << entropy.block_drift_energy << '\n';
    };
    write_ie("global", 0, global_mm_energies);
    for (size_t t = 0; t < trajectories.size(); ++t) {
      std::vector<double> values;
      values.reserve(trajectories[t].size());
      for (const auto &snapshot : trajectories[t])
        values.push_back(snapshot.E_vdw + snapshot.E_elec);
      write_ie("trajectory", t, values);
    }
  } else {
    TitanUI::print_warning("GBSA: failed to write gbsa_ie_diagnostics.csv.");
  }

  // 4. Decomposition (Split into Receptor and Ligand)
  std::string base_dec =
      dir_prefix + "/" + info_output.get_gbsa_decomposition_filename();

  size_t dot_pos = base_dec.find_last_of(".");
  std::string prefix_dec =
      (dot_pos != std::string::npos) ? base_dec.substr(0, dot_pos) : base_dec;

  std::string fname_dec_rec = prefix_dec + "_receptor.dat";
  std::string fname_dec_lig = prefix_dec + "_ligand.dat";

  std::fstream fdec_rec(fname_dec_rec, std::ios::out);
  if (!fdec_rec.is_open()) {
    std::cerr << "[GBSA] ERROR: Cannot open output file: " << fname_dec_rec
              << "\n";
    return;
  }
  std::fstream fdec_lig(fname_dec_lig, std::ios::out);
  if (!fdec_lig.is_open()) {
    std::cerr << "[GBSA] ERROR: Cannot open output file: " << fname_dec_lig
              << "\n";
    return;
  }

  std::string header = "Chain | ID | ResName/AtomName | E_vdw" + eu_str +
                       " | E_elec" + eu_str + " | G_polar" + eu_str +
                       " | G_nonpolar" + eu_str + " | Total" + eu_str +
                       " | IE_status | IE_samples | IE_ESS | -TdS_IE" +
                       eu_str + " | dG_Binding" + eu_str + "\n";
  fdec_rec << "# interaction_entropy_source E_MM_elec_plus_vdw\n" << header;
  fdec_lig << "# interaction_entropy_source E_MM_elec_plus_vdw\n" << header;

  // FIX: Chia năng lượng phân rã cho total_frames thay vì d.count
  double div = (total_frames > 0) ? (double)total_frames : 1.0;

  for (const auto &d : decomp) {
    // Chỉ in ra những unit thực sự có dữ liệu
    if (d.count > 0) {
      // Enforce user-requested unit conversion
      double avg_vdw = (d.E_vdw / div) / Arguments.eu_par;
      double avg_elec = (d.E_elec / div) / Arguments.eu_par;
      double avg_gpol = (d.G_polar / div) / Arguments.eu_par;
      double avg_gnp = (d.G_nonpolar / div) / Arguments.eu_par;
      double avg_total = (d.total / div) / Arguments.eu_par;

      const TitanEntropy::InteractionEntropyResult entropy =
          calc_gbsa_interaction_entropy(d.frame_mm_energies, Arguments);
      const bool entropy_ok = entropy.status == "ok";
      const double minus_TdS =
          entropy_ok ? entropy.minus_tds_energy / Arguments.eu_par
                     : std::numeric_limits<double>::quiet_NaN();
      const double dG_binding =
          entropy_ok ? avg_total + minus_TdS
                     : std::numeric_limits<double>::quiet_NaN();

      std::fstream &target_file = d.is_ligand ? fdec_lig : fdec_rec;
      target_file << std::fixed << d.chain_str << " | " << std::setprecision(4)
                  << d.resnr << " | " << d.name_str << " | " << avg_vdw << " | "
                  << avg_elec << " | " << avg_gpol << " | " << avg_gnp << " | "
                  << avg_total << " | " << entropy.status << " | "
                  << entropy.samples << " | " << entropy.ess << " | "
                  << minus_TdS << " | " << dG_binding << "\n";
    }
  }

  fdec_rec.close();
  fdec_lig.close();

  if (!capture_performance)
    return;
  const double entropy_output_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - entropy_output_begin)
          .count();

  double precompute_ms = 0.0;
  double energy_kernel_ms = 0.0;
  double decomp_merge_ms = 0.0;
  double total_ms = 0.0;
  size_t frame_count = 0;
  for (const auto &trajectory : trajectories) {
    for (const auto &snapshot : trajectory) {
      precompute_ms += snapshot.time_precompute_ms;
      energy_kernel_ms += snapshot.time_energy_kernel_ms;
      decomp_merge_ms += snapshot.time_decomposition_merge_ms;
      total_ms += snapshot.time_total_ms;
      ++frame_count;
    }
  }

  std::ofstream perf(dir_prefix + "/gbsa_performance.dat",
                     std::ios::out | std::ios::trunc);
  if (perf.is_open()) {
    perf << "# TITAN MM/GBSA performance summary\n";
    perf << "timing_granularity natural_boundaries_fused_energy_kernel\n";
    perf << "interaction_entropy_source E_MM_elec_plus_vdw\n";
    perf << "frames " << frame_count << "\n";
    perf << std::fixed << std::setprecision(6);
    perf << "gbsa_precompute_ms_total " << precompute_ms << "\n";
    perf << "gbsa_energy_kernel_ms_total " << energy_kernel_ms << "\n";
    perf << "decomposition_merge_ms_total " << decomp_merge_ms << "\n";
    perf << "entropy_output_ms " << entropy_output_ms << "\n";
    perf << "frame_total_ms_total " << total_ms << "\n";
    perf << "total_ms_total " << total_ms << "\n";
  } else {
    TitanUI::print_warning("GBSA: failed to write gbsa_performance.dat.");
  }

  if (Arguments.gbsa_perf == "detail") {
    std::ofstream frames(dir_prefix + "/gbsa_performance_frames.csv",
                         std::ios::out | std::ios::trunc);
    if (frames.is_open()) {
      frames << "trajectory_id,frame_index,time_ps,gbsa_precompute_ms,"
                "gbsa_energy_kernel_ms,decomposition_merge_ms,"
                "frame_total_ms,status\n";
      frames << std::setprecision(10);
      for (size_t trajectory_id = 0; trajectory_id < trajectories.size();
           ++trajectory_id) {
        for (size_t frame_index = 0;
             frame_index < trajectories[trajectory_id].size(); ++frame_index) {
          const auto &snapshot = trajectories[trajectory_id][frame_index];
          frames << trajectory_id << ',' << frame_index << ',' << snapshot.time
                 << ',' << snapshot.time_precompute_ms << ','
                 << snapshot.time_energy_kernel_ms << ','
                 << snapshot.time_decomposition_merge_ms << ','
                 << snapshot.time_total_ms << ",ok\n";
        }
      }
    } else {
      TitanUI::print_warning(
          "GBSA: failed to write gbsa_performance_frames.csv.");
    }
  }

  if (log != nullptr && log->is_open()) {
    TitanUI::log_section(*log, "MM/GBSA COMPONENT TIMING");
    TitanUI::log_kv(*log, "Timing Granularity",
                    "natural_boundaries_fused_energy_kernel");
    TitanUI::log_kv(*log, "IE Source", "E_MM_elec_plus_vdw");
    TitanUI::log_time(*log, "GBSA Precompute", precompute_ms / 1000.0);
    TitanUI::log_time(*log, "GBSA Energy Kernel (MM + GB pair fused)",
                      energy_kernel_ms / 1000.0);
    TitanUI::log_time(*log, "Decomposition Merge",
                      decomp_merge_ms / 1000.0);
    TitanUI::log_time(*log, "Entropy + Output", entropy_output_ms / 1000.0);
    TitanUI::log_time(*log, "Frame Total", total_ms / 1000.0);
  }
}
