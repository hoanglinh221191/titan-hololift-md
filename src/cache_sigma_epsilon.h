#ifndef CACHE_SIGMA_EPSILON_H
#define CACHE_SIGMA_EPSILON_H

#include "arguments.h"
#include "aligned_allocator.h"
#include "c6c12.h"
#include "miscellaneous.h"
#include <cstdint>
#include <optional>
#include <memory>
#include <string>
#include <vector>

namespace titan_nb { struct Topology; }

// ---------------------------------------------------------------------------
// cache_energy — lightweight energy pair
// ---------------------------------------------------------------------------
struct cache_energy {
  double elec_ener{0.0};
  double vdw_ener{0.0};

  cache_energy() = default;
  cache_energy(double e, double v) : elec_ener(e), vdw_ener(v) {}

  void set_energy(double e, double v) {
    elec_ener = e;
    vdw_ener = v;
  }

  cache_energy &operator+=(const cache_energy &other) {
    elec_ener += other.elec_ener;
    vdw_ener += other.vdw_ener;
    return *this;
  }
};

#include "force.h"

// ---------------------------------------------------------------------------
// cache_force — pair of force vectors (electrostatic + vdW)
// ---------------------------------------------------------------------------
struct cache_force {
  force forces[2];

  cache_force() = default;
  cache_force(double ex, double ey, double ez, double vx, double vy,
              double vz) {
    set_force(ex, ey, ez, vx, vy, vz);
  }
  void set_force(const force &f1, const force &f2) {
    forces[0] = f1;
    forces[1] = f2;
  }

  void set_force(double e_x, double e_y, double e_z, double v_x, double v_y,
                 double v_z) {
    forces[0].Fx = e_x;
    forces[0].Fy = e_y;
    forces[0].Fz = e_z;
    forces[1].Fx = v_x;
    forces[1].Fy = v_y;
    forces[1].Fz = v_z;
  }

  cache_force &operator+=(const cache_force &other) {
    forces[0] += other.forces[0];
    forces[1] += other.forces[1];
    return *this;
  }
};

// ---------------------------------------------------------------------------
// cache_sigma_epsilon — SoA cache for pairwise interaction energies & forces
//
// Force arrays are wrapped in std::optional<ForceArrays> to guarantee zero
// memory allocation when force calculations are not requested.
// ---------------------------------------------------------------------------
struct cache_sigma_epsilon {

  std::shared_ptr<const titan_nb::Topology> nonbonded_topology;

  // --- Nested sub-struct: groups all 6 force arrays for cache locality ---
  struct ForceArrays {
    AlignedDoubleVector elec_x;
    AlignedDoubleVector elec_y;
    AlignedDoubleVector elec_z;
    AlignedDoubleVector vdw_x;
    AlignedDoubleVector vdw_y;
    AlignedDoubleVector vdw_z;

    void assign(std::size_t n, double val = 0.0) {
      elec_x.assign(n, val);
      elec_y.assign(n, val);
      elec_z.assign(n, val);
      vdw_x.assign(n, val);
      vdw_y.assign(n, val);
      vdw_z.assign(n, val);
    }
  };

  // 1D Arrays for Energy (always allocated)
  AlignedDoubleVector elec_ener;
  AlignedDoubleVector vdw_ener;

  // Force arrays: std::nullopt when forces are not requested
  std::optional<ForceArrays> opt_forces;

  std::vector<double> charge;
  std::string force_field_name;
  double cutoff_dis{-1.1};
  std::vector<double> charmm_pair_sigma;
  std::vector<double> charmm_pair_epsilon;
  int charmm_type_count{0};
  std::vector<std::uint64_t> charmm_pair_sparse_keys;
  std::vector<double> charmm_pair_sparse_sigma;
  std::vector<double> charmm_pair_sparse_epsilon;
  std::vector<double> martini_pair_sigma;
  std::vector<double> martini_pair_epsilon;
  int martini_type_count{0};

  // --- Constructors ---

  cache_sigma_epsilon() = default;

  cache_sigma_epsilon(double dis_cutoff)
      : force_field_name{"amber99sb"}, cutoff_dis{dis_cutoff} {}

  // Always-allocate constructor (legacy: unconditionally allocates forces)
  cache_sigma_epsilon(std::size_t MOL_size)
      : force_field_name{"amber99sb"} {
    elec_ener.assign(MOL_size, 0.0);
    vdw_ener.assign(MOL_size, 0.0);
    opt_forces.emplace();
    opt_forces->assign(MOL_size);
  }

  // Conditional constructor: only allocates forces when flag_cal_force == 1
  cache_sigma_epsilon(std::size_t MOL_size, const arguments &Arguments)
      : force_field_name{"amber99sb"} {
    elec_ener.assign(MOL_size, 0.0);
    vdw_ener.assign(MOL_size, 0.0);
    if (Arguments.flag_cal_force == 1) {
      opt_forces.emplace();
      opt_forces->assign(MOL_size);
    }
  }

  // --- Query ---

  [[nodiscard]] bool has_forces() const noexcept {
    return opt_forces.has_value();
  }
  [[nodiscard]] bool has_charmm_pair_table() const noexcept {
    return charmm_type_count > 0 &&
           charmm_pair_sigma.size() ==
               static_cast<std::size_t>(charmm_type_count) *
                   static_cast<std::size_t>(charmm_type_count) &&
           charmm_pair_epsilon.size() == charmm_pair_sigma.size();
  }
  [[nodiscard]] bool has_charmm_pair_sparse() const noexcept {
    return !charmm_pair_sparse_keys.empty() &&
           charmm_pair_sparse_keys.size() == charmm_pair_sparse_sigma.size() &&
           charmm_pair_sparse_keys.size() == charmm_pair_sparse_epsilon.size();
  }
  [[nodiscard]] bool has_charmm_pair_overrides() const noexcept {
    return has_charmm_pair_table() || has_charmm_pair_sparse();
  }
  [[nodiscard]] bool has_martini_pair_table() const noexcept {
    return martini_type_count > 0 &&
           martini_pair_sigma.size() ==
               static_cast<std::size_t>(martini_type_count) *
                   static_cast<std::size_t>(martini_type_count) &&
           martini_pair_epsilon.size() == martini_pair_sigma.size();
  }

  // --- set_size (defined out-of-line in .cpp) ---

  int set_size(std::size_t MOL_size, const arguments &Arguments,
               std::size_t mol_1_size, std::size_t mol_2_size);

  // --- Energy bulk-set (AVX2-friendly, __restrict for full vectorization) ---

  void set_elec_vdw(std::size_t pos, const double *__restrict elec,
                    const double *__restrict vdw, std::size_t sz) {
    double *__restrict dest_elec = elec_ener.data() + pos;
    double *__restrict dest_vdw  = vdw_ener.data() + pos;
#pragma omp simd
    for (std::size_t i = 0; i < sz; ++i) {
      dest_elec[i] = elec[i];
      dest_vdw[i]  = vdw[i];
    }
  }

  void set_elec_vdw(std::size_t pos, double elec, double vdw) {
    elec_ener[pos] = elec;
    vdw_ener[pos] = vdw;
  }

  // --- Energy getter ---

  [[nodiscard]] cache_energy get_energy(std::size_t pos) const {
    return {elec_ener[pos], vdw_ener[pos]};
  }

  // --- Force getter: returns std::optional<cache_force> ---

  [[nodiscard]] std::optional<cache_force> get_force(std::size_t pos) const {
    if (!opt_forces.has_value())
      return std::nullopt;
    cache_force f;
    f.set_force(opt_forces->elec_x[pos], opt_forces->elec_y[pos],
                opt_forces->elec_z[pos], opt_forces->vdw_x[pos],
                opt_forces->vdw_y[pos], opt_forces->vdw_z[pos]);
    return f;
  }

  // --- Force setter ---

  void set_force(std::size_t pos, double ex, double ey, double ez, double vx,
                 double vy, double vz) {
    if (!opt_forces.has_value())
      return;
    opt_forces->elec_x[pos] = ex;
    opt_forces->elec_y[pos] = ey;
    opt_forces->elec_z[pos] = ez;
    opt_forces->vdw_x[pos] = vx;
    opt_forces->vdw_y[pos] = vy;
    opt_forces->vdw_z[pos] = vz;
  }

  // --- Raw pointer getters for AVX2 compute loops ---
  //     Returns nullptr when forces are not allocated.

  [[nodiscard]] double *__restrict ptr_fx_elec() noexcept {
    return opt_forces.has_value() ? opt_forces->elec_x.data() : nullptr;
  }
  [[nodiscard]] double *__restrict ptr_fy_elec() noexcept {
    return opt_forces.has_value() ? opt_forces->elec_y.data() : nullptr;
  }
  [[nodiscard]] double *__restrict ptr_fz_elec() noexcept {
    return opt_forces.has_value() ? opt_forces->elec_z.data() : nullptr;
  }
  [[nodiscard]] double *__restrict ptr_fx_vdw() noexcept {
    return opt_forces.has_value() ? opt_forces->vdw_x.data() : nullptr;
  }
  [[nodiscard]] double *__restrict ptr_fy_vdw() noexcept {
    return opt_forces.has_value() ? opt_forces->vdw_y.data() : nullptr;
  }
  [[nodiscard]] double *__restrict ptr_fz_vdw() noexcept {
    return opt_forces.has_value() ? opt_forces->vdw_z.data() : nullptr;
  }

  // Const overloads
  [[nodiscard]] const double *__restrict ptr_fx_elec() const noexcept {
    return opt_forces.has_value() ? opt_forces->elec_x.data() : nullptr;
  }
  [[nodiscard]] const double *__restrict ptr_fy_elec() const noexcept {
    return opt_forces.has_value() ? opt_forces->elec_y.data() : nullptr;
  }
  [[nodiscard]] const double *__restrict ptr_fz_elec() const noexcept {
    return opt_forces.has_value() ? opt_forces->elec_z.data() : nullptr;
  }
  [[nodiscard]] const double *__restrict ptr_fx_vdw() const noexcept {
    return opt_forces.has_value() ? opt_forces->vdw_x.data() : nullptr;
  }
  [[nodiscard]] const double *__restrict ptr_fy_vdw() const noexcept {
    return opt_forces.has_value() ? opt_forces->vdw_y.data() : nullptr;
  }
  [[nodiscard]] const double *__restrict ptr_fz_vdw() const noexcept {
    return opt_forces.has_value() ? opt_forces->vdw_z.data() : nullptr;
  }

  // --- Raw pointer getters for Energy (always available) ---

  [[nodiscard]] double *__restrict ptr_elec() noexcept {
    return elec_ener.data();
  }
  [[nodiscard]] double *__restrict ptr_vdw() noexcept {
    return vdw_ener.data();
  }
  [[nodiscard]] const double *__restrict ptr_elec() const noexcept {
    return elec_ener.data();
  }
  [[nodiscard]] const double *__restrict ptr_vdw() const noexcept {
    return vdw_ener.data();
  }

  void set_charmm_pair_table(std::vector<double> sigma_table,
                             std::vector<double> epsilon_table,
                             int type_count) {
    charmm_pair_sigma = std::move(sigma_table);
    charmm_pair_epsilon = std::move(epsilon_table);
    charmm_type_count = type_count;
    charmm_pair_sparse_keys.clear();
    charmm_pair_sparse_sigma.clear();
    charmm_pair_sparse_epsilon.clear();
  }

  void set_charmm_pair_sparse(std::vector<std::uint64_t> keys,
                              std::vector<double> sigma_values,
                              std::vector<double> epsilon_values) {
    charmm_pair_sigma.clear();
    charmm_pair_epsilon.clear();
    charmm_type_count = 0;
    charmm_pair_sparse_keys = std::move(keys);
    charmm_pair_sparse_sigma = std::move(sigma_values);
    charmm_pair_sparse_epsilon = std::move(epsilon_values);
  }

  void set_martini_pair_table(std::vector<double> sigma_table,
                              std::vector<double> epsilon_table,
                              int type_count) {
    martini_pair_sigma = std::move(sigma_table);
    martini_pair_epsilon = std::move(epsilon_table);
    martini_type_count = type_count;
  }

  [[nodiscard]] const double *charmm_sigma_ptr() const noexcept {
    return charmm_pair_sigma.data();
  }
  [[nodiscard]] const double *charmm_epsilon_ptr() const noexcept {
    return charmm_pair_epsilon.data();
  }
  [[nodiscard]] int charmm_types() const noexcept { return charmm_type_count; }
  [[nodiscard]] const std::vector<std::uint64_t> &
  charmm_sparse_keys() const noexcept {
    return charmm_pair_sparse_keys;
  }
  [[nodiscard]] const std::vector<double> &charmm_sparse_sigma() const noexcept {
    return charmm_pair_sparse_sigma;
  }
  [[nodiscard]] const std::vector<double> &
  charmm_sparse_epsilon() const noexcept {
    return charmm_pair_sparse_epsilon;
  }

  [[nodiscard]] const double *martini_sigma_ptr() const noexcept {
    return martini_pair_sigma.data();
  }
  [[nodiscard]] const double *martini_epsilon_ptr() const noexcept {
    return martini_pair_epsilon.data();
  }
  [[nodiscard]] int martini_types() const noexcept { return martini_type_count; }
};

#endif // !CACHE_SIGMA_EPSILON_H
