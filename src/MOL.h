#ifndef MOL_H
#define MOL_H

#include "COM.h"
#include "MOL_info.h"
#include "MOL_para.h"
#include "amino_a.h"
#include "arguments.h"
#include "box_type.h"
#include "c6c12.h"
#include "differential_evolution.h"
#include "force.h"
#include "h_bond.h"
#include "index_v2.h"
#include "miscellaneous.h"
#include "titan_simd_dispatch.h"
#include "uti_func.h"
#include <string_view>
using std::string_view;

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using std::pair;
using std::string;
using std::vector;

class MOL;
void solve_pbc(MOL &mol_1, MOL &mol_2, box &box_type);
class Multi_MOL;

// Structure of Arrays (SoA) for 3D coordinates/vectors
struct AlignedVector3D {
  AlignedDoubleVector x;
  AlignedDoubleVector y;
  AlignedDoubleVector z;

  void resize(size_t size) {
    x.resize(size, 0.0);
    y.resize(size, 0.0);
    z.resize(size, 0.0);
  }

  void clear() noexcept {
    x.clear();
    y.clear();
    z.clear();
  }

  void shrink_to_fit() noexcept {
    x.shrink_to_fit();
    y.shrink_to_fit();
    z.shrink_to_fit();
  }

  [[nodiscard]] size_t size() const noexcept { return x.size(); }
};

class MOL {
private:
  int cal_electric_field_1atom(const vector<mol_info> &info,
                               const arguments &Arguments,
                               const Multi_MOL_parameters &parameters,
                               force &electric_field_vector,
                               const vector<size_t> &list_heavy_atoms,
                               size_t pos_o_water) const;
  int helper_cal_electric_field(const vector<mol_info> &info,
                                const arguments &Arguments,
                                const Multi_MOL_parameters &parameters,
                                vector<force> &elec_field,
                                const vector<size_t> &list_heavy_atoms,
                                const vector<size_t> &pos_o_waters,
                                size_t begin, size_t end) const;
  int cal_dipoles_specific_atoms(const vector<size_t> &list_atoms,
                                 const Multi_MOL_parameters &parameters,
                                 const vector<mol_info> &info,
                                 vector<force> &dipoles_out,
                                 std::string_view atoms_type) const;
  int cal_dipoles_water_molecules(const vector<size_t> &list_atoms,
                                  const Multi_MOL_parameters &parameters,
                                  const vector<mol_info> &info,
                                  vector<force> &dipoles_out) const;

public:
  // Layout optimized from largest to smallest
  AlignedVector3D pos; // Replaces raw x, y, z

  // Conditional Allocation of Forces (Lazy Initialization)
  // Ensures 0 memory overhead during analytical runs.
  std::optional<AlignedVector3D> force_vec;
  std::vector<AlignedVector3D> thread_local_forces;

  vector<double> vdw_radius;
  vector<com> Com;
  string force_f;
  string mol_type;
  h_bonds_wrapper list_h_bonds;
  box Box;

  double time_of_snap;
  bool flag_set_time;
  bool compute_forces{false};

  struct BackupParam {
    double charge;
    double epsilon;
    double sigma;
    double radius;
  };
  std::vector<BackupParam> backup_params;

  // --- CONSTRUCTORS (Modern C++) ---
  MOL() noexcept
      : force_f{"AMBER99SB"}, mol_type{"protein"}, time_of_snap{0.0},
        flag_set_time{false}, compute_forces{false} {}

  MOL(const box &box_t, double time, std::string_view ff,
      bool alloc_forces = false)
      : force_f{ff}, mol_type{"protein"}, Box{box_t}, time_of_snap{time},
        flag_set_time{true}, compute_forces{alloc_forces} {}

  MOL(double _x, double _y, double _z, const box &box_t, double time,
      std::string_view ff, std::string_view _mol_type = "protein",
      bool alloc_forces = false)
      : force_f{ff}, mol_type{_mol_type}, Box{box_t}, time_of_snap{time},
        flag_set_time{true}, compute_forces{alloc_forces} {
    pos.x.emplace_back(_x);
    pos.y.emplace_back(_y);
    pos.z.emplace_back(_z);

    if (compute_forces) {
      initialize_forces();
    }
  }

  MOL(const MOL &) = default;
  MOL(MOL &&) noexcept = default;
  MOL &operator=(const MOL &) = default;
  MOL &operator=(MOL &&) noexcept = default;
  ~MOL() = default;

  // --- MEMORY ALLOCATION CONTROL ---
  void initialize_forces(size_t num_threads = 1) {
    if (pos.size() == 0)
      return;
    compute_forces = true;
    if (!force_vec.has_value()) {
      force_vec.emplace();
    }
    force_vec->resize(pos.size());
    thread_local_forces.resize(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
      thread_local_forces[i].resize(pos.size());
    }
  }

  [[nodiscard]] bool force_enabled() const noexcept {
    return compute_forces && force_vec.has_value();
  }

  // Check safety guard before accessing forces.
  void check_forces_allocated() const {
    if (!force_enabled())
      throw std::logic_error("MOL force buffers are not initialized");
    assert(force_enabled() &&
           "Forces accessed but not allocated! Use compute_forces=true.");
  }

  // --- CORE METHODS ---
  void clear_MOL() noexcept {
    pos.clear();
    if (force_vec.has_value()) {
      force_vec->clear();
      force_vec.reset();
    }
    thread_local_forces.clear();

    Com.clear();
    list_h_bonds.dat.clear();
    flag_set_time = false;
    compute_forces = false;
  }

  void release_memory() noexcept {
    clear_MOL();
    pos.shrink_to_fit();
    thread_local_forces.shrink_to_fit();
    Com.shrink_to_fit();
    list_h_bonds.dat.shrink_to_fit();
  }

  [[nodiscard]] bool coordinates_consistent() const noexcept {
    return pos.x.size() == pos.y.size() && pos.x.size() == pos.z.size();
  }

  [[nodiscard]] bool topology_consistent(size_t topology_atom_count) const
      noexcept {
    return coordinates_consistent() && topology_atom_count == size();
  }

  [[nodiscard]] size_t size() const noexcept { return pos.size(); }

  template <typename T> void com_pushback(T &&_com) {
    Com.emplace_back(std::forward<T>(_com));
  }

  [[nodiscard]] com &get_com(size_t i) noexcept { return Com[i]; }

  [[nodiscard]] double get_time() const noexcept { return time_of_snap; }
  void set_time(double _time_of_snap) { time_of_snap = _time_of_snap; }

  [[nodiscard]] string_view get_force_f() const noexcept { return force_f; }
  [[nodiscard]] string_view get_mol_type() const noexcept { return mol_type; }

  // Read-only span views for positions
  [[nodiscard]] std::span<const double> get_pos_x() const noexcept {
    return pos.x;
  }
  [[nodiscard]] std::span<const double> get_pos_y() const noexcept {
    return pos.y;
  }
  [[nodiscard]] std::span<const double> get_pos_z() const noexcept {
    return pos.z;
  }

  // Safe access to forces
  [[nodiscard]] std::span<double> get_force_x() {
    check_forces_allocated();
    return force_vec->x;
  }
  [[nodiscard]] std::span<double> get_force_y() {
    check_forces_allocated();
    return force_vec->y;
  }
  [[nodiscard]] std::span<double> get_force_z() {
    check_forces_allocated();
    return force_vec->z;
  }

  [[nodiscard]] std::span<double> get_thread_force_x(size_t t_id) {
    check_forces_allocated();
    if (t_id >= thread_local_forces.size())
      throw std::out_of_range("MOL thread-local force buffer index is out of range");
    return thread_local_forces[t_id].x;
  }
  [[nodiscard]] std::span<double> get_thread_force_y(size_t t_id) {
    check_forces_allocated();
    if (t_id >= thread_local_forces.size())
      throw std::out_of_range("MOL thread-local force buffer index is out of range");
    return thread_local_forces[t_id].y;
  }
  [[nodiscard]] std::span<double> get_thread_force_z(size_t t_id) {
    check_forces_allocated();
    if (t_id >= thread_local_forces.size())
      throw std::out_of_range("MOL thread-local force buffer index is out of range");
    return thread_local_forces[t_id].z;
  }

  // --- REDUCTION OF THREAD LOCAL FORCES ---
  void reduce_forces() {
    check_forces_allocated();
    const size_t num_threads = thread_local_forces.size();
    const size_t n_atoms = pos.size();

    if (num_threads == 0 || n_atoms == 0)
      return;

    const size_t vec_size = n_atoms - (n_atoms % 4);

    for (size_t t = 0; t < num_threads; ++t) {
      double *target_x = force_vec->x.data();
      double *target_y = force_vec->y.data();
      double *target_z = force_vec->z.data();

      const double *local_x = thread_local_forces[t].x.data();
      const double *local_y = thread_local_forces[t].y.data();
      const double *local_z = thread_local_forces[t].z.data();

      titan_simd_dispatch::add_row_inplace(target_x, local_x,
                                           static_cast<int>(vec_size));
      titan_simd_dispatch::add_row_inplace(target_y, local_y,
                                           static_cast<int>(vec_size));
      titan_simd_dispatch::add_row_inplace(target_z, local_z,
                                           static_cast<int>(vec_size));

      for (size_t i = vec_size; i < n_atoms; ++i) {
        target_x[i] += local_x[i];
        target_y[i] += local_y[i];
        target_z[i] += local_z[i];
      }
    }
  }

  // --- MATH & DISTANCE INLINES (Tối ưu hóa vòng lặp) ---
  inline double dis_between2_atoms(size_t i, size_t j) const noexcept {
    double dx = pos.x[i] - pos.x[j];
    double dy = pos.y[i] - pos.y[j];
    double dz = pos.z[i] - pos.z[j];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  inline double dis_between_atom_and_point(size_t i, double px, double py,
                                           double pz) const noexcept {
    double dx = pos.x[i] - px;
    double dy = pos.y[i] - py;
    double dz = pos.z[i] - pz;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  inline double squared_dis_between_atom_and_point(size_t i, double px,
                                                   double py,
                                                   double pz) const noexcept {
    double dx = pos.x[i] - px;
    double dy = pos.y[i] - py;
    double dz = pos.z[i] - pz;
    return (dx * dx + dy * dy + dz * dz);
  }

  // --- CÁC HÀM XỬ LÝ KHÁC (Khai báo nguyên bản) ---

  int post_proc_hbonds(const MOL &mol, vector<mol_info> &info,
                       vector<mol_info> &counter_info);
  int imp_post_proc(const pair<char, size_t> &atom_dat, const MOL &mol,
                    vector<mol_info> &info, vector<mol_info> &counter_info);
  void take_com(Multi_MOL_parameters &parameters, size_t i, size_t end,
                vector<class index> &Index, vector<mol_info> &info);

  int create_pbc_images(arguments &Arguments, Multi_MOL &out_mol,
                        const vector<class index> &Index);
  int find_best_image_of_residues(Multi_MOL &pbc_images, size_t chain,
                                  vector<class index> &Index,
                                  vector<size_t> &rejected_images);

  // Replaced double* with std::span<double> for Com
  void get_centroid(std::span<double> com) const;

  int find_images_suitable(size_t chain, size_t res, vector<class index> &Index,
                           vector<size_t> &list_image) const;

  int impl_solv_pbc_clust(arguments &Arguments, vector<class index> &Index);

  double find_dis_res_point(size_t chain, size_t res, Multi_MOL &images,
                            vector<class index> &Index, size_t pos_image,
                            std::span<const double> point) const;

  double find_min_dis_2res(size_t chain, size_t chain2, size_t res1,
                           size_t res2, Multi_MOL &images,
                           vector<class index> &Index, size_t pos_image) const;

  int find_cog_chain(size_t ind, arguments &Arguments,
                     vector<class index> &Index,
                     std::span<double> center_of_geometry);

  size_t find_chain_ref_pbc(arguments &Arguments, vector<class index> &Index);

  int cal_dis_charged_atoms(vector<double> &distances,
                            vector<double> &distances_negative_charges,
                            vector<double> &distances_positive_charges,
                            const Multi_MOL_parameters &parameters,
                            std::span<const double> com) const;

  int find_surface_water(const vector<size_t> &list_waters,
                         const vector<size_t> &list_heavy_atoms,
                         vector<size_t> &list_surface) const;

  int find_sphere_water(const vector<size_t> &list_waters,
                        const vector<size_t> &list_heavy_atoms,
                        vector<size_t> &list_surface, double d_crit) const;

  int cal_electric_field(const vector<mol_info> &info,
                         const arguments &Arguments,
                         const Multi_MOL_parameters &parameters,
                         const vector<size_t> &list_heavy_atoms,
                         const vector<size_t> &pos_o_waters,
                         vector<force> &elec_fields) const;

  int cal_angle_dipoles_elecfield(const vector<mol_info> &info,
                                  const arguments &Arguments,
                                  vector<force> &elec_fields,
                                  const Multi_MOL_parameters &parameters,
                                  const vector<size_t> &list_heavy_atoms,
                                  const vector<size_t> &pos_o_waters,
                                  vector<double> &cos_out) const;

  [[nodiscard]] double
  cal_minimum_distance_2mols(const MOL &other,
                             const arguments &Arguments) const;
  [[nodiscard]] double cal_minimum_distance_mol_atom(const MOL &other,
                                                     const arguments &Arguments,
                                                     size_t &index) const;
  int cal_min_max_mol(double &min_x, double &max_x, double &min_y,
                      double &max_y, double &min_z, double &max_z) const;

  void take_com_all_atoms(Multi_MOL_parameters &parameters, size_t I,
                          size_t end, vector<class index> &Index,
                          vector<mol_info> &info, multi_points &com_res);
  int translate(double d_x, double d_y, double d_z);
  bool mindist_mol(const MOL &lig, arguments &Arguments, de_arg &DE_arg) const;

  void mutate_to_alanine(int res_id, const std::vector<mol_info> &info,
                         Multi_MOL_parameters &parameters);
  void restore_mutation(int res_id, const std::vector<mol_info> &info,
                        Multi_MOL_parameters &parameters);
};

#endif
