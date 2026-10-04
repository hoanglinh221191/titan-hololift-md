#ifndef MULTI_MOL_H
#define MULTI_MOL_H

#include "MOL.h"
#include "MOL_info.h"
#include "MOL_para.h"
#include "amino_a.h"
#include "arguments.h"
#include "c6c12.h"
#include "index_v2.h"
#include "info.h"
#include "mol_topology.h"
#include <cstdint>
#include <expected>
#include <fstream>
#include <istream>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using std::fstream;
using std::map;
using std::string;
using std::string_view;
using std::vector;

struct topo_info_temp;
struct cache_sigma_epsilon;
class TitanExecutor;
class groups_cal;

class Multi_MOL {
private:
  vector<unsigned char> pbctopo_frame_certified_;
  vector<string> pbctopo_frame_status_;

  int find_best_image(size_t ind, int chain_ref, arguments &Arguments);
  int find_best_image_envelope(size_t ind, int chain_ref, arguments &Arguments);
  int find_best_image_new(size_t ind, size_t chain_ref, arguments &Arguments,
                          vector<vector<size_t>> &list_chain_combi);
  int impl_solv_pbc_new(Multi_MOL &mol2, size_t begin, size_t end,
                        arguments &Arguments, const amino_acid &amine);
  int impl_solve_pbc_chain(size_t ind, arguments &Arguments,
                           vector<size_t> &rejected_images);
  int find_best_image_of_mol2(Multi_MOL &mol2, size_t i, int chain_ref,
                              arguments &Arguments);
  int aggressive_find_best_image_of_mol2(Multi_MOL &mol2, size_t ind,
                                         int chain_ref, arguments &Arguments);
  int solv_pbc_envi(size_t ind, arguments &Arguments,
                    vector<size_t> &rejected_images, size_t k,
                    Multi_MOL &pbc_images, std::span<double> com_ref_chain);

  void imp_post_proc(int begin, int end, string_view type,
                     const amino_acid &amine);
  int parse_pdb(std::string_view input_name, std::string_view force_f,
                arguments &Arguments, ::info &output_info,
                size_t max_frames = 0);

  int post_process_type_res(class index &in, size_t pos_res);
  int parse_type(const amino_acid &amine);
  string check_head_tail_res(class index &in, size_t pos_res);
  int post_process_coordinates(const amino_acid &amine);
  void post_process_coordinates_gromos(const amino_acid &amine);
  void post_process_coordinates_lig();
  void post_process_coordinates_charmm(const amino_acid &amine);
  void post_process_coordinates_oplsaa(const amino_acid &amine);
  void post_process_coordinates_user_define(string_view type);

  int process_atom_from_topo(fstream &input_topo_protein, string &protein_1,
                             string &lig_name, int &i,
                             vector<topo_info_temp> &info_topo);
  int post_process_atom_from_topo(vector<topo_info_temp> &info_topo);
  int helper_check_side_chain(const arguments &Arguments, size_t begin,
                              size_t end);

  int helper_topo_lg_gromacs_format(std::string_view input_topo,
                                    std::string_view force_f, para &parameters,
                                    fstream &input_topo_protein,
                                    string &ligname,
                                    vector<topo_info_temp> &info_topo);
  int helper_topo_lg_parm_format(std::string_view input_topo,
                                 std::string_view force_f, para &parameters,
                                 fstream &input_topo_protein, string &lig_name);

  int impl_cal_distribution_distance_charged_atoms(
      vector<vector<double>> &distances,
      vector<vector<double>> &distances_negative_charges,
      vector<vector<double>> &distances_positive_charges, size_t begin,
      size_t end);
  int find_water(vector<size_t> &list_atoms, vector<size_t> &list_heavy_atoms);

  int helper_parse_pdb(arguments &Arguments, box &Box_temp, bool &flag_set_time,
                       double &time_of_snap_temp, string_view input_pdb,
                       size_t &I, bool &flag_set_box, ::info &output_info,
                       string_view protein_1, std::string_view input_name,
                       std::string_view force_f);
  int helper_parse_buffer(vector<string> &buffer_string, arguments &Arguments,
                          box &Box_temp, bool &flag_set_time,
                          double &time_of_snap_temp, size_t &I,
                          std::string_view input_name, size_t &count_snap,
                          bool &flag_tune_snap, size_t &limit_snap,
                          bool &flag_end);

public:
  int parse_martini_topology(const arguments &Arguments,
                             std::fstream &log_file,
                             cache_sigma_epsilon *martini_cache = nullptr);
  // Returns number of frames actually read. 0 means EOF.
  std::expected<size_t, std::string> read_pdb_chunk(std::ifstream &input_pdb,
                                                    size_t chunk_size,
                                                    arguments &Arguments,
                                                    ::info &output_info,
                                                    std::string_view force_f);
  // --- MEMORY LAYOUT TỐI ƯU (Từ lớn đến nhỏ) ---
  vector<MOL> m_mol;
  Multi_MOL_parameters parameters;
  vector<double> lcpo_p1;
  vector<double> lcpo_p2;
  vector<double> lcpo_p3;
  vector<double> lcpo_p4;
  vector<class index> Index;
  vector<mol_info> info;
  MolTopology topology;
  string mol_type{"protein"};
  string force_f{"AMBER99SB"};

  // Store global computation intent
  bool compute_forces{false};

  // --- CONSTRUCTORS (Modern C++) ---
  Multi_MOL() noexcept = default;

  Multi_MOL(string_view _force_f, bool alloc_forces = false)
      : force_f{_force_f}, compute_forces{alloc_forces} {}

  Multi_MOL(const Multi_MOL &) = default;
  Multi_MOL(Multi_MOL &&) noexcept = default;
  Multi_MOL &operator=(const Multi_MOL &) = default;
  Multi_MOL &operator=(Multi_MOL &&) noexcept = default;
  ~Multi_MOL() = default;

  // --- CÁC HÀM GET/SET ---
  [[nodiscard]] inline size_t size_index() const noexcept {
    return Index.size();
  }
  [[nodiscard]] class index &get_index(size_t i) { return Index[i]; }
  [[nodiscard]] const class index &get_index(size_t i) const {
    return Index[i];
  }

  void set_ff(string_view force_f_) { force_f = force_f_; }
  void set_mol_type(string_view temp) { mol_type = temp; }
  void set_compute_forces(bool flag) { compute_forces = flag; }

  [[nodiscard]] string_view get_mol_type() const noexcept { return mol_type; }
  [[nodiscard]] bool frame_coordinates_consistent(size_t frame_index) const
      noexcept;
  [[nodiscard]] bool topology_matches_frame(size_t frame_index) const noexcept;
  [[nodiscard]] bool parameters_match_info() const noexcept;
  [[nodiscard]] bool parameters_match_frame(size_t frame_index) const noexcept;
  [[nodiscard]] bool analysis_frame_ready(size_t frame_index) const noexcept;
  [[nodiscard]] bool has_pbctopo_frame_status(size_t frame_index) const
      noexcept {
    return frame_index < pbctopo_frame_status_.size();
  }
  [[nodiscard]] bool pbctopo_frame_certified(size_t frame_index) const
      noexcept {
    return frame_index < pbctopo_frame_certified_.size() &&
           pbctopo_frame_certified_[frame_index] != 0;
  }
  [[nodiscard]] string_view pbctopo_frame_status(size_t frame_index) const
      noexcept {
    static constexpr string_view missing{"missing_status"};
    return frame_index < pbctopo_frame_status_.size()
               ? string_view(pbctopo_frame_status_[frame_index])
               : missing;
  }

  // --- CÁC HÀM XỬ LÝ CHÍNH ---
  int gen_vector_parameters();
  void cache_topology_vdw_radii_from_parameters();
  int generate_index();

  void assign_lcpo_parameters(size_t num_atoms,
                              const vector<string> &atom_names,
                              const vector<string> &res_names);

  int check_dna_or_rna(const amino_acid &amine);
  int post_process_nucleotide();
  int post_process_protein();

  double next_target_time{-1e9}; // Added for -dt parsing logic

  void reset_parser_state() noexcept { next_target_time = -1e9; }
  void begin_frame(size_t frame_index, const box &box_type, double time,
                   std::string_view force_field, size_t reserve_atoms,
                   std::string_view frame_mol_type = {});
  void reserve_atoms(size_t frame_index, size_t atom_count);
  void append_xyz(size_t frame_index, double x, double y, double z);
  void append_topology_if_frame0(const mol_info &info_in);
  void discard_pending_frame(size_t frame_index);

  int take_coordinate(std::string_view input_name, std::string_view _force_f,
                      arguments &Arguments, const amino_acid &amine,
                      ::info &output_info, int flag_raw, size_t max_frames = 0);

  int take_coordinate_from_group(std::string_view input_name,
                                 std::string_view _force_f,
                                 const Multi_MOL &base_mol,
                                 const amino_acid &amine, ::info &output_info,
                                 const std::vector<size_t> &group_indices,
                                 arguments &Arguments);

  void take_topo_lg(string &input_name, string &input_topo,
                    arguments &Arguments, std::string_view _force_f,
                    para &parameter, const amino_acid &amine,
                    ::info &output_info, std::string_view forced_type_mol);

  void topo_lg(std::string_view input_topo, std::string_view force_f,
               para &parameters);
  int topo_protein(std::string_view input_topo, std::string_view force_f,
                   para &parameters, const vector<topo_info_temp> &info_topo);

  void pbc_prob(Multi_MOL &lig, size_t i, size_t num_threads);
  int solve_pbc_int(size_t init, size_t nt);
  int solve_pbc_new(Multi_MOL &mol2, arguments &Arguments,
                    const amino_acid &amine);
  int solv_pbc_pbcn(Multi_MOL &mol2, arguments &Arguments);
  int solve_pbc_pbctopo(Multi_MOL &mol2, arguments &Arguments,
                        TitanExecutor *executor = nullptr,
                        size_t frame_report_offset = 0);

  int emplace(size_t i, box &box_type, double time, std::string_view force_f,
              mol_info &&info_in, MOL &&mol_in);
  int emplace(size_t i, box &box_type, double time, std::string_view force_f,
              const mol_info &info_in, const MOL &mol_in,
              std::string_view _mol_type);
  int emplace(size_t i, box &box_type, double time, std::string_view force_f,
              const mol_info &info_in, MOL &&mol_in,
              std::string_view _mol_type);

  [[nodiscard]] double find_min_dis(size_t ind, size_t ind2, size_t chain,
                                    size_t chain2, Multi_MOL &images) const;
  [[nodiscard]] double find_min_dis2(size_t ind, size_t ind2, size_t chain,
                                     size_t chain2, Multi_MOL &images) const;

  int wrapper_impl_post_proc(arguments &Arguments, string_view type,
                             const amino_acid &amine);
  int add_coms(bool flag, size_t snap_index);
  int take_charge(amino_acid &amine, const para &parameter, size_t end,
                  size_t begin, std::string_view force_f, int flag,
                  const arguments *Arguments = nullptr,
                  TitanExecutor *executor = nullptr,
                  std::string_view molecule_label = {});

  // Replaced raw pointer with std::span
  int tran_multi_mol(std::span<double> TranVec);

  int find_box_from_min_max();
  int align_to_box();

  // Replaced raw pointer with std::span
  int com_mol(const size_t num_snap, std::span<double> com) const;

  int check_side_chain(const arguments &Arguments);
  int set_chainnum();
  int com_dis_2mol(Multi_MOL &rhs, arguments &Arguments,
                   const ::info &info_output, fstream &output_dis);
  int solve_pbc_cluster(arguments &Arguments, ::info &output_info);
  int init_chains_connect(arguments &Arguments, const amino_acid &amine,
                          map<size_t, size_t> &chains_connect);
  int solv_pbc_ref(arguments &Arguments, const amino_acid &amine,
                   map<size_t, size_t> &chains_connect, size_t ind,
                   size_t ref_ind);
  int impl_com_dis_2mol(Multi_MOL &rhs, const arguments &Arguments,
                        const ::info &info_output, fstream &output_dis,
                        size_t i);
  int cal_dipole_moment(const ::info &info_output, const arguments &Arguments,
                        std::string_view prefix_name);
  int impl_cal_dipole_moment(const arguments &Arguments, vector<force> &dipoles,
                             size_t begin, size_t end);
  int cal_distribution_distance_charged_atoms(const arguments &Arguments,
                                              const ::info &output_info);
  int cal_angle_dipole_water_elecField(const arguments &Arguments,
                                       ::info &output_info);
  bool check_and_pop_size_2last_snaps(size_t &I, std::string_view name = " ");

  [[nodiscard]] double total_charge() const;
  int cal_dis_res_2_com(const arguments &Arguments, fstream &output);
  int cal_volumes(vector<double> &vol_out) const;
  int gen_pop(MOL &mol_seed, de_arg &DE_arguments, const vector<mol_info> &info,
              string_view mol_type, vector<force> &tran_data);
};

enum class PdbRouteTarget : std::uint8_t { None = 0, Pro = 1, Lig = 2, Wat = 3 };

struct PdbRouteEntry {
  PdbRouteTarget target{PdbRouteTarget::None};
  size_t output_index{0};
};

struct PdbRoutePlan {
  std::vector<PdbRouteEntry> atom_routes;
  std::vector<size_t> membrane_routes;
  size_t split_begin_mol_2{0};
  size_t pro_atom_count{0};
  size_t lig_atom_count{0};
  size_t wat_atom_count{0};
  size_t membrane_atom_count{0};

  [[nodiscard]] bool uses_explicit_routes() const noexcept {
    return !atom_routes.empty();
  }

  [[nodiscard]] PdbRouteEntry route_for(size_t source_atom_1based) const noexcept {
    if (uses_explicit_routes()) {
      if (source_atom_1based < atom_routes.size())
        return atom_routes[source_atom_1based];
      return {};
    }

    if (split_begin_mol_2 == 0)
      return {};

    if (source_atom_1based < split_begin_mol_2)
      return {PdbRouteTarget::Pro, source_atom_1based - 1};
    return {PdbRouteTarget::Lig, source_atom_1based - split_begin_mol_2};
  }

  [[nodiscard]] size_t
  membrane_route_for(size_t source_atom_1based) const noexcept {
    if (source_atom_1based < membrane_routes.size())
      return membrane_routes[source_atom_1based];
    return std::numeric_limits<size_t>::max();
  }
};

std::expected<PdbRoutePlan, std::string>
build_pdb_route_plan(groups_cal &groups, const arguments &Arguments);

std::expected<size_t, std::string>
read_routed_pdb_chunk(std::istream &input_pdb, std::string_view input_name,
                      size_t chunk_size, std::string_view force_f,
                      arguments &Arguments, ::info &output_info,
                      Multi_MOL &pro_out, Multi_MOL &lig_out,
                      Multi_MOL &water_out, Multi_MOL &membrane_out,
                      const PdbRoutePlan &route_plan,
                      double &next_target_time_state, bool capture_topology);

#endif // MULTI_MOL_H
