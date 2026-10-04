#include <string_view>
using std::string_view;
#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include "Forcefield.h"
#include "Multi_MOL.h"
#include "amino_a.h"
#include "arguments.h"
#include "c6c12.h"
#include "cache_sigma_epsilon.h"
#include "cscd.h"
#include "distance_v2.h"
#include "folder.h"
#include "force.h"
#include "groups.h"
#include "h_groups.h"
#include "info.h"
#include "matrix.h"
#include "miscellaneous.h"
#include "stat_hb_and_sc_contact.h"
#include <thread>

#ifdef __AVX__
#include <immintrin.h>
#else
#warning AVX is not available.Code will not compile!
#endif

using namespace std;

class TitanExecutor;

#ifdef __AVX__
inline double hsum_double_avx(__m256d v) {
  __m128d vlow = _mm256_castpd256_pd128(v);
  __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
  vlow = _mm_add_pd(vlow, vhigh);              // reduce down to 128

  __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
  return _mm_cvtsd_f64(_mm_add_sd(vlow, high64)); // reduce to scalar
}
#endif

inline bool check_in_time(const arguments &Arguments,
                          double time_of_snap_temp) noexcept {
  if (Arguments.flag_begin_set &&
      time_of_snap_temp < Arguments.begin_time - 1e-3)
    return false;
  if (Arguments.flag_end_set && time_of_snap_temp > Arguments.end_time + 1e-3)
    return false;

  if (Arguments.dt > 0) {
    double ref_time = Arguments.flag_begin_set ? Arguments.begin_time : 0.0;
    double diff = time_of_snap_temp - ref_time;
    if (diff < -1e-3)
      return false;
    long long steps = std::lround(diff / Arguments.dt);
    if (std::abs(diff - steps * Arguments.dt) > 1e-3) {
      return false;
    }
  }
  return true;
}
int gen_input_list(arguments &Arguments, vector<string> &input_line,
                   folders &paths, fstream &log, int &input_name);
#include <expected>

struct StreamingContext {
  std::ifstream traj_stream_pro;
  std::ifstream traj_stream_lig;
  // We can hold open streams here
};

int parse_input_pdb(arguments &Arguments, fstream &log, string name_pro,
                    string name_lig, string force_f, int input_name,
                    vector<string> &input_line, bool topo_2_is_open,
                    bool topo_is_open, Multi_MOL &pro_in, Multi_MOL &lig,
                    Multi_MOL &water_traj, groups_cal &Groups_from_index_file,
                    para &parameter, amino_acid &amine, info &output_info);

std::expected<void, std::string> initialize_topology_streaming(
    arguments &Arguments, std::fstream &log, std::string name_pro,
    std::string name_lig, std::string force_f, int input_name,
    std::vector<std::string> &input_line, bool topo_2_is_open,
    bool topo_is_open, Multi_MOL &pro_in, Multi_MOL &lig, Multi_MOL &water_traj,
    Multi_MOL &membrane_traj, groups_cal &Groups_from_index_file,
    para &parameter, amino_acid &amine, info &output_info);
int engine(arguments &Arguments);
int take_amino_acids(amino_acid &amine, int &num_amine, string &force_f,
                     const arguments &Arguments);
void take_min(vector<com> &cen_cas, vector<com> &cen_ab, d_min &dmin,
              size_t begin, size_t end, size_t begin2, size_t end2,
              const arguments &Arguments, int flag, double d_standard);
int take_para(para &parameters, string force_f, arguments &Arguments);

int cal_energy_amber_screening(const MOL &rec, const MOL &lg,
                               const vector<mol_info> &info_rec,
                               const vector<mol_info> &info_lg,
                               cache_sigma_epsilon &CachedData, size_t begin,
                               size_t end, cache_energy &Energy,
                               size_t num_snap, const arguments &Arguments,
                               const Multi_MOL_parameters &rec_parameters,
                               const Multi_MOL_parameters &lg_parameters);

void cal_energy_amber(const MOL &rec, const MOL &lg,
                      const vector<mol_info> &info_rec,
                      const vector<mol_info> &info_lg,
                      cache_sigma_epsilon &CachedData, size_t begin, size_t end,

                      cache_energy &Energy, size_t num_snap,
                      const arguments &Arguments,
                      const Multi_MOL_parameters &rec_parameters,
                      const Multi_MOL_parameters &lg_parameters);

void cal_energy_opls(const MOL &rec, const MOL &lg,
                     const vector<mol_info> &info_rec,
                     const vector<mol_info> &info_lg,
                     cache_sigma_epsilon &CachedData, size_t begin, size_t end,
                     cache_energy &Energy, size_t num_snap,
                     const arguments &Arguments,
                     const Multi_MOL_parameters &rec_parameters,
                     const Multi_MOL_parameters &lg_parameters);

void cal_energy_gromos(const MOL &rec, const MOL &lg,
                       const vector<mol_info> &info_rec,
                       const vector<mol_info> &info_lg,
                       cache_sigma_epsilon &CachedData, size_t begin,
                       size_t end, cache_energy &Energy, size_t num_snap,
                       const arguments &Arguments,
                       const Multi_MOL_parameters &rec_parameters,
                       const Multi_MOL_parameters &lg_parameters);

void cal_energy_martini(const MOL &rec, const MOL &lg,
                        const vector<mol_info> &info_rec,
                        const vector<mol_info> &info_lg,
                        cache_sigma_epsilon &CachedData, size_t begin,
                        size_t end, cache_energy &Energy, size_t num_snap,
                        const arguments &Arguments,
                        const Multi_MOL_parameters &rec_parameters,
                        const Multi_MOL_parameters &lg_parameters);

void cal_energy_force_amber(const MOL &rec, const MOL &lg,
                            const vector<mol_info> &info_rec,
                            const vector<mol_info> &info_lg,
                            cache_sigma_epsilon &CachedData, size_t begin,
                            size_t end, cache_energy &Energy,
                            cache_force &Force, size_t num_snap,
                            const arguments &Arguments,
                            const Multi_MOL_parameters &rec_parameters,
                            const Multi_MOL_parameters &lg_parameters);

void cal_energy_force_opls(const MOL &rec, const MOL &lg,
                           const vector<mol_info> &info_rec,
                           const vector<mol_info> &info_lg,
                           cache_sigma_epsilon &CachedData, size_t begin,
                           size_t end, cache_energy &Energy, cache_force &Force,
                           size_t num_snap, const arguments &Arguments,
                           const Multi_MOL_parameters &rec_parameters,
                           const Multi_MOL_parameters &lg_parameters);

void cal_energy_force_gromos(const MOL &rec, const MOL &lg,
                             const vector<mol_info> &info_rec,
                             const vector<mol_info> &info_lg,
                             cache_sigma_epsilon &CachedData, size_t begin,
                             size_t end, cache_energy &Energy,
                             cache_force &Force, size_t num_snap,
                             const arguments &Arguments,
                             const Multi_MOL_parameters &rec_parameters,
                             const Multi_MOL_parameters &lg_parameters);

void cal_energy_force_martini(const MOL &rec, const MOL &lg,
                              const vector<mol_info> &info_rec,
                              const vector<mol_info> &info_lg,
                              cache_sigma_epsilon &CachedData, size_t begin,
                              size_t end, cache_energy &Energy,
                              cache_force &Force, size_t num_snap,
                              const arguments &Arguments,
                              const Multi_MOL_parameters &rec_parameters,
                              const Multi_MOL_parameters &lg_parameters);

int cal_energy_amber_screening_intrin(
    const MOL &rec, const MOL &lg, const vector<mol_info> &info_rec,
    const vector<mol_info> &info_lg, cache_sigma_epsilon &CachedData,
    size_t begin, size_t end, cache_energy &Energy, size_t num_snap,
    const arguments &Arguments, const Multi_MOL_parameters &rec_parameters,
    const Multi_MOL_parameters &lg_parameters);

int cal_energy_amber_screening_intrin_nofilter(
    const MOL &rec, const MOL &lg, const vector<mol_info> &info_rec,
    const vector<mol_info> &info_lg, cache_sigma_epsilon &CachedData,
    size_t begin, size_t end, cache_energy &Energy, size_t num_snap,
    const arguments &Arguments, const Multi_MOL_parameters &rec_parameters,
    const Multi_MOL_parameters &lg_parameters);

int cal_energy_gromos_intrin(const MOL &rec, const MOL &lg,
                             const vector<mol_info> &info_rec,
                             const vector<mol_info> &info_lg,
                             cache_sigma_epsilon &CachedData, size_t begin,
                             size_t end, cache_energy &Energy, size_t num_snap,
                             const arguments &Arguments,
                             const Multi_MOL_parameters &rec_parameters,
                             const Multi_MOL_parameters &lg_parameters);

int cal_energy_gromos_intrin_nofilter(
    const MOL &rec, const MOL &lg, const vector<mol_info> &info_rec,
    const vector<mol_info> &info_lg, cache_sigma_epsilon &CachedData,
    size_t begin, size_t end, cache_energy &Energy, size_t num_snap,
    const arguments &Arguments, const Multi_MOL_parameters &rec_parameters,
    const Multi_MOL_parameters &lg_parameters);

void cal_energy_opls_intrin(const MOL &rec, const MOL &lg,
                            const vector<mol_info> &info_rec,
                            const vector<mol_info> &info_lg,
                            cache_sigma_epsilon &CachedData, size_t begin,
                            size_t end, cache_energy &Energy, size_t num_snap,
                            const arguments &Arguments,
                            const Multi_MOL_parameters &rec_parameters,
                            const Multi_MOL_parameters &lg_parameters);

void cal_energy_opls_intrin_nofilter(
    const MOL &rec, const MOL &lg, const vector<mol_info> &info_rec,
    const vector<mol_info> &info_lg, cache_sigma_epsilon &CachedData,
    size_t begin, size_t end, cache_energy &Energy, size_t num_snap,
    const arguments &Arguments, const Multi_MOL_parameters &rec_parameters,
    const Multi_MOL_parameters &lg_parameters);

int out(vector<double> &E_e, vector<double> &E_v, vector<double> &time_of_snap,
        fstream &cal_out, arguments &Arguments, int input_name,
        vector<vector<double>> &E_e_vector, vector<vector<double>> &E_v_vector,
        vector<size_t> &size_snap);
int out_force(vector<force> &F_e, vector<force> &F_v,
              vector<double> &time_of_snap, arguments &Arguments,
              int input_name, vector<vector<force>> &F_e_vector,
              vector<vector<force>> &F_v_vector, vector<size_t> &size_snap);
void take_ff_para(amino_acid &amine_ff, para &par);
double min_dis(const MOL &molecule_in_1, const MOL &molecule_in_2, int coor);
void Fix_pbc(MOL &molecule_1, int coor, const double &min_dis, double size);
void solve_pbc(MOL &mol_1, MOL &mol_2, box &box_type);
void fix_dodecahedron(MOL &mol_1, MOL &mol_2, box &box_type);
int pre_proc_pdb(string force_field, Multi_MOL &protein_1, Multi_MOL &protein_2,
                 Multi_MOL &water_traj, Multi_MOL &membrane_traj,
                 groups_cal &groups, int flag, arguments &Arguments,
                 const amino_acid &amine, info &output_info);
// void pre_parameters(vector<sigma_epsilon>& parameters, const Multi_MOL&
// pro_in, const Multi_MOL& lig, int begin, int end);
int write_pqr(const Multi_MOL &pro_in, const Multi_MOL &lig,
              vector<string> &input_line, int input_name, size_t num_snap,
              const arguments &Arguments, folders &paths);
int find_h_bonds(MOL &pro_in, MOL &pro_in_ref, MOL &lig, MOL &lig_ref,
                 h_groups &groups_1, h_groups &groups_2,
                 const arguments &Arguments, size_t num_snap,
                 vector<mol_info> &info_rec, vector<mol_info> &info_lg,
                 vector<vector<size_t>> &Hbonds_vector);
int init_hbond_output_files(const arguments &Arguments);
int finalize_hbond_output_files();
int add_atom_to_MOL(Multi_MOL &MOL_source, Multi_MOL &protein_des,
                    groups_cal &groups, int group_index,
                    const arguments &Arguments, string type,
                    const amino_acid &amine, int flag);
int trans_atom_to_MOL(Multi_MOL &MOL_source, Multi_MOL &protein_des,
                      Multi_MOL &protein_des_2, groups_cal &groups,
                      size_t group_index, size_t group_index_2,
                      const arguments &Arguments, const amino_acid &amine,
                      int flag);
int write_pdb(const Multi_MOL &pro_in, const Multi_MOL &lig,
              const arguments &Arguments, size_t num_snap, folders &paths,
              vector<string> &input_line, int input_name);
int finalize_pdb_output_stream();
size_t check_res_H(vector<mol_info> &protein_in, string_view res_name,
                   string_view atom_name, size_t i);
int check_and_settime(double &time_of_snap, string protein_1,
                      fstream &input_pdb, arguments &Arguments);
void take_coordinate(Multi_MOL &pro_in, Multi_MOL &lig, Multi_MOL &water_traj,
                     groups_cal &groups, string input_name, string force_f,
                     arguments &Arguments, const amino_acid &amine,
                     ::info &output_info, int flag_raw = 0,
                     size_t max_frames = 0);
void reduce_h_form(matrix &A, int I, int K, size_t nt);
int cal_entropy(const Multi_MOL &mol1, const Multi_MOL &mol2,
                groups_cal &groups, arguments &Arguments, const folders &paths,
                size_t file_pos, const info &info_output);
double non_gaussian_entropy(matrix &matrix_in, int k_neighbor, int II);
double non_gaussian_entropy_2nd(matrix &matrix_in, int k_neighbor, int II,
                                int JJ);
int align_complex(Multi_MOL &rec, Multi_MOL &lig, int begin, int end);
int impl_entropy_duan_zhang(folders &paths, vector<vector<double>> &E_e_vector,
                            vector<vector<double>> &E_v_vector,
                            vector<double> &time_of_snap, arguments &Arguments,
                            vector<double> &E_e, vector<double> &E_v);
int find_d_rescom_com(vector<com> &cen_cas, Multi_MOL &lig, d_min &dmin,
                      size_t begin, size_t end, size_t begin2, size_t end2,
                      string force_f, int flag, double d_standard);
int impl_cscd(Multi_MOL &pro_in, Multi_MOL &lig, const arguments &Arguments,
              fstream &log, TitanExecutor &executor, size_t num_snap,
              int sc_flag, int input_name, vector<string> &list_output_name,
              all_cscd &out_dis, const folders &paths,
              vector<fstream> &out_files);
int impl_add_coms_2_mols(Multi_MOL &mol1, Multi_MOL &mol2, arguments &Arguments,
                         size_t snap_index);
int wrap_cal_force_map(TitanExecutor &executor, Multi_MOL &mol1,
                       Multi_MOL &mol2, const arguments &Arguments,
                       size_t num_snap, const vector<d_min> &dmin,
                       const cache_sigma_epsilon &CachedData,
                       vector<energy_res> &Energy_res_mol1,
                       vector<energy_res> &Energy_res_mol2, info &info_output);
istream &safeGetline(istream &is, string &t);
int post_proc_cscd(vector<string> &list_input_name);
bool check_end_cal(arguments &Arguments, vector<string> &input_line,
                   int &input_name);
int print_force_res(vector<energy_res> &Energy_res, string name, folders &paths,
                    int file_num, const arguments &Arguments);
int find_maxforce_res(vector<energy_res> &Energy_res, string name,
                      folders &paths, int file_num, const arguments &Arguments,
                      vector<energy_res> &Energy_res_out);
int stat_maxforce_res(vector<energy_res> &Force_res_stat, string name,
                      const info &infos, arguments &Arguments);
int print_cscd(arguments &Arguments, all_cscd &out_dis,
               vector<string> &list_file_name);
int stat_hbonds(vector<vector<size_t>> &Hbonds_vector, const info &infos,
                arguments &Arguments, vector<double> &time_of_snaps,
                vector<string> &list_name);
bool remove_double_counting(cache_sigma_epsilon &CachedData,
                            vector<cache_energy> &Energy);
int dis_com_2mol(const Multi_MOL &mol1, const Multi_MOL &mol2,
                 const size_t num_snap, const size_t NUM_snap,
                 arguments &Arguments, int input_name, const folders &paths,
                 double E_E, double E_V, const info &out_info);
int add_hb_res(MOL &pro_in, MOL &lig, vector<mol_info> &info_rec,
               vector<mol_info> &info_lg, const arguments &Arguments,
               traj_hb &res_hb_rec_temp);
int stat_hb_res(vector<traj_hb> &hb_reslist, string output_filename);
int set_chainnum(Multi_MOL &pro_in, Multi_MOL &lig, arguments const &Arguments);
int process_time_of_infiles(Multi_MOL &pro_in, arguments &Arguments,
                            info &info_output, vector<double> &time_of_snap);
int process_adding_energies_forces(
    arguments &Arguments, int input_name, vector<double> &E_v,
    vector<double> &E_e, double &E_E_, double &E_V_, force &f_e_, force &f_v_,
    vector<vector<double>> &E_v_vector, vector<vector<double>> &E_e_vector,
    vector<force> &Force_e, vector<force> &Force_v,
    vector<vector<force>> &Force_e_vector,
    vector<vector<force>> &Force_v_vector, vector<size_t> &size_snap,
    size_t num_snap);
int post_process_Energy_force_subgroups(
    vector<double> &time, vector<vector<energy_groups>> &EF_groups,
    folders &paths, arguments &Arguments, info &info_output);
int Energy_force_subgroups(groups_cal &groups, const Multi_MOL &rec,
                           const Multi_MOL &lg,
                           const cache_sigma_epsilon &CachedData,
                           vector<vector<energy_groups>> &EF_subgroups,
                           vector<energy_groups> &Energy_groups_out,
                           size_t num_snap, size_t MAX_NUM,
                           arguments &Arguments, TitanExecutor &executor);
int impl_nocal(Multi_MOL &pro_in, Multi_MOL &lig, fstream &log,
               arguments &Arguments, info &info_output, TitanExecutor &executor,
               amino_acid &amine, para &parameter, folders &paths,
               vector<string> &input_line, h_groups &groups_1,
               h_groups &groups_2, vector<vector<size_t>> &Hbonds_vector,
               int &input_name, vector<vector<double>> &Volumes);
int process_moleculetype(fstream &input_topo_protein, string &lig_name,
                         string &protein_1);
int process_atomtypes(fstream &input_topo_protein, string &protein_1,
                      para &parameters, string force_f, string &lig_name);

int cal_energy_amber_intrin(const MOL &rec, const MOL &lg,
                            const vector<mol_info> &info_rec,
                            const vector<mol_info> &info_lg,
                            cache_sigma_epsilon &CachedData, size_t begin,
                            size_t end, cache_energy &Energy, size_t num_snap,
                            const arguments &Arguments,
                            const Multi_MOL_parameters &rec_parameters,
                            const Multi_MOL_parameters &lg_parameters);

int cal_energy_amber_intrin_nofilter(
    const MOL &rec, const MOL &lg, const vector<mol_info> &info_rec,
    const vector<mol_info> &info_lg, cache_sigma_epsilon &CachedData,
    size_t begin, size_t end, cache_energy &Energy, size_t num_snap,
    const arguments &Arguments, const Multi_MOL_parameters &rec_parameters,
    const Multi_MOL_parameters &lg_parameters);

int cal_energy_martini_intrin(const MOL &rec, const MOL &lg,
                              const vector<mol_info> &info_rec,
                              const vector<mol_info> &info_lg,
                              cache_sigma_epsilon &CachedData, size_t begin,
                              size_t end, cache_energy &Energy, size_t num_snap,
                              const arguments &Arguments,
                              const Multi_MOL_parameters &rec_parameters,
                              const Multi_MOL_parameters &lg_parameters);
int cal_one_ele(const MOL &rec, const MOL &lg, cache_sigma_epsilon &CachedData,
                const arguments &Arguments, double pos_x, double pos_y,
                double pos_z, double temp_sigma, double temp_charge,
                double temp_epsilon_lg,
                const Multi_MOL_parameters &rec_parameters,
                const Multi_MOL_parameters &lg_parameters,
                const vector<mol_info> &info_rec,
                const vector<mol_info> &info_lg, cache_energy &Energy, size_t i,
                size_t j);
int find_pull_direct(Multi_MOL &rec, Multi_MOL &lg,
                     cache_sigma_epsilon &CachedData, de_arg &DE_arguments,
                     arguments &Arguments, info &output_info,
                     TitanExecutor &executor,
                     groups_cal &Groups_from_index_file);
int mindis_subgroups(groups_cal &groups, const Multi_MOL &rec,
                     const Multi_MOL &lg, size_t num_snap, size_t MAX_NUM,
                     arguments &Arguments, TitanExecutor &executor,
                     vector<mindistance_group> &MIN_DIS);
int post_proc_mindis(groups_cal &groups, arguments &Arguments,
                     info &info_output, vector<mindistance_group> &MIN_DIS);
#endif
