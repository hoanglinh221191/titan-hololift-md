#ifndef GBSA_INTRIN_H
#define GBSA_INTRIN_H

#include "MOL.h"
#include "arguments.h"
#include "cache_sigma_epsilon.h"
#include "info.h"
#include <iosfwd>
#include <limits>
#include <string>
#include <unordered_map>

#include <vector>

class TitanExecutor;

// Structure to hold GBSA outputs per snap
struct GBSA_energy_snap {
  double total_energy;
  double E_vdw;
  double E_elec;
  double G_polar;
  double G_nonpolar;
  double time;
  double time_precompute_ms;
  double time_energy_kernel_ms;
  double time_decomposition_merge_ms;
  double time_total_ms;
  double alpb_receptor_size_nm;
  double alpb_ligand_size_nm;
  double alpb_complex_size_nm;
  GBSA_energy_snap()
      : total_energy(0.), E_vdw(0.), E_elec(0.), G_polar(0.), G_nonpolar(0.),
        time(0.), time_precompute_ms(0.), time_energy_kernel_ms(0.),
        time_decomposition_merge_ms(0.), time_total_ms(0.),
        alpb_receptor_size_nm(0.), alpb_ligand_size_nm(0.),
        alpb_complex_size_nm(0.) {}
};

struct GBSA_decomposition {
  bool is_ligand;
  int chainnum;
  int resnr;
  std::string unit_type_str;
  std::string chain_str;
  std::string name_str;
  double E_vdw;
  double E_elec;
  double G_polar;
  double G_nonpolar;
  double total;
  size_t count;
  std::vector<double> frame_mm_energies;
  GBSA_decomposition()
      : is_ligand(false), chainnum(0), resnr(0), E_vdw(0.), E_elec(0.),
        G_polar(0.), G_nonpolar(0.), total(0.), count(0) {}
};

struct GBSASasaCache {
  bool gbnsr6_sizes{false};
  std::vector<double> sasa_rec_free;
  std::vector<double> sasa_lig_free;
  std::vector<double> sasa_complex;
  double alpb_rec_electrostatic_size_nm = 0.0;
  double alpb_lig_electrostatic_size_nm = 0.0;
  double alpb_complex_electrostatic_size_nm = 0.0;
};

class GBSA_Aggregator {
private:
  std::vector<std::vector<GBSA_energy_snap>> trajectories;
  std::vector<GBSA_decomposition> decomp;

  using DecompKey = std::tuple<bool, int, int>; // (is_ligand, chainnum, resnr)
  struct DecompKeyHash {
    size_t operator()(const DecompKey &k) const {
      size_t h = std::hash<bool>()(std::get<0>(k));
      h ^= std::hash<int>()(std::get<1>(k)) + 0x9e3779b9 + (h << 6) + (h >> 2);
      h ^= std::hash<int>()(std::get<2>(k)) + 0x9e3779b9 + (h << 6) + (h >> 2);
      return h;
    }
  };
  std::unordered_map<DecompKey, size_t, DecompKeyHash> decomp_index;

private:
  const arguments &Arguments;
  const info &info_output;
  double saltcon, ext_diel, int_diel;
  int igb;
  std::string prefix;

public:
  GBSA_Aggregator(const arguments &arg, const info &info)
      : Arguments(arg), info_output(info) {
    saltcon = arg.saltcon;
    ext_diel = arg.ext_diel;
    int_diel = arg.int_diel;
    igb = arg.igb;
    prefix = "";
  }

  void add_snapshot(size_t traj_index, GBSA_energy_snap snap);
  void add_decomposition(bool is_ligand, int chainnum, int resnr,
                         bool detail_atom, const mol_info *m_info, double ev,
                         double ee, double gpol, double gnp);
  void merge(const GBSA_Aggregator &other);
  void commit_frame_decompositions(const GBSA_Aggregator &frame_data,
                                   size_t current_frame_idx);
  void print_all_stats(const std::string &dir_prefix, std::fstream *log = nullptr);
  void initialize_trajectory(size_t traj_index);

  const std::vector<GBSA_decomposition> &get_decomp() const { return decomp; }
};

#ifdef __AVX__

void set_gbnsr6_electrostatic_sizes(const MOL &rec, const MOL &lg,
    const Multi_MOL_parameters &rp, const Multi_MOL_parameters &lp,
    const std::vector<mol_info> &ri, const std::vector<mol_info> &li,
    GBSASasaCache &cache);

void calculate_gbnsr6_binding_radii(const MOL &rec, const MOL &lg,
    const AlignedDoubleVector &rvdw, const AlignedDoubleVector &lvdw,
    const std::vector<mol_info> &ri, const std::vector<mol_info> &li,
    AlignedDoubleVector &rf, AlignedDoubleVector &lf,
    AlignedDoubleVector &rc, AlignedDoubleVector &lc,
    double spacing, TitanExecutor *executor = nullptr);

void build_gbsa_sasa_cache(const MOL &rec, const MOL &lg,
                           const Multi_MOL_parameters &rec_parameters,
                           const Multi_MOL_parameters &lg_parameters,
                           GBSASasaCache &cache_out, int point_count,
                           bool compute_alpb_sizes,
                           TitanExecutor *executor = nullptr);

// Calculate Effective Born Radii AVX2 Kernel.
// rgbmax_angstrom truncates the HCT descreening integral (Amber egb rgbmax).
// The default +inf keeps the uncut sum, which is TITAN's historical behaviour.
void cal_effective_born_radii_avx2(
    const MOL &mol, const AlignedDoubleVector &vdw_radii,
    const std::vector<mol_info> &atom_info, AlignedDoubleVector &radii,
    int igb, size_t begin, size_t end,
    const MOL *env_mol = nullptr,
    const AlignedDoubleVector *env_vdw_radii = nullptr,
    const std::vector<mol_info> *env_atom_info = nullptr,
    double rgbmax_angstrom = std::numeric_limits<double>::infinity(),
    double nsr6_spacing = 0.5);

void cal_effective_born_radii_parallel_avx2(
    const MOL &mol, const AlignedDoubleVector &vdw_radii,
    const std::vector<mol_info> &atom_info, AlignedDoubleVector &radii,
    int igb, TitanExecutor &executor, const MOL *env_mol = nullptr,
    const AlignedDoubleVector *env_vdw_radii = nullptr,
    const std::vector<mol_info> *env_atom_info = nullptr,
    double rgbmax_angstrom = std::numeric_limits<double>::infinity(),
    double nsr6_spacing = 0.5);

// The free radius of an atom integrates over its own molecule; the complex
// radius continues that same sum over the partner's atoms in index order.
// The two functions below therefore return, bit for bit, what separate free
// and complex cal_effective_born_radii_*() calls return, while integrating
// each molecule's own pairs once instead of twice.
//
// Free and complex radii of `mol` over [begin, end), with `env_mol` as the
// partner. GBNSR6 (igb 66) falls back to the two separate calls.
void cal_free_and_complex_born_radii_avx2(
    const MOL &mol, const AlignedDoubleVector &vdw_radii,
    const std::vector<mol_info> &atom_info, AlignedDoubleVector &free_radii,
    AlignedDoubleVector &complex_radii, int igb, size_t begin, size_t end,
    const MOL &env_mol, const AlignedDoubleVector &env_vdw_radii,
    const std::vector<mol_info> &env_atom_info,
    double rgbmax_angstrom = std::numeric_limits<double>::infinity(),
    double nsr6_spacing = 0.5);

// All four binding endpoints at once. Not for GBNSR6, whose radii come from
// calculate_gbnsr6_binding_radii().
void cal_binding_born_radii_parallel_avx2(
    const MOL &rec, const AlignedDoubleVector &rec_vdw_radii,
    const std::vector<mol_info> &rec_info, const MOL &lg,
    const AlignedDoubleVector &lg_vdw_radii,
    const std::vector<mol_info> &lg_info, int igb, TitanExecutor &executor,
    double rgbmax_angstrom, AlignedDoubleVector &rec_free_radii,
    AlignedDoubleVector &lg_free_radii, AlignedDoubleVector &rec_complex_radii,
    AlignedDoubleVector &lg_complex_radii);

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
    const GBSASasaCache *sasa_cache = nullptr,
    bool write_pair_cache = true, bool include_rec_onebody = true,
    bool include_lig_onebody = true,
    // ENE-22: topology exclusions/listed pairs between the groups, by local
    // index; ignored unless the sizes match rec and lg.
    const titan_gbsa::TopologyPairs *topology_pairs = nullptr);

#endif // __AVX__

#endif // GBSA_INTRIN_H
