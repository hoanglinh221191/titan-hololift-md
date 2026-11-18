#ifndef CAS_ANALYSIS_H
#define CAS_ANALYSIS_H

#include "MOL.h"
#include "arguments.h"
#include "functions.h"
#include "gbsa_intrin.h"
#include "groups.h"
#include "titan_executor.h"
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

struct CASResidueKey {
  std::string role;
  std::size_t chainnum{};
  std::string chain;
  std::size_t resnr{};
  std::string resname;

  bool operator==(const CASResidueKey &rhs) const noexcept {
    return role == rhs.role && chainnum == rhs.chainnum &&
           chain == rhs.chain && resnr == rhs.resnr && resname == rhs.resname;
  }
};

struct CASResidueKeyHash {
  std::size_t operator()(const CASResidueKey &key) const noexcept;
};

struct CAS_Result {
  CASResidueKey key;
  std::string mutation_status{"valid_mutation"};
  double ddG_vdw{};
  double ddG_elec{};
  double ddG_polar{};
  double ddG_nonpolar{};
  double ddG_total{};
  double distance_sum{};
  int contact_frames{};
};

struct CASMutationAtom {
  size_t atom_index{};
  double charge{};
  double epsilon{};
  double sigma{};
  double radius{};
  // The atom type id selects pair-specific [ nonbond_params ]; a deleted
  // atom gets -1 so an override cannot give it LJ again (ENE-11).
  int32_t type_id{-1};
};

struct CASWorkerScratch {
  AlignedDoubleVector mut_free_radii;
  AlignedDoubleVector mut_complex_radii;
  // Complex radii of the unmutated partner, descreened by the mutant.
  AlignedDoubleVector partner_complex_radii;
  Multi_MOL_parameters parameters;
};

struct CASPerformanceStats {
  std::size_t frames{};
  std::size_t active_residue_evaluations{};
  std::size_t mutation_evaluations{};
  std::size_t skipped_noop_alanine{};
  std::size_t skipped_no_mutatable_sidechain{};
  std::size_t ligand_scan_skipped_nonprotein_frames{};
  double gbsa_mutation_ms{};
  double cas_total_ms{};
};

class CASAnalyzer {
private:
  std::unordered_map<CASResidueKey, CAS_Result, CASResidueKeyHash>
      cas_receptor_data;
  std::unordered_map<CASResidueKey, CAS_Result, CASResidueKeyHash>
      cas_ligand_data;
  std::unordered_map<CASResidueKey, std::vector<CASMutationAtom>,
                     CASResidueKeyHash>
      receptor_mutation_plans;
  std::unordered_map<CASResidueKey, std::vector<CASMutationAtom>,
                     CASResidueKeyHash>
      ligand_mutation_plans;
  std::vector<CASResidueKey> unique_res_rec;
  std::vector<CASResidueKey> unique_res_lig;
  std::vector<CASWorkerScratch> worker_scratch_;
  CASPerformanceStats perf_;
  size_t global_total_frames;
  bool has_reference;
  bool ligand_scan_enabled_{false};
  MOL ref_pro_frame;
  MOL ref_lig_frame;
  std::vector<mol_info> ref_pro_info;
  std::vector<mol_info> ref_lig_info;

public:
  CASAnalyzer();

  void init(Multi_MOL &pro_in, Multi_MOL &lig, const std::string &mol_2_type);

  void process_frame(Multi_MOL &pro_in, Multi_MOL &lig, size_t num_snap,
                     TitanExecutor &executor, size_t num_threads,
                     const std::vector<d_min> &dis_min,
                     cache_sigma_epsilon &CachedData,
                     const arguments &Arguments, info &info_output,
                     const AlignedDoubleVector &rec_free_radii,
                     const AlignedDoubleVector &lg_free_radii,
                     const AlignedDoubleVector &rec_complex_radii,
                     const AlignedDoubleVector &lg_complex_radii,
                     const GBSA_energy_snap &frame_gbsa_total);

  void export_results(const std::string &rec_file, const std::string &lig_file,
                      double eu_par);

  void export_hotspot_pdb(const std::string &out_pdb_filename, double eu_par);

  void export_summary(const std::string &summary_file,
                      const arguments &Arguments) const;

  void export_performance(const std::string &performance_file) const;
};

#endif // CAS_ANALYSIS_H
