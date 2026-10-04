#ifndef NUCLEIC_PROFILER_H
#define NUCLEIC_PROFILER_H

#include "MOL.h"
#include "MOL_info.h"
#include "arguments.h"
#include "index_v2.h"
#include <map>
#include <string>
#include <utility>
#include <vector>

class TitanExecutor;

struct NA_Interaction {
  int frame;
  double time_ps;
  std::string type;
  std::string pro_res;
  std::string na_res;
  double distance;
};

class NucleicProfiler {
private:
  struct ResidueRangeRef {
    size_t chain_pos{};
    size_t res_idx{};
    size_t start_idx{};
    size_t end_idx{};
    int resnr{};
    std::string resname;
    size_t slot_index{};
  };

  struct ProteinResidueSlot {
    int resnr{};
    std::string resname;
  };

  struct ProteinAtomInfo {
    size_t slot_index{};
    double x{};
    double y{};
    double z{};
  };

  struct RecognitionCount {
    int base_contacts{};
    int backbone_contacts{};
  };

  std::vector<NA_Interaction> interaction_timeline;

  std::vector<ProteinResidueSlot> protein_slots_;
  std::vector<ResidueRangeRef> protein_residues_;
  std::vector<ResidueRangeRef> nucleic_residues_;
  std::vector<RecognitionCount> global_recognition_profile_;
  std::vector<ProteinAtomInfo> protein_atom_buffer_;
  std::vector<std::vector<unsigned char>> worker_base_hits_;
  std::vector<std::vector<unsigned char>> worker_backbone_hits_;
  std::vector<int> protein_grid_head_;
  std::vector<int> protein_grid_next_;
  bool initialized_ = false;
  bool coarse_grained_nucleic_ = false;
  bool protein_grid_active_ = false;
  int protein_grid_nx_ = 0;
  int protein_grid_ny_ = 0;
  int protein_grid_nz_ = 0;
  double protein_grid_len_x_ = 0.0;
  double protein_grid_len_y_ = 0.0;
  double protein_grid_len_z_ = 0.0;
  double protein_grid_cell_x_ = 0.0;
  double protein_grid_cell_y_ = 0.0;
  double protein_grid_cell_z_ = 0.0;
  double contact_cutoff_a_ = 3.5;
  size_t total_frames_processed = 0;

  std::vector<double> calc_residue_center(const MOL &frame,
                                          const std::vector<mol_info> &info,
                                          const class index &chain_idx,
                                          size_t res_idx_in_chain,
                                          const std::string &center_type);
  void initialize(const std::vector<class index> &pro_index,
                  const std::vector<class index> &na_index,
                  const std::vector<mol_info> &pro_info,
                  const std::vector<mol_info> &na_info);
  void ensure_worker_storage(size_t worker_capacity);
  void build_protein_grid(const MOL &pro_frame);

  inline bool is_backbone(const std::string &atom_name) const {
    if (coarse_grained_nucleic_)
      return atom_name == "BB1" || atom_name == "BB2" ||
             atom_name == "BB3";
    return (atom_name == "P" || atom_name == "OP1" || atom_name == "OP2" ||
            atom_name == "OP3" || atom_name == "O1P" || atom_name == "O2P" ||
            atom_name == "O3P" ||
            atom_name == "O5'" || atom_name == "C5'" || atom_name == "C4'" ||
            atom_name == "O4'" || atom_name == "C3'" || atom_name == "O3'" ||
            atom_name == "C2'" || atom_name == "C1'" || atom_name == "O2'");
  }

  inline bool is_base(const std::string &atom_name) const {
    if (coarse_grained_nucleic_)
      return atom_name == "SC1" || atom_name == "SC2" ||
             atom_name == "SC3" || atom_name == "SC4" ||
             atom_name == "SC5";
    if (atom_name.empty())
      return false;
    char first = atom_name[0];
    if (first != 'N' && first != 'C' && first != 'O')
      return false;
    if (is_backbone(atom_name))
      return false;
    return true;
  }

public:
  void reset();
  void initialize_if_needed(const std::vector<class index> &pro_index,
                            const std::vector<class index> &na_index,
                            const std::vector<mol_info> &pro_info,
                            const std::vector<mol_info> &na_info);
  void process_frame(const MOL &pro_frame, const MOL &na_frame,
                     const std::vector<mol_info> &pro_info,
                     const std::vector<mol_info> &na_info,
                     const std::vector<class index> &pro_index,
                     const std::vector<class index> &na_index, size_t num_snap,
                     double time_ps, TitanExecutor &executor,
                     size_t requested_workers);

  void export_results(const std::string &csv_filename);
};

#endif // NUCLEIC_PROFILER_H
