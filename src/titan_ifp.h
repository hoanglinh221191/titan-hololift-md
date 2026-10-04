#ifndef TITAN_IFP_H
#define TITAN_IFP_H

#include "MOL.h"
#include "MOL_info.h"
#include "algebra/vec3.h"
#include "arguments.h"
#include "distance_v2.h"
#include "index_v2.h"

#include <array>
#include <bitset>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// ─── Bit positions for 6 interaction types ──────────────────
enum IFPBit : uint8_t {
  BIT_HBOND = 0,
  BIT_SALT = 1,
  BIT_PI_PI = 2,
  BIT_CATION_PI = 3,
  BIT_HYDROPHOBIC = 4,
  BIT_HALOGEN = 5
};

// ─── Per-residue-pair interaction record ────────────────────
struct IFP_Contact {
  size_t resnr_rec{};
  size_t resnr_lig{};
  size_t chainid_rec{};
  size_t chainid_lig{};
  std::string res_name_rec;
  std::string res_name_lig;
  std::bitset<8> interactions{};
};

using IFP_Frame = std::vector<IFP_Contact>;

// ─── Residue key for dedup & occupancy ─────────────────────
// Order: (chainid_rec, resnr_rec, chainid_lig, resnr_lig)
using ResKey = std::tuple<size_t, size_t, size_t, size_t>;

struct ResKeyHash {
  size_t operator()(const ResKey &key) const noexcept {
    const auto mix = [](size_t seed, size_t value) {
      seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
      return seed;
    };
    size_t seed = 0;
    seed = mix(seed, std::get<0>(key));
    seed = mix(seed, std::get<1>(key));
    seed = mix(seed, std::get<2>(key));
    seed = mix(seed, std::get<3>(key));
    return seed;
  }
};

// ─── Free helper functions ─────────────────────────────────

/// Returns [start, end) as 0-based indices into mol.pos.*
/// mol_total_atoms = total atoms in the molecule (fallback for last chain).
inline std::pair<size_t, size_t>
get_atom_range(const std::vector<class index> &Index, size_t chainid,
               size_t respos, size_t mol_total_atoms) {
  // Bounds safety: invalid chainid or respos → empty range
  if (chainid >= Index.size())
    return {0, 0};
  const auto &idx = Index[chainid];
  if (respos >= idx.get_size_v())
    return {0, 0};
  size_t start = idx.get_res_pos_e(respos) - 1; // 1-based → 0-based
  // Not the last residue in this chain → next residue's start
  if (respos + 1 < idx.get_size_v())
    return {start, idx.get_res_pos_e(respos + 1) - 1};
  // Last residue of this chain → use next chain's first atom
  if (chainid + 1 < Index.size())
    return {start, Index[chainid + 1].get_res_pos_e(0) - 1};
  // Last residue of last chain → use molecule end
  return {start, mol_total_atoms};
}



// ─── Main TitanIFP class ───────────────────────────────────
class TitanExecutor;
class TitanIFP {
public:
  struct InteractionCountRecord {
    size_t resnr_rec{}, resnr_lig{};
    size_t chainid_rec{}, chainid_lig{};
    std::string res_name_rec, res_name_lig;
    std::array<size_t, 6> counts{};
  };
  using ContactCountAccumulator =
      std::unordered_map<ResKey, InteractionCountRecord, ResKeyHash>;

  IFP_Frame collect_frame_contacts(const MOL &rec_mol, const MOL &lig_mol,
                                   const std::vector<mol_info> &rec_info,
                                   const std::vector<mol_info> &lig_info,
                                   const std::vector<class index> &rec_index,
                                   const std::vector<class index> &lig_index,
                                   size_t rec_total_atoms,
                                   size_t lig_total_atoms,
                                   const std::vector<d_min> &dis_min,
                                   const IfpArgs &args) const;

  void process_frame(const MOL &rec_mol, const MOL &lig_mol,
                     const std::vector<mol_info> &rec_info,
                     const std::vector<mol_info> &lig_info,
                     const std::vector<class index> &rec_index,
                     const std::vector<class index> &lig_index,
                     size_t rec_total_atoms, size_t lig_total_atoms,
                     const std::vector<d_min> &dis_min, const IfpArgs &args,
                     TitanExecutor &executor, size_t requested_workers,
                     bool build_awh_accumulators);

  void aggregate_occupancy();  // converts counts -> percentage
  void export_csv(const std::string &filename) const;
  void export_rna_annotations_csv(const std::string &filename) const;
  bool has_rna_annotations() const { return !rna_region_map_.empty(); }
  const std::vector<ContactCountAccumulator> &last_frame_contact_accumulators()
      const {
    return worker_contact_accumulators_;
  }
  size_t last_frame_active_workers() const { return last_frame_active_workers_; }
  size_t processed_frame_count() const { return total_frames_; }

private:
  size_t total_frames_{0};  // online frame counter

  // ── Occupancy accumulation ──
  struct OccupancyRecord {
    size_t resnr_rec{}, resnr_lig{};
    size_t chainid_rec{}, chainid_lig{};
    std::string res_name_rec, res_name_lig;
    std::array<size_t, 6> counts{};    // one per IFPBit
    std::array<double, 6> occupancy{}; // filled by aggregate_occupancy()
  };
  std::unordered_map<ResKey, OccupancyRecord, ResKeyHash> occupancy_map_;

  struct RnaRegionRecord {
    size_t resnr_rec{}, resnr_lig{};
    size_t chainid_rec{}, chainid_lig{};
    std::string res_name_rec, res_name_lig;
    std::array<size_t, 3> counts{};
    std::array<double, 3> occupancy{};
  };
  std::unordered_map<ResKey, RnaRegionRecord, ResKeyHash> rna_region_map_;

  struct FrameRnaRegionContact {
    size_t resnr_rec{}, resnr_lig{};
    size_t chainid_rec{}, chainid_lig{};
    std::string res_name_rec, res_name_lig;
    std::array<bool, 3> flags{};
  };

  struct FrameAnalysisResult {
    IFP_Frame contacts;
    std::vector<FrameRnaRegionContact> rna_regions;
  };

  struct PreparedResiduePair {
    dmin_base contact;
    size_t rs{};
    size_t re{};
    size_t ls{};
    size_t le{};
    bool pair_has_nucleic{false};
  };

  FrameAnalysisResult analyze_frame(const MOL &rec_mol, const MOL &lig_mol,
                                   const std::vector<mol_info> &rec_info,
                                   const std::vector<mol_info> &lig_info,
                                   const std::vector<class index> &rec_index,
                                   const std::vector<class index> &lig_index,
                                   size_t rec_total_atoms,
                                   size_t lig_total_atoms,
                                   const std::vector<d_min> &dis_min,
                                   const IfpArgs &args) const;

  void ensure_worker_storage(size_t worker_count);
  std::vector<PreparedResiduePair>
  build_prepared_pairs(const std::vector<d_min> &dis_min,
                       const std::vector<class index> &rec_index,
                       const std::vector<class index> &lig_index,
                       size_t rec_total_atoms, size_t lig_total_atoms) const;
  bool analyze_prepared_pair(const PreparedResiduePair &pair,
                             const MOL &rec_mol, const MOL &lig_mol,
                             const std::vector<mol_info> &rec_info,
                             const std::vector<mol_info> &lig_info,
                             const IfpArgs &args, IFP_Contact &result,
                             std::array<bool, 3> &region_flags) const;
  void accumulate_frame(const FrameAnalysisResult &frame);
  static void accumulate_contact_counts(ContactCountAccumulator &accumulator,
                                        const IFP_Contact &contact);

  std::vector<FrameAnalysisResult> worker_frame_results_;
  std::vector<ContactCountAccumulator> worker_contact_accumulators_;
  size_t last_frame_active_workers_{0};

  // ── Perf cache: (chainid, resnr) → respos ──


  // ── Geometric interaction checkers ──

  /// H-bond: D···A < cutoff AND D-H···A angle ≥ 135°
  bool check_hbond(const MOL &rec_mol, const MOL &lig_mol,
                   const std::vector<mol_info> &rec_info,
                   const std::vector<mol_info> &lig_info, size_t ar, size_t al,
                   size_t rs, size_t re, size_t ls, size_t le, double d,
                   const IfpArgs &args) const;

  /// Salt bridge: acidic-basic pair, distance < cutoff
  static bool check_salt_bridge(const mol_info &info_a, const mol_info &info_b);

  /// Hydrophobic: both atoms C-based, distance < cutoff
  static bool check_hydrophobic(const mol_info &info_a,
                                const mol_info &info_b);

  /// Pi-Pi and Cation-Pi (ring-level, sets is_cation_pi if applicable)
  bool check_pi_pi(const MOL &rec_mol, const MOL &lig_mol, size_t rec_start,
                   size_t rec_end, size_t lig_start, size_t lig_end,
                   const std::vector<mol_info> &rec_info,
                   const std::vector<mol_info> &lig_info,
                   const std::string &rec_res_name,
                   const std::string &lig_res_name,
                   double cutoff,
                   double cutoff_cation_pi, bool &is_cation_pi) const;

  /// Halogen bond: C-X···Acceptor, angle > 160°, distance < cutoff
  bool check_halogen(const MOL &rec_mol, const MOL &lig_mol,
                     const std::vector<mol_info> &rec_info,
                     const std::vector<mol_info> &lig_info, size_t ar,
                     size_t al, size_t rs, size_t re, size_t ls, size_t le,
                     double d) const;

  // ── Classification helpers (static, no state) ──
  static bool is_aromatic_residue(const std::string &res_name);
  static bool is_hydrophobic_atom(const std::string &atom_name,
                                  const std::string &res_name);
  static bool is_hbond_donor(const std::string &atom,
                             const std::string &res_name);
  static bool is_hbond_acceptor(const std::string &atom,
                                const std::string &res_name);
  static bool is_acidic(const std::string &atom, const std::string &res_name);
  static bool is_basic(const std::string &atom, const std::string &res_name);
  static bool is_halogen(const std::string &atom);
  static bool is_halogen_acceptor(const std::string &atom);
  static bool is_cation_atom(const std::string &atom,
                             const std::string &res_name);

  /// Stack-allocated ring atom indices (max 9 for TRP bicyclic).
  struct RingAtoms {
    std::array<size_t, 9> idx{};
    size_t count{0};
    [[nodiscard]] bool empty() const { return count == 0; }
    [[nodiscard]] size_t size() const { return count; }
    [[nodiscard]] const size_t *begin() const { return idx.data(); }
    [[nodiscard]] const size_t *end() const { return idx.data() + count; }
    void push_back(size_t v) {
      if (count < idx.size()) idx[count++] = v;
    }
  };

  /// Get ring atom indices for known aromatic residues.
  /// Returns empty RingAtoms if residue has no known aromatic ring.
  static RingAtoms
  get_ring_atoms(const std::string &res_name,
                 const std::vector<mol_info> &info, size_t start, size_t end);

  /// Compute ring centroid and normal.
  static bool compute_ring_geometry(const MOL &mol,
                                    const RingAtoms &ring_indices,
                                    titan::algebra::Vec3 &com,
                                    titan::algebra::Vec3 &normal);

  /// Find hydrogen bonded to a donor atom (distance < 1.2 Å), returns SIZE_MAX
  /// if not found.
  size_t find_bonded_hydrogen(const MOL &mol, const std::vector<mol_info> &info,
                              size_t donor_idx, size_t res_start,
                              size_t res_end) const;
};

#endif // TITAN_IFP_H
