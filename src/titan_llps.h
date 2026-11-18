#pragma once
#ifndef TITAN_LLPS_H
#define TITAN_LLPS_H

#include "aligned_allocator.h"
#include "titan_neighbor_grid.h"
#include "titan_llps_geometry.h"
#include <exception>
#include "titan_thread_compat.h"
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

class TitanExecutor;
class MOL;
class info;
class mol_info;

// ============================================================================
// Structs
// ============================================================================

/// Per-frame snapshot of LLPS state (macro-node level)
struct LLPSFrameResult {
  int frame_idx;
  float time_ns;
  float gcc_fraction;   // N_gcc_macronodes / N_macronodes
  int n_clusters;       // number of connected components
  int gcc_size;         // macro-nodes in giant connected component
  float gcc_rg_nm;      // Rg of GCC (nm), PBC-corrected
  float mean_degree;    // <k> = 2*edges / n_macronodes
  float density_dense;  // representative macro-nodes/nm^3 in declared window
  float density_dilute;
  titan_llps_geometry::PhaseResult phase;
};

/// Full-trajectory LLPS results
struct LLPSResult {
  std::vector<LLPSFrameResult> frames;
  std::vector<double> ct_lags_ns;
  std::vector<double> ct_values;
  double tau_ns = 0.0;
  bool is_percolated = false;
  double percolation_time_ns = -1.0;
  std::vector<double> pk_mean; // P(k) mean over trajectory
  // STR-04: standard deviation of the per-frame P(k) across frames (1/N)
  std::vector<double> pk_std;
  // Residue-level contact frequency, aggregated over sticker-bead contacts.
  std::vector<std::vector<float>> contact_freq; // [n_residue][n_residue]
  std::vector<int> sticker_residue_indices;     // residue numbers for rows/cols
};

struct LLPSLifetimeDistributionResult {
  std::vector<double> lifetime_ns;
  std::vector<uint64_t> counts;
  uint64_t total_events{0};
  double mean_ns{0.0};
  double median_ns{0.0};
  double p90_ns{0.0};
  double max_ns{0.0};
  // Liquidity v3: censoring. A contact still present at the last analysed
  // frame is right-censored (its "lifetime" is a lower bound); one present
  // from the first to the last frame spans the whole window. Statistics over
  // contacts whose formation AND rupture were both observed are kept apart.
  uint64_t right_censored_events{0};
  uint64_t full_window_events{0};
  uint64_t uncensored_events{0};
  double uncensored_mean_ns{0.0};
  double uncensored_median_ns{0.0};
  double uncensored_p90_ns{0.0};
  double uncensored_p99_ns{0.0};
  double window_ns{0.0};
};

struct LLPSContactDynamicsResult {
  LLPSLifetimeDistributionResult sticker_contact_lifetime;
  LLPSLifetimeDistributionResult protein_rna_contact_lifetime;
  double protein_neighbor_persistence{0.0};
  double rna_neighbor_persistence{0.0};
  double global_neighbor_persistence{0.0};
  double protein_rewiring_rate_per_ns{0.0};
  double rna_rewiring_rate_per_ns{0.0};
  double global_rewiring_rate_per_ns{0.0};
  double interface_new_contacts_per_ns{0.0};
  double interface_lost_contacts_per_ns{0.0};
  double interface_net_change_per_ns{0.0};
  uint64_t interface_new_contacts{0};
  uint64_t interface_lost_contacts{0};
};

enum class LlpsKernelMode : unsigned char { Auto, Scalar, Avx2 };

// ============================================================================
// TitanLLPS
// ============================================================================
class TitanLLPS {
public:
  TitanLLPS(int n_chains, const std::vector<int> &chain_start,
            const std::vector<int> &chain_len, float r_cut_nm = 0.7f,
            float perc_thr = 0.5f, int dimer_size = 1,
            bool contacts_on = false, float voxel_size_nm = 0.5f,
            bool dynamics_enabled = false, bool dynamics_detailed = false,
            float prna_cutoff_nm = 0.0f, const std::string &kernel_mode = "auto",
            const std::string &density_mode = "auto");
  ~TitanLLPS() { try { flush_async(); } catch (...) {} }
  void configure_phase(const titan_llps_geometry::PhaseConfig& config) {
    flush_async(); config.validate(); phase_config_=config;
  }

  void set_sticker_beads(const std::vector<mol_info> &info,
                         const std::string &sticker_str);
  void set_rna_metadata(const std::vector<mol_info> &rna_info);
  void configure_dynamics(bool enabled, bool detailed,
                          float prna_cutoff_nm = 0.0f);
  void configure_trace_outputs(const info &info_output);

  void process_frame(const MOL &mol_frame, const std::vector<mol_info> &info,
                     float box_x_nm, float box_y_nm, float box_z_nm,
                     float time_ns, int frame_idx,
                     const MOL *rna_frame = nullptr,
                     const std::vector<mol_info> *rna_info = nullptr);

  void start_async();
  void process_frame_async(const MOL &mol_frame,
                           const std::vector<mol_info> &info, float box_x_nm,
                           float box_y_nm, float box_z_nm, float time_ns,
                           int frame_idx, const MOL *rna_frame = nullptr,
                           const std::vector<mol_info> *rna_info = nullptr);
  void flush_async();

  void finalize(TitanExecutor &executor, int max_lag_frames);
  void export_results(const info &info_output, std::fstream &log) const;

#ifdef TITAN_LLPS_TESTING
  void debug_build_contacts(const std::vector<float> &pos_nm, float bx,
                            float by, float bz,
                            std::vector<std::pair<int, int>> &macro_contacts) {
    std::vector<uint64_t> packed;
    build_contacts(pos_nm, bx, by, bz, packed);
    macro_contacts.clear();
    macro_contacts.reserve(packed.size());
    for (uint64_t key : packed)
      macro_contacts.push_back(unpack_pair(key));
  }
#endif

  const LLPSResult &result() const { return result_; }
  const LLPSContactDynamicsResult &contact_dynamics() const {
    return contact_dynamics_result_;
  }
  bool is_percolated() const { return result_.is_percolated; }
  int n_frames_processed() const {
    return static_cast<int>(result_.frames.size());
  }
  int n_macronodes() const { return n_macro_; }

private:
  struct FrameWorkItem {
    titan_llps_geometry::Cell cell;
    std::vector<float> pos_nm;
    std::optional<std::vector<float>> rna_nm;
    float bx, by, bz;
    float time_ns;
    int frame_idx;
    int n_rna = 0;
  };

  struct ActiveContactRun {
    double start_time_ns{0.0};
    double last_time_ns{0.0};
    int start_frame_idx{0};
    int last_frame_idx{0};
  };

  struct ContactEventRow {
    double time_ns{0.0};
    std::string contact_type{};
    uint64_t key{0};
    int a_id{-1};
    int b_id{-1};
    int state{0};
    double duration_ns{0.0};
  };

  struct ContactResidueMeta {
    std::string species;
    int chain_index{-1};
    int chain_raw_id{-1};
    std::string chain_label;
    int residue_index{-1};
    std::string residue_name;
  };

  struct NeighborFrameMetric {
    double time_ns{0.0};
    double dt_ns{0.0};
    double protein_persistence{0.0};
    double rna_persistence{0.0};
    double global_persistence{0.0};
    uint64_t protein_changes{0};
    uint64_t rna_changes{0};
    uint64_t global_changes{0};
  };

  struct InterfaceTurnoverFrame {
    double time_ns{0.0};
    double dt_ns{0.0};
    uint64_t new_contacts{0};
    uint64_t lost_contacts{0};
  };

  struct PerformanceCounters {
    uint64_t frames_processed{0};
    uint64_t macro_contact_pairs{0};
    uint64_t sticker_contact_pairs{0};
    uint64_t protein_chain_pairs{0};
    uint64_t protein_rna_pairs{0};
    double pack_seconds{0.0};
    double contact_seconds{0.0};
    double protein_rna_seconds{0.0};
    double graph_seconds{0.0};
    double density_seconds{0.0};
    double polymer_scaling_seconds{0.0};
    double dynamics_seconds{0.0};
    double finalize_ct_seconds{0.0};
    double finalize_contact_map_seconds{0.0};
    double finalize_seconds{0.0};
    double export_seconds{0.0};
    std::size_t max_queue_depth{0};
    std::size_t finalize_executor_workers{1};
  };

  std::queue<FrameWorkItem> work_queue_;
  std::vector<FrameWorkItem> free_work_items_;
  std::mutex queue_mtx_;
  std::condition_variable queue_cv_;
  bool worker_stop_ = false;
  titan::jthread worker_thread_;
  bool async_mode_ = false;
  bool worker_processing_ = false;
  std::exception_ptr worker_error_;
  static constexpr int MAX_QUEUE_SIZE_ = 4;
  std::condition_variable queue_full_cv_;

  void worker_loop(titan::stop_token stoken);
  void process_frame_impl(const std::vector<float> &pos_nm,
                          const float *rna_nm_data, int n_rna, float bx,
                          float by, float bz, float time_ns, int frame_idx);

  int n_chains_;
  int dimer_size_;
  int n_macro_;
  float r_cut_;
  float r_cut2_;
  float perc_thr_;
  bool contacts_on_;
  bool dynamics_enabled_{false};
  bool dynamics_detailed_{false};
  float prna_cutoff_nm_{0.0f};
  float prna_cutoff2_nm_{0.0f};
  LlpsKernelMode kernel_mode_{LlpsKernelMode::Auto};
  std::string density_mode_{"auto"};
  titan_llps_geometry::Cell cell_;
  titan_llps_geometry::PhaseConfig phase_config_;
  mutable std::optional<titan_llps_geometry::Point> gcc_center_;
  double frame_dist2(double dx,double dy,double dz,float bx,float by,float bz) const {
    if (!cell_.metric) return pbc_dist2(dx,dy,dz,bx,by,bz);
    auto p=cell_.delta({dx,dy,dz});return p[0]*p[0]+p[1]*p[1]+p[2]*p[2];
  }
  bool density_disabled_by_budget_{false};

  std::vector<int> chain_start_;
  std::vector<int> chain_len_;
  std::vector<std::vector<int>> macro_chains_;
  std::vector<int> bb_atom_;

  std::vector<bool> sticker_mask_;
  std::vector<int> sticker_atom_idx_;

  int n_pairs_;
  int n_words_;
  // C(t) window (STR-03): the frames from the percolation frame on (every
  // frame while the system has not percolated). The macro-node adjacency of
  // each window frame is kept (n_words_ per stored frame) so that C(t) runs
  // over every origin of the window once the persistent pattern is known;
  // when the history would pass ct_budget_bytes_ every second stored frame is
  // dropped and the stride doubles. pair_contact_cnt_ counts every window
  // frame, so the pattern and its threshold use the same frames.
  void record_ct_frame(const std::vector<uint64_t> &contacts, bool restart);
  std::vector<uint64_t> ct_history_;
  std::vector<uint64_t> ct_bits_buf_;
  std::size_t ct_stored_frames_ = 0;
  int ct_stride_ = 1;
  int ct_window_frames_ = 0;
  std::size_t ct_budget_bytes_ = 0;
  bool ct_budget_warned_ = false;
  std::size_t ct_origin_frames_ = 0;
  double ct_dt_ns_ = 0.0;
  std::vector<int> pair_contact_cnt_;

  int pk_max_k_ = -1;
  std::vector<double> pk_sum_;
  std::vector<double> pk_sq_;

  int n_stickers_;
  std::vector<float> contact_accum_upper_;
  std::vector<int> sticker_seen_gen_;
  int sticker_cur_gen_ = 0;

  bool has_rna_ = false;
  std::vector<int> atom_macro_;
  std::vector<int> seen_gen_;
  int cur_gen_ = 0;

  mutable std::vector<float> shifted_pos_buf_;
  mutable std::vector<float> rna_nm_buf_;
  mutable std::vector<float> pos_nm_buf_;
  mutable AlignedDoubleVector gcc_rg_x_buf_;
  mutable AlignedDoubleVector gcc_rg_y_buf_;
  mutable AlignedDoubleVector gcc_rg_z_buf_;
  mutable std::vector<std::pair<int, int>> contacts_buf_;
  mutable std::vector<uint64_t> macro_contact_keys_buf_;
  mutable std::vector<int> degree_buf_;
  mutable std::vector<int> degree_hist_buf_; // STR-04: nodes per degree, one frame
  mutable std::vector<int> gcc_macros_buf_;
  mutable std::vector<uint64_t> sticker_pair_keys_buf_;
  mutable std::vector<uint64_t> protein_chain_pair_keys_buf_;
  mutable std::vector<uint64_t> protein_rna_pair_keys_buf_;
  mutable std::vector<std::size_t> neighbor_offsets_buf_;
  mutable std::vector<std::size_t> neighbor_write_pos_buf_;
  mutable std::vector<int> neighbor_flat_buf_;
  std::vector<std::size_t> prev_neighbor_offsets_;
  std::vector<int> prev_neighbor_flat_;
  std::vector<std::size_t> current_neighbor_offsets_buf_;
  std::vector<int> current_neighbor_flat_buf_;
  std::vector<int> sticker_local_idx_;
  titan_neighbor::HalfShellGrid contact_grid_;
  titan_neighbor::HalfShellGrid prna_grid_;

  std::vector<int> atom_chain_;
  std::vector<int> atom_residue_;
  std::vector<int> protein_atom_contact_node_;
  std::vector<int> rna_atom_chain_;
  std::vector<int> rna_atom_residue_;
  std::vector<int> rna_atom_contact_node_;
  std::vector<int> rna_chain_raw_ids_;
  std::unordered_map<int, int> rna_chain_raw_to_local_;
  std::vector<ContactResidueMeta> contact_residue_nodes_;
  std::size_t protein_contact_node_count_{0};
  std::unordered_map<uint64_t, uint64_t> detailed_contact_frame_counts_;
  mutable std::vector<uint64_t> detailed_pp_frame_keys_buf_;
  mutable std::vector<uint64_t> detailed_prna_frame_keys_buf_;

  float voxel_size_nm_ = 0.5f;
  int grid_nx_ = 0, grid_ny_ = 0, grid_nz_ = 0;
  mutable std::vector<double> vox_pro_;
  mutable std::vector<double> vox_rna_;
  std::array<double,9> density_reference_cell_{};
  int density_frame_count_ = 0;
  float grid_bx_ = 0, grid_by_ = 0, grid_bz_ = 0;

  int max_rna_chain_len_ = 0;
  std::vector<int> rna_chain_starts_;
  std::vector<double> sum_R2_s_;
  std::vector<uint64_t> count_R2_s_;

  LLPSResult result_;
  LLPSContactDynamicsResult contact_dynamics_result_;
  int percolation_frame_idx_ = -1;

  std::unordered_map<uint64_t, ActiveContactRun> active_sticker_contacts_;
  std::unordered_map<uint64_t, ActiveContactRun> active_prna_contacts_;
  std::vector<double> sticker_contact_lifetimes_ns_;
  std::vector<double> prna_contact_lifetimes_ns_;
  // Liquidity v3: lifetimes whose start and end were both observed.
  std::vector<double> sticker_contact_lifetimes_uncensored_ns_;
  std::vector<double> prna_contact_lifetimes_uncensored_ns_;
  std::vector<ContactEventRow> contact_trace_rows_;
  std::ofstream contact_trace_out_;
  bool contact_trace_streaming_{false};
  std::vector<NeighborFrameMetric> neighbor_frame_metrics_;
  std::vector<InterfaceTurnoverFrame> interface_turnover_frames_;
  std::vector<uint64_t> prev_prna_contacts_;
  double first_contact_time_ns_{0.0};
  double last_contact_time_ns_{0.0};
  bool have_contact_time_window_{false};
  double sum_protein_neighbor_persistence_{0.0};
  double sum_rna_neighbor_persistence_{0.0};
  double sum_global_neighbor_persistence_{0.0};
  uint64_t count_protein_neighbor_persistence_{0};
  uint64_t count_rna_neighbor_persistence_{0};
  uint64_t count_global_neighbor_persistence_{0};
  uint64_t total_protein_neighbor_changes_{0};
  uint64_t total_rna_neighbor_changes_{0};
  uint64_t total_global_neighbor_changes_{0};
  double total_neighbor_dt_ns_{0.0};
  uint64_t total_new_interface_contacts_{0};
  uint64_t total_lost_interface_contacts_{0};
  double total_interface_dt_ns_{0.0};
  PerformanceCounters perf_;

  struct UF {
    std::vector<int> parent, rank_, size_;
    void init(int n);
    void reset(int n);
    int find(int x);
    void merge(int a, int b);
    int largest_component() const;
  } uf_;

  void build_contacts(const std::vector<float> &pos_nm, float bx, float by,
                      float bz,
                      std::vector<uint64_t> &macro_contact_keys,
                      std::vector<uint64_t> *sticker_pair_keys = nullptr,
                      std::vector<uint64_t> *protein_chain_pair_keys = nullptr);
  void build_protein_rna_contacts(const std::vector<float> &pos_nm,
                                  const float *rna_nm_data, int n_rna,
                                  float bx, float by, float bz,
                                  std::vector<uint64_t> &protein_rna_pairs);
  void update_contact_dynamics(double time_ns, int frame_idx,
                               const std::vector<uint64_t> &sticker_pairs,
                               const std::vector<uint64_t> &protein_pairs,
                               const std::vector<uint64_t> &protein_rna_pairs);
  void finalize_contact_dynamics();
  void export_contact_dynamics(const info &info_output,
                               std::fstream &log) const;
  void export_sticker_hotspots(const info &info_output,
                               std::fstream &log) const;

  float compute_gcc_rg(const std::vector<float> &pos_nm, float bx, float by,
                       float bz, const std::vector<int> &gcc_macros) const;

  static float pbc_dist2(float dx, float dy, float dz, float bx, float by,
                         float bz);
  int pair_index(int i, int j) const;
  std::size_t contact_upper_index(int i, int j) const;
  void set_adj_bit(std::vector<uint64_t> &words, int i, int j) const;
  double fit_tau(const std::vector<double> &ct,
                 const std::vector<double> &lags_ns) const;
  bool kernel_allows_avx2() const noexcept;
  static LlpsKernelMode parse_kernel_mode(const std::string &mode);
  void expand_contact_map_result(int n_frames);

  static uint64_t pack_ordered_pair(int a, int b);
  static uint64_t pack_prna_pair(int protein_chain, int rna_chain_local);
  static std::pair<int, int> unpack_pair(uint64_t key);
};

#endif // TITAN_LLPS_H
