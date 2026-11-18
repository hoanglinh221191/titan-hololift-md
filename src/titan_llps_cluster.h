#ifndef TITAN_LLPS_CLUSTER_H
#define TITAN_LLPS_CLUSTER_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class TitanExecutor;
class EnergyDecompositionProvider;
class MOL;
class box;
class info;
class mol_info;
struct LLPSContactDynamicsResult;

struct LLPSNode {
  int id;
  int type;       // 1: Protein, 2: RNA/Ligand
  int chain_idx;
  double x, y, z;
  int cluster_id{0};   // 0: unclassified, -1: noise, >0: cluster
  bool is_core{false}; // DBSCAN core point
};

struct ClusterMetrics {
  double time_ps;
  int num_nodes;
  int num_clusters;
  int largest_cluster_pro_chains;
  int largest_cluster_rna_chains;
  int noise_count;
  double rg_pro_dilute{0.0};
  double rg_pro_dense{0.0};
  double rg_rna_dilute{0.0};
  double rg_rna_dense{0.0};
};

struct LLPSAutocorrSpeciesResult {
  std::vector<double> lag_ns;
  std::vector<double> mean;
  std::vector<double> stddev;
  std::vector<uint64_t> contributing_chains;
  double tau_ns{0.0};
  // Liquidity v2. `mean` above averages per-unit correlations and discards
  // units with occupancy p == 0 or p == 1, i.e. it removes exactly the
  // permanently resident (immobile) population a FRAP plateau measures.
  // `pooled` is the standard pooled membership correlation
  //   C(t) = (<h(0)h(t)> - <h>^2) / (<h> - <h>^2)
  // accumulated over every unit of the species, permanent residents included.
  std::vector<double> pooled;
  std::vector<uint64_t> pooled_pairs;
  double occupancy{0.0};        // <h>, fraction of unit-frames inside dense
  double pooled_tau_ns{0.0};    // decay time of the decaying component
  double mobile_fraction{0.0};  // A in C(t) = A exp(-t/tau) + (1 - A)
  double pooled_plateau{0.0};   // 1 - A, FRAP-plateau analogue (immobile)
  uint64_t units{0};            // units of this species (chains or segments)
  // Material-exchange census over the analysed window. The pooled amplitude
  // above is a variance share and is dominated by whichever units happen to
  // shuttle; a FRAP plateau is a share of MATERIAL, so count units instead:
  // permanent = inside the dense phase in every frame it was seen,
  // exchanging = seen both inside and outside, never = never inside.
  uint64_t units_permanent{0};
  uint64_t units_exchanging{0};
  uint64_t units_never{0};
  // exchanging / (permanent + exchanging): the share of dense-phase material
  // that visited the dilute phase at least once inside the window. This is
  // the quantity to put beside a FRAP mobile fraction, and like a FRAP
  // plateau it grows with observation time.
  double exchange_fraction{0.0};
};

struct LLPSEpisodeSummary {
  uint64_t total_episodes{0};
  uint64_t complete_episodes{0};
  uint64_t censored_episodes{0};
  double mean_complete_ns{0.0};
  double median_complete_ns{0.0};
  double p90_complete_ns{0.0};
  double max_complete_ns{0.0};
};

struct LLPSChainDynamicsResult {
  LLPSAutocorrSpeciesResult protein_membership_autocorr;
  LLPSAutocorrSpeciesResult rna_membership_autocorr;
  LLPSEpisodeSummary protein_residence;
  LLPSEpisodeSummary rna_residence;
  double protein_entry_rate_per_ns{0.0};
  double protein_exit_rate_per_ns{0.0};
  double rna_entry_rate_per_ns{0.0};
  double rna_exit_rate_per_ns{0.0};
  double protein_entry_rate_per_chain_ns{0.0};
  double protein_exit_rate_per_chain_ns{0.0};
  double rna_entry_rate_per_chain_ns{0.0};
  double rna_exit_rate_per_chain_ns{0.0};
};

struct LLPSDenseMSDResult {
  std::vector<double> lag_ns;
  std::vector<double> protein_msd_nm2;
  std::vector<double> rna_msd_nm2;
  std::vector<uint64_t> protein_counts;
  std::vector<uint64_t> rna_counts;
  double protein_diffusion_nm2_per_ns{0.0};
  double rna_diffusion_nm2_per_ns{0.0};
  double protein_fit_start_ns{0.0};
  double protein_fit_end_ns{0.0};
  double rna_fit_start_ns{0.0};
  double rna_fit_end_ns{0.0};
  // Liquidity v2 diagnostics. The MSD log-log exponent on the fitted window
  // tells whether a Fickian regime was reached at all; without it a slope is
  // not a diffusion coefficient. `*_diffusive` is false when no window with
  // exponent in [exponent_lo, exponent_hi] exists, and the corresponding
  // diffusion value is then NaN rather than a plausible-looking number.
  double protein_msd_exponent{0.0};
  double rna_msd_exponent{0.0};
  double protein_msd_intercept_nm2{0.0};
  double rna_msd_intercept_nm2{0.0};
  bool protein_diffusive{false};
  bool rna_diffusive{false};
  // FRAP-comparable mobile fraction: fraction of dense-phase units whose
  // drift-corrected displacement exceeds `mobile_displacement_nm` over
  // `mobile_lag_ns`.
  double mobile_displacement_nm{0.0};
  double mobile_lag_ns{0.0};
  double protein_mobile_fraction_displacement{0.0};
  double rna_mobile_fraction_displacement{0.0};
  uint64_t protein_mobile_pairs{0};
  uint64_t rna_mobile_pairs{0};
  // Cumulative dense-phase translation that was removed (nm).
  double dense_drift_removed_nm{0.0};
  bool drift_corrected{false};
  // Liquidity v3: local log-log exponent per lag (factor-2 window in time),
  // MSD at fixed lags, and the displacement mobile-fraction curve.
  std::vector<double> protein_local_alpha;
  std::vector<double> rna_local_alpha;
  std::vector<double> fixed_lags_ns;
  std::vector<double> protein_msd_fixed;
  std::vector<double> rna_msd_fixed;
  std::vector<double> protein_mobile_curve;
  std::vector<double> rna_mobile_curve;
  std::vector<uint64_t> protein_mobile_curve_pairs;
  std::vector<uint64_t> rna_mobile_curve_pairs;
  // Lateral (in-plane) MSD and D = slope/4 for slab geometry. STR-01: axis
  // 0/1/2 = lattice vector a/b/c with the largest plane spacing; the
  // lateral displacement has no component along slab_normal (unit vector).
  int slab_normal_axis{2};
  std::array<double, 3> slab_normal{0.0, 0.0, 1.0};
  std::vector<double> protein_msd_lat_nm2;
  std::vector<double> rna_msd_lat_nm2;
  std::vector<double> protein_lat_local_alpha;
  std::vector<double> rna_lat_local_alpha;
  double protein_lat_diffusion_nm2_per_ns{0.0};
  double rna_lat_diffusion_nm2_per_ns{0.0};
  double protein_lat_fit_start_ns{0.0}, protein_lat_fit_end_ns{0.0};
  double rna_lat_fit_start_ns{0.0}, rna_lat_fit_end_ns{0.0};
  double protein_lat_exponent{0.0}, rna_lat_exponent{0.0};
  double protein_lat_intercept_nm2{0.0}, rna_lat_intercept_nm2{0.0};
  bool protein_lat_diffusive{false}, rna_lat_diffusive{false};
  // Whole RNA molecule COM MSD (3D and lateral), pooled over RNA molecules.
  std::vector<double> rna_molecule_lag_ns;
  std::vector<double> rna_molecule_msd_nm2;
  std::vector<double> rna_molecule_msd_lat_nm2;
  std::vector<uint64_t> rna_molecule_counts;
  uint64_t rna_molecules{0};
};

struct LLPSClusterLineageResult {
  uint64_t fusion_events{0};
  uint64_t fission_events{0};
};

struct LLPSPaperConfig {
  bool enabled{false};
  double profile_bin_nm{0.2};
  int requested_blocks{0}; // 0 = auto
  std::string geometry{"auto"};
  bool fractal_enabled{false};
  int slab_axis{2}; // reciprocal normal a/b/c, shared with main LLPS phase axis
  double connectivity_cutoff_angstrom{10.0};
};

struct LLPSPaperFractalConfig {
  bool enabled{false};
  double min_scale_nm{0.5};
  double max_scale_fraction{0.75};
  double connectivity_cutoff_angstrom{10.0};
};

struct LLPSPaperFractalFrame {
  int frame{0};
  double time_ns{0.0};
  int dense_nodes{0};
  int dense_protein_nodes{0};
  int dense_rna_nodes{0};
  double df_mass_radius{std::numeric_limits<double>::quiet_NaN()};
  double df_box_count{std::numeric_limits<double>::quiet_NaN()};
  double kf_mass_radius{std::numeric_limits<double>::quiet_NaN()};
  double fit_r2_mass_radius{0.0};
  double fit_r2_box_count{0.0};
  int scale_count_mass_radius{0};
  int scale_count_box_count{0};
  double dense_radius_nm{0.0};
  double dense_rg_nm{0.0};
  double mean_local_valency{std::numeric_limits<double>::quiet_NaN()};
  double protein_rna_neighbor_fraction{
      std::numeric_limits<double>::quiet_NaN()};
  bool valid{false};
  std::string invalid_reason{"not_evaluated"};
};

struct LLPSPaperFractalScaleRow {
  int frame{0};
  double time_ns{0.0};
  std::string method{"mass_radius"};
  double scale_nm{0.0};
  double value{0.0};
  double log_scale{0.0};
  double log_value{0.0};
};

class LLPSPaperFractalAccumulator {
public:
  void configure(const LLPSPaperFractalConfig &config);
  bool enabled() const { return config_.enabled; }
  void process_frame(double time_ns, int frame,
                     const std::vector<LLPSNode> &nodes,
                     const std::vector<bool> &is_dense, int giant_cluster_id,
                     const box &Box);
  void export_outputs(const info &info_output, std::fstream &log,
                      int requested_blocks) const;
  const std::vector<LLPSPaperFractalFrame> &frames() const { return frames_; }
  const std::vector<LLPSPaperFractalScaleRow> &scales() const {
    return scales_;
  }

#ifdef TITAN_LLPS_TESTING
  static LLPSPaperFractalFrame debug_fit_points(
      const std::vector<std::array<double, 3>> &points_nm);
#endif

private:
  LLPSPaperFractalConfig config_;
  std::vector<LLPSPaperFractalFrame> frames_;
  std::vector<LLPSPaperFractalScaleRow> scales_;
};

class LLPSPaperAccumulator {
public:
  void configure(const LLPSPaperConfig &config);
  bool enabled() const { return config_.enabled; }
  void process_frame(double time_ps, int frame_idx,
                     const std::vector<LLPSNode> &nodes,
                     const std::vector<bool> &is_dense,
                     int giant_cluster_id, const box &Box);
  void export_outputs(const info &info_output, std::fstream &log) const;

public:
  struct FrameRow {
    double time_ns{0.0};
    int frame{0};
    std::string geometry{"none"};
    int total_protein_nodes{0};
    int total_rna_nodes{0};
    int dense_protein_nodes{0};
    int dense_rna_nodes{0};
    int dilute_protein_nodes{0};
    int dilute_rna_nodes{0};
    double dense_fraction{0.0};
    double dense_volume_nm3{0.0};
    double dilute_volume_nm3{0.0};
    double box_volume_nm3{0.0};
    double dense_protein_conc{0.0};
    double dilute_protein_conc{0.0};
    double dense_rna_conc{0.0};
    double dilute_rna_conc{0.0};
    double protein_partition{0.0};
    double rna_partition{0.0};
    double protein_transfer_proxy{0.0};
    double rna_transfer_proxy{0.0};
    double dense_protein_rna_ratio{0.0};
    double rna_enrichment{0.0};
    double core_fraction{0.0};
    double interface_fraction{0.0};
    double dilute_fraction{0.0};
    double radius_nm{0.0};
    double interface_width_nm{0.0};
  };

  struct ProfileBin {
    uint64_t samples{0};
    double volume_nm3_sum{0.0};
    double sampled_upper_nm_sum{0.0};
    double protein_count{0.0};
    double rna_count{0.0};
    double dense_protein_count{0.0};
    double dense_rna_count{0.0};
    double sticker_count{0.0};
    double water_count{0.0};
    double ion_count{0.0};
  };

  struct ContactFrameRow {
    double time_ns{0.0};
    int frame{0};
    std::string metric{"cluster_comembership_proxy"};
    uint64_t pp{0};
    uint64_t prna{0};
    uint64_t rr{0};
    uint64_t pp_core{0};
    uint64_t prna_core{0};
    uint64_t rr_core{0};
    uint64_t pp_interface{0};
    uint64_t prna_interface{0};
    uint64_t rr_interface{0};
  };

  struct RunningScalar {
    uint64_t n{0};
    double sum{0.0};
    double sum_sq{0.0};
    void add(double v);
    double mean() const;
    double sem() const;
  };

private:
  LLPSPaperConfig config_;
  std::vector<FrameRow> frames_;
  std::vector<ContactFrameRow> contact_rows_;
  std::vector<ProfileBin> profile_bins_;
  uint64_t skipped_no_dense_{0};
  uint64_t skipped_bad_box_{0};
  std::map<std::string,uint64_t> geometry_failures_;
  std::string observed_geometry_{"none"};
  LLPSPaperFractalAccumulator fractal_;
};

class TitanLLPSCluster {
public:
  TitanLLPSCluster() = default;
  ~TitanLLPSCluster();

  void init(double eps_angstrom, int min_pts, const std::string &out_pdb_name,
            const std::string &out_csv_name = "");
  void enable_fuzzy(bool v) { fuzzy_enabled_ = v; }
  void configure_dynamics(bool enabled, bool detailed, int max_lag_frames,
                          double lineage_overlap_threshold,
                          int min_lineage_cluster_size,
                          double msd_fit_start_ns,
                          double msd_fit_end_ns);
  // Liquidity v2 knobs; see LlpsArgs for the meaning of each value.
  void configure_liquidity(int rna_segment_beads, bool msd_drift_correct,
                           double mobile_displacement_nm,
                           double mobile_lag_ns);
  void configure_trace_outputs(const info &info_output);
  void configure_rna_compaction(const std::string &mode, bool dynamics_enabled,
                                bool detailed, bool has_rna,
                                const std::string &sticker_beads,
                                float protein_rna_cutoff_nm,
                                float intra_cutoff_nm,
                                int longrange_min);
  void configure_rna_information(const std::string &mode,
                                 bool dynamics_enabled, bool detailed,
                                 bool has_rna, int rg_bins,
                                 int longrange_bins);
  void configure_rna_topology(const std::string &mode, bool dynamics_enabled,
                              bool detailed, bool has_rna,
                              float protein_protein_cutoff_nm,
                              float protein_rna_cutoff_nm, int longrange_min,
                              float bridge_geom_min_nm, int patch_gap_max,
                              bool clamp_require_loop,
                              const info &info_output);
  void configure_energy_correlation(
      const std::string &mode, bool dynamics_enabled, bool has_rna, bool nocal,
      const EnergyDecompositionProvider *energy_provider, bool detailed,
      const info &info_output);
  void set_kernel_mode(const std::string &mode) { kernel_mode_ = mode; }
  void configure_paper(const LLPSPaperConfig &config);

  void process_frame(TitanExecutor &executor, const MOL &pro,
                     const std::vector<mol_info> &pro_info, const MOL &lig,
                     const std::vector<mol_info> &lig_info,
                     double current_time_ps, int frame_idx);

  void export_timeseries(const std::string &out_csv) const;
  void finalize_dynamics(TitanExecutor &executor);
  void export_dynamics(const info &info_output, std::fstream &log,
                       const LLPSContactDynamicsResult *contact_result) const;
  void export_performance_report(const info &info_output,
                                 const TitanExecutor &executor) const;
  void export_paper_outputs(const info &info_output, std::fstream &log) const;

  const LLPSChainDynamicsResult &chain_dynamics() const {
    return chain_dynamics_result_;
  }
  const LLPSClusterLineageResult &cluster_lineage() const {
    return cluster_lineage_result_;
  }
  const LLPSDenseMSDResult &dense_msd() const { return dense_msd_result_; }

#ifdef TITAN_LLPS_TESTING
  void debug_seed_dense_msd(
      int total_frames, int max_lag_frames, double fit_start_ns,
      double fit_end_ns, const std::vector<double> &protein_sum,
      const std::vector<uint64_t> &protein_count,
      const std::vector<double> &protein_time_sum,
      const std::vector<uint64_t> &protein_time_count,
      const std::vector<double> &rna_sum,
      const std::vector<uint64_t> &rna_count,
      const std::vector<double> &rna_time_sum,
      const std::vector<uint64_t> &rna_time_count);
  void debug_finalize_dense_msd(TitanExecutor &executor) {
    finalize_dense_msd(executor);
  }
#endif

private:
  struct ChainFrameSummary {
    int species{0};
    int raw_chain_id{0};
    bool inside_dense{false};
    int cluster_id{-1};
    double wrapped_com_x{0.0};
    double wrapped_com_y{0.0};
    double wrapped_com_z{0.0};
    int n_beads{1};
  };

  struct ResidenceEpisodeRow {
    int species{0};
    int raw_chain_id{0};
    uint64_t episode_id{0};
    double start_ns{0.0};
    double end_ns{0.0};
    double duration_ns{0.0};
    bool censored_start{false};
    bool censored_end{false};
  };

  struct MembershipTraceRow {
    double time_ns{0.0};
    int species{0};
    int raw_chain_id{0};
    uint64_t episode_id{0};
    int inside_dense{0};
    std::string event{};
  };

  struct LineageClusterRow {
    double time_ns{0.0};
    int frame_cluster_id{0};
    int protein_chains{0};
    int rna_chains{0};
    std::vector<int> matched_prev_clusters;
    std::vector<double> matched_jaccard;
    // share of each matched previous cluster's members found in this one
    std::vector<double> matched_prev_share;
    std::string event{};
  };

  struct DropletShapeFrame {
    double time_ps{0.0};
    int frame{0};
    int cluster_id{-1};
    int nodes{0};
    int protein_nodes{0};
    int rna_nodes{0};
    double rg_angstrom{0.0};
    double asphericity_angstrom2{0.0};
    double acylindricity_angstrom2{0.0};
    double relative_shape_anisotropy{0.0};
    double sphericity_proxy{0.0};
  };

  struct ChainRuntimeState {
    int species{0};
    int raw_chain_id{0};
    bool seen_any_frame{false};
    bool prev_inside{false};
    bool has_prev_com{false};
    uint64_t current_episode_id{0};
    double episode_start_ns{0.0};
    bool episode_censored_start{false};
    double first_time_ns{0.0};
    double last_time_ns{0.0};
    double prev_wrapped_x{0.0};
    double prev_wrapped_y{0.0};
    double prev_wrapped_z{0.0};
    double unwrapped_x{0.0};
    double unwrapped_y{0.0};
    double unwrapped_z{0.0};
  };

  struct LineageClusterSnapshot {
    int frame_cluster_id{0};
    int protein_chains{0};
    int rna_chains{0};
    std::vector<int> chain_ids;
  };

  struct RunningStats {
    uint64_t n{0};
    double sum{0.0};
    double sum_sq{0.0};
    void add(double value);
    double mean() const;
    double stddev() const;
  };

  struct RNACompactionSample {
    double time_ns{0.0};
    int frame{0};
    int rna_chain_id{0};
    int cluster_id{-1};
    int partner_state_id{0};
    int compaction_state_id{-1};
    double bound_protein_chains{0.0};
    double protein_rna_bead_contacts{0.0};
    double rg_nm{0.0};
    double end_to_end_nm{0.0};
    double relative_shape_anisotropy{0.0};
    double longrange_contact_fraction{0.0};
    double intrachain_contacts{0.0};
    double longrange_intrachain_contacts{0.0};
    double energy_elec_kjmol{0.0};
    double energy_vdw_kjmol{0.0};
    double energy_total_kjmol{0.0};
    int topology_state{0};
    bool any_bridge{false};
    int num_bridge_proteins{0};
    bool any_clamp{false};
    int num_clamp_pairs{0};
    double bridge_residue_span_max{0.0};
    double bridge_geom_span_max_nm{0.0};
    double clamp_supported_longrange_contacts{0.0};
    double bridge_attraction_kjmol{0.0};
    double nonbridge_attraction_kjmol{0.0};
    double local_pp_elec_kjmol{0.0};
    double local_pp_vdw_kjmol{0.0};
    double local_pp_total_kjmol{0.0};
    double local_pp_attraction_strength_kjmol{0.0};
    double prna_attraction_strength_kjmol{0.0};
    double bridge_attraction_strength_kjmol{0.0};
    double nonbridge_attraction_strength_kjmol{0.0};
    double local_pp_sticker_contacts{0.0};
    double local_pp_to_prna_contact_ratio{0.0};
    double local_pp_to_prna_attraction_ratio{0.0};
    double local_pp_to_bridge_attraction_ratio{0.0};
    bool inside_dense{false};
  };

  struct RNACompactionTraceRow {
    double time_ns{0.0};
    int frame{0};
    int rna_chain_id{0};
    int inside_dense{0};
    int cluster_id{-1};
    int bound_protein_chains{0};
    int protein_rna_bead_contacts{0};
    double rg_nm{0.0};
    double end_to_end_nm{0.0};
    double asphericity_nm2{0.0};
    double normalized_asphericity{0.0};
    double acylindricity_nm2{0.0};
    double relative_shape_anisotropy{0.0};
    double aspect_ratio{0.0};
    int intrachain_contacts{0};
    int longrange_intrachain_contacts{0};
    double longrange_contact_fraction{0.0};
  };

  struct RNACompactionTopologyCache {
    bool ready{false};
    std::size_t pro_info_size{0};
    std::size_t lig_info_size{0};
    std::string sticker_signature;
    std::vector<std::size_t> sticker_atom_indices;
    std::vector<int> sticker_chain_ids;
    std::unordered_map<int, std::vector<int>> sticker_indices_by_chain;
    std::vector<int> rna_chain_ids;
    std::vector<int> rna_atom_chain_ids;
    std::vector<int> rna_atom_residue_ids;
  };

  struct RNAChainCompactionStats {
    RunningStats inside_dense;
    RunningStats bound_protein_chains;
    RunningStats protein_rna_bead_contacts;
    RunningStats rg_nm;
    RunningStats end_to_end_nm;
    RunningStats relative_shape_anisotropy;
    RunningStats longrange_contact_fraction;
    RunningStats intrachain_contacts;
    RunningStats longrange_intrachain_contacts;
  };

  struct RNABindingBinStats {
    RunningStats rg_nm;
    RunningStats end_to_end_nm;
    RunningStats asphericity_nm2;
    RunningStats longrange_contact_fraction;
  };

  struct RNAEnergyBinStats {
    RunningStats energy_elec_kjmol;
    RunningStats energy_vdw_kjmol;
    RunningStats energy_total_kjmol;
    RunningStats attraction_strength_kjmol;
    RunningStats rg_nm;
    RunningStats end_to_end_nm;
    RunningStats relative_shape_anisotropy;
    RunningStats longrange_contact_fraction;
  };

  struct RNAInformationSummaryRow {
    std::string scope;
    uint64_t samples{0};
    uint64_t transitions{0};
    double h_compaction_bits{0.0};
    double h_compaction_norm{0.0};
    double h_partner_bits{0.0};
    double h_partner_norm{0.0};
    double h_partner_given_binding_bits{0.0};
    double mi_binding_compaction_bits{0.0};
    double mi_binding_compaction_norm{0.0};
    double h_compaction_given_binding_bits{0.0};
    double transition_entropy_rate_bits_per_step{0.0};
    double transition_entropy_rate_bits_per_ns{0.0};
  };

  struct RNAInformationPerChainRow {
    std::string scope;
    int rna_chain_id{0};
    uint64_t frames{0};
    uint64_t transitions{0};
    double h_compaction_bits{0.0};
    double h_partner_bits{0.0};
    double h_partner_given_binding_bits{0.0};
    double mi_binding_compaction_bits{0.0};
    double h_compaction_given_binding_bits{0.0};
    double transition_entropy_rate_bits_per_step{0.0};
    double transition_entropy_rate_bits_per_ns{0.0};
  };

  struct RNACompactionStateHistogramRow {
    std::string scope;
    int state_id{-1};
    uint64_t frame_count{0};
    double probability{0.0};
    int rg_bin{0};
    double rg_min_nm{0.0};
    double rg_max_nm{0.0};
    int longrange_bin{0};
    double longrange_min{0.0};
    double longrange_max{0.0};
  };

  struct RNAPartnerStateHistogramRow {
    std::string scope;
    int bound_protein_chains{0};
    int partner_state_id{0};
    std::string partner_set;
    uint64_t frame_count{0};
    double probability{0.0};
    double conditional_probability{0.0};
  };

  struct RNATransitionInformationRow {
    std::string scope;
    int from_state_id{-1};
    int to_state_id{-1};
    uint64_t count{0};
    double probability{0.0};
  };

  struct RNABindingCompactionInformationRow {
    std::string scope;
    int binding_value{0};
    int compaction_state_id{-1};
    uint64_t count{0};
    double joint_probability{0.0};
    double binding_probability{0.0};
    double compaction_probability{0.0};
  };

  struct PerformanceCounters {
    uint64_t frames_processed{0};
    uint64_t node_count_accum{0};
    uint64_t dbscan_neighbor_checks{0};
    uint64_t dbscan_cluster_expansions{0};
    uint64_t rna_chain_evaluations{0};
    double extract_seconds{0.0};
    double dbscan_seconds{0.0};
    double cluster_summary_seconds{0.0};
    double fuzzy_seconds{0.0};
    double dynamics_seconds{0.0};
    double rna_compaction_seconds{0.0};
    double finalize_seconds{0.0};
    std::size_t process_executor_workers{1};
    std::size_t finalize_executor_workers{1};
  };

  double eps_{0.0};
  double eps_sq_{0.0};
  int min_pts_{2};
  int frame_count_{0};
  bool fuzzy_enabled_{false};
  int write_pdb_stride_{1};
  bool dynamics_enabled_{false};
  bool dynamics_detailed_{false};
  int dynamics_max_lag_frames_{0};
  int dynamics_ring_cap_{0};
  int dynamics_ring_count_{0};
  int dynamics_total_frames_{0};
  double lineage_overlap_threshold_{0.50}; // containment share (STR-09)
  int min_lineage_cluster_size_{1};
  double msd_fit_start_ns_{-1.0};
  double msd_fit_end_ns_{-1.0};
  double first_frame_time_ns_{0.0};
  double last_frame_time_ns_{0.0};
  bool have_time_window_{false};
  uint64_t next_episode_id_{1};
  bool rna_compaction_enabled_{false};
  bool rna_compaction_detailed_{false};
  double rna_compaction_prna_cutoff_angstrom_{0.0};
  double rna_compaction_intra_cutoff_angstrom_{0.0};
  int rna_compaction_longrange_min_{10};
  std::unordered_set<std::string> rna_compaction_sticker_names_;
  bool rna_information_enabled_{false};
  bool rna_information_detailed_{false};
  int rna_information_rg_bins_{16};
  int rna_information_longrange_bins_{16};
  std::string rna_information_primary_scope_{"none"};
  bool rna_topology_enabled_{false};
  bool rna_topology_detailed_{false};
  double rna_topology_bridge_geom_min_angstrom_{0.0};
  int rna_topology_patch_gap_max_{1};
  bool rna_topology_clamp_require_loop_{true};
  double rna_topology_pp_cutoff_angstrom_{0.0};
  bool energy_correlation_enabled_{false};
  bool energy_correlation_detailed_{false};
  std::string kernel_mode_{"auto"};
  const EnergyDecompositionProvider *energy_provider_{nullptr};

  std::ofstream ts_out_;
  bool streaming_csv_{false};
  std::vector<ClusterMetrics> timeseries_;
  std::ofstream vmd_out_;
  std::ofstream rna_binding_energy_trace_out_;
  std::ofstream prna_pair_energy_trace_out_;
  std::ofstream rna_topology_trace_out_;
  std::ofstream rna_bridge_trace_out_;
  std::ofstream rna_clamp_trace_out_;
  std::ofstream membership_trace_out_;
  std::ofstream rna_compaction_trace_out_;
  std::ofstream droplet_shape_out_;
  bool membership_trace_streaming_{false};
  bool rna_compaction_trace_streaming_{false};
  bool droplet_shape_streaming_{false};

  std::vector<int> head_;
  std::vector<int> nxt_;
  std::vector<int> cell_id_;
  std::vector<int> neighbor_count_;
  std::vector<int> bfs_q_;

  std::unordered_map<uint64_t, int> chain_key_to_index_;
  std::vector<ChainRuntimeState> chain_states_;
  std::vector<int> chain_index_species_;
  std::vector<int> chain_index_raw_id_;
  std::vector<uint64_t> membership_inside_counts_;
  std::vector<uint64_t> membership_frame_counts_;
  std::vector<uint64_t> membership_joint_inside_;
  std::vector<uint64_t> membership_pair_counts_;
  std::vector<double> protein_autocorr_lag_time_sum_;
  std::vector<double> rna_autocorr_lag_time_sum_;
  std::vector<uint64_t> protein_autocorr_lag_time_count_;
  std::vector<uint64_t> rna_autocorr_lag_time_count_;
  std::vector<double> frame_time_ring_;
  std::vector<uint8_t> frame_present_ring_;
  std::vector<uint8_t> frame_inside_ring_;
  std::vector<uint64_t> frame_episode_ring_;
  std::vector<double> frame_unwrapped_x_ring_;
  std::vector<double> frame_unwrapped_y_ring_;
  std::vector<double> frame_unwrapped_z_ring_;
  std::size_t dynamics_ring_chain_capacity_{0};
  std::size_t membership_lag_chain_capacity_{0};
  std::vector<uint8_t> current_present_buf_;
  std::vector<uint8_t> current_inside_buf_;
  std::vector<uint64_t> current_episode_buf_;
  std::vector<double> current_unwrapped_x_buf_;
  std::vector<double> current_unwrapped_y_buf_;
  std::vector<double> current_unwrapped_z_buf_;
  std::vector<double> msd_sum_protein_;
  std::vector<double> msd_sum_rna_;
  std::vector<uint64_t> msd_count_protein_;
  std::vector<uint64_t> msd_count_rna_;
  std::vector<double> msd_time_sum_protein_;
  std::vector<double> msd_time_sum_rna_;
  std::vector<uint64_t> msd_time_count_protein_;
  std::vector<uint64_t> msd_time_count_rna_;
  // --- Liquidity v2 ---
  int rna_segment_beads_{0};
  bool msd_drift_correct_{true};
  double mobile_displacement_nm_{2.0};
  double mobile_lag_ns_{-1.0};
  // Cumulative translation of the dense phase, integrated from per-frame
  // minimum-image displacements of the units resident at both frames.
  double dense_drift_x_{0.0};
  double dense_drift_y_{0.0};
  double dense_drift_z_{0.0};
  std::vector<double> frame_dx_buf_;
  std::vector<double> frame_dy_buf_;
  std::vector<double> frame_dz_buf_;
  std::vector<uint8_t> frame_drift_use_buf_;
  std::vector<int> frame_weight_buf_;
  // Pooled membership correlation accumulators (permanent residents kept).
  std::vector<uint64_t> pooled_joint_protein_;
  std::vector<uint64_t> pooled_pairs_protein_;
  std::vector<uint64_t> pooled_joint_rna_;
  std::vector<uint64_t> pooled_pairs_rna_;
  uint64_t pooled_inside_protein_{0};
  uint64_t pooled_frames_protein_{0};
  uint64_t pooled_inside_rna_{0};
  uint64_t pooled_frames_rna_{0};
  // Displacement-threshold (FRAP analogue) accumulators, per lag.
  std::vector<uint64_t> disp_gt_protein_;
  std::vector<uint64_t> disp_all_protein_;
  std::vector<uint64_t> disp_gt_rna_;
  std::vector<uint64_t> disp_all_rna_;
  // --- Liquidity v3 ---
  // In-plane (lateral) squared displacement for slab geometries: motion
  // along the slab normal is confined by the slab thickness, so D is also
  // reported from the displacement without its normal component. STR-01:
  // the slab axis is the lattice direction with the largest plane spacing
  // and the normal is perpendicular to the two other lattice vectors
  // (titan_llps_geometry::slab_normal), set from each frame's cell.
  std::vector<double> msd_lat_sum_protein_;
  std::vector<double> msd_lat_sum_rna_;
  int slab_normal_axis_{2};
  std::array<double, 3> slab_normal_{0.0, 0.0, 1.0};
  // Whole RNA molecule COM track (unwrapped, drift removed, nm), one per RNA
  // chain, so that a molecular MSD exists beside the segment MSD.
  struct RnaMoleculeTrack {
    bool has_prev{false};
    double prev_x{0.0}, prev_y{0.0}, prev_z{0.0};
    double ux{0.0}, uy{0.0}, uz{0.0};
    std::vector<double> t_ns, x, y, z;
    std::vector<uint8_t> inside;
  };
  std::map<int, RnaMoleculeTrack> rna_molecule_tracks_;
  uint64_t protein_entry_events_{0};
  uint64_t protein_exit_events_{0};
  uint64_t rna_entry_events_{0};
  uint64_t rna_exit_events_{0};
  std::vector<ResidenceEpisodeRow> residence_rows_;
  std::vector<MembershipTraceRow> membership_trace_rows_;
  std::vector<LineageClusterRow> lineage_rows_;
  std::vector<DropletShapeFrame> droplet_shape_rows_;
  std::vector<LineageClusterSnapshot> prev_lineage_clusters_;

  LLPSChainDynamicsResult chain_dynamics_result_;
  LLPSClusterLineageResult cluster_lineage_result_;
  LLPSDenseMSDResult dense_msd_result_;
  LLPSPaperAccumulator paper_accumulator_;
  std::vector<RNACompactionSample> rna_compaction_samples_;
  std::unordered_map<int, RNAChainCompactionStats> rna_chain_compaction_stats_;
  std::unordered_map<int, RNABindingBinStats> rna_binding_bin_stats_;
  std::unordered_map<int, RNAEnergyBinStats> rna_energy_bin_stats_;
  std::vector<RNACompactionTraceRow> rna_compaction_trace_rows_;
  RNACompactionTopologyCache rna_compaction_topology_cache_;
  std::unordered_map<std::string, int> rna_partner_state_to_id_;
  std::vector<std::string> rna_partner_state_labels_;
  std::vector<double> rna_information_rg_edges_;
  std::vector<double> rna_information_longrange_edges_;
  std::vector<RNAInformationSummaryRow> rna_information_summary_rows_;
  std::vector<RNAInformationPerChainRow> rna_information_per_chain_rows_;
  std::vector<RNACompactionStateHistogramRow>
      rna_compaction_state_histogram_rows_;
  std::vector<RNAPartnerStateHistogramRow> rna_partner_state_histogram_rows_;
  std::vector<RNATransitionInformationRow> rna_transition_information_rows_;
  std::vector<RNABindingCompactionInformationRow>
      rna_binding_compaction_information_rows_;
  PerformanceCounters perf_;

  void resize_dynamics_ring_storage(std::size_t chain_count);
  void resize_membership_lag_storage(std::size_t chain_count);
  std::vector<LLPSNode> extract_nodes(const MOL &pro,
                                      const std::vector<mol_info> &pro_info,
                                      const MOL &lig,
                                      const std::vector<mol_info> &lig_info) const;
  void run_dbscan(TitanExecutor &executor, std::vector<LLPSNode> &nodes,
                  const box &Box);
  void fuzzy_refine(TitanExecutor &executor, const std::vector<LLPSNode> &nodes,
                    const box &Box, int giant_cluster_id,
                    std::vector<bool> &is_dense, std::vector<double> &bfactor,
                    std::vector<double> &occupancy);
  void write_frame_to_pdb(const std::vector<LLPSNode> &nodes, int frame_idx,
                          const std::vector<double> &bfactor,
                          const std::vector<double> &occupancy);

  void ensure_chain_capacity(std::size_t chain_index);
  void update_dynamics(const box &Box, double time_ns, int giant_cluster_id,
                       const std::vector<std::vector<int>> &cluster_pro,
                       const std::vector<std::vector<int>> &cluster_rna,
                       const std::vector<ChainFrameSummary> &frame_chains);
  void update_lineage(
      double time_ns, const std::vector<std::vector<int>> &cluster_pro,
      const std::vector<std::vector<int>> &cluster_rna);
  void finalize_residence_summary();
  void finalize_membership_autocorr(TitanExecutor &executor);
  void finalize_dense_msd(TitanExecutor &executor);
  void finalize_rna_information(TitanExecutor &executor);
  void update_rna_compaction(
      const MOL &pro, const std::vector<mol_info> &pro_info, const MOL &lig,
      const std::vector<mol_info> &lig_info, const box &Box, double time_ns,
      int frame_idx, const std::vector<ChainFrameSummary> &frame_chains);
  void refresh_rna_compaction_topology_cache(
      const std::vector<mol_info> &pro_info,
      const std::vector<mol_info> &lig_info);
  void export_rna_compaction_outputs(const info &info_output) const;
  void export_rna_information_outputs(const info &info_output) const;
  void export_rna_topology_outputs(const info &info_output) const;
  int register_rna_partner_state(const std::vector<int> &protein_chain_ids);
  static std::string encode_hybrid36(int n);
  static std::string species_label(int species);
  static uint64_t pack_chain_key(int species, int raw_chain_id);
};

#endif // TITAN_LLPS_CLUSTER_H
