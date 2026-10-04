#ifndef TITAN_HBOND_KINETICS_H
#define TITAN_HBOND_KINETICS_H

#include "Multi_MOL.h"
#include "arguments.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

class TitanExecutor;
class info;

namespace titan_hbond_kinetics {

struct KeyAggregate {
  std::uint64_t total_active_frames{0};
  std::uint64_t event_count{0};
  std::uint64_t total_lifetime_frames{0};
  std::uint64_t max_lifetime_frames{0};
};

class StreamingKinetics {
public:
  explicit StreamingKinetics(std::size_t maxlag = 500);

  void reset(std::size_t maxlag);
  void process_active_ids(std::vector<int> active_ids);
  void finalize();
  void ensure_id(std::size_t id);

  [[nodiscard]] std::size_t maxlag() const noexcept { return maxlag_; }
  [[nodiscard]] std::uint64_t frame_count() const noexcept {
    return frame_count_;
  }
  [[nodiscard]] const std::vector<std::uint64_t> &origin_counts() const noexcept {
    return origin_counts_;
  }
  [[nodiscard]] const std::vector<std::uint64_t> &
  intermittent_hits() const noexcept {
    return intermittent_hits_;
  }
  [[nodiscard]] const std::vector<std::uint64_t> &continuous_hits() const noexcept {
    return continuous_hits_;
  }
  [[nodiscard]] const std::vector<KeyAggregate> &key_stats() const noexcept {
    return key_stats_;
  }
  [[nodiscard]] const std::vector<std::uint64_t> &
  lifetime_histogram() const noexcept {
    return lifetime_histogram_;
  }

private:
  void finalize_run(int id);

  std::size_t maxlag_{500};
  std::uint64_t frame_count_{0};
  std::vector<std::uint64_t> origin_counts_;
  std::vector<std::uint64_t> intermittent_hits_;
  std::vector<std::uint64_t> continuous_hits_;
  std::vector<KeyAggregate> key_stats_;
  std::vector<std::uint64_t> current_run_lengths_;
  std::vector<int> previous_active_ids_;
  std::deque<std::vector<int>> history_;
  std::vector<std::uint64_t> lifetime_histogram_;
};

} // namespace titan_hbond_kinetics

class TitanHBondKineticsAnalyzer {
public:
  TitanHBondKineticsAnalyzer() = default;

  void configure(const HBondKineticsArgs &args);
  void init(const Multi_MOL &mol1, const Multi_MOL *mol2,
            bool water_stream_available, const arguments &args,
            std::fstream &log);
  void process_frame(const MOL &mol1_frame, const MOL *mol2_frame,
                     const MOL *water_frame, TitanExecutor &executor);
  void finalize_trajectory(const std::string &base_dir, const info &info_output,
                           const arguments &args,
                           const std::vector<std::string> &input_line,
                           int input_name, std::fstream &log);

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }
  [[nodiscard]] bool initialized() const noexcept { return initialized_; }

private:
  enum class ScopeKind : std::size_t { Inter = 0, IntraMol1 = 1, IntraMol2 = 2 };
  enum class InteractionKind : std::size_t { HBond = 0, Salt = 1, Water = 2 };

  struct AtomLabel {
    std::size_t atom_index{0};
    std::size_t chain_id{0};
    std::size_t resnr{0};
    std::string chain;
    std::string res_name;
    std::string atom_name;
  };

  struct DonorSite {
    std::size_t donor_index{0};
    std::size_t hydrogen_index{0};
  };

  struct MoleculeCache {
    bool available{false};
    std::string label;
    std::vector<AtomLabel> atoms;
    std::vector<DonorSite> donors;
    std::vector<std::size_t> acceptors;
    std::vector<std::size_t> acidic_atoms;
    std::vector<std::size_t> basic_atoms;
    std::vector<std::size_t> heavy_atoms;
  };

  struct HBondEvent {
    int donor_mol{0};
    int donor_idx{0};
    int hydrogen_idx{0};
    int acceptor_mol{0};
    int acceptor_idx{0};

    [[nodiscard]] bool operator<(const HBondEvent &rhs) const noexcept;
    [[nodiscard]] bool operator==(const HBondEvent &rhs) const noexcept;
  };

  struct SaltEvent {
    int acidic_mol{0};
    int acidic_idx{0};
    int basic_mol{0};
    int basic_idx{0};

    [[nodiscard]] bool operator<(const SaltEvent &rhs) const noexcept;
    [[nodiscard]] bool operator==(const SaltEvent &rhs) const noexcept;
  };

  struct WaterEvent {
    int water_index{0};
    int atom_a_mol{0};
    int atom_a_idx{0};
    int atom_b_mol{0};
    int atom_b_idx{0};

    [[nodiscard]] bool operator<(const WaterEvent &rhs) const noexcept;
    [[nodiscard]] bool operator==(const WaterEvent &rhs) const noexcept;
  };

  struct HBondDescriptor {
    std::string scope;
    std::string donor_mol;
    AtomLabel donor;
    std::string hydrogen_mol;
    AtomLabel hydrogen;
    std::string acceptor_mol;
    AtomLabel acceptor;
  };

  struct SaltDescriptor {
    std::string scope;
    std::string acidic_mol;
    AtomLabel acidic;
    std::string basic_mol;
    AtomLabel basic;
  };

  struct WaterDescriptor {
    std::string scope;
    int water_index{0};
    std::string atom_a_mol;
    AtomLabel atom_a;
    std::string atom_b_mol;
    AtomLabel atom_b;
  };

  template <typename Descriptor> struct SeriesState {
    bool requested{false};
    bool enabled{false};
    std::string scope_label;
    std::string status{"inactive"};
    titan_hbond_kinetics::StreamingKinetics kinetics;
    std::unordered_map<std::string, int> id_by_key;
    std::vector<Descriptor> descriptors;

    void reset(std::size_t maxlag) {
      kinetics.reset(maxlag);
      id_by_key.clear();
      descriptors.clear();
    }
  };

  void reset_trajectory_state();
  void configure_requested_kinds(const HBondKineticsArgs &args);
  void prepare_molecule_cache(const Multi_MOL &mol, const std::string &label,
                              MoleculeCache &cache);
  void configure_scope_states(const arguments &args, std::fstream &log);
  void update_frame_timing(double time_ps);

  void detect_hbonds_between(const MoleculeCache &donor_cache,
                             const MOL &donor_frame, int donor_mol,
                             const MoleculeCache &acceptor_cache,
                             const MOL &acceptor_frame, int acceptor_mol,
                             bool exclude_same_residue, TitanExecutor &executor,
                             std::vector<HBondEvent> &out_events) const;
  void detect_salts_between(const MoleculeCache &acid_cache,
                            const MOL &acid_frame, int acid_mol,
                            const MoleculeCache &base_cache,
                            const MOL &base_frame, int base_mol,
                            bool exclude_same_residue, TitanExecutor &executor,
                            std::vector<SaltEvent> &out_events) const;
  void detect_water_inter(const MOL &water_frame, const MoleculeCache &mol1_cache,
                          const MOL &mol1_frame, const MoleculeCache &mol2_cache,
                          const MOL &mol2_frame, TitanExecutor &executor,
                          std::vector<WaterEvent> &out_events) const;
  void detect_water_intra(const MOL &water_frame, const MoleculeCache &cache,
                          const MOL &frame, int mol_id,
                          TitanExecutor &executor,
                          std::vector<WaterEvent> &out_events) const;
  void collect_nearby_heavy_atoms(const MOL &water_frame, std::size_t oxygen_idx,
                                  const MoleculeCache &cache, const MOL &frame,
                                  double cutoff_sq,
                                  std::vector<std::size_t> &hits) const;

  int intern_hbond_event(ScopeKind scope, const HBondEvent &event);
  int intern_salt_event(ScopeKind scope, const SaltEvent &event);
  int intern_water_event(ScopeKind scope, const WaterEvent &event);
  void process_hbond_scope(ScopeKind scope, std::vector<HBondEvent> events);
  void process_salt_scope(ScopeKind scope, std::vector<SaltEvent> events);
  void process_water_scope(ScopeKind scope, std::vector<WaterEvent> events);
  void process_empty_scope_frames();

  void export_summary(const std::string &filename, std::fstream &log) const;
  void export_hbond_pairs(const std::string &filename, std::fstream &log) const;
  void export_salt_pairs(const std::string &filename, std::fstream &log) const;
  void export_water_pairs(const std::string &filename, std::fstream &log) const;
  void export_ct_file(const std::string &filename, InteractionKind kind,
                      std::fstream &log) const;
  void export_lifetime_file(const std::string &filename, InteractionKind kind,
                            std::fstream &log) const;

  [[nodiscard]] const MoleculeCache &molecule_cache(int mol_id) const;
  [[nodiscard]] const char *scope_label(ScopeKind scope) const;
  [[nodiscard]] std::size_t scope_index(ScopeKind scope) const;
  [[nodiscard]] double frame_dt_ps() const;

  bool requested_{false};
  bool enabled_{false};
  bool initialized_{false};
  bool request_hbond_{true};
  bool request_salt_{true};
  bool request_water_{true};
  bool allow_inter_{true};
  bool allow_intra_mol1_{true};
  bool allow_intra_mol2_{true};
  bool water_stream_available_{false};
  bool warned_missing_water_{false};
  bool warned_missing_mol2_{false};

  std::size_t maxlag_{500};
  double hbond_cutoff_ang_{3.5};
  double hbond_cutoff_sq_{12.25};
  double angle_cutoff_deg_{135.0};
  double angle_cos_cutoff_{-0.7071067811865476};
  double salt_cutoff_ang_{4.0};
  double salt_cutoff_sq_{16.0};
  double water_cutoff_ang_{3.5};
  double water_cutoff_sq_{12.25};
  double fallback_dt_ps_{0.0};
  double previous_time_ps_{0.0};
  bool has_previous_time_{false};
  double observed_dt_sum_ps_{0.0};
  std::size_t observed_dt_count_{0};

  std::uint64_t frames_seen_{0};
  std::uint64_t water_missing_frames_{0};

  MoleculeCache mol1_cache_;
  MoleculeCache mol2_cache_;

  std::vector<std::size_t> water_hit_buffer_a_;
  std::vector<std::size_t> water_hit_buffer_b_;

  std::vector<SeriesState<HBondDescriptor>> hbond_series_;
  std::vector<SeriesState<SaltDescriptor>> salt_series_;
  std::vector<SeriesState<WaterDescriptor>> water_series_;
};

#endif // TITAN_HBOND_KINETICS_H
