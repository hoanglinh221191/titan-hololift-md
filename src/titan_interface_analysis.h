#ifndef TITAN_INTERFACE_ANALYSIS_H
#define TITAN_INTERFACE_ANALYSIS_H

#include "Multi_MOL.h"
#include "arguments.h"
#include "titan_contact_geometry.h"
#include "titan_contact_primitives.h"
#include "titan_segment_config.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

class TitanExecutor;
class info;

class TitanInterfaceAnalyzer {
public:
  void configure(const InterfaceAnalysisArgs &args,
                 const SegmentArgs &segments);
  void init(const Multi_MOL &mol1, const Multi_MOL *mol2,
            const arguments &args, std::fstream &log);
  void process_frame(const Multi_MOL &mol1, const Multi_MOL *mol2,
                     std::size_t snap_idx, std::size_t global_frame_idx,
                     TitanExecutor &executor);
  void finalize_trajectory(const std::string &base_dir, const info &info_output,
                           const arguments &args,
                           const std::vector<std::string> &input_line,
                           int input_name, std::fstream &log);
  void finalize_all(const std::string &base_dir, const info &info_output,
                    const arguments &args, std::fstream &log);

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }
  [[nodiscard]] bool initialized() const noexcept { return initialized_; }
  [[nodiscard]] bool has_partner() const noexcept { return partner_index_ >= 0; }

private:
  struct ResidueIdentity {
    std::uint8_t molecule{1};
    std::size_t chain_number{0};
    std::size_t residue_number{0};

    bool operator==(const ResidueIdentity &other) const noexcept {
      return molecule == other.molecule &&
             chain_number == other.chain_number &&
             residue_number == other.residue_number;
    }
  };

  struct ResidueIdentityHash {
    std::size_t operator()(const ResidueIdentity &key) const noexcept;
  };

  struct AtomRef {
    std::uint8_t molecule{1};
    std::size_t atom_index{0};
  };

  struct EntityResidue {
    titan_segments::ResidueDescriptor descriptor;
    std::string segment{"all"};
  };

  struct EntityRuntime {
    titan_segments::Entity definition;
    std::vector<EntityResidue> residues;
    std::vector<AtomRef> atoms;
  };

  struct ResidueLocation {
    std::size_t entity_index{0};
    std::size_t residue_index{0};
  };

  struct RuntimePoint {
    double x{0.0};
    double y{0.0};
    double z{0.0};
    std::size_t residue_index{0};
  };

  struct SegmentChannel {
    std::string interface_name;
    std::string entity_a;
    std::string segment_a;
    std::string entity_b;
    std::string segment_b;
  };

  struct FrameRecord {
    std::size_t frame_index{0};
    double time_ps{0.0};
    std::string state;
    std::string proximal_entity;
    std::uint32_t core_contacts{0};
    std::uint32_t core_a_partner_contacts{0};
    std::uint32_t core_b_partner_contacts{0};
    std::vector<std::uint32_t> segment_contacts;
  };

  struct SasaRecord {
    std::size_t frame_index{0};
    double time_ps{0.0};
    double sasa_core_a{0.0};
    double sasa_core_b{0.0};
    double sasa_partner{0.0};
    double sasa_core_pair{0.0};
    double sasa_complex{0.0};
    double dimer_buried_total{0.0};
    double partner_buried_total{0.0};
  };

  struct MetricSummary {
    std::size_t samples{0};
    std::size_t blocks{0};
    double mean{0.0};
    double sem{0.0};
    double ci95_low{0.0};
    double ci95_high{0.0};
    double bootstrap_low{0.0};
    double bootstrap_high{0.0};
    std::string status;
  };

  void build_runtime(const Multi_MOL &mol1, const Multi_MOL *mol2,
                     std::fstream &log);
  void build_entities_for_molecule(const Multi_MOL &mol,
                                   std::uint8_t molecule);
  void build_atom_membership(const Multi_MOL &mol, std::uint8_t molecule);
  void build_segment_channels();
  void populate_runtime_points(
      const Multi_MOL &mol, std::size_t snap_idx,
      const titan_contact_geometry::ContactGeometryCache &geometry,
      std::vector<titan_contact_geometry::ContactPoint> &point_scratch);
  std::vector<std::pair<std::size_t, std::size_t>>
  contact_pairs(std::size_t entity_a, std::size_t entity_b,
                const box &frame_box) const;
  void add_segment_counts(
      const std::string &interface_name, std::size_t entity_a,
      std::size_t entity_b,
      const std::vector<std::pair<std::size_t, std::size_t>> &pairs,
      std::vector<std::uint32_t> &counts) const;
  void sample_sasa(const Multi_MOL &mol1, const Multi_MOL *mol2,
                   std::size_t snap_idx, std::size_t global_frame_idx,
                   TitanExecutor &executor);

  [[nodiscard]] MetricSummary
  summarize_metric(const std::vector<double> &values,
                   const UncertaintyArgs &uncertainty,
                   unsigned int seed_offset = 0) const;
  void export_records(const std::string &timeseries_filename,
                      const std::string &population_filename,
                      const std::string &segment_filename,
                      const std::string &segment_timeseries_filename,
                      const std::string &metrics_filename,
                      const std::string &sasa_filename,
                      const std::vector<FrameRecord> &records,
                      const std::vector<SasaRecord> &sasa_records,
                      const arguments &args, std::fstream &log) const;
  void reset_trajectory();

  bool requested_{false};
  bool enabled_{false};
  bool initialized_{false};
  bool sasa_enabled_{false};
  bool warned_invalid_box_{false};
  std::size_t trajectory_count_{0};
  std::size_t valid_frame_counter_{0};

  std::string config_file_;
  titan_segments::Config config_;
  titan_contact::ContactMode contact_mode_{
      titan_contact::ContactMode::SidechainCOM};
  double cutoff_a_{6.5};
  int state_min_contacts_{1};
  int sasa_stride_{10};
  int sasa_points_{96};

  std::vector<EntityRuntime> entities_;
  int core_a_index_{-1};
  int core_b_index_{-1};
  int partner_index_{-1};

  titan_contact_geometry::ContactGeometryCache geometry_mol1_;
  titan_contact_geometry::ContactGeometryCache geometry_mol2_;
  bool geometry_mol1_active_{false};
  bool geometry_mol2_active_{false};
  std::vector<titan_contact_geometry::ContactPoint> point_scratch_mol1_;
  std::vector<titan_contact_geometry::ContactPoint> point_scratch_mol2_;
  std::vector<std::vector<RuntimePoint>> runtime_points_;
  std::unordered_map<ResidueIdentity, ResidueLocation, ResidueIdentityHash>
      residue_locations_;
  std::vector<SegmentChannel> segment_channels_;

  std::vector<FrameRecord> current_records_;
  std::vector<FrameRecord> pooled_records_;
  std::vector<SasaRecord> current_sasa_records_;
  std::vector<SasaRecord> pooled_sasa_records_;
};

#endif // TITAN_INTERFACE_ANALYSIS_H
