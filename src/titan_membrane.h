#ifndef TITAN_MEMBRANE_H
#define TITAN_MEMBRANE_H

#include "Multi_MOL.h"
#include "aligned_allocator.h"
#include "arguments.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

class TitanExecutor;
class info;

class TitanMembraneAnalyzer {
public:
  struct FrameSummary {
    std::size_t frame{0};
    double time_ps{0.0};
    double mean_apl_a2{0.0};
    double mean_thickness_a{0.0};
    double interdigitation{0.0};
    double penetration_depth_a{0.0};
    std::size_t upper_count{0};
    std::size_t lower_count{0};
  };

  TitanMembraneAnalyzer() = default;

  void configure(const MembraneArgs &args);
  void init(const Multi_MOL &lipids, std::fstream &log);
  void process_frame(const Multi_MOL &lipids, std::size_t snap_idx,
                     TitanExecutor &executor);
  void process_frame(const Multi_MOL &lipids, const Multi_MOL *protein,
                     std::size_t snap_idx, TitanExecutor &executor);
  void finalize_trajectory(const std::string &base_dir, const info &info_output,
                           const arguments &args,
                           const std::vector<std::string> &input_line,
                           int input_name, std::fstream &log);

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }
  [[nodiscard]] bool initialized() const noexcept { return initialized_; }
  [[nodiscard]] const std::vector<FrameSummary> &frames() const noexcept {
    return frame_summaries_;
  }
  [[nodiscard]] std::size_t lipid_count() const noexcept {
    return lipid_plans_.size();
  }

  [[nodiscard]] static double order_parameter(double dx, double dy, double dz,
                                              double nx, double ny,
                                              double nz) noexcept;

private:
  struct LipidPlan {
    std::string id;
    std::string chain;
    std::string resname;
    std::size_t resnr{0};
    std::size_t head_atom{0};
    std::vector<std::vector<std::size_t>> tails;
  };

  struct ThicknessCell {
    double sum{0.0};
    double sum_sq{0.0};
    std::size_t count{0};
  };

  struct OrderAccumulator {
    double sum{0.0};
    double sum_sq{0.0};
    std::size_t count{0};
  };

  struct FlipState {
    int stable{0};
    int pending{0};
    int pending_count{0};
  };

  struct FlipEvent {
    std::size_t frame{0};
    double time_ps{0.0};
    std::string lipid_id;
    std::string from_leaflet;
    std::string to_leaflet;
  };

  struct AplRow {
    std::size_t frame{0};
    double time_ps{0.0};
    std::string lipid_id;
    std::string leaflet;
    double apl_a2{0.0};
  };

  struct LeafletRow {
    std::size_t frame{0};
    double time_ps{0.0};
    std::string lipid_id;
    std::string leaflet;
    double signed_distance_a{0.0};
  };

  struct ProteinResiduePlan {
    std::string label;
    std::string chain;
    std::string resname;
    std::size_t resnr{0};
    std::vector<std::size_t> atoms;
    std::vector<std::size_t> sidechain_atoms;
  };

  struct ContactRow {
    std::size_t lipid_index{0};
    std::size_t frame{0};
    double time_ps{0.0};
    std::string protein_residue;
    std::string protein_chain;
    std::size_t protein_resnr{0};
    std::string protein_resname;
    std::string lipid_id;
    std::string lipid_resname;
    std::string leaflet;
    std::string bead_name;
    std::string bead_class;
    double distance_a{0.0};
  };

  struct CouplingCounter {
    double sum{0.0};
    double sum_sq{0.0};
    std::size_t count{0};
  };

  struct ProximityAccumulator {
    double near_sum{0.0};
    double near_sum_sq{0.0};
    double far_sum{0.0};
    double far_sum_sq{0.0};
    std::size_t near_count{0};
    std::size_t far_count{0};
  };

  struct EnrichmentAccumulator {
    std::size_t near_count{0};
    std::size_t bulk_count{0};
  };

  struct CurvatureCell {
    double sum{0.0};
    double sum_sq{0.0};
    std::size_t count{0};
  };

  void reset_trajectory_state();
  void build_topology_cache(const Multi_MOL &lipids, std::fstream &log);
  [[nodiscard]] std::array<double, 3>
  resolve_normal(const MOL &frame) const;
  void build_tangent_basis(const std::array<double, 3> &normal,
                           std::array<double, 3> &u,
                           std::array<double, 3> &v) const noexcept;
  void classify_leaflets(const MOL &frame, const box &active_box,
                         const std::array<double, 3> &normal,
                         std::vector<int> &leaflets,
                         AlignedDoubleVector &signed_distance) const;
  void compute_apl(const MOL &frame, const box &active_box,
                   const std::array<double, 3> &normal,
                   const std::vector<int> &leaflets, TitanExecutor &executor,
                   std::vector<double> &apl, double &surface_area) const;
  void update_order_parameters(const MOL &frame, const box &active_box,
                               const std::array<double, 3> &normal,
                               const std::vector<int> *leaflets = nullptr,
                               std::span<const std::uint8_t> near_lipids = {});
  void update_flipflop(const std::vector<int> &leaflets, std::size_t frame,
                       double time_ps);
  void compute_interdigitation(const MOL &frame,
                               const std::array<double, 3> &normal,
                               const std::vector<int> &leaflets,
                               double &overlap, double &penetration) const;
  void update_thickness_map(const MOL &frame, const box &active_box,
                            const std::array<double, 3> &normal,
                            const std::vector<int> &leaflets,
                            double &mean_thickness,
                            std::vector<double> *per_lipid_thickness =
                                nullptr);
  void build_protein_cache(const Multi_MOL &protein);
  [[nodiscard]] std::array<double, 3>
  residue_com(const MOL &frame, const ProteinResiduePlan &plan,
              const Multi_MOL &protein) const;
  void update_protein_lipid_contacts(const Multi_MOL *protein,
                                     const Multi_MOL &lipids,
                                     const MOL &lipid_frame,
                                     const box &active_box,
                                     const std::array<double, 3> &normal,
                                     const std::vector<int> &leaflets,
                                     std::size_t snap_idx, double time_ps,
                                     TitanExecutor &executor,
                                     std::span<std::uint8_t> near_lipids);
  void update_curvature_grid(const std::array<double, 3> &normal,
                             const std::vector<int> &leaflets,
                             const box &active_box,
                             std::vector<double> *per_lipid_curvature =
                                 nullptr);
  void update_proximity_coupling(
      const std::vector<double> &apl,
      const std::vector<double> &per_lipid_thickness,
      const std::vector<double> &per_lipid_curvature,
      const std::vector<int> &leaflets,
      std::span<const std::uint8_t> near_lipids);
  void record_proximity_value(const std::string &observable,
                              const std::string &group,
                              const std::string &leaflet,
                              const std::string &resname, double value,
                              bool near);
  void record_proximity_direct(const std::string &observable,
                               const std::string &group,
                               const std::string &leaflet,
                               const std::string &resname, double near_value,
                               double far_value);
  [[nodiscard]] std::pair<int, int> resolved_grid(const MOL *frame) const;
  [[nodiscard]] std::string leaflet_name(int leaflet) const;
  void load_user_leaflets();

  void export_leaflets(const std::string &filename) const;
  void export_apl_timeseries(const std::string &filename) const;
  void export_apl_per_lipid(const std::string &filename) const;
  void export_thickness_timeseries(const std::string &filename) const;
  void export_thickness_map(const std::string &filename) const;
  void export_order_parameters(const std::string &filename) const;
  void export_interdigitation(const std::string &filename) const;
  void export_flipflop_events(const std::string &filename) const;
  void export_protein_lipid_contacts(const std::string &filename) const;
  void export_lipid_enrichment(const std::string &filename) const;
  void export_curvature_map(const std::string &filename) const;
  void export_proximity_coupling(const std::string &filename) const;
  void export_coupling_summary(const std::string &filename) const;
  void export_summary(const std::string &filename) const;
  void accumulate_trajectory();
  void export_average(const std::string &base_dir,
                      const info &info_output) const;

  MembraneArgs args_{};
  bool requested_{false};
  bool enabled_{false};
  bool initialized_{false};
  std::vector<LipidPlan> lipid_plans_;
  std::unordered_map<std::string, int> user_leaflets_;

  mutable std::array<double, 3> previous_normal_{0.0, 0.0, 1.0};
  mutable bool have_previous_normal_{false};
  AlignedDoubleVector head_x_;
  AlignedDoubleVector head_y_;
  AlignedDoubleVector head_z_;

  std::vector<FrameSummary> frame_summaries_;
  std::vector<AplRow> apl_rows_;
  std::vector<LeafletRow> leaflet_rows_;
  std::map<std::string, OrderAccumulator> order_accumulators_;
  std::vector<FlipState> flip_states_;
  std::vector<FlipEvent> flip_events_;
  std::vector<ThicknessCell> thickness_cells_;
  int grid_x_{0};
  int grid_y_{0};
  std::size_t skipped_frames_{0};
  std::size_t unknown_lipids_{0};
  box reference_box_;
  bool have_reference_box_{false};
  std::vector<ProteinResiduePlan> protein_plans_;
  bool protein_cache_ready_{false};
  std::vector<ContactRow> contact_rows_;
  std::map<std::string, std::size_t> protein_contact_counts_;
  std::map<std::string, std::size_t> lipid_contact_counts_;
  std::set<std::string> near_lipid_ids_;
  std::map<std::string, EnrichmentAccumulator> enrichment_counts_;
  std::vector<CurvatureCell> curvature_cells_;
  std::map<std::string, ProximityAccumulator> proximity_accumulators_;
  std::size_t contact_frames_{0};
  std::size_t contact_skipped_frames_{0};
  std::size_t aggregate_contact_frames_{0};
  std::size_t aggregate_contact_rows_{0};

  ThicknessCell aggregate_apl_;
  ThicknessCell aggregate_thickness_;
  ThicknessCell aggregate_interdigitation_;
  std::vector<ThicknessCell> aggregate_thickness_cells_;
  std::map<std::string, ThicknessCell> aggregate_order_parameters_;
  std::map<std::string, CouplingCounter> aggregate_enrichment_;
  std::vector<CouplingCounter> aggregate_curvature_cells_;
  std::map<std::string, ProximityAccumulator> aggregate_proximity_;
  std::size_t aggregate_trajectories_{0};
};

#endif // TITAN_MEMBRANE_H
