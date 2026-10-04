#ifndef TITAN_AWH_IFP_H
#define TITAN_AWH_IFP_H

#include "MOL.h"
#include "MOL_info.h"
#include "arguments.h"
#include "info.h"
#include "titan_ifp.h"

#include <cstdint>
#include <array>
#include <string>
#include <vector>

class TitanAwhIfp {
public:
  struct AwhFrameContext {
    bool matched{false};
    int replica_id{-1};
    std::int64_t global_frame_index{-1};
    double time_ps{0.0};
    double rc_value{0.0};
    size_t bin_idx{0};
    double delta_to_center{0.0};
  };

  bool initialize(const arguments &args, const info &info_output,
                  size_t expected_walkers);

  AwhFrameContext prepare_frame(int replica_id, double time_ps,
                                std::int64_t global_frame_index);
  void consume_frame(
      const AwhFrameContext &frame_ctx,
      const std::vector<TitanIFP::ContactCountAccumulator> &worker_accumulators,
      size_t active_workers, const MOL &rec_mol, const MOL &lig_mol);

  void export_results(const arguments &args, const info &info_output,
                      const std::vector<mol_info> &rec_info,
                      const std::vector<mol_info> &lig_info) const;

  bool enabled() const { return initialized_; }
  size_t matched_frames() const { return matched_frames_; }
  size_t unmatched_frames() const { return unmatched_frames_; }

private:
  struct RcSample {
    double time_ps{0.0};
    double rc_value{0.0};
  };

  struct WalkerSeries {
    std::vector<RcSample> samples;
    double tolerance_ps{1e-3};
  };

  struct WalkerDiagnostics {
    std::string xvg_path{};
    std::size_t rc_samples{0};
    double rc_min{0.0};
    double rc_max{0.0};
    double time_min_ps{0.0};
    double time_max_ps{0.0};
    double tolerance_ps{0.0};
    std::size_t matched_frames{0};
    std::size_t unmatched_frames{0};
  };

  using ContactRecord = TitanIFP::InteractionCountRecord;

  struct RepresentativeFrame {
    bool valid{false};
    std::int64_t global_frame_index{-1};
    int replica_id{-1};
    double time_ps{0.0};
    double rc_value{0.0};
    double delta_to_center{0.0};
    MOL rec_frame;
    MOL lig_frame;
  };

  struct MatchedFrameMeta {
    int replica_id{-1};
    std::int64_t global_frame_index{-1};
    double rc_value{0.0};
    double time_ps{0.0};
  };

  struct BinData {
    size_t frame_count{0};
    TitanIFP::ContactCountAccumulator contacts;
    std::vector<size_t> walker_frame_counts;
    std::vector<MatchedFrameMeta> matched_frames;
    RepresentativeFrame representative;
  };

  bool initialized_{false};
  std::vector<WalkerSeries> walkers_;
  std::vector<WalkerDiagnostics> walker_diagnostics_;
  std::vector<BinData> bins_;
  std::vector<double> bin_edges_;
  double rc_min_{0.0};
  double rc_max_{0.0};
  double bin_width_{0.0};
  std::string representative_prefix_;
  size_t matched_frames_{0};
  size_t unmatched_frames_{0};

  static bool include_time_sample(const arguments &args, double time_ps);
  static double infer_tolerance_ps(const std::vector<RcSample> &samples);
  bool lookup_rc_value(int replica_id, double time_ps, double &rc_value) const;
  size_t assign_bin(double rc_value) const;
  double bin_left(size_t bin_idx) const;
  double bin_right(size_t bin_idx) const;
  double bin_center(size_t bin_idx) const;
  static std::string interaction_label(int bit);
  static std::string build_contact_label(const ContactRecord &record, int bit);
  static void write_representative_pdb(const std::string &filename,
                                       const RepresentativeFrame &frame,
                                       const std::vector<mol_info> &rec_info,
                                       const std::vector<mol_info> &lig_info);
};

#endif // TITAN_AWH_IFP_H
