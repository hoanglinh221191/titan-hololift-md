#ifndef TITAN_S2_H
#define TITAN_S2_H

#include "Multi_MOL.h"
#include "algebra/mat3.h"
#include "algebra/matrix3_buffer.h"
#include "titan_s2_kernels.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <vector>

class TitanExecutor;

struct TitanS2Config {
  bool flag_s2{false};
  std::string vectors{"both"};
  std::string residues{"all"};
  std::string align{"rms"};
  std::string maxlag{"auto"};
  std::string fit_window{"auto"};
  bool pdb{false};
  bool average_pdb{false};
};

struct TitanS2OutputFiles {
  std::string order_csv{"s2_order_parameters.csv"};
  std::string correlation_dat{"s2_correlation_functions.dat"};
  std::string summary_dat{"s2_summary.dat"};
  std::string pdb{"s2_bfactor.pdb"};
  std::string average_csv{"s2_order_parameters_average.csv"};
  std::string average_summary_dat{"s2_summary_average.dat"};
  std::string average_pdb{"s2_bfactor_average.pdb"};
};

class TitanS2Analyzer {
public:
  struct VectorPlan {
    std::size_t residue_index{0};
    std::size_t ordinal{0};
    std::size_t resnr{0};
    std::size_t chainnum{0};
    std::string chain{" "};
    std::string resname{"UNK"};
    std::string kind;
    int from_atom{-1};
    int to_atom{-1};
  };

  struct OrderParameterRow {
    std::size_t plan_index{0};
    double s2{0.0};
    double raw_plateau{0.0};
    double tau_ps{0.0};
    double weighted_residual{0.0};
    double fit_r2{std::numeric_limits<double>::quiet_NaN()};
    double fit_window_start_ps{std::numeric_limits<double>::quiet_NaN()};
    double fit_window_end_ps{std::numeric_limits<double>::quiet_NaN()};
    std::uint64_t samples{0};
    bool fit_ok{false};
    bool used_tail_fallback{false};
    std::size_t correlation_rows{0};
    std::size_t fit_rows{0};
  };

  TitanS2Analyzer() = default;

  void configure(const TitanS2Config &args);

  template <typename S2Args>
  void configure(const S2Args &args) {
    TitanS2Config config;
    if constexpr (requires { args.flag_s2; })
      config.flag_s2 = args.flag_s2;
    if constexpr (requires { args.vectors; })
      config.vectors = args.vectors;
    else if constexpr (requires { args.vector; })
      config.vectors = args.vector;
    if constexpr (requires { args.residues; })
      config.residues = args.residues;
    if constexpr (requires { args.align; })
      config.align = args.align;
    if constexpr (requires { args.maxlag; })
      config.maxlag = args.maxlag;
    if constexpr (requires { args.fit_window; })
      config.fit_window = args.fit_window;
    else if constexpr (requires { args.plateau_window; })
      config.fit_window = args.plateau_window;
    if constexpr (requires { args.pdb; })
      config.pdb = args.pdb;
    else if constexpr (requires { args.export_pdb; })
      config.pdb = args.export_pdb;
    else if constexpr (requires { args.bfactor_pdb; }) {
      config.pdb = args.bfactor_pdb;
      config.average_pdb = args.bfactor_pdb;
    }
    if constexpr (requires { args.average_pdb; })
      config.average_pdb = args.average_pdb;
    configure(config);
  }

  void init(const Multi_MOL &mol1, std::fstream &log);

  template <typename Args>
  void init(const Multi_MOL &mol1, const Args &, std::fstream &log) {
    init(mol1, log);
  }

  void process_frame(const Multi_MOL &mol1, std::size_t snap_idx,
                     TitanExecutor &executor);
  void finalize_trajectory(const std::string &base_dir,
                           const TitanS2OutputFiles &files,
                           bool export_average, TitanExecutor &executor,
                           std::fstream &log);
  void finalize_all(const std::string &base_dir,
                    const TitanS2OutputFiles &files) const;

  template <typename Info, typename Args>
  void finalize_trajectory(const std::string &base_dir,
                           const Info &info_output, const Args &args,
                           const std::vector<std::string> &input_line,
                           int input_name, TitanExecutor &executor,
                           std::fstream &log) {
    TitanS2OutputFiles files;
    if constexpr (requires {
                    info_output.get_s2_order_parameters_filename(
                        args, input_line, input_name);
                  })
      files.order_csv = info_output.get_s2_order_parameters_filename(
          args, input_line, input_name);
    if constexpr (requires {
                    info_output.get_s2_correlation_filename(args, input_line,
                                                            input_name);
                  })
      files.correlation_dat = info_output.get_s2_correlation_filename(
          args, input_line, input_name);
    else if constexpr (requires {
                         info_output.get_s2_correlation_functions_filename(
                             args, input_line, input_name);
                       })
      files.correlation_dat =
          info_output.get_s2_correlation_functions_filename(args, input_line,
                                                            input_name);
    if constexpr (requires {
                    info_output.get_s2_summary_filename(args, input_line,
                                                        input_name);
                  })
      files.summary_dat =
          info_output.get_s2_summary_filename(args, input_line, input_name);
    if constexpr (requires {
                    info_output.get_s2_pdb_filename(args, input_line,
                                                    input_name);
                  })
      files.pdb =
          info_output.get_s2_pdb_filename(args, input_line, input_name);
    else if constexpr (requires {
                         info_output.get_s2_bfactor_filename(args, input_line,
                                                             input_name);
                       })
      files.pdb =
          info_output.get_s2_bfactor_filename(args, input_line, input_name);
    if constexpr (requires {
                    info_output.get_s2_order_parameters_average_filename();
                  })
      files.average_csv =
          info_output.get_s2_order_parameters_average_filename();
    if constexpr (requires { info_output.get_s2_summary_average_filename(); })
      files.average_summary_dat =
          info_output.get_s2_summary_average_filename();
    if constexpr (requires { info_output.get_s2_pdb_average_filename(); })
      files.average_pdb = info_output.get_s2_pdb_average_filename();
    else if constexpr (requires {
                         info_output.get_s2_bfactor_average_filename();
                       })
      files.average_pdb = info_output.get_s2_bfactor_average_filename();
    const bool multi_input = input_line.size() > 1 || args.flag_listfile ||
                             args.flag_pattern_input || args.flag_xdr;
    finalize_trajectory(base_dir, files, multi_input, executor, log);
  }

  template <typename Info>
  void finalize_all(const std::string &base_dir, const Info &info_output,
                    std::fstream &log) const {
    TitanS2OutputFiles files;
    files.average_csv = info_output.get_s2_order_parameters_average_filename();
    files.average_summary_dat = info_output.get_s2_summary_average_filename();
    if constexpr (requires { info_output.get_s2_pdb_average_filename(); })
      files.average_pdb = info_output.get_s2_pdb_average_filename();
    else if constexpr (requires {
                         info_output.get_s2_bfactor_average_filename();
                       })
      files.average_pdb = info_output.get_s2_bfactor_average_filename();
    finalize_all(base_dir, files);
    if (log.is_open())
      log << "S2: wrote final aggregate outputs to " << base_dir << '\n';
  }

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }
  [[nodiscard]] bool initialized() const noexcept { return initialized_; }
  [[nodiscard]] const std::vector<VectorPlan> &plans() const noexcept {
    return plans_;
  }
  [[nodiscard]] std::vector<OrderParameterRow>
  compute_order_parameters(TitanExecutor &executor) const;

private:
  struct Residue {
    std::size_t ordinal{0};
    std::size_t resnr{0};
    std::size_t chainnum{0};
    std::string chain{" "};
    std::string resname{"UNK"};
    std::map<std::string, int> atom_by_name;
  };

  struct ResidueRange {
    bool has_chain{false};
    std::string chain;
    std::size_t begin{0};
    std::size_t end{0};
  };

  struct AggregateCell {
    double sum{0.0};
    double sum_sq{0.0};
    std::size_t count{0};
  };

  struct AggregateResult {
    AggregateCell raw_plateau;
    AggregateCell s2;
  };

  void reset_trajectory_state();
  void build_topology_cache(const Multi_MOL &mol1);
  void parse_residue_selection();
  [[nodiscard]] bool selected(const Residue &residue) const;
  [[nodiscard]] bool wants_nh() const noexcept;
  [[nodiscard]] bool wants_ca_c() const noexcept;
  void build_alignment_trace();
  void ensure_alignment_reference(const MOL &frame);
  void append_vectors(const MOL &frame);
  [[nodiscard]] std::size_t resolve_maxlag() const;
  [[nodiscard]] std::vector<titan_s2_kernel::CorrelationRow>
  fit_rows(const std::vector<titan_s2_kernel::CorrelationRow> &rows) const;
  [[nodiscard]] double weighted_fit_r2(
      const std::vector<titan_s2_kernel::CorrelationRow> &rows,
      const titan_s2_kernel::FitResult &fit) const noexcept;
  [[nodiscard]] std::string residue_label(const VectorPlan &plan) const;
  [[nodiscard]] std::string aggregate_key(const VectorPlan &plan) const;
  void export_order_csv(const std::string &filename,
                        const std::vector<OrderParameterRow> &rows) const;
  void export_correlation_dat(const std::string &filename,
                              TitanExecutor &executor) const;
  void export_summary(const std::string &filename,
                      const std::vector<OrderParameterRow> &rows) const;
  void export_pdb(const std::string &filename,
                  const std::map<std::string, double> &s2_by_key,
                  bool average) const;
  [[nodiscard]] std::vector<std::string> plan_signature() const;
  bool accumulate_trajectory(const std::vector<OrderParameterRow> &rows);
  void export_average(const std::string &base_dir,
                      const TitanS2OutputFiles &files) const;

  TitanS2Config args_{};
  bool requested_{false};
  bool enabled_{false};
  bool initialized_{false};
  std::vector<Residue> residues_;
  std::vector<ResidueRange> ranges_;
  std::vector<VectorPlan> plans_;
  std::vector<int> alignment_atoms_;
  std::vector<titan_s2_kernel::BondVectorsSoA> vectors_;
  std::vector<double> time_ps_;
  AlignedDoubleVector scratch_from_x_;
  AlignedDoubleVector scratch_from_y_;
  AlignedDoubleVector scratch_from_z_;
  AlignedDoubleVector scratch_to_x_;
  AlignedDoubleVector scratch_to_y_;
  AlignedDoubleVector scratch_to_z_;
  titan_s2_kernel::BondVectorsSoA scratch_frame_vectors_;
  std::array<double, 9> scratch_rotation_{};
  titan::algebra::Matrix3Buffer alignment_reference_matrix_;
  titan::algebra::Matrix3Buffer alignment_mobile_matrix_;
  titan::algebra::Mat3 alignment_rotation_{
      titan::algebra::Mat3::identity()};
  std::vector<mol_info> reference_info_;
  MOL first_snapshot_;
  bool have_first_snapshot_{false};
  bool have_alignment_reference_{false};
  std::uint64_t frames_seen_{0};
  std::uint64_t frames_used_{0};
  std::uint64_t raw_box_fallback_frames_{0};
  std::uint64_t missing_nh_skipped_{0};
  std::uint64_t pro_nh_skipped_{0};

  std::map<std::string, AggregateResult> aggregate_results_;
  std::map<std::string, VectorPlan> aggregate_plans_;
  std::vector<std::string> aggregate_plan_signature_;
  std::size_t aggregate_trajectories_{0};
  std::vector<mol_info> aggregate_reference_info_;
  MOL aggregate_first_snapshot_;
  bool have_aggregate_snapshot_{false};
};

#endif // TITAN_S2_H
