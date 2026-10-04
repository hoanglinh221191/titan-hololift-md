#ifndef TITAN_OBSERVABLE_FEATURES_H
#define TITAN_OBSERVABLE_FEATURES_H

#include <Eigen/Dense>

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

struct FeatureColumn {
  std::string source;
  std::string name;
  std::string unit;
  bool periodic{false};
  std::string residue;
  std::string role;
};

struct FeatureFrame {
  std::uint64_t frame{0};
  double time_ps{0.0};
  Eigen::VectorXd values;
};

class TitanObservableFeatureRouter {
public:
  void configure(bool export_enabled, bool consume_tica,
                 std::string requested_source);
  void begin_trajectory(std::vector<FeatureColumn> columns,
                        std::string resolved_source);
  [[nodiscard]] bool active() const noexcept { return active_; }
  [[nodiscard]] bool export_enabled() const noexcept { return export_enabled_; }
  [[nodiscard]] bool consume_tica() const noexcept { return consume_tica_; }
  [[nodiscard]] int feature_dim() const noexcept {
    return static_cast<int>(columns_.size());
  }
  [[nodiscard]] const std::vector<FeatureColumn> &columns() const noexcept {
    return columns_;
  }
  [[nodiscard]] const std::string &resolved_source() const noexcept {
    return resolved_source_;
  }

  [[nodiscard]] bool accept_frame(std::uint64_t frame, double time_ps,
                                  const Eigen::VectorXd &values);
  void export_outputs(const std::string &feature_filename,
                      const std::string &metadata_filename,
                      const std::string &summary_filename,
                      std::fstream &log) const;

private:
  void export_feature_matrix(const std::string &filename,
                             std::fstream &log) const;
  void export_metadata(const std::string &filename, std::fstream &log) const;
  void export_summary(const std::string &filename, std::fstream &log) const;

  bool export_enabled_{false};
  bool consume_tica_{false};
  bool active_{false};
  std::string requested_source_{"auto"};
  std::string resolved_source_{"auto"};
  std::vector<FeatureColumn> columns_;
  std::vector<FeatureFrame> frames_;
  std::uint64_t frames_seen_{0};
  std::uint64_t frames_accepted_{0};
  std::uint64_t frames_skipped_nonfinite_{0};
  std::uint64_t frames_skipped_dimension_{0};
};

#endif // TITAN_OBSERVABLE_FEATURES_H
