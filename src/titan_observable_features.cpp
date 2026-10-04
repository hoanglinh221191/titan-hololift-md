#include "titan_observable_features.h"

#include "titan_ui.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>

namespace {

std::string csv_escape(const std::string &value) {
  if (value.find_first_of(",\"\n\r") == std::string::npos)
    return value;
  std::string out = "\"";
  for (const char c : value) {
    if (c == '"')
      out += "\"\"";
    else
      out += c;
  }
  out += '"';
  return out;
}

bool vector_is_finite(const Eigen::VectorXd &values) {
  for (Eigen::Index i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values(i)))
      return false;
  }
  return true;
}

} // namespace

void TitanObservableFeatureRouter::configure(bool export_enabled,
                                             bool consume_tica,
                                             std::string requested_source) {
  export_enabled_ = export_enabled;
  consume_tica_ = consume_tica;
  requested_source_ = std::move(requested_source);
}

void TitanObservableFeatureRouter::begin_trajectory(
    std::vector<FeatureColumn> columns, std::string resolved_source) {
  columns_ = std::move(columns);
  resolved_source_ = std::move(resolved_source);
  frames_.clear();
  frames_seen_ = 0;
  frames_accepted_ = 0;
  frames_skipped_nonfinite_ = 0;
  frames_skipped_dimension_ = 0;
  active_ = export_enabled_ || consume_tica_;
}

bool TitanObservableFeatureRouter::accept_frame(std::uint64_t frame,
                                                double time_ps,
                                                const Eigen::VectorXd &values) {
  ++frames_seen_;
  if (values.size() != static_cast<Eigen::Index>(columns_.size())) {
    ++frames_skipped_dimension_;
    return false;
  }
  if (!vector_is_finite(values)) {
    ++frames_skipped_nonfinite_;
    return false;
  }
  ++frames_accepted_;
  if (export_enabled_) {
    FeatureFrame record;
    record.frame = frame;
    record.time_ps = time_ps;
    record.values = values;
    frames_.push_back(std::move(record));
  }
  return true;
}

void TitanObservableFeatureRouter::export_outputs(
    const std::string &feature_filename, const std::string &metadata_filename,
    const std::string &summary_filename, std::fstream &log) const {
  if (!export_enabled_)
    return;
  export_feature_matrix(feature_filename, log);
  export_metadata(metadata_filename, log);
  export_summary(summary_filename, log);
}

void TitanObservableFeatureRouter::export_feature_matrix(
    const std::string &filename, std::fstream &log) const {
  std::ofstream out(filename);
  if (!out) {
    TitanUI::log_kv(log, "Feature Export", "failed to open " + filename);
    return;
  }
  out << "frame,time_ps";
  for (const FeatureColumn &column : columns_)
    out << ',' << csv_escape(column.source + ":" + column.name);
  out << '\n';
  out << std::fixed << std::setprecision(8);
  for (const FeatureFrame &frame : frames_) {
    out << frame.frame << ',' << frame.time_ps;
    for (Eigen::Index i = 0; i < frame.values.size(); ++i)
      out << ',' << frame.values(i);
    out << '\n';
  }
  TitanUI::log_kv(log, "Feature Matrix", filename);
}

void TitanObservableFeatureRouter::export_metadata(const std::string &filename,
                                                   std::fstream &log) const {
  std::ofstream out(filename);
  if (!out) {
    TitanUI::log_kv(log, "Feature Metadata", "failed to open " + filename);
    return;
  }
  out << "column,source,name,unit,periodic,residue,role\n";
  for (std::size_t i = 0; i < columns_.size(); ++i) {
    const FeatureColumn &column = columns_[i];
    out << i << ',' << csv_escape(column.source) << ','
        << csv_escape(column.name) << ',' << csv_escape(column.unit) << ','
        << (column.periodic ? "true" : "false") << ','
        << csv_escape(column.residue) << ',' << csv_escape(column.role) << '\n';
  }
  TitanUI::log_kv(log, "Feature Metadata", filename);
}

void TitanObservableFeatureRouter::export_summary(const std::string &filename,
                                                  std::fstream &log) const {
  std::ofstream out(filename);
  if (!out) {
    TitanUI::log_kv(log, "Feature Summary", "failed to open " + filename);
    return;
  }
  out << "module: titan_observable_features\n";
  out << "requested_source: " << requested_source_ << '\n';
  out << "resolved_source: " << resolved_source_ << '\n';
  out << "feature_dim: " << columns_.size() << '\n';
  out << "frames_seen: " << frames_seen_ << '\n';
  out << "frames_accepted: " << frames_accepted_ << '\n';
  out << "frames_skipped_nonfinite: " << frames_skipped_nonfinite_ << '\n';
  out << "frames_skipped_dimension: " << frames_skipped_dimension_ << '\n';
  out << "tica_consumed: " << (consume_tica_ ? "true" : "false") << '\n';
  TitanUI::log_kv(log, "Feature Summary", filename);
}
