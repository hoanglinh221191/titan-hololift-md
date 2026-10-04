#include "titan_uncertainty.h"

#include "titan_ui.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace titan_uncertainty {
namespace {

struct SidecarRow {
  std::string module;
  std::string observable;
  std::string source_file;
  std::size_t n_samples{0};
  std::size_t n_blocks{0};
  double mean{std::numeric_limits<double>::quiet_NaN()};
  double sem{std::numeric_limits<double>::quiet_NaN()};
  double ci95_low{std::numeric_limits<double>::quiet_NaN()};
  double ci95_high{std::numeric_limits<double>::quiet_NaN()};
  double boot_low{std::numeric_limits<double>::quiet_NaN()};
  double boot_high{std::numeric_limits<double>::quiet_NaN()};
  std::string status;
  std::string note;
};

struct SummaryCounters {
  std::size_t modules_requested{0};
  std::size_t sidecars_written{0};
  std::size_t rows_ok{0};
  std::size_t rows_insufficient{0};
  std::size_t rows_point_estimate_only{0};
};

std::string trim(std::string value) {
  const auto first =
      std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
      });
  const auto last =
      std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
      }).base();
  if (first >= last)
    return {};
  return std::string(first, last);
}

std::vector<std::string> split_csv_line(const std::string &line) {
  std::vector<std::string> out;
  std::string field;
  bool quoted = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (quoted) {
      if (c == '"' && i + 1 < line.size() && line[i + 1] == '"') {
        field.push_back('"');
        ++i;
      } else if (c == '"') {
        quoted = false;
      } else {
        field.push_back(c);
      }
    } else if (c == '"') {
      quoted = true;
    } else if (c == ',') {
      out.push_back(trim(field));
      field.clear();
    } else {
      field.push_back(c);
    }
  }
  out.push_back(trim(field));
  return out;
}

std::vector<std::string> split_ws_line(const std::string &line) {
  std::istringstream in(line);
  std::vector<std::string> out;
  for (std::string field; in >> field;)
    out.push_back(field);
  return out;
}

bool parse_double(const std::string &text, double &out) {
  try {
    std::size_t pos = 0;
    out = std::stod(text, &pos);
    return pos > 0 && std::isfinite(out);
  } catch (...) {
    return false;
  }
}

std::string format_double(double value) {
  if (!std::isfinite(value))
    return "nan";
  std::ostringstream out;
  out << std::setprecision(12) << value;
  return out.str();
}

std::string csv_escape(const std::string &value) {
  if (value.find_first_of(",\"\n\r") == std::string::npos)
    return value;
  std::string out = "\"";
  for (const char c : value) {
    if (c == '"')
      out += "\"\"";
    else
      out.push_back(c);
  }
  out.push_back('"');
  return out;
}

std::vector<fs::path> matching_files(const std::string &base) {
  std::vector<fs::path> files;
  const fs::path base_path(base);
  if (fs::exists(base_path))
    files.push_back(base_path);

  const std::string stem = base_path.stem().string();
  const std::string ext = base_path.extension().string();
  for (const auto &entry : fs::directory_iterator(fs::current_path())) {
    if (!entry.is_regular_file())
      continue;
    const fs::path p = entry.path().filename();
    if (p == base_path.filename())
      continue;
    const std::string name = p.string();
    if (name.rfind(stem + "_", 0) == 0 && p.extension().string() == ext &&
        name.find("_average") == std::string::npos)
      files.push_back(p);
  }
  std::sort(files.begin(), files.end());
  return files;
}

std::vector<double> read_table_column(const fs::path &path,
                                      const std::vector<std::string> &names,
                                      bool fallback_last_numeric = false) {
  std::ifstream in(path);
  if (!in)
    return {};
  std::string line;
  if (!std::getline(in, line))
    return {};
  const bool comma = line.find(',') != std::string::npos;
  const auto header = comma ? split_csv_line(line) : split_ws_line(line);
  int index = -1;
  for (const auto &name : names) {
    for (std::size_t i = 0; i < header.size(); ++i) {
      if (header[i] == name) {
        index = static_cast<int>(i);
        break;
      }
    }
    if (index >= 0)
      break;
  }

  std::vector<double> values;
  while (std::getline(in, line)) {
    if (trim(line).empty())
      continue;
    const auto fields = comma ? split_csv_line(line) : split_ws_line(line);
    double value = 0.0;
    if (index >= 0 && static_cast<std::size_t>(index) < fields.size()) {
      if (parse_double(fields[static_cast<std::size_t>(index)], value))
        values.push_back(value);
      continue;
    }
    if (fallback_last_numeric) {
      for (auto it = fields.rbegin(); it != fields.rend(); ++it) {
        if (parse_double(*it, value)) {
          values.push_back(value);
          break;
        }
      }
    }
  }
  return values;
}

std::map<std::string, std::string> read_key_value_dat(const fs::path &path) {
  std::map<std::string, std::string> values;
  std::ifstream in(path);
  if (!in)
    return values;
  for (std::string line; std::getline(in, line);) {
    const auto colon = line.find(':');
    if (colon == std::string::npos)
      continue;
    values[trim(line.substr(0, colon))] = trim(line.substr(colon + 1));
  }
  return values;
}

bool parse_fit_window(const std::map<std::string, std::string> &kv,
                      double &tmin, double &tmax) {
  const auto it = kv.find("fit_window_ps");
  if (it == kv.end())
    return false;
  const auto comma = it->second.find(',');
  if (comma == std::string::npos)
    return false;
  double a = 0.0;
  double b = 0.0;
  if (!parse_double(it->second.substr(0, comma), a) ||
      !parse_double(it->second.substr(comma + 1), b) || !(b > a))
    return false;
  tmin = a;
  tmax = b;
  return true;
}

std::string sibling_output(const fs::path &source, const std::string &from_stem,
                           const std::string &to_stem,
                           const std::string &extension) {
  std::string stem = source.stem().string();
  if (stem.rfind(from_stem, 0) == 0)
    stem = to_stem + stem.substr(from_stem.size());
  else
    stem = to_stem;
  return (source.parent_path() / (stem + extension)).string();
}

std::vector<XYPoint> read_xy_table(const fs::path &path,
                                   const std::vector<std::string> &x_names,
                                   const std::vector<std::string> &y_names) {
  std::ifstream in(path);
  if (!in)
    return {};
  std::string line;
  if (!std::getline(in, line))
    return {};
  const bool comma = line.find(',') != std::string::npos;
  const auto header = comma ? split_csv_line(line) : split_ws_line(line);
  int x_index = -1;
  int y_index = -1;
  for (const auto &name : x_names) {
    for (std::size_t i = 0; i < header.size(); ++i) {
      if (header[i] == name) {
        x_index = static_cast<int>(i);
        break;
      }
    }
    if (x_index >= 0)
      break;
  }
  for (const auto &name : y_names) {
    for (std::size_t i = 0; i < header.size(); ++i) {
      if (header[i] == name) {
        y_index = static_cast<int>(i);
        break;
      }
    }
    if (y_index >= 0)
      break;
  }
  if (x_index < 0 || y_index < 0)
    return {};
  std::vector<XYPoint> out;
  while (std::getline(in, line)) {
    if (trim(line).empty())
      continue;
    const auto fields = comma ? split_csv_line(line) : split_ws_line(line);
    if (static_cast<std::size_t>(x_index) >= fields.size() ||
        static_cast<std::size_t>(y_index) >= fields.size())
      continue;
    double x = 0.0;
    double y = 0.0;
    if (parse_double(fields[static_cast<std::size_t>(x_index)], x) &&
        parse_double(fields[static_cast<std::size_t>(y_index)], y))
      out.push_back(XYPoint{x, y});
  }
  return out;
}

std::vector<double> block_means_for(const std::vector<double> &values,
                                    const std::string &blocks_spec) {
  std::vector<double> finite;
  finite.reserve(values.size());
  for (const double value : values) {
    if (std::isfinite(value))
      finite.push_back(value);
  }
  const auto blocks = partition_blocks(finite.size(), blocks_spec);
  std::vector<double> means;
  means.reserve(blocks.size());
  for (const auto &block : blocks) {
    if (block.begin >= block.end || block.end > finite.size())
      continue;
    const double sum = std::accumulate(finite.begin() + block.begin,
                                       finite.begin() + block.end, 0.0);
    means.push_back(sum / static_cast<double>(block.end - block.begin));
  }
  return means;
}

std::vector<XYPoint> filter_points(const std::vector<XYPoint> &points,
                                   double tmin, double tmax, bool log_transform) {
  std::vector<XYPoint> out;
  out.reserve(points.size());
  for (const auto &point : points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || point.x < tmin ||
        point.x > tmax)
      continue;
    if (log_transform) {
      if (!(point.x > 0.0) || !(point.y > 0.0))
        continue;
      out.push_back(XYPoint{std::log(point.x), std::log(point.y)});
    } else {
      out.push_back(point);
    }
  }
  return out;
}

SidecarRow make_fit_row(const arguments &args, std::string module,
                        std::string observable, const fs::path &source,
                        const BlockFitResult &fit, double point_value,
                        double scale, bool use_intercept = false,
                        bool exp_intercept = false) {
  SidecarRow row;
  row.module = std::move(module);
  row.observable = std::move(observable);
  row.source_file = source.filename().string();
  row.n_samples = fit.point.n;
  row.n_blocks = fit.block_count;
  row.mean = point_value;
  const Stats &stats = use_intercept ? fit.intercept_stats : fit.slope_stats;
  row.sem = std::isfinite(stats.sem) ? stats.sem * scale : stats.sem;
  row.ci95_low = std::isfinite(stats.ci95_low) ? stats.ci95_low * scale
                                               : stats.ci95_low;
  row.ci95_high = std::isfinite(stats.ci95_high) ? stats.ci95_high * scale
                                                 : stats.ci95_high;
  if (use_intercept && exp_intercept) {
    if (std::isfinite(stats.sem)) {
      const auto lo = std::isfinite(stats.ci95_low) ? std::exp(stats.ci95_low)
                                                    : stats.ci95_low;
      const auto hi = std::isfinite(stats.ci95_high) ? std::exp(stats.ci95_high)
                                                     : stats.ci95_high;
      row.ci95_low = lo;
      row.ci95_high = hi;
    }
    row.sem = std::numeric_limits<double>::quiet_NaN();
  }
  const double boot_low =
      use_intercept ? fit.intercept_bootstrap_ci95_low
                    : fit.slope_bootstrap_ci95_low;
  const double boot_high =
      use_intercept ? fit.intercept_bootstrap_ci95_high
                    : fit.slope_bootstrap_ci95_high;
  row.boot_low = std::isfinite(boot_low) ? boot_low * scale : boot_low;
  row.boot_high = std::isfinite(boot_high) ? boot_high * scale : boot_high;
  if (use_intercept && exp_intercept) {
    row.boot_low = std::isfinite(boot_low) ? std::exp(boot_low) : boot_low;
    row.boot_high = std::isfinite(boot_high) ? std::exp(boot_high) : boot_high;
  }
  if (!fit.point.ok) {
    row.status = fit.point.status;
    row.note = "point_estimate_unchanged";
  } else if (stats.n < 2) {
    row.status = "insufficient_blocks";
    row.note = "fit_point_estimate_only";
  } else {
    row.status = "ok";
    row.note = args.uncertainty.mode == "bootstrap"
                   ? "fit_block_bootstrap_over_block_estimates"
                   : "fit_contiguous_block_sem";
  }
  return row;
}

SidecarRow make_temporal_row(const arguments &args, std::string module,
                             std::string observable, const fs::path &source,
                             const std::vector<double> &values) {
  SidecarRow row;
  row.module = std::move(module);
  row.observable = std::move(observable);
  row.source_file = source.filename().string();
  row.n_samples = values.size();

  const auto raw_stats = summarize_values(values);
  row.mean = raw_stats.mean;
  const auto means = block_means_for(values, args.uncertainty.blocks);
  row.n_blocks = means.size();
  if (means.size() >= 2) {
    const auto block_stats = summarize_values(means);
    row.sem = block_stats.sem;
    row.ci95_low = block_stats.ci95_low;
    row.ci95_high = block_stats.ci95_high;
    row.status = "ok";
    if (args.uncertainty.mode == "bootstrap") {
      const auto ci =
          bootstrap_ci95(means, args.uncertainty.bootstrap_samples,
                         args.uncertainty.seed);
      row.boot_low = ci.first;
      row.boot_high = ci.second;
      row.note = "block_bootstrap_over_block_means";
    } else {
      row.note = "contiguous_block_sem";
    }
  } else {
    row.status = "insufficient_blocks";
    row.note = "point_estimate_unchanged";
  }
  return row;
}

SidecarRow make_entity_row(std::string module, std::string observable,
                           const fs::path &source,
                           const std::vector<double> &values) {
  SidecarRow row;
  row.module = std::move(module);
  row.observable = std::move(observable);
  row.source_file = source.filename().string();
  const auto stats = summarize_values(values);
  row.n_samples = stats.n;
  row.mean = stats.mean;
  row.sem = stats.sem;
  row.ci95_low = stats.ci95_low;
  row.ci95_high = stats.ci95_high;
  row.n_blocks = 0;
  row.status = stats.n > 0 ? "point_estimate_only" : "missing_data";
  row.note = "entity_distribution_not_time_uncertainty";
  return row;
}

void write_sidecar(const std::string &filename,
                   const std::vector<SidecarRow> &rows) {
  std::ofstream out(filename);
  out << "module,observable,source_file,n_samples,n_blocks,mean,sem,ci95_low,"
         "ci95_high,bootstrap_ci95_low,bootstrap_ci95_high,status,note\n";
  for (const auto &row : rows) {
    out << csv_escape(row.module) << ',' << csv_escape(row.observable) << ','
        << csv_escape(row.source_file) << ',' << row.n_samples << ','
        << row.n_blocks << ',' << format_double(row.mean) << ','
        << format_double(row.sem) << ',' << format_double(row.ci95_low) << ','
        << format_double(row.ci95_high) << ',' << format_double(row.boot_low)
        << ',' << format_double(row.boot_high) << ','
        << csv_escape(row.status) << ',' << csv_escape(row.note) << '\n';
  }
}

void accumulate_counters(const std::vector<SidecarRow> &rows,
                         SummaryCounters &counters) {
  ++counters.sidecars_written;
  for (const auto &row : rows) {
    if (row.status == "ok")
      ++counters.rows_ok;
    else if (row.status == "insufficient_blocks" ||
             row.status == "insufficient_points" ||
             row.status == "singular_fit" || row.status == "missing_data")
      ++counters.rows_insufficient;
    else if (row.status == "point_estimate_only")
      ++counters.rows_point_estimate_only;
  }
}

void add_temporal_rows(const arguments &args, std::vector<SidecarRow> &rows,
                       const std::string &module,
                       const std::string &base_filename,
                       const std::vector<std::pair<std::string,
                                                   std::vector<std::string>>>
                           &observables,
                       bool fallback_last_numeric = false) {
  for (const auto &file : matching_files(base_filename)) {
    for (const auto &observable : observables) {
      const auto values =
          read_table_column(file, observable.second, fallback_last_numeric);
      rows.push_back(make_temporal_row(args, module, observable.first, file,
                                       values));
    }
  }
}

void add_entity_rows(std::vector<SidecarRow> &rows, const std::string &module,
                     const std::string &base_filename,
                     const std::vector<std::pair<std::string,
                                                 std::vector<std::string>>>
                         &observables) {
  for (const auto &file : matching_files(base_filename)) {
    for (const auto &observable : observables) {
      const auto values = read_table_column(file, observable.second, false);
      rows.push_back(make_entity_row(module, observable.first, file, values));
    }
  }
}

void add_record_value_rows(
    std::vector<SidecarRow> &rows, const std::string &module,
    const std::string &base_filename,
    const std::vector<std::string> &label_columns,
    const std::vector<std::pair<std::string, std::vector<std::string>>>
        &observables) {
  for (const auto &file : matching_files(base_filename)) {
    std::ifstream in(file);
    if (!in)
      continue;
    std::string line;
    if (!std::getline(in, line))
      continue;
    const bool comma = line.find(',') != std::string::npos;
    const auto header = comma ? split_csv_line(line) : split_ws_line(line);
    std::vector<int> label_indices;
    for (const auto &column : label_columns) {
      int index = -1;
      for (std::size_t i = 0; i < header.size(); ++i) {
        if (header[i] == column) {
          index = static_cast<int>(i);
          break;
        }
      }
      if (index >= 0)
        label_indices.push_back(index);
    }
    std::vector<std::pair<std::string, int>> obs_indices;
    for (const auto &observable : observables) {
      int index = -1;
      for (const auto &name : observable.second) {
        for (std::size_t i = 0; i < header.size(); ++i) {
          if (header[i] == name) {
            index = static_cast<int>(i);
            break;
          }
        }
        if (index >= 0)
          break;
      }
      if (index >= 0)
        obs_indices.emplace_back(observable.first, index);
    }
    for (; std::getline(in, line);) {
      if (trim(line).empty())
        continue;
      const auto fields = comma ? split_csv_line(line) : split_ws_line(line);
      std::string label;
      for (const int index : label_indices) {
        if (index < 0 || static_cast<std::size_t>(index) >= fields.size())
          continue;
        if (!label.empty())
          label.push_back('|');
        label += fields[static_cast<std::size_t>(index)];
      }
      for (const auto &[observable, index] : obs_indices) {
        if (index < 0 || static_cast<std::size_t>(index) >= fields.size())
          continue;
        double value = 0.0;
        if (!parse_double(fields[static_cast<std::size_t>(index)], value))
          continue;
        SidecarRow row;
        row.module = module;
        row.observable = label.empty() ? observable : observable + "|" + label;
        row.source_file = file.filename().string();
        row.n_samples = 1;
        row.mean = value;
        row.status = "point_estimate_only";
        row.note = "entity_distribution_not_time_uncertainty";
        rows.push_back(row);
      }
    }
  }
}

void add_key_value_rows(
    std::vector<SidecarRow> &rows, const std::string &module,
    const std::string &filename,
    const std::vector<std::pair<std::string, std::vector<std::string>>>
        &observables) {
  if (!fs::exists(filename))
    return;
  const auto kv = read_key_value_dat(filename);
  for (const auto &observable : observables) {
    double value = 0.0;
    bool found = false;
    for (const auto &key : observable.second) {
      const auto it = kv.find(key);
      if (it != kv.end() && parse_double(it->second, value)) {
        found = true;
        break;
      }
    }
    if (!found)
      continue;
    SidecarRow row;
    row.module = module;
    row.observable = observable.first;
    row.source_file = fs::path(filename).filename().string();
    row.n_samples = 1;
    row.mean = value;
    row.status = "point_estimate_only";
    row.note = "replicate_aggregate_point_estimate";
    rows.push_back(row);
  }
}

void add_msd_fit_rows(const arguments &args, std::vector<SidecarRow> &rows) {
  for (const auto &file : matching_files("msd_timeseries.dat")) {
    const auto points =
        read_xy_table(file, {"lag_time_ps"}, {"msd_A2"});
    const auto diffusion_summary = read_key_value_dat(
        sibling_output(file, "msd_timeseries", "msd_diffusion_summary", ".dat"));
    const auto alpha_summary = read_key_value_dat(
        sibling_output(file, "msd_timeseries", "msd_alpha_fit", ".dat"));
    double tmin = 0.0;
    double tmax = 0.0;
    if (!parse_fit_window(diffusion_summary, tmin, tmax) && !points.empty()) {
      tmin = points.front().x;
      tmax = points.back().x;
    }
    const auto linear = linear_fit_with_blocks(
        points, tmin, tmax, args.uncertainty.blocks,
        args.uncertainty.bootstrap_samples, args.uncertainty.seed);
    rows.push_back(make_fit_row(args, "MSD", "slope_A2_per_ps", file, linear,
                                linear.point.slope, 1.0));
    rows.push_back(make_fit_row(args, "MSD", "D_A2_per_ps", file, linear,
                                linear.point.ok ? linear.point.slope / 6.0
                                                : std::numeric_limits<double>::quiet_NaN(),
                                1.0 / 6.0));
    rows.push_back(make_fit_row(args, "MSD", "D_cm2_per_s", file, linear,
                                linear.point.ok ? linear.point.slope / 6.0e4
                                                : std::numeric_limits<double>::quiet_NaN(),
                                1.0 / 6.0e4));

    double amin = tmin;
    double amax = tmax;
    (void)parse_fit_window(alpha_summary, amin, amax);
    const auto logfit = loglog_fit_with_blocks(
        points, amin, amax, args.uncertainty.blocks,
        args.uncertainty.bootstrap_samples, args.uncertainty.seed + 17U);
    rows.push_back(make_fit_row(args, "MSD", "alpha", file, logfit,
                                logfit.point.slope, 1.0));
    rows.push_back(make_fit_row(
        args, "MSD", "prefactor_A2_per_ps_alpha", file, logfit,
        logfit.point.ok ? std::exp(logfit.point.intercept)
                        : std::numeric_limits<double>::quiet_NaN(),
        1.0, true, true));
  }
}

void write_summary(const arguments &args, const std::string &filename,
                   const std::map<std::string, std::size_t> &sidecar_rows,
                   const SummaryCounters &counters) {
  std::ofstream out(filename);
  out << "implementation_version: 2\n";
  out << "uncertainty_mode: " << args.uncertainty.mode << '\n';
  out << "block_count: " << args.uncertainty.blocks << '\n';
  out << "bootstrap_samples: " << args.uncertainty.bootstrap_samples << '\n';
  out << "seed: " << args.uncertainty.seed << '\n';
  out << "fit_aware_modules: MSD,S2,Membrane\n";
  out << "trajectory_policy: trajectory_equal_means\n";
  out << "note: Point estimates are unchanged; uncertainty files are opt-in "
         "diagnostics.\n";
  out << "sidecar_status_table:\n";
  out << "  modules_requested: " << counters.modules_requested << '\n';
  out << "  sidecars_written: " << counters.sidecars_written << '\n';
  out << "  rows_ok: " << counters.rows_ok << '\n';
  out << "  rows_insufficient: " << counters.rows_insufficient << '\n';
  out << "  rows_point_estimate_only: "
      << counters.rows_point_estimate_only << '\n';
  out << "sidecars:\n";
  for (const auto &[name, rows] : sidecar_rows)
    out << "  " << name << ": rows=" << rows << '\n';
}

} // namespace

std::vector<Block> partition_blocks(std::size_t frame_count,
                                    const std::string &blocks_spec) {
  if (frame_count < 10)
    return {};
  std::size_t count = 0;
  if (blocks_spec == "auto") {
    if (frame_count < 50)
      count = 5;
    else
      count = std::min<std::size_t>(
          20, static_cast<std::size_t>(std::floor(std::sqrt(frame_count))));
  } else {
    int parsed = 0;
    try {
      parsed = std::stoi(blocks_spec);
    } catch (...) {
      throw std::invalid_argument("invalid block count");
    }
    if (parsed < 1)
      throw std::invalid_argument("invalid block count");
    count = static_cast<std::size_t>(parsed);
  }
  count = std::max<std::size_t>(1, std::min(count, frame_count));
  std::vector<Block> blocks;
  blocks.reserve(count);
  const std::size_t base = frame_count / count;
  const std::size_t rem = frame_count % count;
  std::size_t begin = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t len = base + (i < rem ? 1 : 0);
    blocks.push_back(Block{begin, begin + len});
    begin += len;
  }
  return blocks;
}

Stats summarize_values(const std::vector<double> &values) {
  std::vector<double> finite;
  finite.reserve(values.size());
  for (const double value : values) {
    if (std::isfinite(value))
      finite.push_back(value);
  }
  Stats out;
  out.n = finite.size();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  if (finite.empty()) {
    out.mean = out.variance = out.sem = out.ci95_low = out.ci95_high = nan;
    return out;
  }
  const double sum = std::accumulate(finite.begin(), finite.end(), 0.0);
  out.mean = sum / static_cast<double>(finite.size());
  if (finite.size() < 2) {
    out.variance = out.sem = out.ci95_low = out.ci95_high = nan;
    return out;
  }
  double ss = 0.0;
  for (const double value : finite) {
    const double d = value - out.mean;
    ss += d * d;
  }
  out.variance = ss / static_cast<double>(finite.size() - 1);
  out.sem = std::sqrt(out.variance / static_cast<double>(finite.size()));
  out.ci95_low = out.mean - 1.96 * out.sem;
  out.ci95_high = out.mean + 1.96 * out.sem;
  return out;
}

Stats summarize_blocks(const std::vector<double> &values,
                       const std::string &blocks_spec) {
  return summarize_values(block_means_for(values, blocks_spec));
}

std::pair<double, double>
bootstrap_ci95(const std::vector<double> &block_means, int samples,
               unsigned int seed) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  if (block_means.size() < 2 || samples < 1)
    return {nan, nan};
  std::mt19937 rng(seed);
  std::uniform_int_distribution<std::size_t> pick(0, block_means.size() - 1);
  std::vector<double> means;
  means.reserve(static_cast<std::size_t>(samples));
  for (int s = 0; s < samples; ++s) {
    double sum = 0.0;
    for (std::size_t i = 0; i < block_means.size(); ++i)
      sum += block_means[pick(rng)];
    means.push_back(sum / static_cast<double>(block_means.size()));
  }
  std::sort(means.begin(), means.end());
  const auto index = [&](double q) {
    const double pos = q * static_cast<double>(means.size() - 1);
    return static_cast<std::size_t>(std::round(pos));
  };
  return {means[index(0.025)], means[index(0.975)]};
}

BootstrapStatistic bootstrap_statistic(const std::vector<double> &values,
                                       int samples, unsigned int seed) {
  BootstrapStatistic out;
  out.samples = summarize_values(values);
  const auto ci = bootstrap_ci95(values, samples, seed);
  out.ci95_low = ci.first;
  out.ci95_high = ci.second;
  return out;
}

FitEstimate linear_fit(const std::vector<XYPoint> &points) {
  FitEstimate fit;
  fit.n = points.size();
  if (points.size() < 3) {
    fit.status = "insufficient_points";
    return fit;
  }
  double sx = 0.0;
  double sy = 0.0;
  double sxx = 0.0;
  double syy = 0.0;
  double sxy = 0.0;
  std::size_t n = 0;
  for (const auto &point : points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y))
      continue;
    sx += point.x;
    sy += point.y;
    sxx += point.x * point.x;
    syy += point.y * point.y;
    sxy += point.x * point.y;
    ++n;
  }
  fit.n = n;
  if (n < 3) {
    fit.status = "insufficient_points";
    return fit;
  }
  const double dn = static_cast<double>(n);
  const double denom = dn * sxx - sx * sx;
  if (!(std::abs(denom) > 0.0)) {
    fit.status = "singular_fit";
    return fit;
  }
  fit.slope = (dn * sxy - sx * sy) / denom;
  fit.intercept = (sy - fit.slope * sx) / dn;
  const double rden = (dn * sxx - sx * sx) * (dn * syy - sy * sy);
  fit.r2 = rden > 0.0 ? ((dn * sxy - sx * sy) * (dn * sxy - sx * sy)) / rden
                      : 0.0;
  fit.ok = true;
  fit.status = "ok";
  return fit;
}

BlockFitResult fit_with_blocks_impl(const std::vector<XYPoint> &points,
                                    double tmin, double tmax,
                                    const std::string &blocks_spec,
                                    int bootstrap_samples, unsigned int seed,
                                    bool log_transform) {
  BlockFitResult out;
  const auto selected = filter_points(points, tmin, tmax, log_transform);
  out.point = linear_fit(selected);
  if (!out.point.ok)
    return out;

  const auto blocks = partition_blocks(selected.size(), blocks_spec);
  out.block_count = blocks.size();
  std::vector<double> slopes;
  std::vector<double> intercepts;
  for (const auto &block : blocks) {
    if (block.begin >= block.end || block.end > selected.size())
      continue;
    std::vector<XYPoint> block_points(selected.begin() + block.begin,
                                      selected.begin() + block.end);
    const auto fit = linear_fit(block_points);
    if (fit.ok) {
      slopes.push_back(fit.slope);
      intercepts.push_back(fit.intercept);
    }
  }
  out.slope_stats = summarize_values(slopes);
  out.intercept_stats = summarize_values(intercepts);
  const auto slope_ci = bootstrap_ci95(slopes, bootstrap_samples, seed);
  out.slope_bootstrap_ci95_low = slope_ci.first;
  out.slope_bootstrap_ci95_high = slope_ci.second;
  const auto intercept_ci =
      bootstrap_ci95(intercepts, bootstrap_samples, seed + 7919U);
  out.intercept_bootstrap_ci95_low = intercept_ci.first;
  out.intercept_bootstrap_ci95_high = intercept_ci.second;
  return out;
}

BlockFitResult linear_fit_with_blocks(const std::vector<XYPoint> &points,
                                      double tmin, double tmax,
                                      const std::string &blocks_spec,
                                      int bootstrap_samples,
                                      unsigned int seed) {
  return fit_with_blocks_impl(points, tmin, tmax, blocks_spec,
                              bootstrap_samples, seed, false);
}

BlockFitResult loglog_fit_with_blocks(const std::vector<XYPoint> &points,
                                      double tmin, double tmax,
                                      const std::string &blocks_spec,
                                      int bootstrap_samples,
                                      unsigned int seed) {
  return fit_with_blocks_impl(points, tmin, tmax, blocks_spec,
                              bootstrap_samples, seed, true);
}

void export_uncertainty_outputs(const arguments &args, const info &out_info,
                                std::fstream &log) {
  if (args.uncertainty.mode == "off")
    return;

  std::map<std::string, std::size_t> sidecar_rows;
  SummaryCounters counters;

  if (args.cmap.flag_cmap) {
    ++counters.modules_requested;
    std::vector<SidecarRow> rows;
    add_temporal_rows(args, rows, "CMap", "contact_count_timeseries.csv",
                      {{"contact_count", {"contact_count", "contacts"}}},
                      true);
    add_entity_rows(rows, "CMap", "contact_residue_occupancy.csv",
                    {{"residue_occupancy", {"occupancy", "probability"}}});
    write_sidecar(out_info.get_cmap_uncertainty_filename(), rows);
    accumulate_counters(rows, counters);
    sidecar_rows[out_info.get_cmap_uncertainty_filename()] = rows.size();
  }

  if (args.qnative.flag_qnative) {
    ++counters.modules_requested;
    std::vector<SidecarRow> rows;
    add_temporal_rows(args, rows, "QNative", "qnative_timeseries.csv",
                      {{"Q", {"Q"}},
                       {"formed_native_contacts", {"formed_native_contacts"}}});
    add_entity_rows(rows, "QNative", "qnative_contact_occupancy.csv",
                    {{"native_contact_occupancy",
                      {"occupancy", "probability"}}});
    write_sidecar(out_info.get_qnative_uncertainty_filename(), rows);
    accumulate_counters(rows, counters);
    sidecar_rows[out_info.get_qnative_uncertainty_filename()] = rows.size();
  }

  if (args.dihedral.flag_dihedral) {
    ++counters.modules_requested;
    std::vector<SidecarRow> rows;
    add_entity_rows(rows, "Dihedral", "dihedral_rotamer_populations.csv",
                    {{"rotamer_occupancy", {"occupancy"}},
                     {"valid_samples", {"valid_samples"}}});
    add_entity_rows(rows, "Dihedral", "dihedral_ramachandran.csv",
                    {{"ramachandran_occupancy", {"occupancy"}},
                     {"valid_samples", {"valid_samples"}}});
    write_sidecar(out_info.get_dihedral_uncertainty_filename(), rows);
    accumulate_counters(rows, counters);
    sidecar_rows[out_info.get_dihedral_uncertainty_filename()] = rows.size();
  }

  if (args.msd.flag_msd) {
    ++counters.modules_requested;
    std::vector<SidecarRow> rows;
    add_temporal_rows(args, rows, "MSD", "msd_timeseries.dat",
                      {{"msd_A2", {"msd_A2"}}}, true);
    add_msd_fit_rows(args, rows);
    write_sidecar(out_info.get_msd_uncertainty_filename(), rows);
    accumulate_counters(rows, counters);
    sidecar_rows[out_info.get_msd_uncertainty_filename()] = rows.size();
  }

  if (args.s2.flag_s2) {
    ++counters.modules_requested;
    std::vector<SidecarRow> rows;
    add_record_value_rows(rows, "S2", "s2_order_parameters.csv",
                          {"residue", "vector_type"},
                          {{"plateau_C_raw", {"plateau_C_raw"}},
                           {"S2_clipped", {"S2_clipped", "S2"}},
                           {"tau_ps", {"tau_ps"}},
                           {"fit_R2", {"fit_R2"}}});
    write_sidecar(out_info.get_s2_uncertainty_filename(), rows);
    accumulate_counters(rows, counters);
    sidecar_rows[out_info.get_s2_uncertainty_filename()] = rows.size();
  }

  if (args.membrane.flag_membrane) {
    ++counters.modules_requested;
    std::vector<SidecarRow> rows;
    add_temporal_rows(args, rows, "Membrane", "membrane_apl_timeseries.csv",
                      {{"mean_apl_A2", {"mean_apl_A2"}}});
    add_temporal_rows(args, rows, "Membrane",
                      "membrane_thickness_timeseries.csv",
                      {{"mean_thickness_A",
                        {"mean_thickness_A", "thickness_A"}}});
    add_temporal_rows(args, rows, "Membrane", "membrane_interdigitation.dat",
                      {{"interdigitation", {"overlap_coefficient"}},
                       {"penetration_depth_A", {"penetration_depth_A"}}});
    add_record_value_rows(rows, "Membrane", "membrane_order_parameters.csv",
                          {"resname", "tail", "bond"},
                          {{"order_parameter", {"mean_S", "S"}},
                           {"order_parameter_std", {"std_S"}}});
    add_key_value_rows(rows, "Membrane", "membrane_summary_average.dat",
                       {{"replicate_mean_apl_A2", {"mean_apl_A2"}},
                        {"replicate_mean_thickness_A", {"mean_thickness_A"}},
                        {"replicate_mean_interdigitation",
                         {"mean_interdigitation"}}});
    write_sidecar(out_info.get_membrane_uncertainty_filename(), rows);
    accumulate_counters(rows, counters);
    sidecar_rows[out_info.get_membrane_uncertainty_filename()] = rows.size();
  }

  if (args.edecomp.flag_edecomp) {
    ++counters.modules_requested;
    std::vector<SidecarRow> rows;
    for (const auto &file :
         {"edecomp_mm_residue_ranking.csv", "edecomp_mm_atom_ranking.csv",
          "edecomp_mmgbsa_residue_ranking.csv",
          "edecomp_mmgbsa_atom_ranking.csv",
          "edecomp_mmpbsa_residue_ranking.csv",
          "edecomp_mmpbsa_atom_ranking.csv"}) {
      add_entity_rows(rows, "EDecomp", file,
                      {{"abs_mean", {"abs_mean"}},
                       {"mean_abs", {"mean_abs"}}});
    }
    write_sidecar(out_info.get_edecomp_uncertainty_filename(), rows);
    accumulate_counters(rows, counters);
    sidecar_rows[out_info.get_edecomp_uncertainty_filename()] = rows.size();
  }

  write_summary(args, out_info.get_uncertainty_summary_filename(),
                sidecar_rows, counters);
  TitanUI::log_kv(log, "Uncertainty Summary",
                  out_info.get_uncertainty_summary_filename());
}

} // namespace titan_uncertainty
