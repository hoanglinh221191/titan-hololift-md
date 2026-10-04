#include "titan_llps_cluster.h"
#include "titan_llps_profile_geometry.h"
#include "box_type.h"
#include "info.h"
#include "titan_ui.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;
constexpr double kEps = 1.0e-12;

double deg_to_rad(double v) { return v * kPi / 180.0; }

double box_volume_nm3(const box &Box) {
  const double a = Box.get_a();
  const double b = Box.get_b();
  const double c = Box.get_c();
  if (a <= 0.0 || b <= 0.0 || c <= 0.0)
    return 0.0;
  const double ca = std::cos(deg_to_rad(Box.get_alpha()));
  const double cb = std::cos(deg_to_rad(Box.get_beta()));
  const double cg = std::cos(deg_to_rad(Box.get_gamma()));
  const double metric =
      1.0 + 2.0 * ca * cb * cg - ca * ca - cb * cb - cg * cg;
  return a * b * c * std::sqrt(std::max(0.0, metric)) / 1000.0;
}

uint64_t choose2(uint64_t n) { return (n > 1) ? (n * (n - 1) / 2) : 0; }

double safe_partition(double dense, double dilute) {
  if (dense <= 0.0 && dilute <= 0.0)
    return 0.0;
  return (dense + kEps) / (dilute + kEps);
}

double transfer_proxy(double partition) {
  if (partition <= 0.0)
    return 0.0;
  return -std::log(partition);
}

std::string region_label(double center_nm, double core_limit_nm,
                         double interface_limit_nm) {
  if (center_nm <= core_limit_nm)
    return "core";
  if (center_nm <= interface_limit_nm)
    return "interface";
  return "dilute";
}

struct BlockStats {
  int blocks{0};
  double mean{0.0};
  double sem{0.0};
};

struct LinearFitResult {
  double slope{std::numeric_limits<double>::quiet_NaN()};
  double intercept{std::numeric_limits<double>::quiet_NaN()};
  double r2{0.0};
  int n{0};
};

LinearFitResult fit_line(const std::vector<std::pair<double, double>> &xy) {
  LinearFitResult out;
  out.n = static_cast<int>(xy.size());
  if (xy.size() < 3)
    return out;
  double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0, syy = 0.0;
  for (const auto &p : xy) {
    sx += p.first;
    sy += p.second;
    sxx += p.first * p.first;
    sxy += p.first * p.second;
    syy += p.second * p.second;
  }
  const double n = static_cast<double>(xy.size());
  const double denom = n * sxx - sx * sx;
  if (std::abs(denom) < kEps)
    return out;
  out.slope = (n * sxy - sx * sy) / denom;
  out.intercept = (sy - out.slope * sx) / n;
  const double ss_tot = syy - sy * sy / n;
  double ss_res = 0.0;
  for (const auto &p : xy) {
    const double yhat = out.slope * p.first + out.intercept;
    const double d = p.second - yhat;
    ss_res += d * d;
  }
  out.r2 = ss_tot > kEps ? std::max(0.0, 1.0 - ss_res / ss_tot) : 0.0;
  return out;
}

uint64_t box_key(int ix, int iy, int iz) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(ix)) << 42) ^
         (static_cast<uint64_t>(static_cast<uint32_t>(iy)) << 21) ^
         static_cast<uint64_t>(static_cast<uint32_t>(iz));
}

std::vector<double> select_scales(const std::vector<double> &candidates,
                                  double min_scale_nm) {
  std::vector<double> selected;
  selected.reserve(candidates.size());
  for (double scale : candidates) {
    if (scale >= min_scale_nm)
      selected.push_back(scale);
  }
  if (selected.size() >= 3)
    return selected;
  return candidates;
}

void compute_dense_neighbor_topology(
    const std::vector<std::array<double, 3>> &points_nm,
    const std::vector<int> &types, double cutoff_nm,
    LLPSPaperFractalFrame &row) {
  if (points_nm.size() < 2 || points_nm.size() != types.size()) {
    row.mean_local_valency = std::numeric_limits<double>::quiet_NaN();
    row.protein_rna_neighbor_fraction =
        std::numeric_limits<double>::quiet_NaN();
    return;
  }

  const double cutoff = std::max(0.1, cutoff_nm);
  const double cutoff2 = cutoff * cutoff;
  uint64_t neighbor_pairs = 0;
  uint64_t protein_rna_pairs = 0;
  for (std::size_t i = 0; i < points_nm.size(); ++i) {
    for (std::size_t j = i + 1; j < points_nm.size(); ++j) {
      double d2 = 0.0;
      for (int dim = 0; dim < 3; ++dim) {
        const double delta = points_nm[i][dim] - points_nm[j][dim];
        d2 += delta * delta;
      }
      if (d2 > cutoff2 + kEps)
        continue;
      ++neighbor_pairs;
      if ((types[i] == 1 && types[j] != 1) ||
          (types[i] != 1 && types[j] == 1))
        ++protein_rna_pairs;
    }
  }

  row.mean_local_valency =
      2.0 * static_cast<double>(neighbor_pairs) /
      static_cast<double>(points_nm.size());
  row.protein_rna_neighbor_fraction =
      neighbor_pairs > 0
          ? static_cast<double>(protein_rna_pairs) /
                static_cast<double>(neighbor_pairs)
          : std::numeric_limits<double>::quiet_NaN();
}

void compute_fractal_frame_from_points(
    const std::vector<std::array<double, 3>> &points_nm,
    const LLPSPaperFractalConfig &config, LLPSPaperFractalFrame &row,
    int frame, double time_ns,
    std::vector<LLPSPaperFractalScaleRow> *scale_rows) {
  row.frame = frame;
  row.time_ns = time_ns;
  row.dense_nodes = static_cast<int>(points_nm.size());
  row.invalid_reason = "insufficient_scales";
  if (points_nm.empty())
    return;

  std::array<double, 3> center = {0.0, 0.0, 0.0};
  std::array<double, 3> min_nm = {std::numeric_limits<double>::max(),
                                  std::numeric_limits<double>::max(),
                                  std::numeric_limits<double>::max()};
  std::array<double, 3> max_nm = {std::numeric_limits<double>::lowest(),
                                  std::numeric_limits<double>::lowest(),
                                  std::numeric_limits<double>::lowest()};
  for (const auto &p : points_nm) {
    for (int d = 0; d < 3; ++d) {
      center[d] += p[d];
      min_nm[d] = std::min(min_nm[d], p[d]);
      max_nm[d] = std::max(max_nm[d], p[d]);
    }
  }
  const double inv_n = 1.0 / static_cast<double>(points_nm.size());
  for (double &v : center)
    v *= inv_n;

  double r2_sum = 0.0;
  std::vector<double> distances;
  distances.reserve(points_nm.size());
  for (const auto &p : points_nm) {
    double r2 = 0.0;
    for (int d = 0; d < 3; ++d) {
      const double delta = p[d] - center[d];
      r2 += delta * delta;
    }
    r2_sum += r2;
    const double r = std::sqrt(r2);
    distances.push_back(r);
    row.dense_radius_nm = std::max(row.dense_radius_nm, r);
  }
  row.dense_rg_nm = std::sqrt(r2_sum * inv_n);
  if (row.dense_radius_nm <= kEps)
    return;

  const double max_fraction =
      std::clamp(config.max_scale_fraction, 0.25, 1.0);
  const std::vector<double> fractions = {0.25, 0.35, 0.50,
                                         0.70, 0.90, 1.00};
  std::vector<double> mass_candidates;
  for (double fraction : fractions) {
    if (fraction <= max_fraction || mass_candidates.size() < 3)
      mass_candidates.push_back(fraction * row.dense_radius_nm);
  }
  const auto mass_scales =
      select_scales(mass_candidates, std::max(0.1, config.min_scale_nm));
  std::vector<std::pair<double, double>> mass_fit_xy;
  for (double scale : mass_scales) {
    int count = 0;
    for (double r : distances) {
      if (r <= scale + kEps)
        ++count;
    }
    if (count <= 0)
      continue;
    const double log_scale = std::log(scale);
    const double log_count = std::log(static_cast<double>(count));
    mass_fit_xy.push_back({log_scale, log_count});
    if (scale_rows) {
      LLPSPaperFractalScaleRow scale_row;
      scale_row.frame = frame;
      scale_row.time_ns = time_ns;
      scale_row.method = "mass_radius";
      scale_row.scale_nm = scale;
      scale_row.value = static_cast<double>(count);
      scale_row.log_scale = log_scale;
      scale_row.log_value = log_count;
      scale_rows->push_back(scale_row);
    }
  }
  const LinearFitResult mass_fit = fit_line(mass_fit_xy);
  row.scale_count_mass_radius = mass_fit.n;
  row.df_mass_radius = mass_fit.slope;
  row.kf_mass_radius = std::isfinite(mass_fit.intercept)
                           ? std::exp(mass_fit.intercept)
                           : std::numeric_limits<double>::quiet_NaN();
  row.fit_r2_mass_radius = mass_fit.r2;

  std::array<double, 3> extent_nm = {max_nm[0] - min_nm[0],
                                     max_nm[1] - min_nm[1],
                                     max_nm[2] - min_nm[2]};
  const double max_extent =
      std::max(extent_nm[0], std::max(extent_nm[1], extent_nm[2]));
  if (max_extent <= kEps)
    return;

  const std::vector<int> divisions = {2, 3, 4};
  std::vector<double> box_candidates;
  for (int div : divisions) {
    const double scale = max_extent / static_cast<double>(div);
    const double fraction = scale / std::max(kEps, max_extent);
    if (fraction <= max_fraction || box_candidates.size() < 3)
      box_candidates.push_back(scale);
  }
  const auto box_scales =
      select_scales(box_candidates, std::max(0.1, config.min_scale_nm));
  std::vector<std::pair<double, double>> box_fit_xy;
  for (double scale : box_scales) {
    if (scale <= kEps)
      continue;
    int div = std::max(1, static_cast<int>(std::ceil(max_extent / scale)));
    std::unordered_set<uint64_t> occupied;
    occupied.reserve(points_nm.size());
    for (const auto &p : points_nm) {
      int idx[3] = {0, 0, 0};
      for (int d = 0; d < 3; ++d) {
        const double shifted = p[d] - min_nm[d];
        idx[d] = std::min(div - 1,
                          std::max(0, static_cast<int>(shifted / scale)));
      }
      occupied.insert(box_key(idx[0], idx[1], idx[2]));
    }
    if (occupied.empty())
      continue;
    const double log_inv_scale = std::log(1.0 / scale);
    const double log_count =
        std::log(static_cast<double>(occupied.size()));
    box_fit_xy.push_back({log_inv_scale, log_count});
    if (scale_rows) {
      LLPSPaperFractalScaleRow scale_row;
      scale_row.frame = frame;
      scale_row.time_ns = time_ns;
      scale_row.method = "box_count";
      scale_row.scale_nm = scale;
      scale_row.value = static_cast<double>(occupied.size());
      scale_row.log_scale = log_inv_scale;
      scale_row.log_value = log_count;
      scale_rows->push_back(scale_row);
    }
  }
  const LinearFitResult box_fit = fit_line(box_fit_xy);
  row.scale_count_box_count = box_fit.n;
  row.df_box_count = box_fit.slope;
  row.fit_r2_box_count = box_fit.r2;

  if (row.scale_count_mass_radius < 3 || row.scale_count_box_count < 3) {
    row.invalid_reason = "insufficient_scales";
    return;
  }
  if (!std::isfinite(row.df_mass_radius) || row.df_mass_radius < 0.5 ||
      row.df_mass_radius > 3.2) {
    row.invalid_reason = "df_out_of_range";
    return;
  }
  row.valid = true;
  row.invalid_reason = "";
}

template <typename Fn>
BlockStats block_reduce(const std::vector<LLPSPaperAccumulator::FrameRow> &rows,
                        int requested_blocks, Fn value_of) {
  BlockStats out;
  if (rows.empty())
    return out;
  int blocks = requested_blocks > 0 ? requested_blocks : 5;
  blocks = std::min<int>(blocks, static_cast<int>(rows.size()));
  if (rows.size() < 20 && requested_blocks <= 0)
    blocks = std::min<int>(blocks, std::max<int>(1, rows.size() / 2));
  blocks = std::max(1, blocks);
  out.blocks = blocks;

  std::vector<double> block_means;
  block_means.reserve(static_cast<std::size_t>(blocks));
  for (int b = 0; b < blocks; ++b) {
    const std::size_t begin =
        static_cast<std::size_t>((static_cast<uint64_t>(b) * rows.size()) /
                                 static_cast<uint64_t>(blocks));
    const std::size_t end =
        static_cast<std::size_t>((static_cast<uint64_t>(b + 1) * rows.size()) /
                                 static_cast<uint64_t>(blocks));
    if (end <= begin)
      continue;
    double sum = 0.0;
    for (std::size_t i = begin; i < end; ++i)
      sum += value_of(rows[i]);
    block_means.push_back(sum / static_cast<double>(end - begin));
  }
  if (block_means.empty())
    return out;
  out.mean = std::accumulate(block_means.begin(), block_means.end(), 0.0) /
             static_cast<double>(block_means.size());
  if (block_means.size() > 1) {
    double var = 0.0;
    for (double v : block_means) {
      const double d = v - out.mean;
      var += d * d;
    }
    var /= static_cast<double>(block_means.size() - 1);
    out.sem =
        std::sqrt(var) / std::sqrt(static_cast<double>(block_means.size()));
  }
  return out;
}

std::string format_paper_double(double value) {
  if (!std::isfinite(value))
    return "nan";
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(8) << value;
  return oss.str();
}

template <typename Fn>
BlockStats fractal_block_reduce(
    const std::vector<LLPSPaperFractalFrame> &rows, int requested_blocks,
    Fn value_of) {
  std::vector<const LLPSPaperFractalFrame *> valid_rows;
  valid_rows.reserve(rows.size());
  for (const auto &row : rows) {
    if (row.valid)
      valid_rows.push_back(&row);
  }

  BlockStats out;
  if (valid_rows.empty()) {
    out.mean = std::numeric_limits<double>::quiet_NaN();
    out.sem = std::numeric_limits<double>::quiet_NaN();
    return out;
  }

  int blocks = requested_blocks > 0 ? requested_blocks : 5;
  blocks = std::min<int>(blocks, static_cast<int>(valid_rows.size()));
  if (valid_rows.size() < 20 && requested_blocks <= 0)
    blocks = std::min<int>(
        blocks, std::max<int>(1, static_cast<int>(valid_rows.size() / 2)));
  blocks = std::max(1, blocks);
  out.blocks = blocks;

  std::vector<double> block_means;
  block_means.reserve(static_cast<std::size_t>(blocks));
  for (int b = 0; b < blocks; ++b) {
    const std::size_t begin = static_cast<std::size_t>(
        (static_cast<uint64_t>(b) * valid_rows.size()) /
        static_cast<uint64_t>(blocks));
    const std::size_t end = static_cast<std::size_t>(
        (static_cast<uint64_t>(b + 1) * valid_rows.size()) /
        static_cast<uint64_t>(blocks));
    if (end <= begin)
      continue;
    double sum = 0.0;
    for (std::size_t i = begin; i < end; ++i)
      sum += value_of(*valid_rows[i]);
    block_means.push_back(sum / static_cast<double>(end - begin));
  }
  if (block_means.empty()) {
    out.mean = std::numeric_limits<double>::quiet_NaN();
    out.sem = std::numeric_limits<double>::quiet_NaN();
    return out;
  }
  out.mean = std::accumulate(block_means.begin(), block_means.end(), 0.0) /
             static_cast<double>(block_means.size());
  out.sem = 0.0;
  if (block_means.size() > 1) {
    double var = 0.0;
    for (double v : block_means) {
      const double d = v - out.mean;
      var += d * d;
    }
    var /= static_cast<double>(block_means.size() - 1);
    out.sem =
        std::sqrt(var) / std::sqrt(static_cast<double>(block_means.size()));
  }
  return out;
}

double drift_slope_per_ns(
    const std::vector<LLPSPaperAccumulator::FrameRow> &rows) {
  if (rows.size() < 2)
    return 0.0;
  double sx = 0.0;
  double sy = 0.0;
  double sxx = 0.0;
  double sxy = 0.0;
  for (const auto &row : rows) {
    sx += row.time_ns;
    sy += row.dense_fraction;
    sxx += row.time_ns * row.time_ns;
    sxy += row.time_ns * row.dense_fraction;
  }
  const double n = static_cast<double>(rows.size());
  const double denom = n * sxx - sx * sx;
  if (std::abs(denom) < kEps)
    return 0.0;
  return (n * sxy - sx * sy) / denom;
}

} // namespace

void LLPSPaperAccumulator::RunningScalar::add(double v) {
  ++n;
  sum += v;
  sum_sq += v * v;
}

double LLPSPaperAccumulator::RunningScalar::mean() const {
  return n > 0 ? sum / static_cast<double>(n) : 0.0;
}

double LLPSPaperAccumulator::RunningScalar::sem() const {
  if (n < 2)
    return 0.0;
  const double m = mean();
  const double var =
      std::max(0.0, sum_sq / static_cast<double>(n) - m * m);
  return std::sqrt(var / static_cast<double>(n));
}

void LLPSPaperFractalAccumulator::configure(
    const LLPSPaperFractalConfig &config) {
  config_ = config;
  config_.min_scale_nm = std::max(0.1, config_.min_scale_nm);
  config_.max_scale_fraction =
      std::clamp(config_.max_scale_fraction, 0.25, 1.0);
  frames_.clear();
  scales_.clear();
}

void LLPSPaperFractalAccumulator::process_frame(
    double time_ns, int frame, const std::vector<LLPSNode> &nodes,
    const std::vector<bool> &is_dense, int giant_cluster_id, const box &Box) {
  if (!config_.enabled)
    return;

  LLPSPaperFractalFrame row;
  row.frame = frame;
  row.time_ns = time_ns;
  if (nodes.empty() || is_dense.size() != nodes.size() ||
      giant_cluster_id <= 0) {
    row.invalid_reason = "no_dense_cluster";
    frames_.push_back(row);
    return;
  }

  std::vector<std::array<double, 3>> points_nm;
  points_nm.reserve(nodes.size());
  std::vector<int> dense_types;
  dense_types.reserve(nodes.size());
  const bool use_mic = BoxMatrix::valid_box(Box);
  std::optional<TitanExactPeriodicBox> bmat;
  if (use_mic)
    bmat.emplace(Box);
  std::size_t first_dense = nodes.size();
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (!is_dense[i])
      continue;
    if (first_dense == nodes.size())
      first_dense = i;
    ++row.dense_nodes;
    if (nodes[i].type == 1)
      ++row.dense_protein_nodes;
    else
      ++row.dense_rna_nodes;
    dense_types.push_back(nodes[i].type);
    points_nm.push_back({nodes[i].x / 10.0, nodes[i].y / 10.0, nodes[i].z / 10.0});
  }
  if (use_mic) {
    std::vector<std::array<double,3>> points_a, lifted;
    for (const auto &p : points_nm) points_a.push_back({10*p[0],10*p[1],10*p[2]});
    if (!titan_lift_radius_graph(*bmat, points_a, config_.connectivity_cutoff_angstrom, lifted)) {
      row.invalid_reason = "periodic_embedding_not_finite";
      frames_.push_back(row);return;
    }
    points_nm.clear();
    for (const auto &p : lifted) points_nm.push_back({p[0]/10.,p[1]/10.,p[2]/10.});
  }

  if (row.dense_nodes < 8) {
    row.invalid_reason = "dense_nodes_lt_8";
    frames_.push_back(row);
    return;
  }

  compute_dense_neighbor_topology(points_nm, dense_types, config_.min_scale_nm,
                                  row);
  compute_fractal_frame_from_points(points_nm, config_, row, frame, time_ns,
                                    &scales_);
  frames_.push_back(row);
}

void LLPSPaperFractalAccumulator::export_outputs(
    const info &info_output, std::fstream &log, int requested_blocks) const {
  if (!config_.enabled)
    return;

  const std::string timeseries_file =
      info_output.get_llps_paper_fractal_timeseries_filename();
  const std::string scales_file =
      info_output.get_llps_paper_fractal_scales_filename();
  const std::string summary_file =
      info_output.get_llps_paper_fractal_summary_filename();
  const std::string readme_file =
      info_output.get_llps_paper_fractal_readme_filename();

  {
    std::ofstream out(timeseries_file);
    out << "frame,time_ns,dense_nodes,dense_protein_nodes,dense_rna_nodes,"
           "df_mass_radius,df_box_count,kf_mass_radius,fit_r2_mass_radius,"
           "fit_r2_box_count,scale_count_mass_radius,scale_count_box_count,"
           "dense_radius_nm,dense_rg_nm,mean_local_valency,"
           "protein_rna_neighbor_fraction,valid,invalid_reason\n";
    for (const auto &row : frames_) {
      out << row.frame << "," << format_paper_double(row.time_ns) << ","
          << row.dense_nodes << "," << row.dense_protein_nodes << ","
          << row.dense_rna_nodes << ","
          << format_paper_double(row.df_mass_radius) << ","
          << format_paper_double(row.df_box_count) << ","
          << format_paper_double(row.kf_mass_radius) << ","
          << format_paper_double(row.fit_r2_mass_radius) << ","
          << format_paper_double(row.fit_r2_box_count) << ","
          << row.scale_count_mass_radius << "," << row.scale_count_box_count
          << "," << format_paper_double(row.dense_radius_nm) << ","
          << format_paper_double(row.dense_rg_nm) << ","
          << format_paper_double(row.mean_local_valency) << ","
          << format_paper_double(row.protein_rna_neighbor_fraction) << ","
          << (row.valid ? "yes" : "no") << "," << row.invalid_reason << "\n";
    }
  }

  {
    std::ofstream out(scales_file);
    out << "frame,time_ns,method,scale_nm,value,log_scale,log_value\n";
    for (const auto &row : scales_) {
      out << row.frame << "," << format_paper_double(row.time_ns) << ","
          << row.method << "," << format_paper_double(row.scale_nm) << ","
          << format_paper_double(row.value) << ","
          << format_paper_double(row.log_scale) << ","
          << format_paper_double(row.log_value) << "\n";
    }
  }

  const auto df_mass =
      fractal_block_reduce(frames_, requested_blocks,
                           [](const LLPSPaperFractalFrame &r) {
                             return r.df_mass_radius;
                           });
  const auto df_box =
      fractal_block_reduce(frames_, requested_blocks,
                           [](const LLPSPaperFractalFrame &r) {
                             return r.df_box_count;
                           });
  const auto kf_mass =
      fractal_block_reduce(frames_, requested_blocks,
                           [](const LLPSPaperFractalFrame &r) {
                             return r.kf_mass_radius;
                           });
  const auto dense_radius =
      fractal_block_reduce(frames_, requested_blocks,
                           [](const LLPSPaperFractalFrame &r) {
                             return r.dense_radius_nm;
                           });
  const auto dense_rg =
      fractal_block_reduce(frames_, requested_blocks,
                           [](const LLPSPaperFractalFrame &r) {
                             return r.dense_rg_nm;
                           });
  const auto fit_r2_mass =
      fractal_block_reduce(frames_, requested_blocks,
                           [](const LLPSPaperFractalFrame &r) {
                             return r.fit_r2_mass_radius;
                           });
  const auto fit_r2_box =
      fractal_block_reduce(frames_, requested_blocks,
                           [](const LLPSPaperFractalFrame &r) {
                             return r.fit_r2_box_count;
                           });
  const std::size_t frames_valid = static_cast<std::size_t>(
      std::count_if(frames_.begin(), frames_.end(),
                    [](const LLPSPaperFractalFrame &row) {
                      return row.valid;
                    }));
  {
    std::ofstream out(summary_file);
    out << "enabled: yes\n";
    out << "frames_total: " << frames_.size() << "\n";
    out << "frames_valid: " << frames_valid << "\n";
    out << "frames_invalid: " << (frames_.size() - frames_valid) << "\n";
    out << "mean_df_mass_radius: " << format_paper_double(df_mass.mean)
        << "\n";
    out << "sem_df_mass_radius: " << format_paper_double(df_mass.sem) << "\n";
    out << "mean_df_box_count: " << format_paper_double(df_box.mean) << "\n";
    out << "sem_df_box_count: " << format_paper_double(df_box.sem) << "\n";
    out << "mean_kf_mass_radius: " << format_paper_double(kf_mass.mean)
        << "\n";
    out << "sem_kf_mass_radius: " << format_paper_double(kf_mass.sem) << "\n";
    out << "mean_dense_radius_nm: "
        << format_paper_double(dense_radius.mean) << "\n";
    out << "sem_dense_radius_nm: " << format_paper_double(dense_radius.sem)
        << "\n";
    out << "mean_dense_rg_nm: " << format_paper_double(dense_rg.mean)
        << "\n";
    out << "sem_dense_rg_nm: " << format_paper_double(dense_rg.sem) << "\n";
    out << "mean_fit_r2_mass_radius: "
        << format_paper_double(fit_r2_mass.mean) << "\n";
    out << "mean_fit_r2_box_count: " << format_paper_double(fit_r2_box.mean)
        << "\n";
    out << "interpretation: morphology_descriptor_not_phase_boundary\n";
  }

  {
    std::ofstream out(readme_file);
    out << "# TITAN LLPS Paper Fractal Outputs\n\n";
    out << "These files report dense-phase fractal morphology descriptors "
           "from bead/residue occupancy in the LLPS DBSCAN dense phase.\n\n";
    out << "Primary citation files:\n\n";
    out << "- `llps_paper_fractal_timeseries.csv`: per-frame fractal "
           "descriptors.\n";
    out << "- `llps_paper_fractal_scales.csv`: scale-by-scale audit data for "
           "log-log fits.\n";
    out << "- `llps_paper_fractal_summary.dat`: block-averaged summary "
           "statistics.\n\n";
    out << "The descriptors quantify dense-phase morphology. They are not "
           "standalone evidence for LLPS, binodal boundaries, or spinodal "
           "behavior.\n";
  }

  TitanUI::log_kv(log, timeseries_file, "emitted");
  TitanUI::log_kv(log, scales_file, "emitted");
  TitanUI::log_kv(log, summary_file, "emitted");
  TitanUI::log_kv(log, readme_file, "emitted");
}

#ifdef TITAN_LLPS_TESTING
LLPSPaperFractalFrame LLPSPaperFractalAccumulator::debug_fit_points(
    const std::vector<std::array<double, 3>> &points_nm) {
  LLPSPaperFractalConfig config;
  config.enabled = true;
  config.min_scale_nm = 0.1;
  config.max_scale_fraction = 1.0;
  LLPSPaperFractalFrame row;
  compute_fractal_frame_from_points(points_nm, config, row, 0, 0.0, nullptr);
  return row;
}
#endif

void LLPSPaperAccumulator::configure(const LLPSPaperConfig &config) {
  config_ = config;
  if (!std::isfinite(config_.profile_bin_nm) || config_.profile_bin_nm <= 0 ||
      config_.slab_axis < 0 || config_.slab_axis > 2 ||
      !std::isfinite(config_.connectivity_cutoff_angstrom) || config_.connectivity_cutoff_angstrom <= 0)
    throw std::invalid_argument("Invalid LLPS paper profile geometry parameters");
  config_.profile_bin_nm = std::max(0.01, config_.profile_bin_nm);
  if (config_.geometry != "droplet" && config_.geometry != "slab")
    config_.geometry = "auto";
  LLPSPaperFractalConfig fractal_config;
  fractal_config.enabled = config_.fractal_enabled;
  fractal_config.connectivity_cutoff_angstrom = config_.connectivity_cutoff_angstrom;
  fractal_.configure(fractal_config);
  frames_.clear();
  contact_rows_.clear();
  profile_bins_.clear();
  skipped_no_dense_ = 0;
  skipped_bad_box_ = 0;
  geometry_failures_.clear();
  observed_geometry_ = "none";
}

void LLPSPaperAccumulator::process_frame(
    double time_ps, int frame_idx, const std::vector<LLPSNode> &nodes,
    const std::vector<bool> &is_dense, int giant_cluster_id, const box &Box) {
  if (!config_.enabled)
    return;
  const double volume_nm3 = box_volume_nm3(Box);
  if (volume_nm3 <= 0.0 || nodes.empty() || is_dense.size() != nodes.size()) {
    fractal_.process_frame(time_ps / 1000.0, frame_idx, nodes, is_dense,
                           giant_cluster_id, Box);
    ++skipped_bad_box_;
    return;
  }


  FrameRow row;
  row.time_ns = time_ps / 1000.0;
  row.frame = frame_idx;
  row.box_volume_nm3 = volume_nm3;

  int first_dense = -1;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const bool protein = nodes[i].type == 1;
    row.total_protein_nodes += protein ? 1 : 0;
    row.total_rna_nodes += protein ? 0 : 1;
    if (is_dense[i]) {
      if (first_dense < 0)
        first_dense = static_cast<int>(i);
      row.dense_protein_nodes += protein ? 1 : 0;
      row.dense_rna_nodes += protein ? 0 : 1;
    }
  }
  row.dilute_protein_nodes = row.total_protein_nodes - row.dense_protein_nodes;
  row.dilute_rna_nodes = row.total_rna_nodes - row.dense_rna_nodes;
  row.dense_fraction =
      static_cast<double>(row.dense_protein_nodes + row.dense_rna_nodes) /
      static_cast<double>(nodes.size());
  row.dilute_fraction = 1.0 - row.dense_fraction;

  if (first_dense < 0 || giant_cluster_id <= 0) {
    ++skipped_no_dense_;
    row.geometry = "none";
    row.dilute_volume_nm3 = volume_nm3;
    if (row.dilute_volume_nm3 > 0.0) {
      row.dilute_protein_conc =
          static_cast<double>(row.dilute_protein_nodes) / row.dilute_volume_nm3;
      row.dilute_rna_conc =
          static_cast<double>(row.dilute_rna_nodes) / row.dilute_volume_nm3;
    }
    fractal_.process_frame(row.time_ns, frame_idx, nodes, is_dense,
                           giant_cluster_id, Box);
    frames_.push_back(row);
    return;
  }

  std::vector<std::array<double,3>> points;
  points.reserve(nodes.size());
  for (const auto &node : nodes) points.push_back({node.x,node.y,node.z});
  const auto profile = TitanLLPSProfileGeometry::build(Box, points, is_dense,
      config_.geometry, config_.slab_axis, config_.connectivity_cutoff_angstrom,
      config_.profile_bin_nm);
  if (!profile.valid) {
    if (profile.reason == "triclinic_requires_explicit_geometry")
      throw std::invalid_argument("LLPS paper profiles require explicit -llps-paper-geometry droplet or slab for triclinic cells");
    ++geometry_failures_[profile.reason];
    fractal_.process_frame(row.time_ns, frame_idx, nodes, is_dense, giant_cluster_id, Box);
    return;
  }
  const std::string geometry = profile.geometry;
  if (observed_geometry_ != "none" && observed_geometry_ != geometry) {
    ++geometry_failures_["geometry_switch_requires_separate_profile"];return;
  }
  observed_geometry_ = geometry;row.geometry = geometry;
  const double radius_nm = profile.radius_nm;
  const auto &dense_distance_nm = profile.distance_nm;
  row.radius_nm = radius_nm;
  row.dense_volume_nm3 = profile.dense_envelope_nm3;
  // These envelope and interface quantities remain geometric proxies.
  row.interface_width_nm = std::max(config_.profile_bin_nm,
      (geometry == "slab" ? .5 : .2) * radius_nm);
  row.dilute_volume_nm3 = std::max(kEps, volume_nm3 - row.dense_volume_nm3);
  row.dense_volume_nm3 = std::max(kEps, row.dense_volume_nm3);
  row.dense_protein_conc =
      static_cast<double>(row.dense_protein_nodes) / row.dense_volume_nm3;
  row.dilute_protein_conc =
      static_cast<double>(row.dilute_protein_nodes) / row.dilute_volume_nm3;
  row.dense_rna_conc =
      static_cast<double>(row.dense_rna_nodes) / row.dense_volume_nm3;
  row.dilute_rna_conc =
      static_cast<double>(row.dilute_rna_nodes) / row.dilute_volume_nm3;
  row.protein_partition =
      safe_partition(row.dense_protein_conc, row.dilute_protein_conc);
  row.rna_partition = safe_partition(row.dense_rna_conc, row.dilute_rna_conc);
  row.protein_transfer_proxy = transfer_proxy(row.protein_partition);
  row.rna_transfer_proxy = transfer_proxy(row.rna_partition);
  row.dense_protein_rna_ratio =
      static_cast<double>(row.dense_protein_nodes) /
      std::max(1.0, static_cast<double>(row.dense_rna_nodes));
  const double global_rna_fraction =
      static_cast<double>(row.total_rna_nodes) /
      std::max(1.0, static_cast<double>(row.total_protein_nodes +
                                        row.total_rna_nodes));
  const double dense_rna_fraction =
      static_cast<double>(row.dense_rna_nodes) /
      std::max(1.0, static_cast<double>(row.dense_protein_nodes +
                                        row.dense_rna_nodes));
  row.rna_enrichment = safe_partition(dense_rna_fraction, global_rna_fraction);

  uint64_t core_p = 0;
  uint64_t core_r = 0;
  uint64_t interface_p = 0;
  uint64_t interface_r = 0;
  const double core_limit = 0.60 * radius_nm;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (!is_dense[i])
      continue;
    const bool is_core = dense_distance_nm[i] <= core_limit;
    if (is_core) {
      if (nodes[i].type == 1)
        ++core_p;
      else
        ++core_r;
    } else {
      if (nodes[i].type == 1)
        ++interface_p;
      else
        ++interface_r;
    }
  }
  const double dense_nodes =
      static_cast<double>(row.dense_protein_nodes + row.dense_rna_nodes);
  row.core_fraction = static_cast<double>(core_p + core_r) /
                      std::max(1.0, dense_nodes);
  row.interface_fraction =
      static_cast<double>(interface_p + interface_r) / std::max(1.0, dense_nodes);

  ContactFrameRow contact;
  contact.time_ns = row.time_ns;
  contact.frame = frame_idx;
  const uint64_t dense_p = static_cast<uint64_t>(row.dense_protein_nodes);
  const uint64_t dense_r = static_cast<uint64_t>(row.dense_rna_nodes);
  contact.pp = choose2(dense_p);
  contact.prna = dense_p * dense_r;
  contact.rr = choose2(dense_r);
  contact.pp_core = choose2(core_p);
  contact.prna_core = core_p * core_r;
  contact.rr_core = choose2(core_r);
  contact.pp_interface = choose2(interface_p);
  contact.prna_interface = interface_p * interface_r;
  contact.rr_interface = choose2(interface_r);
  contact_rows_.push_back(contact);

  const double profile_limit_nm = profile.limit_nm;
  const std::size_t needed_bins = static_cast<std::size_t>(
      std::ceil(profile_limit_nm / config_.profile_bin_nm));
  if (profile_bins_.size() < needed_bins) profile_bins_.resize(needed_bins);
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (dense_distance_nm[i] > profile_limit_nm) continue;
    const std::size_t bin = std::min<std::size_t>(needed_bins - 1,
        static_cast<std::size_t>(dense_distance_nm[i] / config_.profile_bin_nm));
    auto &dst = profile_bins_[bin];
    if (nodes[i].type == 1) {
      dst.protein_count += 1.0;dst.sticker_count += 1.0;
      if (is_dense[i]) dst.dense_protein_count += 1.0;
    } else {
      dst.rna_count += 1.0;
      if (is_dense[i]) dst.dense_rna_count += 1.0;
    }
  }
  for (std::size_t bin = 0; bin < needed_bins; ++bin) {
    auto &dst = profile_bins_[bin];
    const double r0 = bin * config_.profile_bin_nm;
    const double upper = std::min(r0 + config_.profile_bin_nm, profile_limit_nm);
    const double volume = profile.shell_volume(r0,upper);
    if (volume > 0) {
      dst.volume_nm3_sum += volume;dst.sampled_upper_nm_sum += upper;++dst.samples;
    }
  }

  fractal_.process_frame(row.time_ns, frame_idx, nodes, is_dense,
                         giant_cluster_id, Box);
  frames_.push_back(row);
}

void LLPSPaperAccumulator::export_outputs(const info &info_output,
                                          std::fstream &log) const {
  if (!config_.enabled)
    return;
  const std::string geometry =
      (observed_geometry_ == "slab") ? "slab" : "droplet";
  const std::string profile_file =
      (geometry == "slab") ? info_output.get_llps_slab_profiles_filename()
                            : info_output.get_llps_radial_profiles_filename();
  const int requested_blocks = config_.requested_blocks;

  const auto protein_dense =
      block_reduce(frames_, requested_blocks,
                   [](const FrameRow &r) { return r.dense_protein_conc; });
  const auto protein_dilute =
      block_reduce(frames_, requested_blocks,
                   [](const FrameRow &r) { return r.dilute_protein_conc; });
  const auto rna_dense =
      block_reduce(frames_, requested_blocks,
                   [](const FrameRow &r) { return r.dense_rna_conc; });
  const auto rna_dilute =
      block_reduce(frames_, requested_blocks,
                   [](const FrameRow &r) { return r.dilute_rna_conc; });
  const auto protein_partition =
      block_reduce(frames_, requested_blocks,
                   [](const FrameRow &r) { return r.protein_partition; });
  const auto rna_partition =
      block_reduce(frames_, requested_blocks,
                   [](const FrameRow &r) { return r.rna_partition; });
  const auto dense_ratio =
      block_reduce(frames_, requested_blocks,
                   [](const FrameRow &r) { return r.dense_protein_rna_ratio; });
  const auto rna_enrichment =
      block_reduce(frames_, requested_blocks,
                   [](const FrameRow &r) { return r.rna_enrichment; });
  const auto radius =
      block_reduce(frames_, requested_blocks,
                   [](const FrameRow &r) { return r.radius_nm; });
  const auto dense_fraction =
      block_reduce(frames_, requested_blocks,
                   [](const FrameRow &r) { return r.dense_fraction; });

  {
    std::ofstream out(info_output.get_llps_phase_partition_summary_filename());
    out << std::fixed << std::setprecision(8);
    out << "# TITAN LLPS paper phase/partition summary\n";
    out << "# Concentrations are node-count densities in nm^-3; transfer "
           "proxies are -ln(K) in kBT-like units.\n";
    out << "Metric Mean SEM Blocks Unit\n";
    auto emit = [&](const std::string &name, const BlockStats &s,
                    const std::string &unit) {
      out << name << " " << s.mean << " " << s.sem << " " << s.blocks << " "
          << unit << "\n";
    };
    emit("protein_dense_concentration", protein_dense, "nodes_nm^-3");
    emit("protein_dilute_concentration", protein_dilute, "nodes_nm^-3");
    emit("rna_dense_concentration", rna_dense, "nodes_nm^-3");
    emit("rna_dilute_concentration", rna_dilute, "nodes_nm^-3");
    emit("protein_partition_coefficient", protein_partition, "unitless");
    emit("rna_partition_coefficient", rna_partition, "unitless");
    emit("protein_transfer_free_energy_proxy",
         block_reduce(frames_, requested_blocks,
                      [](const FrameRow &r) {
                        return r.protein_transfer_proxy;
                      }),
         "kBT_proxy");
    emit("rna_transfer_free_energy_proxy",
         block_reduce(frames_, requested_blocks,
                      [](const FrameRow &r) { return r.rna_transfer_proxy; }),
         "kBT_proxy");
    emit("dense_protein_rna_ratio", dense_ratio, "node_ratio");
    emit("rna_enrichment", rna_enrichment, "unitless");
    emit("dense_fraction", dense_fraction, "fraction");
    emit("condensate_radius_or_halfwidth", radius, "nm");
  }

  {
    std::ofstream out(profile_file);
    out << std::fixed << std::setprecision(8);
    out << "BinStart_nm,BinEnd_nm,BinCenter_nm,Geometry,Frames,"
           "ShellVolumeMean_nm3,ProteinDensity,RNADensity,StickerDensity,"
           "WaterDensity,IonDensity,DenseProteinDensity,DenseRNADensity,"
           "RegionLabel\n";
    const double core_limit = 0.60 * radius.mean;
    const double interface_limit = std::max(core_limit, radius.mean);
    for (std::size_t i = 0; i < profile_bins_.size(); ++i) {
      const auto &bin = profile_bins_[i];
      const double r0 = static_cast<double>(i) * config_.profile_bin_nm;
      if (bin.samples == 0) continue;
      const double r1 = bin.sampled_upper_nm_sum / bin.samples;
      const double center = 0.5 * (r0 + r1);
      const double volume =
          bin.samples > 0 ? bin.volume_nm3_sum / static_cast<double>(bin.samples)
                          : 0.0;
      const double norm = std::max(kEps, bin.volume_nm3_sum);
      out << r0 << "," << r1 << "," << center << "," << geometry << ","
          << bin.samples << "," << volume << ","
          << bin.protein_count / norm << "," << bin.rna_count / norm << ","
          << bin.sticker_count / norm << "," << bin.water_count / norm << ","
          << bin.ion_count / norm << "," << bin.dense_protein_count / norm
          << "," << bin.dense_rna_count / norm << ","
          << region_label(center, core_limit, interface_limit) << "\n";
    }
  }

  {
    std::ofstream out(info_output.get_llps_composition_timeseries_filename());
    out << std::fixed << std::setprecision(8);
    out << "Time_ns,Frame,Geometry,TotalProteinNodes,TotalRNANodes,"
           "DenseProteinNodes,DenseRNANodes,DiluteProteinNodes,"
           "DiluteRNANodes,DenseFraction,CoreFraction,InterfaceFraction,"
           "DiluteFraction,DenseProteinConc,DenseRNAConc,DiluteProteinConc,"
           "DiluteRNAConc,ProteinPartition,RNAPartition,"
           "DenseProteinRNARatio,RNAEnrichment,RadiusOrHalfWidth_nm,"
           "InterfaceWidth_nm\n";
    for (const auto &r : frames_) {
      out << r.time_ns << "," << r.frame << "," << r.geometry << ","
          << r.total_protein_nodes << "," << r.total_rna_nodes << ","
          << r.dense_protein_nodes << "," << r.dense_rna_nodes << ","
          << r.dilute_protein_nodes << "," << r.dilute_rna_nodes << ","
          << r.dense_fraction << "," << r.core_fraction << ","
          << r.interface_fraction << "," << r.dilute_fraction << ","
          << r.dense_protein_conc << "," << r.dense_rna_conc << ","
          << r.dilute_protein_conc << "," << r.dilute_rna_conc << ","
          << r.protein_partition << "," << r.rna_partition << ","
          << r.dense_protein_rna_ratio << "," << r.rna_enrichment << ","
          << r.radius_nm << "," << r.interface_width_nm << "\n";
    }
  }

  {
    std::ofstream out(info_output.get_llps_contact_taxonomy_filename());
    out << "Time_ns,Frame,Metric,PP,PRNA,RR,PP_Core,PRNA_Core,RR_Core,"
           "PP_Interface,PRNA_Interface,RR_Interface\n";
    out << std::fixed << std::setprecision(8);
    for (const auto &r : contact_rows_) {
      out << r.time_ns << "," << r.frame << "," << r.metric << "," << r.pp
          << "," << r.prna << "," << r.rr << "," << r.pp_core << ","
          << r.prna_core << "," << r.rr_core << "," << r.pp_interface << ","
          << r.prna_interface << "," << r.rr_interface << "\n";
    }
  }

  {
    RunningScalar pp;
    RunningScalar prna;
    RunningScalar rr;
    for (const auto &r : contact_rows_) {
      pp.add(static_cast<double>(r.pp));
      prna.add(static_cast<double>(r.prna));
      rr.add(static_cast<double>(r.rr));
    }
    std::ofstream out(info_output.get_llps_contact_taxonomy_summary_filename());
    out << std::fixed << std::setprecision(8);
    out << "# Contact taxonomy currently reports dense-cluster co-membership "
           "pair proxies from DBSCAN labels.\n";
    out << "Metric Mean SEM Frames\n";
    out << "PP_cluster_comembership_proxy " << pp.mean() << " " << pp.sem()
        << " " << pp.n << "\n";
    out << "PRNA_cluster_comembership_proxy " << prna.mean() << " "
        << prna.sem() << " " << prna.n << "\n";
    out << "RR_cluster_comembership_proxy " << rr.mean() << " " << rr.sem()
        << " " << rr.n << "\n";
  }

  const double slope = drift_slope_per_ns(frames_);
  const double span_ns = frames_.empty()
                             ? 0.0
                             : (frames_.back().time_ns - frames_.front().time_ns);
  const bool insufficient_blocks = dense_fraction.blocks < 2;
  const bool drift_warning = std::abs(slope * span_ns) > 0.10;
  const bool short_warning = frames_.size() < 10;
  {
    std::ofstream out(info_output.get_llps_convergence_report_filename());
    out << std::fixed << std::setprecision(8);
    out << "frames_used " << frames_.size() << "\n";
    out << "frames_skipped_no_stable_dense_phase " << skipped_no_dense_ << "\n";
    out << "frames_skipped_bad_box " << skipped_bad_box_ << "\n";
    for (const auto &[reason,count] : geometry_failures_)
      out << "frames_skipped_" << reason << " " << count << "\n";
    out << "blocks " << dense_fraction.blocks << "\n";
    out << "dense_fraction_mean " << dense_fraction.mean << "\n";
    out << "dense_fraction_sem " << dense_fraction.sem << "\n";
    out << "dense_fraction_drift_slope_per_ns " << slope << "\n";
    out << "time_span_ns " << span_ns << "\n";
    out << "insufficient_blocks " << (insufficient_blocks ? "yes" : "no")
        << "\n";
    out << "short_trajectory_warning " << (short_warning ? "yes" : "no")
        << "\n";
    out << "equilibration_drift_warning " << (drift_warning ? "yes" : "no")
        << "\n";
    out << "stable_dense_phase "
        << (skipped_no_dense_ < frames_.size() ? "yes" : "no") << "\n";
  }

  {
    std::ofstream out(info_output.get_llps_paper_summary_md_filename());
    out << "# TITAN LLPS Paper Summary\n\n";
    out << "- Frames used: " << frames_.size() << "\n";
    out << "- Geometry: " << geometry << "\n";
    out << "- Profiles use exact lattice geometry; slab normal is reciprocal axis " << config_.slab_axis << " (0=a, 1=b, 2=c).\n";
    out << "- Radial shells stop at the conservative inscribed domain. Slab terminal bins are clipped to half the face height. BinEnd is the sampled mean upper edge across frames.\n";
    out << "- Dense/dilute envelope concentrations and interface widths are geometric proxies, not an automatically validated binodal.\n";
    for (const auto &[reason,count] : geometry_failures_)
      out << "- Skipped geometry " << reason << ": " << count << " frames.\n";
    out << "- Protein dense concentration: " << protein_dense.mean << " +/- "
        << protein_dense.sem << " nodes/nm^3\n";
    out << "- RNA dense concentration: " << rna_dense.mean << " +/- "
        << rna_dense.sem << " nodes/nm^3\n";
    out << "- Protein partition coefficient: " << protein_partition.mean
        << " +/- " << protein_partition.sem << "\n";
    out << "- RNA partition coefficient: " << rna_partition.mean << " +/- "
        << rna_partition.sem << "\n";
    out << "- Dense protein:RNA ratio: " << dense_ratio.mean << " +/- "
        << dense_ratio.sem << "\n";
    out << "- RNA enrichment: " << rna_enrichment.mean << " +/- "
        << rna_enrichment.sem << "\n";
    out << "- Dense fraction drift slope: " << slope << " per ns\n";
    if (skipped_no_dense_ >= frames_.size())
      out << "- Status: no_stable_dense_phase\n";
    else if (insufficient_blocks || drift_warning || short_warning)
      out << "- Status: convergence_warning\n";
    else
      out << "- Status: ok\n";
  }

  {
    std::ofstream out(info_output.get_llps_paper_summary_json_filename());
    out << std::fixed << std::setprecision(8);
    out << "{\n";
    out << "  \"frames_used\": " << frames_.size() << ",\n";
    out << "  \"frames_skipped_no_stable_dense_phase\": " << skipped_no_dense_
        << ",\n";
    out << "  \"geometry\": \"" << geometry << "\",\n";
    out << "  \"protein_dense_concentration_mean\": " << protein_dense.mean
        << ",\n";
    out << "  \"protein_dense_concentration_sem\": " << protein_dense.sem
        << ",\n";
    out << "  \"rna_dense_concentration_mean\": " << rna_dense.mean << ",\n";
    out << "  \"rna_dense_concentration_sem\": " << rna_dense.sem << ",\n";
    out << "  \"protein_partition_mean\": " << protein_partition.mean << ",\n";
    out << "  \"protein_partition_sem\": " << protein_partition.sem << ",\n";
    out << "  \"rna_partition_mean\": " << rna_partition.mean << ",\n";
    out << "  \"rna_partition_sem\": " << rna_partition.sem << ",\n";
    out << "  \"dense_protein_rna_ratio_mean\": " << dense_ratio.mean << ",\n";
    out << "  \"rna_enrichment_mean\": " << rna_enrichment.mean << ",\n";
    out << "  \"dense_fraction_mean\": " << dense_fraction.mean << ",\n";
    out << "  \"dense_fraction_sem\": " << dense_fraction.sem << ",\n";
    out << "  \"dense_fraction_drift_slope_per_ns\": " << slope << ",\n";
    out << "  \"insufficient_blocks\": "
        << (insufficient_blocks ? "true" : "false") << ",\n";
    out << "  \"equilibration_drift_warning\": "
        << (drift_warning ? "true" : "false") << ",\n";
    out << "  \"short_trajectory_warning\": "
        << (short_warning ? "true" : "false") << "\n";
    out << "}\n";
  }

  TitanUI::log_section(log, "LLPS Paper Output Audit");
  TitanUI::log_kv(log, info_output.get_llps_phase_partition_summary_filename(),
                  "emitted");
  TitanUI::log_kv(log, profile_file, "emitted");
  TitanUI::log_kv(log, info_output.get_llps_composition_timeseries_filename(),
                  "emitted");
  TitanUI::log_kv(log, info_output.get_llps_contact_taxonomy_filename(),
                  "emitted");
  TitanUI::log_kv(log, info_output.get_llps_contact_taxonomy_summary_filename(),
                  "emitted");
  TitanUI::log_kv(log, info_output.get_llps_convergence_report_filename(),
                  "emitted");
  TitanUI::log_kv(log, info_output.get_llps_paper_summary_md_filename(),
                  "emitted");
  TitanUI::log_kv(log, info_output.get_llps_paper_summary_json_filename(),
                  "emitted");

  fractal_.export_outputs(info_output, log, config_.requested_blocks);
}

void TitanLLPSCluster::configure_paper(const LLPSPaperConfig &config) {
  auto checked = config;
  checked.connectivity_cutoff_angstrom = eps_;
  paper_accumulator_.configure(checked);
}

void TitanLLPSCluster::export_paper_outputs(const info &info_output,
                                            std::fstream &log) const {
  paper_accumulator_.export_outputs(info_output, log);
}
