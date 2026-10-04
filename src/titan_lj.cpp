#include "titan_lj.h"

#include "titan_exact_periodic_box.h"
#include "titan_neighbor_grid.h"
#include "titan_executor.h"
#include "titan_simd_dispatch.h"
#include "titan_ui.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <thread>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kFeatureEps = 1.0e-12;
constexpr std::array<const char *, TitanLJAnalyzer::kTicaFeatureDim>
    kLJFeatureLabels{{
        "rho_bulk_nm^-3",
        "largest_cluster_fraction",
        "largest_cluster_rg_over_cutoff",
        "largest_cluster_rmax_over_cutoff",
        "cluster_count_fraction",
        "mean_cn",
        "std_cn",
        "isolated_fraction",
        "dense_fraction",
        "mean_dense_q4",
        "mean_dense_q6",
        "mean_dense_q4_bar",
        "mean_dense_q6_bar",
        "solid_like_fraction",
        "largest_solid_cluster_fraction",
        "fcc_like_fraction",
        "hcp_like_fraction",
        "shell1_enrichment",
        "shell2_enrichment",
        "shell3_enrichment",
    }};

constexpr std::array<double, 13> kFactorials{{
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5040.0,
    40320.0,
    362880.0,
    3628800.0,
    39916800.0,
    479001600.0,
}};

double assoc_legendre_compat(int l, int m, double x) {
  if (l < 0 || m < 0 || m > l)
    return 0.0;

  x = std::clamp(x, -1.0, 1.0);
  const double one_minus_x2 = std::max(0.0, 1.0 - x * x);

  double p_mm = 1.0;
  if (m > 0) {
    double fact = 1.0;
    const double root = std::sqrt(one_minus_x2);
    for (int i = 1; i <= m; ++i) {
      p_mm *= -fact * root;
      fact += 2.0;
    }
  }

  if (l == m)
    return p_mm;

  double p_mmp1 = x * static_cast<double>(2 * m + 1) * p_mm;
  if (l == m + 1)
    return p_mmp1;

  double p_lm2 = p_mm;
  double p_lm1 = p_mmp1;
  for (int n = m + 2; n <= l; ++n) {
    const double p_lm =
        ((static_cast<double>(2 * n - 1) * x * p_lm1) -
         (static_cast<double>(n + m - 1) * p_lm2)) /
        static_cast<double>(n - m);
    p_lm2 = p_lm1;
    p_lm1 = p_lm;
  }
  return p_lm1;
}

template <int L> double spherical_harmonic_norm(int m) {
  return std::sqrt(((2.0 * static_cast<double>(L) + 1.0) / (4.0 * kPi)) *
                   (kFactorials[static_cast<std::size_t>(L - m)] /
                    kFactorials[static_cast<std::size_t>(L + m)]));
}

template <int L>
void accumulate_qlm(double dx, double dy, double dz,
                    std::array<std::complex<double>, L + 1> &qlm) {
  const double r2 = dx * dx + dy * dy + dz * dz;
  if (r2 <= kFeatureEps)
    return;

  const double inv_r = 1.0 / std::sqrt(r2);
  const double cos_theta =
      std::clamp(dz * inv_r, -1.0 + 1.0e-12, 1.0 - 1.0e-12);
  const double phi = std::atan2(dy, dx);

  for (int m = 0; m <= L; ++m) {
    const double P_lm = assoc_legendre_compat(L, m, cos_theta);
    const double norm = spherical_harmonic_norm<L>(m);
    const double phase = static_cast<double>(m) * phi;
    qlm[static_cast<std::size_t>(m)] +=
        norm * P_lm * std::complex<double>(std::cos(phase), std::sin(phase));
  }
}

template <int L>
double finalize_ql(std::array<std::complex<double>, L + 1> &qlm,
                   int neighbor_count) {
  if (neighbor_count <= 0)
    return 0.0;

  const double inv_n = 1.0 / static_cast<double>(neighbor_count);
  double power = 0.0;
  for (int m = 0; m <= L; ++m) {
    qlm[static_cast<std::size_t>(m)] *= inv_n;
    const double mag2 = std::norm(qlm[static_cast<std::size_t>(m)]);
    power += (m == 0) ? mag2 : 2.0 * mag2;
  }
  return std::sqrt((4.0 * kPi / (2.0 * static_cast<double>(L) + 1.0)) * power);
}

template <int L>
double ql_from_normalized(const std::array<std::complex<double>, L + 1> &qlm) {
  double power = 0.0;
  for (int m = 0; m <= L; ++m) {
    const double mag2 = std::norm(qlm[static_cast<std::size_t>(m)]);
    power += (m == 0) ? mag2 : 2.0 * mag2;
  }
  return std::sqrt((4.0 * kPi / (2.0 * static_cast<double>(L) + 1.0)) * power);
}

double sum_avx2(const double *data, std::size_t n) {
  return titan_simd_dispatch::sum_doubles(data, n);
}

double centered_sq_sum_avx2(const double *x, const double *y, const double *z,
                            std::size_t n, double cx, double cy, double cz) {
  return titan_simd_dispatch::centered_sq_sum_xyz(x, y, z, n, cx, cy, cz);
}

std::size_t effective_worker_count(std::size_t requested_threads,
                                   std::size_t work_items) noexcept {
  if (requested_threads == 0 || work_items < 512)
    return 1;
  return std::max<std::size_t>(
      1, std::min<std::size_t>(requested_threads, work_items / 256));
}

double determinant3x3(const TitanExactPeriodicBox &box_matrix) {
  return box_matrix.m[0][0] *
             (box_matrix.m[1][1] * box_matrix.m[2][2] -
              box_matrix.m[1][2] * box_matrix.m[2][1]) -
         box_matrix.m[0][1] *
             (box_matrix.m[1][0] * box_matrix.m[2][2] -
              box_matrix.m[1][2] * box_matrix.m[2][0]) +
         box_matrix.m[0][2] *
             (box_matrix.m[1][0] * box_matrix.m[2][1] -
              box_matrix.m[1][1] * box_matrix.m[2][0]);
}

double shell_volume_a3(double r_inner, double r_outer) {
  return (4.0 / 3.0) * kPi *
         (r_outer * r_outer * r_outer - r_inner * r_inner * r_inner);
}

template <typename Accessor>
std::pair<double, double> mean_sem(std::size_t n, Accessor accessor) {
  if (n == 0)
    return {0.0, 0.0};

  double mean = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    mean += accessor(i);
  mean /= static_cast<double>(n);

  if (n < 2)
    return {mean, 0.0};

  double sum_sq = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double diff = accessor(i) - mean;
    sum_sq += diff * diff;
  }
  const double variance = sum_sq / static_cast<double>(n - 1);
  const double sem = std::sqrt(std::max(0.0, variance) / static_cast<double>(n));
  return {mean, sem};
}

} // namespace

bool TitanLJAnalyzer::has_periodic_box(const box &box_value) noexcept {
  return box_value.get_a() > 0.0 && box_value.get_b() > 0.0 &&
         box_value.get_c() > 0.0;
}

void TitanLJAnalyzer::configure(const LjArgs &args) {
  requested_ = args.flag_lj;
  cutoff_a_ = args.lj_cluster_cutoff;
  cutoff_sq_a_ = cutoff_a_ * cutoff_a_;
  min_cluster_size_ = std::max(1, args.lj_min_cluster_size);
  rho_bin_a_ = args.lj_rho_bin;
  rho_max_input_a_ = args.lj_rho_max;
  solid_q6_threshold_ = args.lj_solid_q6;
  solid_min_cn_ = std::max(1, args.lj_solid_min_cn);
  crystal_tol_ = args.lj_crystal_tol;
  enabled_ = requested_ && initialized_ && atom_count_ > 0;
}

void TitanLJAnalyzer::prepare_atom_buffers(std::size_t atom_count) {
  parent_.resize(atom_count);
  rank_.resize(atom_count);
  size_.resize(atom_count);
  atom_degree_.resize(atom_count);
  root_seen_.resize(atom_count);
  component_root_.resize(atom_count);
  atom_to_local_.resize(atom_count);
}

void TitanLJAnalyzer::reset_trajectory_state() {
  frames_.clear();
  feature_history_.clear();
  edges_.clear();
  cluster_members_.clear();
  cluster_graph_.clear();
  bfs_queue_.clear();
  visited_.clear();
  unwrap_x_.clear();
  unwrap_y_.clear();
  unwrap_z_.clear();
  rho_cluster_mean_nm3_.clear();
  rho_cluster_m2_nm3_.clear();
  rho_cluster_count_.clear();
  rho_all_mean_nm3_.clear();
  rho_all_m2_nm3_.clear();
  rho_all_count_.clear();
  radial_bin_count_ = 0;
  fixed_rho_max_a_ = 0.0;
  profile_layout_ready_ = false;
  latest_feature_vector_.resize(0);
  q4_values_.clear();
  q6_values_.clear();
  solid_like_.clear();
}

void TitanLJAnalyzer::init(const Multi_MOL &mol1, const arguments &args,
                           std::fstream &log) {
  if (initialized_) {
    enabled_ = requested_ && atom_count_ > 0;
    return;
  }

  const std::size_t info_atoms = mol1.info.size();
  const std::size_t frame_atoms =
      mol1.m_mol.empty() ? 0 : mol1.m_mol.front().size();
  atom_count_ = (info_atoms > 0 && frame_atoms > 0)
                    ? std::min(info_atoms, frame_atoms)
                    : std::max(info_atoms, frame_atoms);

  initialized_ = true;
  enabled_ = requested_ && atom_count_ > 0;
  if (atom_count_ == 0) {
    TitanUI::print_warning(
        "LJ analysis: no atoms were found in molecule 1. Module is disabled.");
    if (log.is_open())
      TitanUI::log_kv(log, "LJ Atoms", "0");
    return;
  }

  prepare_atom_buffers(atom_count_);
  reset_trajectory_state();
  latest_feature_vector_ = Eigen::VectorXd::Zero(kTicaFeatureDim);
  if (!mol1.m_mol.empty())
    ensure_profile_layout(mol1.m_mol.front());

  if (log.is_open()) {
    TitanUI::log_kv(log, "LJ Atoms", std::to_string(atom_count_));
    TitanUI::log_kv(log, "LJ Cluster Cutoff",
                    std::to_string(cutoff_a_) + " A");
    TitanUI::log_kv(log, "LJ Min Cluster Size",
                    std::to_string(min_cluster_size_));
    TitanUI::log_kv(log, "LJ rho(r) Bin", std::to_string(rho_bin_a_) + " A");
    TitanUI::log_kv(log, "LJ rho(r) Max",
                    rho_max_input_a_ > 0.0
                        ? std::to_string(rho_max_input_a_) + " A"
                        : "auto");
    TitanUI::log_kv(log, "LJ Solid-like q6 Threshold",
                    std::to_string(solid_q6_threshold_));
    TitanUI::log_kv(log, "LJ Solid-like Min CN",
                    std::to_string(solid_min_cn_));
    TitanUI::log_kv(log, "LJ Crystal Classification Tol",
                    std::to_string(crystal_tol_));
    TitanUI::log_kv(log, "LJ Phase-2 Feature Dim",
                    std::to_string(kTicaFeatureDim));
  }
}

void TitanLJAnalyzer::ensure_profile_layout(const MOL &frame) {
  if (profile_layout_ready_)
    return;

  if (rho_max_input_a_ > 0.0) {
    fixed_rho_max_a_ = rho_max_input_a_;
  } else if (has_periodic_box(frame.Box)) {
    fixed_rho_max_a_ =
        TitanExactPeriodicBox(frame.Box).contained_radius();
  } else {
    const auto xs = frame.get_pos_x();
    const auto ys = frame.get_pos_y();
    const auto zs = frame.get_pos_z();
    if (!xs.empty() && !ys.empty() && !zs.empty()) {
      const auto [xmin, xmax] = std::minmax_element(xs.begin(), xs.end());
      const auto [ymin, ymax] = std::minmax_element(ys.begin(), ys.end());
      const auto [zmin, zmax] = std::minmax_element(zs.begin(), zs.end());
      const double span = std::max({*xmax - *xmin, *ymax - *ymin, *zmax - *zmin});
      fixed_rho_max_a_ = 0.5 * span;
    }
  }

  fixed_rho_max_a_ = std::max(fixed_rho_max_a_, rho_bin_a_);
  radial_bin_count_ = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::ceil(fixed_rho_max_a_ / rho_bin_a_)));
  rho_cluster_mean_nm3_.assign(radial_bin_count_, 0.0);
  rho_cluster_m2_nm3_.assign(radial_bin_count_, 0.0);
  rho_cluster_count_.assign(radial_bin_count_, 0);
  rho_all_mean_nm3_.assign(radial_bin_count_, 0.0);
  rho_all_m2_nm3_.assign(radial_bin_count_, 0.0);
  rho_all_count_.assign(radial_bin_count_, 0);
  profile_layout_ready_ = true;
}

void TitanLJAnalyzer::reset_union_find(std::size_t atom_count) {
  for (std::size_t i = 0; i < atom_count; ++i) {
    parent_[i] = static_cast<int>(i);
    rank_[i] = 0;
    size_[i] = 1;
    atom_degree_[i] = 0;
    root_seen_[i] = 0;
  }
}

int TitanLJAnalyzer::find_root(int atom_index) {
  int root = atom_index;
  while (parent_[static_cast<std::size_t>(root)] != root) {
    root = parent_[static_cast<std::size_t>(root)];
  }
  while (parent_[static_cast<std::size_t>(atom_index)] != atom_index) {
    const int next = parent_[static_cast<std::size_t>(atom_index)];
    parent_[static_cast<std::size_t>(atom_index)] = root;
    atom_index = next;
  }
  return root;
}

void TitanLJAnalyzer::unite_atoms(int lhs, int rhs) {
  lhs = find_root(lhs);
  rhs = find_root(rhs);
  if (lhs == rhs)
    return;
  if (rank_[static_cast<std::size_t>(lhs)] < rank_[static_cast<std::size_t>(rhs)])
    std::swap(lhs, rhs);
  parent_[static_cast<std::size_t>(rhs)] = lhs;
  size_[static_cast<std::size_t>(lhs)] += size_[static_cast<std::size_t>(rhs)];
  if (rank_[static_cast<std::size_t>(lhs)] ==
      rank_[static_cast<std::size_t>(rhs)]) {
    ++rank_[static_cast<std::size_t>(lhs)];
  }
}

void TitanLJAnalyzer::build_contact_graph(const MOL &frame, std::size_t atom_count,
                                          bool periodic) {
  edges_.clear();
  const auto xs = frame.get_pos_x();
  const auto ys = frame.get_pos_y();
  const auto zs = frame.get_pos_z();
  if (atom_count == 0 || xs.empty() || ys.empty() || zs.empty())
    return;

  double lx = frame.Box.get_a();
  double ly = frame.Box.get_b();
  double lz = frame.Box.get_c();
  double shift_x = 0.0;
  double shift_y = 0.0;
  double shift_z = 0.0;

  if (!periodic) {
    const auto [xmin, xmax] =
        std::minmax_element(xs.begin(), xs.begin() + atom_count);
    const auto [ymin, ymax] =
        std::minmax_element(ys.begin(), ys.begin() + atom_count);
    const auto [zmin, zmax] =
        std::minmax_element(zs.begin(), zs.begin() + atom_count);
    shift_x = -(*xmin) + cutoff_a_;
    shift_y = -(*ymin) + cutoff_a_;
    shift_z = -(*zmin) + cutoff_a_;
    lx = (*xmax - *xmin) + 2.0 * cutoff_a_;
    ly = (*ymax - *ymin) + 2.0 * cutoff_a_;
    lz = (*zmax - *zmin) + 2.0 * cutoff_a_;
  }

  titan_neighbor::OrthorhombicGridSpec spec;
  spec.cutoff = cutoff_a_;
  spec.len_x = lx;
  spec.len_y = ly;
  spec.len_z = lz;
  spec.shift_x = shift_x;
  spec.shift_y = shift_y;
  spec.shift_z = shift_z;
  spec.periodic = periodic;

  std::optional<TitanExactPeriodicBox> box_matrix_storage;
  if (periodic)
    box_matrix_storage.emplace(frame.Box);
  const TitanExactPeriodicBox *box_matrix =
      box_matrix_storage.has_value() ? &box_matrix_storage.value() : nullptr;
  if(box_matrix) {spec.len_x=box_matrix->heights[0];spec.len_y=box_matrix->heights[1];spec.len_z=box_matrix->heights[2];}
  const double cutoff_sq = cutoff_a_ * cutoff_a_;

  neighbor_grid_.for_each_pair(
      atom_count, spec,
      [&](std::size_t idx, double &x, double &y, double &z) {
        x = xs[idx];
        y = ys[idx];
        z = zs[idx];
        if(box_matrix) box_matrix->grid_coordinate(x,y,z);
      },
      [&](std::size_t idx) { return idx < atom_count; },
      [&](std::size_t lhs, std::size_t rhs) {
        double dx = xs[lhs] - xs[rhs];
        double dy = ys[lhs] - ys[rhs];
        double dz = zs[lhs] - zs[rhs];
        if (box_matrix != nullptr)
          box_matrix->apply_mic(dx, dy, dz);
        if (dx * dx + dy * dy + dz * dz > cutoff_sq)
          return;
        const int ilhs = static_cast<int>(lhs);
        const int irhs = static_cast<int>(rhs);
        unite_atoms(ilhs, irhs);
        ++atom_degree_[lhs];
        ++atom_degree_[rhs];
        edges_.emplace_back(ilhs, irhs);
      });
}

int TitanLJAnalyzer::find_largest_cluster_root(std::size_t atom_count,
                                               int &n_clusters_out) {
  n_clusters_out = 0;
  int largest_root = -1;
  int largest_size = 0;

  for (std::size_t i = 0; i < atom_count; ++i) {
    const int root = component_root_[i];
    if (root < 0 || root_seen_[static_cast<std::size_t>(root)] != 0)
      continue;

    const int component_size = size_[static_cast<std::size_t>(root)];
    root_seen_[static_cast<std::size_t>(root)] = 1;
    if (component_size >= min_cluster_size_)
      ++n_clusters_out;
    if (component_size > largest_size) {
      largest_size = component_size;
      largest_root = root;
    }
  }

  return largest_root;
}

void TitanLJAnalyzer::build_largest_cluster_members(std::size_t atom_count,
                                                    int largest_root) {
  cluster_members_.clear();
  std::fill(atom_to_local_.begin(), atom_to_local_.begin() + atom_count, -1);
  if (largest_root < 0)
    return;

  for (std::size_t i = 0; i < atom_count; ++i) {
    if (component_root_[i] != largest_root)
      continue;
    atom_to_local_[i] = static_cast<int>(cluster_members_.size());
    cluster_members_.push_back(static_cast<int>(i));
  }
}

void TitanLJAnalyzer::unwrap_largest_cluster(const MOL &frame, bool periodic) {
  const std::size_t cluster_size = cluster_members_.size();
  unwrap_x_.assign(cluster_size, 0.0);
  unwrap_y_.assign(cluster_size, 0.0);
  unwrap_z_.assign(cluster_size, 0.0);
  if (cluster_size == 0)
    return;

  if(periodic) {
    const TitanExactPeriodicBox cell(frame.Box);
    std::vector<std::array<double,3>> points,lifted;
    for(int atom:cluster_members_)points.push_back({frame.pos.x[atom],frame.pos.y[atom],frame.pos.z[atom]});
    const bool valid=titan_lift_radius_graph(cell,points,cutoff_a_,lifted);
    for(std::size_t i=0;i<cluster_size;++i) {
      unwrap_x_[i]=valid?lifted[i][0]:std::numeric_limits<double>::quiet_NaN();
      unwrap_y_[i]=valid?lifted[i][1]:std::numeric_limits<double>::quiet_NaN();
      unwrap_z_[i]=valid?lifted[i][2]:std::numeric_limits<double>::quiet_NaN();
    }
    return;
  }

  std::vector<std::pair<int, int>> local_edges;
  local_edges.reserve(edges_.size());
  for (const auto &[lhs, rhs] : edges_) {
    if (lhs < 0 || rhs < 0)
      continue;
    if (static_cast<std::size_t>(lhs) >= atom_to_local_.size() ||
        static_cast<std::size_t>(rhs) >= atom_to_local_.size()) {
      continue;
    }
    const int local_lhs = atom_to_local_[static_cast<std::size_t>(lhs)];
    const int local_rhs = atom_to_local_[static_cast<std::size_t>(rhs)];
    if (local_lhs < 0 || local_rhs < 0)
      continue;
    local_edges.emplace_back(local_lhs, local_rhs);
  }
  cluster_graph_ = titan_sparse::CSRTopology::from_undirected_edges(
      static_cast<int>(cluster_size), local_edges);

  visited_.assign(cluster_size, 0);
  bfs_queue_.clear();
  bfs_queue_.reserve(cluster_size);

  const auto xs = frame.get_pos_x();
  const auto ys = frame.get_pos_y();
  const auto zs = frame.get_pos_z();
  const int ref_atom = cluster_members_.front();
  unwrap_x_[0] = xs[static_cast<std::size_t>(ref_atom)];
  unwrap_y_[0] = ys[static_cast<std::size_t>(ref_atom)];
  unwrap_z_[0] = zs[static_cast<std::size_t>(ref_atom)];
  visited_[0] = 1;
  bfs_queue_.push_back(0);

  std::optional<TitanExactPeriodicBox> box_matrix_storage;
  if (periodic)
    box_matrix_storage.emplace(frame.Box);
  const TitanExactPeriodicBox *box_matrix =
      box_matrix_storage.has_value() ? &box_matrix_storage.value() : nullptr;

  for (std::size_t head = 0; head < bfs_queue_.size(); ++head) {
    const int local_u = bfs_queue_[head];
    const int global_u = cluster_members_[static_cast<std::size_t>(local_u)];
    for (int idx = cluster_graph_.row_begin(local_u);
         idx < cluster_graph_.row_end(local_u); ++idx) {
      const int local_v = cluster_graph_.col_index(idx);
      if (local_v < 0 || visited_[static_cast<std::size_t>(local_v)] != 0)
        continue;

      const int global_v = cluster_members_[static_cast<std::size_t>(local_v)];
      double dx = xs[static_cast<std::size_t>(global_v)] -
                  xs[static_cast<std::size_t>(global_u)];
      double dy = ys[static_cast<std::size_t>(global_v)] -
                  ys[static_cast<std::size_t>(global_u)];
      double dz = zs[static_cast<std::size_t>(global_v)] -
                  zs[static_cast<std::size_t>(global_u)];
      if (box_matrix != nullptr)
        box_matrix->apply_mic(dx, dy, dz);

      unwrap_x_[static_cast<std::size_t>(local_v)] =
          unwrap_x_[static_cast<std::size_t>(local_u)] + dx;
      unwrap_y_[static_cast<std::size_t>(local_v)] =
          unwrap_y_[static_cast<std::size_t>(local_u)] + dy;
      unwrap_z_[static_cast<std::size_t>(local_v)] =
          unwrap_z_[static_cast<std::size_t>(local_u)] + dz;
      visited_[static_cast<std::size_t>(local_v)] = 1;
      bfs_queue_.push_back(local_v);
    }
  }

  for (std::size_t local_idx = 0; local_idx < cluster_size; ++local_idx) {
    if (visited_[local_idx] != 0)
      continue;

    const int global_idx = cluster_members_[local_idx];
    double dx = xs[static_cast<std::size_t>(global_idx)] -
                xs[static_cast<std::size_t>(ref_atom)];
    double dy = ys[static_cast<std::size_t>(global_idx)] -
                ys[static_cast<std::size_t>(ref_atom)];
    double dz = zs[static_cast<std::size_t>(global_idx)] -
                zs[static_cast<std::size_t>(ref_atom)];
    if (box_matrix != nullptr)
      box_matrix->apply_mic(dx, dy, dz);

    unwrap_x_[local_idx] = unwrap_x_[0] + dx;
    unwrap_y_[local_idx] = unwrap_y_[0] + dy;
    unwrap_z_[local_idx] = unwrap_z_[0] + dz;
  }
}

TitanLJAnalyzer::ClusterGeometry
TitanLJAnalyzer::compute_cluster_geometry(TitanExecutor &executor) const {
  ClusterGeometry geometry;
  const std::size_t cluster_size = cluster_members_.size();
  if (cluster_size == 0)
    return geometry;
  if(!std::isfinite(unwrap_x_[0])) {geometry.centroid_x=geometry.centroid_y=geometry.centroid_z=geometry.rg_a=geometry.rmax_a=std::numeric_limits<double>::quiet_NaN();return geometry;}

  geometry.centroid_x = sum_avx2(unwrap_x_.data(), cluster_size) /
                        static_cast<double>(cluster_size);
  geometry.centroid_y = sum_avx2(unwrap_y_.data(), cluster_size) /
                        static_cast<double>(cluster_size);
  geometry.centroid_z = sum_avx2(unwrap_z_.data(), cluster_size) /
                        static_cast<double>(cluster_size);

  const std::size_t worker_count =
      effective_worker_count(executor.capacity(), cluster_size);
  if (worker_count <= 1) {
    const double rg_sq = centered_sq_sum_avx2(
        unwrap_x_.data(), unwrap_y_.data(), unwrap_z_.data(), cluster_size,
        geometry.centroid_x, geometry.centroid_y, geometry.centroid_z);
    geometry.rg_a = std::sqrt(rg_sq / static_cast<double>(cluster_size));
    double max_r2 = 0.0;
    for (std::size_t i = 0; i < cluster_size; ++i) {
      const double dx = unwrap_x_[i] - geometry.centroid_x;
      const double dy = unwrap_y_[i] - geometry.centroid_y;
      const double dz = unwrap_z_[i] - geometry.centroid_z;
      max_r2 = std::max(max_r2, dx * dx + dy * dy + dz * dz);
    }
    geometry.rmax_a = std::sqrt(max_r2);
    return geometry;
  }

  std::vector<double> partial_rg_sq(worker_count, 0.0);
  std::vector<double> partial_rmax_sq(worker_count, 0.0);
  executor.parallel_for(
      cluster_size, worker_count, 1,
      [&](std::size_t worker, std::size_t begin, std::size_t end) {
        double rg_sq = centered_sq_sum_avx2(
            unwrap_x_.data() + begin, unwrap_y_.data() + begin,
            unwrap_z_.data() + begin, end - begin, geometry.centroid_x,
            geometry.centroid_y, geometry.centroid_z);
        double max_r2 = 0.0;
        for (std::size_t i = begin; i < end; ++i) {
          const double dx = unwrap_x_[i] - geometry.centroid_x;
          const double dy = unwrap_y_[i] - geometry.centroid_y;
          const double dz = unwrap_z_[i] - geometry.centroid_z;
          max_r2 = std::max(max_r2, dx * dx + dy * dy + dz * dz);
        }
        partial_rg_sq[worker] = rg_sq;
        partial_rmax_sq[worker] = max_r2;
      });

  const double total_rg_sq =
      std::accumulate(partial_rg_sq.begin(), partial_rg_sq.end(), 0.0);
  geometry.rg_a = std::sqrt(total_rg_sq / static_cast<double>(cluster_size));
  geometry.rmax_a =
      std::sqrt(*std::max_element(partial_rmax_sq.begin(), partial_rmax_sq.end()));
  return geometry;
}

TitanLJAnalyzer::LocalOrderSummary TitanLJAnalyzer::compute_local_order_metrics(
    const MOL &frame, std::size_t atom_count, bool periodic,
    TitanExecutor &executor) {
  LocalOrderSummary summary;
  if (atom_count == 0)
    return summary;

  const titan_sparse::CSRTopology graph =
      titan_sparse::CSRTopology::from_undirected_edges(
          static_cast<int>(atom_count), edges_);

  q4_values_.assign(atom_count, 0.0);
  q6_values_.assign(atom_count, 0.0);
  solid_like_.assign(atom_count, 0);
  std::vector<double> q4_bar_values(atom_count, 0.0);
  std::vector<double> q6_bar_values(atom_count, 0.0);
  std::vector<std::array<std::complex<double>, 5>> q4m_local(atom_count);
  std::vector<std::array<std::complex<double>, 7>> q6m_local(atom_count);

  const auto xs = frame.get_pos_x();
  const auto ys = frame.get_pos_y();
  const auto zs = frame.get_pos_z();

  std::optional<TitanExactPeriodicBox> box_matrix_storage;
  if (periodic)
    box_matrix_storage.emplace(frame.Box);
  const TitanExactPeriodicBox *box_matrix =
      box_matrix_storage.has_value() ? &box_matrix_storage.value() : nullptr;

  const std::size_t worker_count =
      effective_worker_count(executor.capacity(), atom_count);
  executor.parallel_for(
      atom_count, worker_count, 1,
      [&](std::size_t /*worker*/, std::size_t begin, std::size_t end) {
        for (std::size_t atom = begin; atom < end; ++atom) {
          const int neighbor_count = graph.degree(static_cast<int>(atom));
          if (neighbor_count <= 0)
            continue;

          std::array<std::complex<double>, 5> q4m{};
          std::array<std::complex<double>, 7> q6m{};
          for (int idx = graph.row_begin(static_cast<int>(atom));
               idx < graph.row_end(static_cast<int>(atom)); ++idx) {
            const int nbr = graph.col_index(idx);
            if (nbr < 0)
              continue;
            double dx = xs[static_cast<std::size_t>(nbr)] - xs[atom];
            double dy = ys[static_cast<std::size_t>(nbr)] - ys[atom];
            double dz = zs[static_cast<std::size_t>(nbr)] - zs[atom];
            if (box_matrix != nullptr)
              box_matrix->apply_mic(dx, dy, dz);
            accumulate_qlm<4>(dx, dy, dz, q4m);
            accumulate_qlm<6>(dx, dy, dz, q6m);
          }
          q4_values_[atom] = finalize_ql<4>(q4m, neighbor_count);
          q6_values_[atom] = finalize_ql<6>(q6m, neighbor_count);
          q4m_local[atom] = q4m;
          q6m_local[atom] = q6m;
        }
      });

  executor.parallel_for(
      atom_count, worker_count, 1,
      [&](std::size_t /*worker*/, std::size_t begin, std::size_t end) {
        for (std::size_t atom = begin; atom < end; ++atom) {
        const int neighbor_count =
            graph.degree(static_cast<int>(atom));
        if (neighbor_count <= 0)
          continue;

        auto q4bar = q4m_local[atom];
        auto q6bar = q6m_local[atom];
        for (int idx = graph.row_begin(static_cast<int>(atom));
             idx < graph.row_end(static_cast<int>(atom)); ++idx) {
          const int nbr = graph.col_index(idx);
          if (nbr < 0)
            continue;
          for (std::size_t m = 0; m < q4bar.size(); ++m)
            q4bar[m] += q4m_local[static_cast<std::size_t>(nbr)][m];
          for (std::size_t m = 0; m < q6bar.size(); ++m)
            q6bar[m] += q6m_local[static_cast<std::size_t>(nbr)][m];
        }
        const double inv_count =
            1.0 / static_cast<double>(neighbor_count + 1);
        for (auto &value : q4bar)
          value *= inv_count;
        for (auto &value : q6bar)
          value *= inv_count;
        q4_bar_values[atom] = ql_from_normalized<4>(q4bar);
        q6_bar_values[atom] = ql_from_normalized<6>(q6bar);
      }
      });

  double dense_q4_sum = 0.0;
  double dense_q6_sum = 0.0;
  double dense_q4_bar_sum = 0.0;
  double dense_q6_bar_sum = 0.0;
  int dense_count = 0;
  int solid_like_count = 0;
  int fcc_count = 0;
  int hcp_count = 0;
  int bcc_count = 0;
  int ico_count = 0;
  int other_solid_count = 0;

  struct CrystalReference {
    double q4_bar;
    double q6_bar;
    int *counter;
  };

  std::array<CrystalReference, 4> crystal_refs{{
      {0.191, 0.575, &fcc_count},
      {0.097, 0.485, &hcp_count},
      {0.036, 0.511, &bcc_count},
      {0.00014, 0.663, &ico_count},
  }};

  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    if (atom_degree_[atom] >= solid_min_cn_) {
      dense_q4_sum += q4_values_[atom];
      dense_q6_sum += q6_values_[atom];
      dense_q4_bar_sum += q4_bar_values[atom];
      dense_q6_bar_sum += q6_bar_values[atom];
      ++dense_count;
      if (q6_bar_values[atom] >= solid_q6_threshold_) {
        solid_like_[atom] = 1;
        ++solid_like_count;

        double best_d2 = std::numeric_limits<double>::infinity();
        int *best_counter = nullptr;
        for (const auto &ref : crystal_refs) {
          const double dq4 = q4_bar_values[atom] - ref.q4_bar;
          const double dq6 = q6_bar_values[atom] - ref.q6_bar;
          const double d2 = dq4 * dq4 + dq6 * dq6;
          if (d2 < best_d2) {
            best_d2 = d2;
            best_counter = ref.counter;
          }
        }
        if (best_counter != nullptr && best_d2 <= crystal_tol_ * crystal_tol_) {
          ++(*best_counter);
        } else {
          ++other_solid_count;
        }
      }
    }
  }

  if (dense_count > 0) {
    const double inv_dense_count = 1.0 / static_cast<double>(dense_count);
    summary.mean_dense_q4 = dense_q4_sum * inv_dense_count;
    summary.mean_dense_q6 = dense_q6_sum * inv_dense_count;
    summary.mean_dense_q4_bar = dense_q4_bar_sum * inv_dense_count;
    summary.mean_dense_q6_bar = dense_q6_bar_sum * inv_dense_count;
  }
  summary.solid_like_fraction =
      static_cast<double>(solid_like_count) / static_cast<double>(atom_count);
  summary.fcc_like_fraction =
      static_cast<double>(fcc_count) / static_cast<double>(atom_count);
  summary.hcp_like_fraction =
      static_cast<double>(hcp_count) / static_cast<double>(atom_count);
  summary.bcc_like_fraction =
      static_cast<double>(bcc_count) / static_cast<double>(atom_count);
  summary.ico_like_fraction =
      static_cast<double>(ico_count) / static_cast<double>(atom_count);
  summary.other_solid_fraction =
      static_cast<double>(other_solid_count) / static_cast<double>(atom_count);

  if (solid_like_count == 0)
    return summary;

  std::vector<int> solid_parent(atom_count, -1);
  std::vector<int> solid_rank(atom_count, 0);
  std::vector<int> solid_size(atom_count, 0);
  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    if (solid_like_[atom] == 0)
      continue;
    solid_parent[atom] = static_cast<int>(atom);
    solid_size[atom] = 1;
  }

  auto solid_find = [&](int atom_index) {
    int root = atom_index;
    while (solid_parent[static_cast<std::size_t>(root)] != root)
      root = solid_parent[static_cast<std::size_t>(root)];
    while (solid_parent[static_cast<std::size_t>(atom_index)] != atom_index) {
      const int next = solid_parent[static_cast<std::size_t>(atom_index)];
      solid_parent[static_cast<std::size_t>(atom_index)] = root;
      atom_index = next;
    }
    return root;
  };

  auto solid_unite = [&](int lhs, int rhs) {
    lhs = solid_find(lhs);
    rhs = solid_find(rhs);
    if (lhs == rhs)
      return;
    if (solid_rank[static_cast<std::size_t>(lhs)] <
        solid_rank[static_cast<std::size_t>(rhs)]) {
      std::swap(lhs, rhs);
    }
    solid_parent[static_cast<std::size_t>(rhs)] = lhs;
    solid_size[static_cast<std::size_t>(lhs)] +=
        solid_size[static_cast<std::size_t>(rhs)];
    if (solid_rank[static_cast<std::size_t>(lhs)] ==
        solid_rank[static_cast<std::size_t>(rhs)]) {
      ++solid_rank[static_cast<std::size_t>(lhs)];
    }
  };

  for (const auto &[lhs, rhs] : edges_) {
    if (lhs < 0 || rhs < 0)
      continue;
    if (static_cast<std::size_t>(lhs) >= atom_count ||
        static_cast<std::size_t>(rhs) >= atom_count) {
      continue;
    }
    if (solid_like_[static_cast<std::size_t>(lhs)] == 0 ||
        solid_like_[static_cast<std::size_t>(rhs)] == 0) {
      continue;
    }
    solid_unite(lhs, rhs);
  }

  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    if (solid_like_[atom] == 0)
      continue;
    const int root = solid_find(static_cast<int>(atom));
    summary.largest_solid_cluster_size =
        std::max(summary.largest_solid_cluster_size,
                 solid_size[static_cast<std::size_t>(root)]);
  }
  return summary;
}

void TitanLJAnalyzer::accumulate_radial_density(const MOL &frame, bool periodic,
                                                const ClusterGeometry &geometry,
                                                std::array<double, kFeatureShellCount>
                                                    &shell_all_nm3,
                                                TitanExecutor &executor) {
  if (!profile_layout_ready_ || radial_bin_count_ == 0)
    return;

  shell_all_nm3.fill(0.0);
  if(!std::isfinite(geometry.centroid_x)) {shell_all_nm3.fill(std::numeric_limits<double>::quiet_NaN());return;}

  double effective_rho_max_a = fixed_rho_max_a_;
  if (periodic) {
    effective_rho_max_a =
        std::min(fixed_rho_max_a_,
                 TitanExactPeriodicBox(frame.Box).contained_radius());
  }

  std::size_t valid_bins = static_cast<std::size_t>(
      std::floor(effective_rho_max_a / rho_bin_a_ + 1e-12));
  valid_bins = std::min(valid_bins, radial_bin_count_);
  if (valid_bins == 0) return;

  const std::size_t cluster_size = cluster_members_.size();
  const auto xs = frame.get_pos_x();
  const auto ys = frame.get_pos_y();
  const auto zs = frame.get_pos_z();
  const std::size_t atom_count = std::min<std::size_t>(atom_count_, xs.size());

  std::optional<TitanExactPeriodicBox> box_matrix_storage;
  if (periodic)
    box_matrix_storage.emplace(frame.Box);
  const TitanExactPeriodicBox *box_matrix =
      box_matrix_storage.has_value() ? &box_matrix_storage.value() : nullptr;

  const std::size_t worker_count = effective_worker_count(
      executor.capacity(), std::max(atom_count, cluster_size));
  std::vector<std::vector<double>> local_cluster(
      worker_count, std::vector<double>(valid_bins, 0.0));
  std::vector<std::vector<double>> local_all(
      worker_count, std::vector<double>(valid_bins, 0.0));

  executor.parallel_for(
      atom_count, worker_count, 1,
      [&](std::size_t worker, std::size_t begin, std::size_t end) {
        auto &all_bins = local_all[worker];
        for (std::size_t atom = begin; atom < end; ++atom) {
          double dx = xs[atom] - geometry.centroid_x;
          double dy = ys[atom] - geometry.centroid_y;
          double dz = zs[atom] - geometry.centroid_z;
          if (box_matrix != nullptr)
            box_matrix->apply_mic(dx, dy, dz);
          const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
          const std::size_t bin =
              static_cast<std::size_t>(std::floor(r / rho_bin_a_));
          if (bin < valid_bins)
            all_bins[bin] += 1.0;
        }
      });

  executor.parallel_for(
      cluster_size, worker_count, 1,
      [&](std::size_t worker, std::size_t begin, std::size_t end) {
        auto &cluster_bins = local_cluster[worker];
        for (std::size_t local_idx = begin; local_idx < end; ++local_idx) {
          const double dx = unwrap_x_[local_idx] - geometry.centroid_x;
          const double dy = unwrap_y_[local_idx] - geometry.centroid_y;
          const double dz = unwrap_z_[local_idx] - geometry.centroid_z;
          const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
          const std::size_t bin =
              static_cast<std::size_t>(std::floor(r / rho_bin_a_));
          if (bin < valid_bins)
            cluster_bins[bin] += 1.0;
        }
      });

  std::vector<double> cluster_counts(valid_bins, 0.0);
  std::vector<double> all_counts(valid_bins, 0.0);
  for (std::size_t worker = 0; worker < worker_count; ++worker) {
    for (std::size_t bin = 0; bin < valid_bins; ++bin) {
      cluster_counts[bin] += local_cluster[worker][bin];
      all_counts[bin] += local_all[worker][bin];
    }
  }

  for (std::size_t bin = 0; bin < valid_bins; ++bin) {
    const double r_inner = static_cast<double>(bin) * rho_bin_a_;
    const double r_outer = r_inner + rho_bin_a_;
    const double shell_vol = shell_volume_a3(r_inner, r_outer);
    if (shell_vol <= 0.0)
      continue;

    const double rho_cluster_nm3 = (cluster_counts[bin] / shell_vol) * 1000.0;
    const double rho_all_nm3 = (all_counts[bin] / shell_vol) * 1000.0;
    if (bin < shell_all_nm3.size())
      shell_all_nm3[bin] = rho_all_nm3;

    const std::uint64_t cluster_n = ++rho_cluster_count_[bin];
    const double cluster_delta = rho_cluster_nm3 - rho_cluster_mean_nm3_[bin];
    rho_cluster_mean_nm3_[bin] += cluster_delta / static_cast<double>(cluster_n);
    const double cluster_delta2 = rho_cluster_nm3 - rho_cluster_mean_nm3_[bin];
    rho_cluster_m2_nm3_[bin] += cluster_delta * cluster_delta2;

    const std::uint64_t all_n = ++rho_all_count_[bin];
    const double all_delta = rho_all_nm3 - rho_all_mean_nm3_[bin];
    rho_all_mean_nm3_[bin] += all_delta / static_cast<double>(all_n);
    const double all_delta2 = rho_all_nm3 - rho_all_mean_nm3_[bin];
    rho_all_m2_nm3_[bin] += all_delta * all_delta2;
  }
}

void TitanLJAnalyzer::process_frame(const Multi_MOL &mol1, std::size_t snap_idx,
                                    TitanExecutor &executor) {
  if (!enabled_ || snap_idx >= mol1.m_mol.size())
    return;

  const MOL &frame = mol1.m_mol[snap_idx];
  const std::size_t atom_count =
      std::min<std::size_t>(atom_count_, frame.size());
  if (atom_count == 0)
    return;

  ensure_profile_layout(frame);
  reset_union_find(atom_count);

  const bool periodic = has_periodic_box(frame.Box);
  build_contact_graph(frame, atom_count, periodic);

  for (std::size_t i = 0; i < atom_count; ++i)
    component_root_[i] = find_root(static_cast<int>(i));

  int n_clusters = 0;
  std::fill(root_seen_.begin(), root_seen_.begin() + atom_count, 0);
  const int largest_root = find_largest_cluster_root(atom_count, n_clusters);
  const int largest_cluster_size =
      (largest_root >= 0) ? size_[static_cast<std::size_t>(largest_root)] : 0;

  build_largest_cluster_members(atom_count, largest_root);
  unwrap_largest_cluster(frame, periodic);
  const ClusterGeometry geometry = compute_cluster_geometry(executor);
  const LocalOrderSummary local_order =
      compute_local_order_metrics(frame, atom_count, periodic, executor);
  std::array<double, kFeatureShellCount> shell_all_nm3{};
  accumulate_radial_density(frame, periodic, geometry, shell_all_nm3,
                            executor);

  double volume_a3 = 0.0;
  if (periodic) {
    const TitanExactPeriodicBox box_matrix(frame.Box);
    volume_a3 = std::abs(determinant3x3(box_matrix));
  }

  FrameResult result;
  result.frame_index = static_cast<int>(frames_.size() + 1);
  result.time_ps = frame.time_of_snap;
  result.volume_a3 = volume_a3;
  if (volume_a3 > 0.0) {
    result.rho_bulk_a3 = static_cast<double>(atom_count) / volume_a3;
    result.rho_bulk_nm3 = result.rho_bulk_a3 * 1000.0;
  }
  result.n_clusters = n_clusters;
  result.largest_cluster_size = largest_cluster_size;
  result.largest_cluster_fraction =
      static_cast<double>(largest_cluster_size) /
      static_cast<double>(atom_count);
  result.largest_cluster_rg_a = geometry.rg_a;
  result.largest_cluster_rmax_a = geometry.rmax_a;
  result.mean_dense_q4 = local_order.mean_dense_q4;
  result.mean_dense_q6 = local_order.mean_dense_q6;
  result.mean_dense_q4_bar = local_order.mean_dense_q4_bar;
  result.mean_dense_q6_bar = local_order.mean_dense_q6_bar;
  result.solid_like_fraction = local_order.solid_like_fraction;
  result.largest_solid_cluster_size = local_order.largest_solid_cluster_size;
  result.largest_solid_cluster_fraction =
      static_cast<double>(local_order.largest_solid_cluster_size) /
      static_cast<double>(atom_count);
  result.fcc_like_fraction = local_order.fcc_like_fraction;
  result.hcp_like_fraction = local_order.hcp_like_fraction;
  result.bcc_like_fraction = local_order.bcc_like_fraction;
  result.ico_like_fraction = local_order.ico_like_fraction;
  result.other_solid_fraction = local_order.other_solid_fraction;
  frames_.push_back(result);

  double mean_cn = 0.0;
  double mean_cn_sq = 0.0;
  std::size_t isolated_atoms = 0;
  std::size_t dense_atoms = 0;
  for (std::size_t i = 0; i < atom_count; ++i) {
    const double cn = static_cast<double>(atom_degree_[i]);
    mean_cn += cn;
    mean_cn_sq += cn * cn;
    if (atom_degree_[i] == 0)
      ++isolated_atoms;
    if (atom_degree_[i] >= 6)
      ++dense_atoms;
  }
  mean_cn /= static_cast<double>(atom_count);
  const double variance_cn =
      std::max(0.0, mean_cn_sq / static_cast<double>(atom_count) -
                        mean_cn * mean_cn);
  const double std_cn = std::sqrt(variance_cn);
  const double inv_cutoff = cutoff_a_ > 0.0 ? 1.0 / cutoff_a_ : 0.0;
  const double inv_atom_count = 1.0 / static_cast<double>(atom_count);
  const double rho_ref_nm3 = std::max(result.rho_bulk_nm3, kFeatureEps);

  std::array<double, kTicaFeatureDim> feature_row{};
  feature_row[0] = result.rho_bulk_nm3;
  feature_row[1] = result.largest_cluster_fraction;
  feature_row[2] = result.largest_cluster_rg_a * inv_cutoff;
  feature_row[3] = result.largest_cluster_rmax_a * inv_cutoff;
  feature_row[4] = static_cast<double>(result.n_clusters) * inv_atom_count;
  feature_row[5] = mean_cn;
  feature_row[6] = std_cn;
  feature_row[7] = static_cast<double>(isolated_atoms) * inv_atom_count;
  feature_row[8] = static_cast<double>(dense_atoms) * inv_atom_count;
  feature_row[9] = result.mean_dense_q4;
  feature_row[10] = result.mean_dense_q6;
  feature_row[11] = result.mean_dense_q4_bar;
  feature_row[12] = result.mean_dense_q6_bar;
  feature_row[13] = result.solid_like_fraction;
  feature_row[14] = result.largest_solid_cluster_fraction;
  feature_row[15] = result.fcc_like_fraction;
  feature_row[16] = result.hcp_like_fraction;
  for (std::size_t shell = 0; shell < kFeatureShellCount; ++shell)
    feature_row[17 + shell] = shell_all_nm3[shell] / rho_ref_nm3;

  latest_feature_vector_.resize(kTicaFeatureDim);
  for (int i = 0; i < kTicaFeatureDim; ++i)
    latest_feature_vector_[i] = feature_row[static_cast<std::size_t>(i)];
  feature_history_.push_back(feature_row);
}

void TitanLJAnalyzer::export_timeseries(const std::string &filename,
                                        std::fstream &log) const {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("LJ analysis: failed to open " + filename);
    return;
  }

  out << "# TITAN LJ condensation timeseries\n";
  out << "# atoms " << atom_count_ << "\n";
  out << "# cluster_cutoff_A " << std::fixed << std::setprecision(6)
      << cutoff_a_ << "\n";
  out << "# min_cluster_size " << min_cluster_size_ << "\n";
  out << "# rho_bin_A " << std::fixed << std::setprecision(6) << rho_bin_a_
      << "\n";
  out << "# rho_max_A " << std::fixed << std::setprecision(6)
      << fixed_rho_max_a_ << "\n";
  out << "# solid_q6_threshold " << std::fixed << std::setprecision(6)
      << solid_q6_threshold_ << "\n";
  out << "# solid_min_cn " << solid_min_cn_ << "\n";
  out << "# crystal_tol " << std::fixed << std::setprecision(6)
      << crystal_tol_ << "\n";
  out << "# columns = frame time_ps volume_A3 rho_bulk_A^-3 rho_bulk_nm^-3 "
         "n_clusters largest_cluster_size largest_cluster_fraction "
         "largest_cluster_rg_A largest_cluster_rmax_A mean_dense_q4 "
         "mean_dense_q6 mean_dense_q4_bar mean_dense_q6_bar "
         "solid_like_fraction largest_solid_cluster_size "
         "largest_solid_cluster_fraction fcc_like_fraction hcp_like_fraction "
         "bcc_like_fraction ico_like_fraction other_solid_fraction\n";

  out << std::fixed << std::setprecision(6);
  for (const auto &frame : frames_) {
    out << std::setw(8) << frame.frame_index << " " << std::setw(14)
        << frame.time_ps << " " << std::setw(16) << frame.volume_a3 << " "
        << std::setw(14) << frame.rho_bulk_a3 << " " << std::setw(14)
        << frame.rho_bulk_nm3 << " " << std::setw(8) << frame.n_clusters << " "
        << std::setw(12) << frame.largest_cluster_size << " "
        << std::setw(14) << frame.largest_cluster_fraction << " "
        << std::setw(14) << frame.largest_cluster_rg_a << " "
        << std::setw(14) << frame.largest_cluster_rmax_a << " "
        << std::setw(14) << frame.mean_dense_q4 << " " << std::setw(14)
        << frame.mean_dense_q6 << " " << std::setw(14)
        << frame.mean_dense_q4_bar << " " << std::setw(14)
        << frame.mean_dense_q6_bar << " " << std::setw(14)
        << frame.solid_like_fraction << " " << std::setw(12)
        << frame.largest_solid_cluster_size << " " << std::setw(14)
        << frame.largest_solid_cluster_fraction << " " << std::setw(14)
        << frame.fcc_like_fraction << " " << std::setw(14)
        << frame.hcp_like_fraction << " " << std::setw(14)
        << frame.bcc_like_fraction << " " << std::setw(14)
        << frame.ico_like_fraction << " " << std::setw(14)
        << frame.other_solid_fraction << "\n";
  }

  if (log.is_open())
    TitanUI::log_kv(log, "LJ Timeseries", filename);
}

void TitanLJAnalyzer::export_rho_profile(const std::string &filename,
                                         std::fstream &log) const {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("LJ analysis: failed to open " + filename);
    return;
  }

  out << "# TITAN LJ radial density profile around the largest-cluster centroid\n";
  out << "# distances in Angstrom, densities in atoms / nm^3\n";
  out << "# atoms " << atom_count_ << "\n";
  out << "# frames " << frames_.size() << "\n";
  out << "# cluster_cutoff_A " << std::fixed << std::setprecision(6)
      << cutoff_a_ << "\n";
  out << "# rho_bin_A " << std::fixed << std::setprecision(6) << rho_bin_a_
      << "\n";
  out << "# rho_max_A " << std::fixed << std::setprecision(6)
      << fixed_rho_max_a_ << "\n";
  out << "# columns = bin r_inner_A r_outer_A r_center_A "
         "rho_cluster_nm^-3 sem_cluster_nm^-3 rho_all_nm^-3 sem_all_nm^-3 "
         "cumulative_cluster_atoms cumulative_all_atoms cluster_samples "
         "all_samples\n";

  double cumulative_cluster = 0.0;
  double cumulative_all = 0.0;
  out << std::fixed << std::setprecision(6);
  for (std::size_t bin = 0; bin < radial_bin_count_; ++bin) {
    const double r_inner = static_cast<double>(bin) * rho_bin_a_;
    const double r_outer = r_inner + rho_bin_a_;
    const double r_center = 0.5 * (r_inner + r_outer);
    const double shell_vol = shell_volume_a3(r_inner, r_outer);

    double sem_cluster = 0.0;
    if (rho_cluster_count_[bin] > 1) {
      const double variance =
          rho_cluster_m2_nm3_[bin] /
          static_cast<double>(rho_cluster_count_[bin] - 1);
      sem_cluster = std::sqrt(std::max(0.0, variance) /
                              static_cast<double>(rho_cluster_count_[bin]));
    }

    double sem_all = 0.0;
    if (rho_all_count_[bin] > 1) {
      const double variance =
          rho_all_m2_nm3_[bin] / static_cast<double>(rho_all_count_[bin] - 1);
      sem_all = std::sqrt(std::max(0.0, variance) /
                          static_cast<double>(rho_all_count_[bin]));
    }

    cumulative_cluster += (rho_cluster_mean_nm3_[bin] * shell_vol) / 1000.0;
    cumulative_all += (rho_all_mean_nm3_[bin] * shell_vol) / 1000.0;

    out << std::setw(6) << (bin + 1) << " " << std::setw(12) << r_inner << " "
        << std::setw(12) << r_outer << " " << std::setw(12) << r_center << " "
        << std::setw(16) << rho_cluster_mean_nm3_[bin] << " " << std::setw(16)
        << sem_cluster << " " << std::setw(16) << rho_all_mean_nm3_[bin] << " "
        << std::setw(16) << sem_all << " " << std::setw(18)
        << cumulative_cluster << " " << std::setw(18) << cumulative_all << " "
        << std::setw(10) << rho_cluster_count_[bin] << " " << std::setw(10)
        << rho_all_count_[bin] << "\n";
  }

  if (log.is_open())
    TitanUI::log_kv(log, "LJ rho(r)", filename);
}

void TitanLJAnalyzer::export_feature_timeseries(const std::string &filename,
                                                std::fstream &log) const {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("LJ analysis: failed to open " + filename);
    return;
  }

  out << "# TITAN LJ phase-2 feature timeseries for TICA/MSM\n";
  out << "# atoms " << atom_count_ << "\n";
  out << "# solid_q6_threshold " << std::fixed << std::setprecision(6)
      << solid_q6_threshold_ << "\n";
  out << "# solid_min_cn " << solid_min_cn_ << "\n";
  out << "# crystal_tol " << std::fixed << std::setprecision(6)
      << crystal_tol_ << "\n";
  out << "# columns = frame time_ps";
  for (const char *label : kLJFeatureLabels)
    out << " " << label;
  out << "\n";

  out << std::fixed << std::setprecision(6);
  const std::size_t n_rows = std::min(frames_.size(), feature_history_.size());
  for (std::size_t i = 0; i < n_rows; ++i) {
    out << std::setw(8) << frames_[i].frame_index << " " << std::setw(14)
        << frames_[i].time_ps;
    for (double value : feature_history_[i])
      out << " " << std::setw(14) << value;
    out << "\n";
  }

  if (log.is_open())
    TitanUI::log_kv(log, "LJ Phase-2 Features", filename);
}

void TitanLJAnalyzer::export_summary(const std::string &filename,
                                     std::fstream &log) const {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("LJ analysis: failed to open " + filename);
    return;
  }

  const auto bulk_density = mean_sem(
      frames_.size(), [&](std::size_t i) { return frames_[i].rho_bulk_nm3; });
  const auto n_clusters =
      mean_sem(frames_.size(), [&](std::size_t i) {
        return static_cast<double>(frames_[i].n_clusters);
      });
  const auto largest_cluster_size =
      mean_sem(frames_.size(), [&](std::size_t i) {
        return static_cast<double>(frames_[i].largest_cluster_size);
      });
  const auto largest_cluster_fraction = mean_sem(
      frames_.size(),
      [&](std::size_t i) { return frames_[i].largest_cluster_fraction; });
  const auto largest_cluster_rg = mean_sem(
      frames_.size(),
      [&](std::size_t i) { return frames_[i].largest_cluster_rg_a; });
  const auto largest_cluster_rmax = mean_sem(
      frames_.size(),
      [&](std::size_t i) { return frames_[i].largest_cluster_rmax_a; });
  const auto mean_dense_q4 = mean_sem(
      frames_.size(), [&](std::size_t i) { return frames_[i].mean_dense_q4; });
  const auto mean_dense_q6 = mean_sem(
      frames_.size(), [&](std::size_t i) { return frames_[i].mean_dense_q6; });
  const auto mean_dense_q4_bar = mean_sem(frames_.size(), [&](std::size_t i) {
    return frames_[i].mean_dense_q4_bar;
  });
  const auto mean_dense_q6_bar = mean_sem(frames_.size(), [&](std::size_t i) {
    return frames_[i].mean_dense_q6_bar;
  });
  const auto solid_like_fraction =
      mean_sem(frames_.size(), [&](std::size_t i) {
        return frames_[i].solid_like_fraction;
      });
  const auto largest_solid_cluster_size =
      mean_sem(frames_.size(), [&](std::size_t i) {
        return static_cast<double>(frames_[i].largest_solid_cluster_size);
      });
  const auto largest_solid_cluster_fraction =
      mean_sem(frames_.size(), [&](std::size_t i) {
        return frames_[i].largest_solid_cluster_fraction;
      });
  const auto fcc_like_fraction =
      mean_sem(frames_.size(), [&](std::size_t i) {
        return frames_[i].fcc_like_fraction;
      });
  const auto hcp_like_fraction =
      mean_sem(frames_.size(), [&](std::size_t i) {
        return frames_[i].hcp_like_fraction;
      });
  const auto bcc_like_fraction =
      mean_sem(frames_.size(), [&](std::size_t i) {
        return frames_[i].bcc_like_fraction;
      });
  const auto ico_like_fraction =
      mean_sem(frames_.size(), [&](std::size_t i) {
        return frames_[i].ico_like_fraction;
      });
  const auto other_solid_fraction =
      mean_sem(frames_.size(), [&](std::size_t i) {
        return frames_[i].other_solid_fraction;
      });

  out << "module: lj_condensation\n";
  out << "frames: " << frames_.size() << "\n";
  out << "atoms: " << atom_count_ << "\n";
  out << std::fixed << std::setprecision(6);
  out << "cluster_cutoff_A: " << cutoff_a_ << "\n";
  out << "min_cluster_size: " << min_cluster_size_ << "\n";
  out << "rho_bin_A: " << rho_bin_a_ << "\n";
  out << "rho_max_A: " << fixed_rho_max_a_ << "\n";
  out << "solid_q6_threshold: " << solid_q6_threshold_ << "\n";
  out << "solid_min_cn: " << solid_min_cn_ << "\n";
  out << "crystal_tol: " << crystal_tol_ << "\n";
  out << "mean_bulk_density_nm^-3: " << bulk_density.first << "\n";
  out << "sem_bulk_density_nm^-3: " << bulk_density.second << "\n";
  out << "mean_n_clusters: " << n_clusters.first << "\n";
  out << "sem_n_clusters: " << n_clusters.second << "\n";
  out << "mean_largest_cluster_size: " << largest_cluster_size.first << "\n";
  out << "sem_largest_cluster_size: " << largest_cluster_size.second << "\n";
  out << "mean_largest_cluster_fraction: " << largest_cluster_fraction.first
      << "\n";
  out << "sem_largest_cluster_fraction: " << largest_cluster_fraction.second
      << "\n";
  out << "mean_largest_cluster_rg_A: " << largest_cluster_rg.first << "\n";
  out << "sem_largest_cluster_rg_A: " << largest_cluster_rg.second << "\n";
  out << "mean_largest_cluster_rmax_A: " << largest_cluster_rmax.first << "\n";
  out << "sem_largest_cluster_rmax_A: " << largest_cluster_rmax.second << "\n";
  out << "mean_dense_q4: " << mean_dense_q4.first << "\n";
  out << "sem_dense_q4: " << mean_dense_q4.second << "\n";
  out << "mean_dense_q6: " << mean_dense_q6.first << "\n";
  out << "sem_dense_q6: " << mean_dense_q6.second << "\n";
  out << "mean_dense_q4_bar: " << mean_dense_q4_bar.first << "\n";
  out << "sem_dense_q4_bar: " << mean_dense_q4_bar.second << "\n";
  out << "mean_dense_q6_bar: " << mean_dense_q6_bar.first << "\n";
  out << "sem_dense_q6_bar: " << mean_dense_q6_bar.second << "\n";
  out << "mean_solid_like_fraction: " << solid_like_fraction.first << "\n";
  out << "sem_solid_like_fraction: " << solid_like_fraction.second << "\n";
  out << "mean_largest_solid_cluster_size: "
      << largest_solid_cluster_size.first << "\n";
  out << "sem_largest_solid_cluster_size: "
      << largest_solid_cluster_size.second << "\n";
  out << "mean_largest_solid_cluster_fraction: "
      << largest_solid_cluster_fraction.first << "\n";
  out << "sem_largest_solid_cluster_fraction: "
      << largest_solid_cluster_fraction.second << "\n";
  out << "mean_fcc_like_fraction: " << fcc_like_fraction.first << "\n";
  out << "sem_fcc_like_fraction: " << fcc_like_fraction.second << "\n";
  out << "mean_hcp_like_fraction: " << hcp_like_fraction.first << "\n";
  out << "sem_hcp_like_fraction: " << hcp_like_fraction.second << "\n";
  out << "mean_bcc_like_fraction: " << bcc_like_fraction.first << "\n";
  out << "sem_bcc_like_fraction: " << bcc_like_fraction.second << "\n";
  out << "mean_ico_like_fraction: " << ico_like_fraction.first << "\n";
  out << "sem_ico_like_fraction: " << ico_like_fraction.second << "\n";
  out << "mean_other_solid_fraction: " << other_solid_fraction.first << "\n";
  out << "sem_other_solid_fraction: " << other_solid_fraction.second << "\n";

  if (log.is_open())
    TitanUI::log_kv(log, "LJ Summary", filename);
}

void TitanLJAnalyzer::finalize_trajectory(
    const std::string &timeseries_filename, const std::string &rho_filename,
    const std::string &feature_filename, const std::string &summary_filename,
    std::fstream &log) {
  if (!enabled_ || frames_.empty()) {
    reset_trajectory_state();
    return;
  }

  export_timeseries(timeseries_filename, log);
  export_rho_profile(rho_filename, log);
  export_feature_timeseries(feature_filename, log);
  export_summary(summary_filename, log);

  TitanUI::print_info("[LJ] Exported timeseries to " + timeseries_filename);
  TitanUI::print_info("[LJ] Exported rho(r) to " + rho_filename);
  TitanUI::print_info("[LJ] Exported phase-2 features to " + feature_filename);
  TitanUI::print_info("[LJ] Exported summary to " + summary_filename);

  reset_trajectory_state();
}
