#include "postproc/postproc_common.h"

namespace postproc_detail {

std::vector<std::pair<double, double>>
read_projection_csv_pairs(const std::string &path) {
  std::ifstream in(path);
  if (!in.is_open())
    return {};

  std::vector<std::pair<double, double>> projections;
  std::string line;
  bool first_line = true;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    if (first_line) {
      first_line = false;
      if (line.find("Frame") != std::string::npos)
        continue;
    }
    std::stringstream ss(line);
    std::string token;
    std::vector<std::string> cols;
    while (std::getline(ss, token, ','))
      cols.push_back(token);
    if (cols.size() < 3)
      continue;
    try {
      projections.emplace_back(std::stod(cols[1]), std::stod(cols[2]));
    } catch (...) {
      continue;
    }
  }
  return projections;
}

int select_effective_msm_clusters(const arguments &args,
                                  const TitanKinetics &tica_analyzer) {
  const int requested_k = std::max(2, args.msm.msm_clusters);
  if (args.msm.msm_clusters_user_set) {
    return requested_k;
  }

  // Auto mode is intentionally conservative for DHAM-facing workflows.
  // We only reduce the default cluster count when the TICA dataset is too
  // short to support a stable microstate model.
  const int64_t n_frames = tica_analyzer.num_frames();
  const int64_t n_pairs = tica_analyzer.num_pairs();
  const int64_t by_frames = std::max<int64_t>(2, n_frames / 5);
  const int64_t by_pairs =
      (n_pairs > 0) ? std::max<int64_t>(2, n_pairs / 3) : by_frames;
  const int64_t auto_cap =
      std::min<int64_t>(requested_k, std::min(by_frames, by_pairs));
  return std::clamp(static_cast<int>(auto_cap), 2, requested_k);
}

std::string describe_msm_cluster_selection(const arguments &args,
                                           int effective_k,
                                           const TitanKinetics &tica_analyzer) {
  const std::string mode =
      args.msm.msm_clusters_user_set ? "user" : "auto";
  return mode + " (requested=" + std::to_string(args.msm.msm_clusters) +
         ", effective=" + std::to_string(effective_k) +
         ", frames=" + std::to_string(tica_analyzer.num_frames()) +
         ", lagged_pairs=" + std::to_string(tica_analyzer.num_pairs()) + ")";
}

void rebuild_msm_projections_from_dccm(const arguments &args,
                                       const DCCMNetwork &dccm_analyzer,
                                       const TitanKinetics &tica_analyzer,
                                       TitanMSM &msm_analyzer,
                                       std::fstream &log) {
  const size_t n_frames = dccm_analyzer.get_total_frames();
  const int64_t expected_frames = tica_analyzer.num_frames();
  const int n_ca = static_cast<int>(dccm_analyzer.get_resnrs().size());
  const int feat_dim = tica_analyzer.feature_dim();

  if (n_frames == 0 || n_ca <= 0 || feat_dim <= 0) {
    TitanUI::print_warning(
        "MSM: Cannot rebuild TICA projections from cached DCCM trajectory.");
    return;
  }
  if (expected_frames > 0 && static_cast<int64_t>(n_frames) != expected_frames) {
    TitanUI::print_warning(
        "MSM: DCCM cache holds " + std::to_string(n_frames) +
        " frames, but TICA accumulated " + std::to_string(expected_frames) +
        ". Refusing partial projection rebuild.");
    return;
  }

  const auto &traj_x = dccm_analyzer.get_traj_x();
  const auto &traj_y = dccm_analyzer.get_traj_y();
  const auto &traj_z = dccm_analyzer.get_traj_z();
  const auto &traj_times = dccm_analyzer.get_traj_times();
  const auto &traj_replicas = dccm_analyzer.get_traj_replicas();
  const auto &global_indices = dccm_analyzer.get_global_frame_indices();

  if (traj_x.size() != n_frames || traj_y.size() != n_frames ||
      traj_z.size() != n_frames) {
    TitanUI::print_warning(
        "MSM: DCCM trajectory cache is inconsistent. Skipping projection rebuild.");
    return;
  }

  msm_analyzer.clear_projected();

  Eigen::VectorXd features(feat_dim);
  Eigen::MatrixXd mobile(n_ca, 3);
  Eigen::MatrixXd ref_centered(n_ca, 3);
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  bool ref_initialized = false;
  size_t n_breaks = 0;

  for (size_t f = 0; f < n_frames; ++f) {
    if (traj_x[f].size() < static_cast<size_t>(n_ca) ||
        traj_y[f].size() < static_cast<size_t>(n_ca) ||
        traj_z[f].size() < static_cast<size_t>(n_ca)) {
      TitanUI::print_warning(
          "MSM: Encountered truncated cached CA coordinates. Aborting projection rebuild.");
      msm_analyzer.clear_projected();
      return;
    }

    for (int a = 0; a < n_ca; ++a) {
      const size_t ai = static_cast<size_t>(a);
      mobile(a, 0) = traj_x[f][ai];
      mobile(a, 1) = traj_y[f][ai];
      mobile(a, 2) = traj_z[f][ai];
    }

    if (!ref_initialized) {
      ref_centered = mobile;
      ref_centered.rowwise() -= ref_centered.colwise().mean().eval();
      mobile = ref_centered;
      ref_initialized = true;
    } else {
      TitanAlignment::align_frame_inplace(ref_centered, mobile, R);
    }

    int feat_idx = 0;
    for (int a = 0; a < n_ca; ++a) {
      for (int b = a + 1; b < n_ca; ++b) {
        const double dx = mobile(a, 0) - mobile(b, 0);
        const double dy = mobile(a, 1) - mobile(b, 1);
        const double dz = mobile(a, 2) - mobile(b, 2);
        features[feat_idx++] = std::sqrt(dx * dx + dy * dy + dz * dz);
      }
    }

    const int replica_id =
        (traj_replicas.size() > f) ? traj_replicas[f] : -1;
    const int64_t global_idx =
        (global_indices.size() > f) ? global_indices[f]
                                    : static_cast<int64_t>(f);

    msm_analyzer.add_projected_frame(
        tica_analyzer.project_frame(features), replica_id, global_idx);
    if (traj_times.size() > f)
      msm_analyzer.record_frame_time(traj_times[f], replica_id);

    if (args.msm.msm_auto_boundary && f + 1 < n_frames &&
        is_trajectory_boundary(traj_replicas, traj_times, f, f + 1)) {
      msm_analyzer.add_trajectory_break();
      ++n_breaks;
    }
  }

  TitanUI::log_kv(log, "MSM Projection Rebuild", "from cached DCCM trajectory");
  TitanUI::log_kv(log, "MSM Projected Frames",
                  std::to_string(msm_analyzer.n_frames()));
  if (args.msm.msm_auto_boundary) {
    TitanUI::log_kv(log, "MSM Rebuilt Trajectory Breaks",
                    std::to_string(n_breaks));
  }
}

} // namespace postproc_detail
