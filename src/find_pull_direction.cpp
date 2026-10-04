#include "MOL.h"
#include "differential_evolution.h"
#include "functions.h"
#include "pull_de_variants.h"
#include "pull_direction_pathway.h"
#include "titan_simd_dispatch.h"
#include "titan_ui.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string_view>

using std::string_view;

const double PI = 3.14159265;

void cartesian_to_spherical(double x, double y, double z, double &theta,
                            double &phi) {
  const double r = std::sqrt(x * x + y * y + z * z);
  if (r < 1e-6) {
    theta = 0.0;
    phi = 0.0;
  } else {
    theta = std::acos(z / r);
    phi = std::atan2(y, x);
    if (phi < 0.0)
      phi += 2.0 * 3.14159265358979323846;
  }
}

double normalize_angle_2pi(double angle) {
  angle = std::fmod(angle, 2.0 * PI);
  if (angle < 0.0)
    angle += 2.0 * PI;
  return angle;
}

double clamp_theta(double theta) {
  if (theta < 0.0)
    return -theta;
  if (theta > PI)
    return 2.0 * PI - theta;
  return theta;
}

void mass_weighted_com(const MOL &mol, const Multi_MOL_parameters &parameters,
                       std::span<double> com) {
  com[0] = 0.0;
  com[1] = 0.0;
  com[2] = 0.0;
  double total_mass = 0.0;
  for (size_t i = 0; i < mol.size(); ++i) {
    const double mass = parameters[i].get_mass();
    com[0] += mol.pos.x[i] * mass;
    com[1] += mol.pos.y[i] * mass;
    com[2] += mol.pos.z[i] * mass;
    total_mass += mass;
  }
  if (total_mass > 0.0) {
    com[0] /= total_mass;
    com[1] /= total_mass;
    com[2] /= total_mass;
  }
}

double cal_score_function(const cache_energy &energy_trial,
                          const cache_energy &energy_ref, int flag_function,
                          double dist, double limit) {
  double base_score = 0.0;
  if (flag_function == 1) {
    base_score = energy_trial.elec_ener + energy_trial.vdw_ener -
                 energy_ref.elec_ener - energy_ref.vdw_ener +
                 2.0 * std::fabs(energy_trial.elec_ener + energy_trial.vdw_ener);
  } else {
    base_score = -1.0 *
                 std::exp(1.6863 * (-energy_trial.elec_ener -
                                    energy_trial.vdw_ener +
                                    energy_ref.elec_ener + energy_ref.vdw_ener));
  }
  const double k_spring = 1000.0;
  const double penalty =
      dist > limit ? k_spring * (dist - limit) * (dist - limit) : 0.0;
  return base_score + penalty;
}

bool check_radius(const de_arg &DE_arguments, const force &tran_vector) {
  return (tran_vector.Fz < DE_arguments.scan_radius &&
          tran_vector.Fz > DE_arguments.min_scan_radius);
}

bool parse_angle(const de_arg &DE_arguments, const force &tran_vector,
                 double &crit_angle) {
  if (!DE_arguments.flag_old_vec) {
    crit_angle = 0.0;
    return true;
  }

  const double sin_theta = std::sin(tran_vector.Fx);
  const double nx = sin_theta * std::cos(tran_vector.Fy);
  const double ny = sin_theta * std::sin(tran_vector.Fy);
  const double nz = std::cos(tran_vector.Fx);

  const double old_mod = DE_arguments.old_vector.modul();
  if (old_mod < 1e-9)
    return true;

  const double ox = DE_arguments.old_vector.Fx / old_mod;
  const double oy = DE_arguments.old_vector.Fy / old_mod;
  const double oz = DE_arguments.old_vector.Fz / old_mod;

  double cos_val = nx * ox + ny * oy + nz * oz;
  if (cos_val > 1.0)
    cos_val = 1.0;
  if (cos_val < -1.0)
    cos_val = -1.0;

  crit_angle = std::acos(cos_val) * 57.2957795;
  return (crit_angle < DE_arguments.restrain_angle);
}

namespace {

constexpr double kPullScoreImproveEps = 1.0e-8;
constexpr double kPullInvalidScore = std::numeric_limits<double>::infinity();
constexpr double kPatternInitAngleStepDeg = 12.0;
constexpr double kPatternMinAngleStepDeg = 0.5;
constexpr double kPatternMinRhoStep = 0.02;
constexpr std::size_t kPatternMaxSweeps = 32;
constexpr std::size_t kVariantPatternTopK = 3;
constexpr double kVariantPatternMinSeedDistanceNorm = 0.18;

enum class PullEnergyBackend { Amber, Charmm, Opls, Martini, Gromos };

struct PullWorkerContext {
  MOL scratch{};
  cache_sigma_epsilon eval_cache{};
  std::size_t eval_quota_remaining{0};
  std::size_t evaluations_done{0};
};

struct PullPatternStats {
  bool enabled{false};
  bool improved{false};
  bool stopped_by_eval_budget{false};
  std::size_t sweeps{0};
  std::size_t accepted_moves{0};
  std::size_t evaluations_used{0};
   std::size_t seeds_attempted{0};
   std::size_t distinct_seed_basins{0};
  double initial_score{kPullInvalidScore};
  double final_score{kPullInvalidScore};
  double wall_seconds{0.0};
};

struct PullDirectStats {
  bool enabled{false};
  bool used_fallback_budget{false};
  bool stopped_by_eval_budget{false};
  bool stopped_by_local_threshold{false};
  bool stopped_by_stagnation{false};
  std::size_t effective_eval_budget{0};
  std::size_t expansions{0};
  std::size_t batches{0};
  std::size_t max_batch_cells{0};
  std::size_t cells_remaining{0};
  std::size_t evaluated_candidates{0};
  std::size_t best_expansion{0};
  std::size_t phase_evaluations_used{0};
  double best_cell_size{0.0};
  double wall_seconds{0.0};
  std::array<double, 3> best_half_width_norm{};
};

struct PullDirectCell {
  std::array<double, 3> center_norm{};
  std::array<double, 3> half_width{};
  PullCandidateState state{};
  std::size_t depth{0};
  std::size_t id{0};
};

struct PullGeometryContext {
  bool has_old_vector{false};
  double old_nx{0.0};
  double old_ny{0.0};
  double old_nz{1.0};
  double restrain_angle_deg{0.0};
};

struct PullEvalBudget {
  std::size_t max_evals{0};
  std::size_t block_size{8};
  std::atomic_size_t reserved{0};
};

struct PullPreparedGeometry {
  force translation{};
  double angle_deg{0.0};
  bool valid{false};
};

struct PullAcceptedUpdate {
  std::size_t index{0};
  PullCandidateState state{};
};

struct PullAcceptedBuffer {
  std::vector<PullAcceptedUpdate> entries{};
  std::size_t count{0};
};

using PullRotationMatrix = std::array<std::array<double, 3>, 3>;

struct PullAlignedFrameData {
  MOL rec_frame{};
  MOL lg_frame{};
  force rec_com_original{};
  force lg_com_original{};
  force rec_com_aligned{};
  force lg_com_aligned{};
  force original_delta{};
  force aligned_delta{};
  force aligned_pull_vector{};
  box aligned_box{};
};

cache_sigma_epsilon make_pull_eval_cache(const cache_sigma_epsilon &source) {
  cache_sigma_epsilon scratch;
  scratch.charge = source.charge;
  scratch.force_field_name = source.force_field_name;
  scratch.cutoff_dis = source.cutoff_dis;
  scratch.charmm_pair_sigma = source.charmm_pair_sigma;
  scratch.charmm_pair_epsilon = source.charmm_pair_epsilon;
  scratch.charmm_type_count = source.charmm_type_count;
  scratch.charmm_pair_sparse_keys = source.charmm_pair_sparse_keys;
  scratch.charmm_pair_sparse_sigma = source.charmm_pair_sparse_sigma;
  scratch.charmm_pair_sparse_epsilon = source.charmm_pair_sparse_epsilon;
  scratch.martini_pair_sigma = source.martini_pair_sigma;
  scratch.martini_pair_epsilon = source.martini_pair_epsilon;
  scratch.martini_type_count = source.martini_type_count;
  return scratch;
}

std::array<double, 3> force_to_array(const force &value) {
  return {value.Fx, value.Fy, value.Fz};
}

force array_to_force(const std::array<double, 3> &value) {
  force converted;
  converted.Fx = value[0];
  converted.Fy = value[1];
  converted.Fz = value[2];
  return converted;
}

double vector_dot(const std::array<double, 3> &lhs,
                  const std::array<double, 3> &rhs) {
  return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

std::array<double, 3> vector_cross(const std::array<double, 3> &lhs,
                                   const std::array<double, 3> &rhs) {
  return {lhs[1] * rhs[2] - lhs[2] * rhs[1],
          lhs[2] * rhs[0] - lhs[0] * rhs[2],
          lhs[0] * rhs[1] - lhs[1] * rhs[0]};
}

double vector_norm(const std::array<double, 3> &value) {
  return std::sqrt(vector_dot(value, value));
}

std::array<double, 3> normalize_vector(const std::array<double, 3> &value) {
  const double norm = vector_norm(value);
  if (norm < 1.0e-12)
    return {0.0, 0.0, 1.0};
  return {value[0] / norm, value[1] / norm, value[2] / norm};
}

PullRotationMatrix identity_rotation_matrix() {
  return {{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
}

PullRotationMatrix rotation_matrix_to_positive_z(const force &translation) {
  const std::array<double, 3> source =
      normalize_vector(force_to_array(translation));
  const std::array<double, 3> target{0.0, 0.0, 1.0};
  const double cos_theta = std::clamp(vector_dot(source, target), -1.0, 1.0);
  if (cos_theta > 1.0 - 1.0e-12)
    return identity_rotation_matrix();
  if (cos_theta < -1.0 + 1.0e-12) {
    return {{{1.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, -1.0}}};
  }

  const std::array<double, 3> axis = vector_cross(source, target);
  const double axis_norm = vector_norm(axis);
  if (axis_norm < 1.0e-12)
    return identity_rotation_matrix();

  const double x = axis[0];
  const double y = axis[1];
  const double z = axis[2];
  const double scale = (1.0 - cos_theta) / (axis_norm * axis_norm);

  return {{{1.0 + scale * (-z * z - y * y), -z + scale * x * y,
            y + scale * x * z},
           {z + scale * x * y, 1.0 + scale * (-z * z - x * x),
            -x + scale * y * z},
           {-y + scale * x * z, x + scale * y * z,
            1.0 + scale * (-y * y - x * x)}}};
}

std::array<double, 3>
apply_rotation(const PullRotationMatrix &rotation,
               const std::array<double, 3> &point) {
  return {
      rotation[0][0] * point[0] + rotation[0][1] * point[1] +
          rotation[0][2] * point[2],
      rotation[1][0] * point[0] + rotation[1][1] * point[1] +
          rotation[1][2] * point[2],
      rotation[2][0] * point[0] + rotation[2][1] * point[1] +
          rotation[2][2] * point[2],
  };
}

void update_bounds_from_mol(const MOL &mol, std::array<double, 3> &mins,
                            std::array<double, 3> &maxs) {
  for (std::size_t i = 0; i < mol.size(); ++i) {
    mins[0] = std::min(mins[0], mol.pos.x[i]);
    mins[1] = std::min(mins[1], mol.pos.y[i]);
    mins[2] = std::min(mins[2], mol.pos.z[i]);
    maxs[0] = std::max(maxs[0], mol.pos.x[i]);
    maxs[1] = std::max(maxs[1], mol.pos.y[i]);
    maxs[2] = std::max(maxs[2], mol.pos.z[i]);
  }
}

void rotate_mol_about_origin(MOL &mol, const std::array<double, 3> &origin,
                             const PullRotationMatrix &rotation) {
  for (std::size_t i = 0; i < mol.size(); ++i) {
    const std::array<double, 3> relative{mol.pos.x[i] - origin[0],
                                         mol.pos.y[i] - origin[1],
                                         mol.pos.z[i] - origin[2]};
    const auto rotated = apply_rotation(rotation, relative);
    mol.pos.x[i] = rotated[0];
    mol.pos.y[i] = rotated[1];
    mol.pos.z[i] = rotated[2];
  }
}

void shift_mol(MOL &mol, const std::array<double, 3> &shift) {
  for (std::size_t i = 0; i < mol.size(); ++i) {
    mol.pos.x[i] += shift[0];
    mol.pos.y[i] += shift[1];
    mol.pos.z[i] += shift[2];
  }
}

PullAlignedFrameData build_pull_aligned_frame_data(
    const Multi_MOL &rec, const Multi_MOL &lg, const force &pull_vector) {
  PullAlignedFrameData data;
  data.rec_frame = rec.m_mol[0];
  data.lg_frame = lg.m_mol[0];

  double rec_com_arr[3]{};
  double lg_com_arr[3]{};
  mass_weighted_com(rec.m_mol[0], rec.parameters, rec_com_arr);
  mass_weighted_com(lg.m_mol[0], lg.parameters, lg_com_arr);
  data.rec_com_original = array_to_force(
      std::array<double, 3>{rec_com_arr[0], rec_com_arr[1], rec_com_arr[2]});
  data.lg_com_original = array_to_force(
      std::array<double, 3>{lg_com_arr[0], lg_com_arr[1], lg_com_arr[2]});
  data.original_delta = data.lg_com_original - data.rec_com_original;

  const PullRotationMatrix rotation = rotation_matrix_to_positive_z(pull_vector);
  rotate_mol_about_origin(data.rec_frame, force_to_array(data.rec_com_original),
                          rotation);
  rotate_mol_about_origin(data.lg_frame, force_to_array(data.rec_com_original),
                          rotation);
  data.aligned_pull_vector = array_to_force(
      apply_rotation(rotation, force_to_array(pull_vector)));

  std::array<double, 3> mins{std::numeric_limits<double>::max(),
                             std::numeric_limits<double>::max(),
                             std::numeric_limits<double>::max()};
  std::array<double, 3> maxs{std::numeric_limits<double>::lowest(),
                             std::numeric_limits<double>::lowest(),
                             std::numeric_limits<double>::lowest()};
  update_bounds_from_mol(data.rec_frame, mins, maxs);
  update_bounds_from_mol(data.lg_frame, mins, maxs);

  constexpr double kPullAlignedPaddingAng = 15.0;
  const std::array<double, 3> shift{kPullAlignedPaddingAng - mins[0],
                                    kPullAlignedPaddingAng - mins[1],
                                    kPullAlignedPaddingAng - mins[2]};
  shift_mol(data.rec_frame, shift);
  shift_mol(data.lg_frame, shift);

  const double lx = (maxs[0] - mins[0]) + 2.0 * kPullAlignedPaddingAng;
  const double ly = (maxs[1] - mins[1]) + 2.0 * kPullAlignedPaddingAng;
  const double lz = (maxs[2] - mins[2]) + 2.0 * kPullAlignedPaddingAng;
  data.aligned_box = box{};
  data.aligned_box.set_a(lx);
  data.aligned_box.set_b(ly);
  data.aligned_box.set_c(lz);
  data.aligned_box.set_alpha(90.0);
  data.aligned_box.set_beta(90.0);
  data.aligned_box.set_gamma(90.0);
  data.rec_frame.Box = data.aligned_box;
  data.lg_frame.Box = data.aligned_box;

  mass_weighted_com(data.rec_frame, rec.parameters, rec_com_arr);
  mass_weighted_com(data.lg_frame, lg.parameters, lg_com_arr);
  data.rec_com_aligned = array_to_force(
      std::array<double, 3>{rec_com_arr[0], rec_com_arr[1], rec_com_arr[2]});
  data.lg_com_aligned = array_to_force(
      std::array<double, 3>{lg_com_arr[0], lg_com_arr[1], lg_com_arr[2]});
  data.aligned_delta = data.lg_com_aligned - data.rec_com_aligned;
  return data;
}

void write_pull_snapshot_atom_block(std::ostream &out, const MOL &mol,
                                    const vector<mol_info> &info,
                                    std::size_t serial_offset) {
  std::size_t serial = 0;
  for (std::size_t i = 0; i < mol.size(); ++i) {
    serial = std::min<std::size_t>(99999, serial_offset + i + 1);
    out << std::setw(4) << "ATOM" << std::setw(7) << serial << std::setw(5)
        << info[i].get_atom() << std::right << std::setw(5)
        << info[i].get_original_res_name() << std::setw(1) << info[i].get_chain()
        << std::setw(4) << info[i].get_resnr() << std::setw(12) << mol.pos.x[i]
        << std::setw(8) << mol.pos.y[i] << std::setw(8) << mol.pos.z[i] << " "
        << std::setw(7) << 1.00 << " " << std::setw(6) << 0.00 << '\n';
  }
}

bool write_pull_direction_aligned_snapshot(
    const PullAlignedFrameData &aligned_data, const Multi_MOL &rec,
    const Multi_MOL &lg, const info &output_info) {
  std::fstream out(output_info.get_pull_direction_aligned_pdb_filename(),
                   std::ios::out);
  if (!out.is_open())
    return false;

  out << std::fixed << std::setprecision(3);
  out << "REMARK    GENERATED BY TITAN PULL-DIRECT\n";
  out << "REMARK    ROTATED SO THAT BEST PULL VECTOR OF MOL2 POINTS TO +Z\n";
  out << "TITLE     frame t= " << std::setw(9) << aligned_data.rec_frame.time_of_snap
      << '\n';
  out << "CRYST1" << std::setw(9) << aligned_data.aligned_box.get_a()
      << std::setw(9) << aligned_data.aligned_box.get_b() << std::setw(9)
      << aligned_data.aligned_box.get_c() << std::setw(7)
      << aligned_data.aligned_box.get_alpha() << std::setw(7)
      << aligned_data.aligned_box.get_beta() << std::setw(7)
      << aligned_data.aligned_box.get_gamma() << '\n';
  out << "MODEL" << std::setw(9) << 1 << '\n';
  write_pull_snapshot_atom_block(out, aligned_data.rec_frame, rec.info, 0);
  out << "TER\n";
  write_pull_snapshot_atom_block(out, aligned_data.lg_frame, lg.info,
                                 aligned_data.rec_frame.size());
  out << "TER\nENDMDL\nEND\n";
  return true;
}

bool write_pull_direction_system_info(const PullAlignedFrameData &aligned_data,
                                      const force &pull_vector,
                                      const info &output_info) {
  std::fstream out(output_info.get_pull_direction_system_info_filename(),
                   std::ios::out);
  if (!out.is_open())
    return false;

  out << std::fixed << std::setprecision(4);
  out << "Pull Direction System Info\n";
  out << "==========================\n\n";
  out << "Original frame\n";
  out << "--------------\n";
  out << "COM mol1               : (" << aligned_data.rec_com_original.Fx << ", "
      << aligned_data.rec_com_original.Fy << ", " << aligned_data.rec_com_original.Fz
      << ")\n";
  out << "COM mol2               : (" << aligned_data.lg_com_original.Fx << ", "
      << aligned_data.lg_com_original.Fy << ", " << aligned_data.lg_com_original.Fz
      << ")\n";
  out << "Original pull vector   : (" << pull_vector.Fx << ", " << pull_vector.Fy
      << ", " << pull_vector.Fz << ")\n";
  out << "Original COM delta     : (" << aligned_data.original_delta.Fx << ", "
      << aligned_data.original_delta.Fy << ", " << aligned_data.original_delta.Fz
      << ")\n\n";
  out << "Aligned frame (+Z)\n";
  out << "------------------\n";
  out << "COM mol1               : (" << aligned_data.rec_com_aligned.Fx << ", "
      << aligned_data.rec_com_aligned.Fy << ", " << aligned_data.rec_com_aligned.Fz
      << ")\n";
  out << "COM mol2               : (" << aligned_data.lg_com_aligned.Fx << ", "
      << aligned_data.lg_com_aligned.Fy << ", " << aligned_data.lg_com_aligned.Fz
      << ")\n";
  out << "Aligned COM delta      : (" << aligned_data.aligned_delta.Fx << ", "
      << aligned_data.aligned_delta.Fy << ", " << aligned_data.aligned_delta.Fz
      << ")\n";
  out << "distance_along_+z      : " << aligned_data.aligned_delta.Fz << "\n";
  out << "Aligned box (A)        : (" << aligned_data.aligned_box.get_a() << ", "
      << aligned_data.aligned_box.get_b() << ", " << aligned_data.aligned_box.get_c()
      << ")\n";
  return true;
}

PullEnergyBackend detect_pull_energy_backend(const arguments &Arguments) {
  if (Arguments.force_f.find("CHARMM") != string::npos)
    return PullEnergyBackend::Charmm;
  if (Arguments.force_f.find("OPLSAA") != string::npos)
    return PullEnergyBackend::Opls;
  if (Arguments.force_f.find("AMBER") != string::npos)
    return PullEnergyBackend::Amber;
  if (Arguments.force_f.find("MARTINI") != string::npos)
    return PullEnergyBackend::Martini;
  return PullEnergyBackend::Gromos;
}

void eval_pull_energy(PullEnergyBackend backend, const MOL &rec, const MOL &lg,
                      const vector<mol_info> &info_rec,
                      const vector<mol_info> &info_lg,
                      cache_sigma_epsilon &CachedData, size_t begin,
                      size_t end, cache_energy &Energy,
                      const arguments &Arguments,
                      const Multi_MOL_parameters &rec_parameters,
                      const Multi_MOL_parameters &lg_parameters) {
#if defined(__AVX__)
  switch (backend) {
  case PullEnergyBackend::Amber:
    cal_energy_amber_intrin(rec, lg, info_rec, info_lg, CachedData, begin, end,
                            Energy, 0, Arguments, rec_parameters,
                            lg_parameters);
    return;
  case PullEnergyBackend::Opls:
    cal_energy_opls_intrin(rec, lg, info_rec, info_lg, CachedData, begin, end,
                           Energy, 0, Arguments, rec_parameters,
                           lg_parameters);
    return;
  case PullEnergyBackend::Martini:
    cal_energy_martini_intrin(rec, lg, info_rec, info_lg, CachedData, begin,
                              end, Energy, 0, Arguments, rec_parameters,
                              lg_parameters);
    return;
  case PullEnergyBackend::Gromos:
    cal_energy_gromos_intrin(rec, lg, info_rec, info_lg, CachedData, begin,
                             end, Energy, 0, Arguments, rec_parameters,
                             lg_parameters);
    return;
  case PullEnergyBackend::Charmm:
    cal_energy_amber_intrin(rec, lg, info_rec, info_lg, CachedData, begin, end,
                            Energy, 0, Arguments, rec_parameters,
                            lg_parameters);
    return;
  }
#endif

  switch (backend) {
  case PullEnergyBackend::Amber:
  case PullEnergyBackend::Charmm:
    cal_energy_amber(rec, lg, info_rec, info_lg, CachedData, begin, end, Energy,
                     0, Arguments, rec_parameters, lg_parameters);
    break;
  case PullEnergyBackend::Opls:
    cal_energy_opls(rec, lg, info_rec, info_lg, CachedData, begin, end, Energy,
                    0, Arguments, rec_parameters, lg_parameters);
    break;
  case PullEnergyBackend::Martini:
    cal_energy_martini(rec, lg, info_rec, info_lg, CachedData, begin, end,
                       Energy, 0, Arguments, rec_parameters, lg_parameters);
    break;
  case PullEnergyBackend::Gromos:
    cal_energy_gromos(rec, lg, info_rec, info_lg, CachedData, begin, end,
                      Energy, 0, Arguments, rec_parameters, lg_parameters);
    break;
  }
}

cache_energy pull_reference_energy(const Multi_MOL &rec, const Multi_MOL &lg,
                                   cache_sigma_epsilon &CachedData,
                                   const arguments &Arguments,
                                   TitanExecutor &executor,
                                   PullEnergyBackend backend) {
  const std::size_t requested_workers = std::max<std::size_t>(1, Arguments.nt);
  const std::size_t worker_slots = std::max<std::size_t>(
      1, std::min(executor.capacity(), requested_workers));
  std::vector<cache_energy> parts(worker_slots);
  const std::size_t lg_size = lg.m_mol[0].size();
  executor.parallel_for(
      lg_size, requested_workers,
      std::max<std::size_t>(1, lg_size / worker_slots),
      [&](std::size_t worker_id, std::size_t begin, std::size_t end) {
        eval_pull_energy(backend, rec.m_mol[0], lg.m_mol[0], rec.info, lg.info,
                         CachedData, begin, end, parts[worker_id], Arguments,
                         rec.parameters, lg.parameters);
      });
  cache_energy total;
  for (const auto &part : parts)
    total += part;
  return total;
}

void translate_seed_to_scratch(const MOL &seed, MOL &scratch, double dX,
                               double dY, double dZ) {
  titan_simd_dispatch::translate_xyz(
      scratch.pos.x.data(), scratch.pos.y.data(), scratch.pos.z.data(),
      seed.pos.x.data(), seed.pos.y.data(), seed.pos.z.data(), seed.size(), dX,
      dY, dZ);
}

force sanitize_spherical_candidate(force spherical) {
  spherical.Fx = clamp_theta(spherical.Fx);
  spherical.Fy = normalize_angle_2pi(spherical.Fy);
  spherical.Fz = std::fabs(spherical.Fz);
  return spherical;
}

force make_force_xyz(double x, double y, double z) {
  force value;
  value.Fx = x;
  value.Fy = y;
  value.Fz = z;
  return value;
}

bool score_improves(double lhs, double rhs) {
  return lhs + kPullScoreImproveEps < rhs;
}

std::size_t effective_direct_eval_budget(const de_arg &DE_arguments) {
  if (DE_arguments.pull_max_evals != 0)
    return DE_arguments.pull_max_evals;

  const std::size_t fallback_iters =
      std::max<std::size_t>(1, DE_arguments.ite_num);
  const std::size_t fallback_pop =
      std::max<std::size_t>(8, DE_arguments.pop_num);
  return fallback_iters * fallback_pop;
}

std::size_t effective_total_eval_budget(const de_arg &DE_arguments) {
  if (optimizer_is_direct_family(DE_arguments.pull_opt))
    return std::max<std::size_t>(effective_direct_eval_budget(DE_arguments), 1024);
  return DE_arguments.pull_max_evals;
}

std::size_t clamp_pattern_tail_budget(std::size_t total_budget) {
  if (total_budget == 0)
    return 0;
  return std::clamp<std::size_t>(total_budget / 8, 48, 160);
}

double direct_local_threshold_size(const de_arg &DE_arguments) {
  const double rho_range =
      std::max(1.0e-6, DE_arguments.scan_radius - DE_arguments.min_scan_radius);
  const double theta_norm = (4.0 * PI / 180.0) / PI;
  const double phi_norm = (8.0 * PI / 180.0) / (2.0 * PI);
  const double rho_norm = 0.08 / rho_range;
  return std::max(theta_norm, std::max(phi_norm, rho_norm));
}

std::array<double, 3>
variant_pattern_initial_steps(const PullDeVariantStats &stats,
                              const de_arg &DE_arguments) {
  const double min_angle_step = kPatternMinAngleStepDeg * PI / 180.0;
  const double default_angle_step = kPatternInitAngleStepDeg * PI / 180.0;
  const double rho_range =
      std::max(1.0e-6, DE_arguments.scan_radius - DE_arguments.min_scan_radius);
  const double default_rho_step = std::max(0.1, 0.25 * rho_range);
  const double population_scale =
      std::max(1.0, std::sqrt(static_cast<double>(
                        std::max<std::size_t>(1, stats.final_population))));
  const double theta_seed =
      std::max(stats.final_theta_spread_rad, default_angle_step);
  const double phi_seed =
      std::max(stats.final_phi_spread_rad, default_angle_step);
  const double rho_seed = std::max(stats.final_rho_spread, default_rho_step);
  return {std::max(min_angle_step, theta_seed / population_scale),
          std::max(min_angle_step, phi_seed / population_scale),
          std::max(kPatternMinRhoStep, rho_seed / population_scale)};
}

double variant_seed_distance_norm(const force &lhs, const force &rhs,
                                  const de_arg &DE_arguments) {
  const double theta_norm = std::fabs(lhs.Fx - rhs.Fx) / PI;
  double phi_delta = std::fabs(lhs.Fy - rhs.Fy);
  if (phi_delta > PI)
    phi_delta = 2.0 * PI - phi_delta;
  const double phi_norm = phi_delta / PI;
  const double rho_range =
      std::max(1.0e-6, DE_arguments.scan_radius - DE_arguments.min_scan_radius);
  const double rho_norm = std::fabs(lhs.Fz - rhs.Fz) / rho_range;
  return std::sqrt(theta_norm * theta_norm + phi_norm * phi_norm +
                   rho_norm * rho_norm);
}

std::vector<std::size_t> rank_valid_candidates_by_score(
    const std::vector<PullCandidateState> &population) {
  std::vector<std::size_t> ranked;
  ranked.reserve(population.size());
  for (std::size_t i = 0; i < population.size(); ++i) {
    if (population[i].valid)
      ranked.push_back(i);
  }
  std::stable_sort(ranked.begin(), ranked.end(),
                   [&](std::size_t lhs, std::size_t rhs) {
                     return population[lhs].score < population[rhs].score;
                   });
  return ranked;
}

std::vector<std::size_t> select_variant_pattern_seed_indices(
    const std::vector<PullCandidateState> &population, const de_arg &DE_arguments,
    std::size_t top_k, double min_seed_distance_norm) {
  const auto ranked = rank_valid_candidates_by_score(population);
  std::vector<std::size_t> selected;
  selected.reserve(std::min(top_k, ranked.size()));
  for (const std::size_t candidate_index : ranked) {
    bool separated = true;
    for (const std::size_t selected_index : selected) {
      if (variant_seed_distance_norm(population[candidate_index].spherical,
                                     population[selected_index].spherical,
                                     DE_arguments) < min_seed_distance_norm) {
        separated = false;
        break;
      }
    }
    if (!separated)
      continue;
    selected.push_back(candidate_index);
    if (selected.size() >= top_k)
      break;
  }
  if (selected.empty() && !ranked.empty())
    selected.push_back(ranked.front());
  return selected;
}

std::string variant_pattern_section_title(PullDeVariant variant) {
  switch (variant) {
  case PullDeVariant::NlShadeLbc:
    return "NL-SHADE-LBC Pattern Refinement";
  case PullDeVariant::Jso:
    return "JSO Pattern Refinement";
  default:
    return "DE Variant Pattern Refinement";
  }
}

PullGeometryContext make_pull_geometry_context(const de_arg &DE_arguments) {
  PullGeometryContext context;
  context.restrain_angle_deg = DE_arguments.restrain_angle;
  if (!DE_arguments.flag_old_vec)
    return context;

  const double old_mod = DE_arguments.old_vector.modul();
  if (old_mod < 1.0e-12)
    return context;

  context.has_old_vector = true;
  context.old_nx = DE_arguments.old_vector.Fx / old_mod;
  context.old_ny = DE_arguments.old_vector.Fy / old_mod;
  context.old_nz = DE_arguments.old_vector.Fz / old_mod;
  return context;
}

std::size_t total_pull_evaluations_used(
    const std::vector<PullWorkerContext> &workers) {
  std::size_t total = 0;
  for (const auto &worker : workers)
    total += worker.evaluations_done;
  return total;
}

bool reserve_eval_budget(PullEvalBudget &budget, PullWorkerContext &worker) {
  if (budget.max_evals == 0) {
    ++worker.evaluations_done;
    return true;
  }
  if (worker.eval_quota_remaining != 0) {
    --worker.eval_quota_remaining;
    ++worker.evaluations_done;
    return true;
  }

  std::size_t current = budget.reserved.load(std::memory_order_relaxed);
  while (current < budget.max_evals) {
    const std::size_t grant =
        std::min(budget.block_size, budget.max_evals - current);
    if (budget.reserved.compare_exchange_weak(current, current + grant,
                                              std::memory_order_relaxed,
                                              std::memory_order_relaxed)) {
      worker.eval_quota_remaining = grant - 1;
      ++worker.evaluations_done;
      return true;
    }
  }
  return false;
}

std::size_t pull_parallel_grain(std::size_t total_items,
                                std::size_t requested_workers,
                                std::size_t ligand_atom_count) {
  const std::size_t workers = std::max<std::size_t>(1, requested_workers);
  if (total_items <= workers)
    return 1;
  const std::size_t min_items_per_worker =
      ligand_atom_count >= 2048 ? 4
      : ligand_atom_count >= 1024 ? 6
      : ligand_atom_count >= 256  ? 8
                                : 12;
  return std::max<std::size_t>(
      min_items_per_worker,
      (total_items + workers * 2 - 1) / (workers * 2));
}

std::size_t pull_parallel_workers(std::size_t total_items,
                                  std::size_t requested_workers,
                                  std::size_t min_grain,
                                  std::size_t max_workers) {
  if (total_items == 0)
    return 1;
  const std::size_t resolved_grain = std::max<std::size_t>(1, min_grain);
  const std::size_t grain_limited =
      (total_items + resolved_grain - 1) / resolved_grain;
  const std::size_t requested = std::max<std::size_t>(1, requested_workers);
  return std::max<std::size_t>(
      1, std::min(max_workers, std::min(requested, grain_limited)));
}

std::size_t pull_eval_budget_block_size(std::size_t population_size,
                                        std::size_t worker_slots) {
  const std::size_t workers = std::max<std::size_t>(1, worker_slots);
  const std::size_t per_worker =
      std::max<std::size_t>(1, (population_size + workers - 1) / workers);
  return std::clamp<std::size_t>(std::max<std::size_t>(8, per_worker / 2), 8,
                                 64);
}

PullPreparedGeometry prepare_pull_geometry(const force &candidate_spherical,
                                          const de_arg &DE_arguments,
                                          const PullGeometryContext &context) {
  PullPreparedGeometry geometry;
  if (!check_radius(DE_arguments, candidate_spherical))
    return geometry;

  const double sin_theta = std::sin(candidate_spherical.Fx);
  const double cos_theta = std::cos(candidate_spherical.Fx);
  const double cos_phi = std::cos(candidate_spherical.Fy);
  const double sin_phi = std::sin(candidate_spherical.Fy);
  geometry.translation = make_force_xyz(candidate_spherical.Fz * sin_theta * cos_phi,
                                        candidate_spherical.Fz * sin_theta * sin_phi,
                                        candidate_spherical.Fz * cos_theta);

  if (!context.has_old_vector) {
    geometry.angle_deg = 0.0;
    geometry.valid = true;
    return geometry;
  }

  double cos_val = sin_theta * cos_phi * context.old_nx +
                   sin_theta * sin_phi * context.old_ny +
                   cos_theta * context.old_nz;
  cos_val = std::clamp(cos_val, -1.0, 1.0);
  geometry.angle_deg = std::acos(cos_val) * 57.2957795;
  geometry.valid = geometry.angle_deg < context.restrain_angle_deg;
  return geometry;
}

force normalized_to_spherical(const std::array<double, 3> &center_norm,
                              const de_arg &DE_arguments) {
  const double theta = PI * center_norm[0];
  const double phi = 2.0 * PI * center_norm[1];
  const double rho = DE_arguments.min_scan_radius +
                     (DE_arguments.scan_radius - DE_arguments.min_scan_radius) *
                         center_norm[2];
  return sanitize_spherical_candidate(make_force_xyz(theta, phi, rho));
}

double direct_cell_size_metric(const PullDirectCell &cell) {
  return std::max(cell.half_width[0],
                  std::max(cell.half_width[1], cell.half_width[2]));
}

double direct_cell_order_score(const PullDirectCell &cell) {
  return cell.state.valid ? cell.state.score : kPullInvalidScore;
}

bool direct_cell_preferred(const PullDirectCell &lhs, const PullDirectCell &rhs) {
  const double lhs_size = direct_cell_size_metric(lhs);
  const double rhs_size = direct_cell_size_metric(rhs);
  if (lhs.state.valid != rhs.state.valid)
    return lhs.state.valid;
  if (lhs.state.valid && rhs.state.valid &&
      std::fabs(lhs.state.score - rhs.state.score) > kPullScoreImproveEps) {
    return lhs.state.score < rhs.state.score;
  }
  if (std::fabs(lhs_size - rhs_size) > 1.0e-12)
    return lhs_size > rhs_size;
  if (lhs.depth != rhs.depth)
    return lhs.depth < rhs.depth;
  return lhs.id < rhs.id;
}

std::size_t select_direct_cell_index(const std::vector<PullDirectCell> &frontier) {
  double largest_size = 0.0;
  for (const auto &cell : frontier)
    largest_size = std::max(largest_size, direct_cell_size_metric(cell));
  const double size_cutoff = largest_size * 0.5;

  std::size_t chosen = frontier.size();
  for (std::size_t i = 0; i < frontier.size(); ++i) {
    if (direct_cell_size_metric(frontier[i]) + 1.0e-12 < size_cutoff)
      continue;
    if (chosen == frontier.size() ||
        direct_cell_preferred(frontier[i], frontier[chosen])) {
      chosen = i;
    }
  }

  if (chosen != frontier.size())
    return chosen;

  chosen = 0;
  for (std::size_t i = 1; i < frontier.size(); ++i) {
    if (direct_cell_preferred(frontier[i], frontier[chosen]))
      chosen = i;
  }
  return chosen;
}

bool direct_cell_dominates(const PullDirectCell &lhs, const PullDirectCell &rhs) {
  const double lhs_size = direct_cell_size_metric(lhs);
  const double rhs_size = direct_cell_size_metric(rhs);
  const double lhs_score = direct_cell_order_score(lhs);
  const double rhs_score = direct_cell_order_score(rhs);

  const bool no_worse_size = lhs_size + 1.0e-12 >= rhs_size;
  const bool no_worse_score = lhs_score <= rhs_score + kPullScoreImproveEps;
  const bool strictly_better =
      lhs_size > rhs_size + 1.0e-12 ||
      score_improves(lhs_score, rhs_score);
  return no_worse_size && no_worse_score && strictly_better;
}

std::vector<std::size_t> select_direct_cell_batch_indices(
    const std::vector<PullDirectCell> &frontier, std::size_t max_cells,
    double current_best_score) {
  std::vector<std::size_t> pareto;
  pareto.reserve(frontier.size());
  for (std::size_t i = 0; i < frontier.size(); ++i) {
    bool dominated = false;
    for (std::size_t j = 0; j < frontier.size(); ++j) {
      if (i == j)
        continue;
      if (direct_cell_dominates(frontier[j], frontier[i])) {
        dominated = true;
        break;
      }
    }
    if (!dominated)
      pareto.push_back(i);
  }

  if (pareto.empty())
    return {select_direct_cell_index(frontier)};

  std::size_t best_score_index = pareto.front();
  for (const std::size_t index : pareto) {
    if (direct_cell_order_score(frontier[index]) <
        direct_cell_order_score(frontier[best_score_index]) -
            kPullScoreImproveEps) {
      best_score_index = index;
    } else if (std::fabs(direct_cell_order_score(frontier[index]) -
                         direct_cell_order_score(frontier[best_score_index])) <=
                   kPullScoreImproveEps &&
               direct_cell_size_metric(frontier[index]) >
                   direct_cell_size_metric(frontier[best_score_index]) +
                       1.0e-12) {
      best_score_index = index;
    }
  }

  std::sort(pareto.begin(), pareto.end(),
            [&](std::size_t lhs, std::size_t rhs) {
              const double lhs_gap = std::fabs(direct_cell_order_score(frontier[lhs]) -
                                               current_best_score);
              const double rhs_gap = std::fabs(direct_cell_order_score(frontier[rhs]) -
                                               current_best_score);
              if (std::fabs(lhs_gap - rhs_gap) > kPullScoreImproveEps)
                return lhs_gap < rhs_gap;
              const double lhs_size = direct_cell_size_metric(frontier[lhs]);
              const double rhs_size = direct_cell_size_metric(frontier[rhs]);
              if (std::fabs(lhs_size - rhs_size) > 1.0e-12)
                return lhs_size > rhs_size;
              const double lhs_score = direct_cell_order_score(frontier[lhs]);
              const double rhs_score = direct_cell_order_score(frontier[rhs]);
              if (std::fabs(lhs_score - rhs_score) > kPullScoreImproveEps)
                return lhs_score < rhs_score;
              return frontier[lhs].id < frontier[rhs].id;
            });

  std::vector<std::size_t> selected;
  selected.reserve(std::min(max_cells, pareto.size()));
  selected.push_back(best_score_index);
  for (const std::size_t index : pareto) {
    if (selected.size() >= max_cells)
      break;
    if (index == best_score_index)
      continue;
    selected.push_back(index);
  }

  std::sort(selected.begin(), selected.end());
  return selected;
}

std::size_t direct_split_dimension(const PullDirectCell &cell) {
  std::size_t dimension = 0;
  double max_width = cell.half_width[0];
  for (std::size_t dim = 1; dim < 3; ++dim) {
    if (cell.half_width[dim] > max_width + 1.0e-12) {
      max_width = cell.half_width[dim];
      dimension = dim;
    }
  }
  return dimension;
}

double angle_to_previous_vector(const force &translation,
                                const force &old_vector) {
  const double old_mod = old_vector.modul();
  const double new_mod = translation.modul();
  if (old_mod < 1.0e-12 || new_mod < 1.0e-12)
    return 0.0;
  force scaled_new{translation};
  force scaled_old{old_vector};
  scaled_new *= old_mod;
  scaled_old *= new_mod;
  return 360.0 / PI *
         std::atan2(force(scaled_new - scaled_old).modul(),
                    force(scaled_new + scaled_old).modul());
}

force sample_initial_spherical_candidate(std::mt19937 &mt,
                                         de_arg &DE_arguments) {
  const double PI_VAL = 3.14159265358979323846;
  std::uniform_real_distribution<double> dist_phi(0.0, 2.0 * PI_VAL);
  std::uniform_real_distribution<double> dist_cos_theta(-1.0, 1.0);
  std::uniform_real_distribution<double> dist_rho(DE_arguments.min_scan_radius,
                                                  DE_arguments.scan_radius);
  std::normal_distribution<double> dist_noise(0.0, 0.1);

  double theta_ref = 0.0;
  double phi_ref = 0.0;
  if (DE_arguments.flag_old_vec) {
    cartesian_to_spherical(DE_arguments.old_vector.Fx, DE_arguments.old_vector.Fy,
                           DE_arguments.old_vector.Fz, theta_ref, phi_ref);
  }

  for (int attempts = 0; attempts < 500; ++attempts) {
    force spherical{std::acos(dist_cos_theta(mt)), dist_phi(mt), dist_rho(mt)};
    double crit_angle = 180.0;
    if (!DE_arguments.flag_old_vec || parse_angle(DE_arguments, spherical, crit_angle))
      return spherical;
  }

  force fallback{theta_ref + dist_noise(mt), phi_ref + dist_noise(mt),
                 dist_rho(mt)};
  fallback.Fx = clamp_theta(fallback.Fx);
  fallback.Fy = normalize_angle_2pi(fallback.Fy);
  fallback.Fz = std::fabs(fallback.Fz);
  return fallback;
}

std::mt19937 make_candidate_rng(unsigned int seed, std::size_t iteration,
                                std::size_t index) {
  std::seed_seq seq{seed, static_cast<unsigned int>(iteration),
                    static_cast<unsigned int>(iteration >> 32),
                    static_cast<unsigned int>(index),
                    static_cast<unsigned int>(index >> 32),
                    static_cast<unsigned int>(seed ^ 0x9e3779b9U)};
  return std::mt19937(seq);
}

void choose_distinct_triplet(std::mt19937 &mt, std::size_t pop_size,
                             std::size_t self, std::size_t abc[3]) {
  std::uniform_int_distribution<std::size_t> choose(0, pop_size - 1);
  abc[0] = choose(mt);
  abc[1] = choose(mt);
  abc[2] = choose(mt);
  while (abc[0] == abc[1] || abc[0] == abc[2] || abc[1] == abc[2] ||
         abc[0] == self || abc[1] == self || abc[2] == self) {
    abc[0] = choose(mt);
    abc[1] = choose(mt);
    abc[2] = choose(mt);
  }
}

bool build_trial_spherical(const std::vector<PullCandidateState> &population,
                           std::size_t index, std::size_t iteration,
                           de_arg &DE_arguments, force &trial_spherical) {
  std::mt19937 mt =
      make_candidate_rng(DE_arguments.pull_seed, iteration + 1, index);
  std::uniform_real_distribution<double> prob(0.0, 1.0);
  std::uniform_int_distribution<std::size_t> choose(0, population.size() - 1);

  std::size_t abc[3]{};
  choose_distinct_triplet(mt, population.size(), index, abc);
  const double crit = prob(mt);
  const std::size_t random_index = choose(mt);

  if (DE_arguments.algorithm == 0) {
    if (!(crit < DE_arguments.cross_prob || index == random_index))
      return false;
    trial_spherical.Fx = population[abc[0]].spherical.Fx +
                         DE_arguments.diff_weight *
                             (population[abc[1]].spherical.Fx -
                              population[abc[2]].spherical.Fx);
    trial_spherical.Fy = population[abc[0]].spherical.Fy +
                         DE_arguments.diff_weight *
                             (population[abc[1]].spherical.Fy -
                              population[abc[2]].spherical.Fy);
    trial_spherical.Fz = population[abc[0]].spherical.Fz +
                         DE_arguments.diff_weight *
                             (population[abc[1]].spherical.Fz -
                              population[abc[2]].spherical.Fz);
  } else if (crit < DE_arguments.cross_prob) {
    trial_spherical.Fx = population[abc[0]].spherical.Fx +
                         DE_arguments.F *
                             (population[abc[1]].spherical.Fx -
                              population[abc[2]].spherical.Fx);
    trial_spherical.Fy = population[abc[0]].spherical.Fy +
                         DE_arguments.F *
                             (population[abc[1]].spherical.Fy -
                              population[abc[2]].spherical.Fy);
    trial_spherical.Fz = population[abc[0]].spherical.Fz +
                         DE_arguments.F *
                             (population[abc[1]].spherical.Fz -
                              population[abc[2]].spherical.Fz);
  } else {
    const double blend = 0.5 * (DE_arguments.F + 1.0);
    trial_spherical.Fx = population[abc[0]].spherical.Fx +
                         blend * (population[abc[1]].spherical.Fx +
                                  population[abc[2]].spherical.Fx -
                                  2.0 * population[abc[0]].spherical.Fx);
    trial_spherical.Fy = population[abc[0]].spherical.Fy +
                         blend * (population[abc[1]].spherical.Fy +
                                  population[abc[2]].spherical.Fy -
                                  2.0 * population[abc[0]].spherical.Fy);
    trial_spherical.Fz = population[abc[0]].spherical.Fz +
                         blend * (population[abc[1]].spherical.Fz +
                                  population[abc[2]].spherical.Fz -
                                  2.0 * population[abc[0]].spherical.Fz);
  }

  trial_spherical = sanitize_spherical_candidate(trial_spherical);
  return true;
}

bool evaluate_pull_candidate(const force &candidate_spherical,
                             PullCandidateState &result, const Multi_MOL &rec,
                             const Multi_MOL &lg, PullWorkerContext &worker,
                             const cache_energy &energy_ref,
                             const arguments &Arguments,
                             const de_arg &DE_arguments,
                             PullEnergyBackend backend,
                             const PullGeometryContext &geometry_context,
                             PullEvalBudget &eval_budget) {
  const PullPreparedGeometry geometry =
      prepare_pull_geometry(candidate_spherical, DE_arguments, geometry_context);
  if (!geometry.valid)
    return false;
  if (!reserve_eval_budget(eval_budget, worker))
    return false;

  result.energy = {};
  result.spherical = candidate_spherical;
  result.translation = geometry.translation;
  result.angle_deg = geometry.angle_deg;
  result.valid = false;

  translate_seed_to_scratch(lg.m_mol[0], worker.scratch, result.translation.Fx,
                            result.translation.Fy, result.translation.Fz);
  const std::size_t scratch_size = worker.scratch.size();
  eval_pull_energy(backend, rec.m_mol[0], worker.scratch, rec.info, lg.info,
                   worker.eval_cache, 0, scratch_size, result.energy,
                   Arguments, rec.parameters, lg.parameters);
  result.score = cal_score_function(result.energy, energy_ref,
                                    Arguments.de.de_flag_score_function, 0.0,
                                    Arguments.de.dis_along_path);
  result.valid = std::isfinite(result.score);
  return result.valid;
}

PullDeBatchEvaluation evaluate_pull_candidate_batch(
    std::span<const force> proposals, std::span<PullCandidateState> results,
    const Multi_MOL &rec, const Multi_MOL &lg,
    std::vector<PullWorkerContext> &workers, const cache_energy &energy_ref,
    const arguments &Arguments, const de_arg &DE_arguments,
    PullEnergyBackend backend, const PullGeometryContext &geometry_context,
    PullEvalBudget &eval_budget, TitanExecutor &executor,
    std::size_t requested_workers, std::size_t ligand_atom_count) {
  PullDeBatchEvaluation batch;
  if (proposals.empty())
    return batch;

  const std::size_t before = total_pull_evaluations_used(workers);
  const std::size_t grain =
      pull_parallel_grain(proposals.size(), requested_workers, ligand_atom_count);
  executor.parallel_for(
      proposals.size(), requested_workers, grain,
      [&](std::size_t worker_id, std::size_t begin, std::size_t end) {
        auto &worker = workers[worker_id];
        for (std::size_t i = begin; i < end; ++i) {
          results[i] = PullCandidateState{};
          results[i].spherical = proposals[i];
          evaluate_pull_candidate(proposals[i], results[i], rec, lg, worker,
                                  energy_ref, Arguments, DE_arguments, backend,
                                  geometry_context, eval_budget);
        }
      });
  const std::size_t after = total_pull_evaluations_used(workers);
  batch.evaluations_used = after - before;
  batch.budget_exhausted =
      (eval_budget.max_evals != 0 && after >= eval_budget.max_evals);
  return batch;
}

std::size_t best_candidate_index(
    const std::vector<PullCandidateState> &population) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < population.size(); ++i) {
    if (population[i].score < population[best].score)
      best = i;
  }
  return best;
}

PullPatternStats refine_pull_candidate_pattern(
    PullCandidateState &best_candidate, const Multi_MOL &rec,
    const Multi_MOL &lg, std::vector<PullWorkerContext> &workers,
    const cache_energy &energy_ref, const arguments &Arguments,
    de_arg &DE_arguments, PullEnergyBackend backend,
    const PullGeometryContext &geometry_context, PullEvalBudget &eval_budget,
    TitanExecutor &executor, double initial_theta_step_rad,
    double initial_phi_step_rad, double initial_rho_step) {
  PullPatternStats stats;
  stats.enabled = true;
  stats.initial_score = best_candidate.score;

  const auto started_at = std::chrono::steady_clock::now();
  const std::size_t evals_before = total_pull_evaluations_used(workers);
  const std::size_t requested_workers = std::max<std::size_t>(1, Arguments.nt);
  const std::size_t ligand_atom_count = lg.m_mol[0].size();
  const double angle_to_rad = PI / 180.0;
  const double min_theta_step = kPatternMinAngleStepDeg * angle_to_rad;
  const double min_phi_step = kPatternMinAngleStepDeg * angle_to_rad;
  double theta_step = std::max(initial_theta_step_rad,
                               kPatternMinAngleStepDeg * angle_to_rad);
  double phi_step =
      std::max(initial_phi_step_rad, kPatternMinAngleStepDeg * angle_to_rad);
  double rho_step = std::max(
      initial_rho_step,
      std::max(0.1, 0.25 * (DE_arguments.scan_radius - DE_arguments.min_scan_radius)));
  if (rho_step <= 0.0)
    rho_step = 0.1;

  while (stats.sweeps < kPatternMaxSweeps &&
         (theta_step > min_theta_step || phi_step > min_phi_step ||
          rho_step > kPatternMinRhoStep)) {
    if (eval_budget.max_evals != 0 &&
        total_pull_evaluations_used(workers) >= eval_budget.max_evals) {
      stats.stopped_by_eval_budget = true;
      break;
    }

    std::array<force, 6> proposals{
        sanitize_spherical_candidate(
            make_force_xyz(best_candidate.spherical.Fx + theta_step,
                           best_candidate.spherical.Fy,
                           best_candidate.spherical.Fz)),
        sanitize_spherical_candidate(
            make_force_xyz(best_candidate.spherical.Fx - theta_step,
                           best_candidate.spherical.Fy,
                           best_candidate.spherical.Fz)),
        sanitize_spherical_candidate(
            make_force_xyz(best_candidate.spherical.Fx,
                           best_candidate.spherical.Fy + phi_step,
                           best_candidate.spherical.Fz)),
        sanitize_spherical_candidate(
            make_force_xyz(best_candidate.spherical.Fx,
                           best_candidate.spherical.Fy - phi_step,
                           best_candidate.spherical.Fz)),
        sanitize_spherical_candidate(
            make_force_xyz(best_candidate.spherical.Fx,
                           best_candidate.spherical.Fy,
                           best_candidate.spherical.Fz + rho_step)),
        sanitize_spherical_candidate(
            make_force_xyz(best_candidate.spherical.Fx,
                           best_candidate.spherical.Fy,
                           best_candidate.spherical.Fz - rho_step))};
    std::array<PullCandidateState, 6> trial_results{};

    executor.parallel_for(
        proposals.size(), requested_workers,
        pull_parallel_grain(proposals.size(), requested_workers,
                            ligand_atom_count),
        [&](std::size_t worker_id, std::size_t begin, std::size_t end) {
          auto &worker = workers[worker_id];
          for (std::size_t i = begin; i < end; ++i) {
            evaluate_pull_candidate(proposals[i], trial_results[i], rec, lg,
                                    worker, energy_ref, Arguments, DE_arguments,
                                    backend, geometry_context, eval_budget);
          }
        });

    ++stats.sweeps;
    std::size_t best_trial = trial_results.size();
    double best_trial_score = best_candidate.score;
    for (std::size_t i = 0; i < trial_results.size(); ++i) {
      const auto &trial = trial_results[i];
      if (!trial.valid)
        continue;
      if (trial.score + kPullScoreImproveEps < best_trial_score) {
        best_trial = i;
        best_trial_score = trial.score;
      }
    }

    if (best_trial < trial_results.size()) {
      best_candidate = trial_results[best_trial];
      stats.improved = true;
      ++stats.accepted_moves;
      continue;
    }

    theta_step *= 0.5;
    phi_step *= 0.5;
    rho_step *= 0.5;
  }

  stats.final_score = best_candidate.score;
  stats.evaluations_used = total_pull_evaluations_used(workers) - evals_before;
  stats.wall_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    started_at)
          .count();
  return stats;
}

PullPatternStats refine_variant_multi_seed_pattern(
    PullCandidateState &best_candidate,
    const std::vector<PullCandidateState> &candidate_pool, const Multi_MOL &rec,
    const Multi_MOL &lg, std::vector<PullWorkerContext> &workers,
    const cache_energy &energy_ref, const arguments &Arguments,
    de_arg &DE_arguments, PullEnergyBackend backend,
    const PullGeometryContext &geometry_context, PullEvalBudget &eval_budget,
    TitanExecutor &executor, const PullDeVariantStats &variant_stats) {
  PullPatternStats aggregate;
  aggregate.enabled = true;
  aggregate.initial_score = best_candidate.score;
  const auto started_at = std::chrono::steady_clock::now();

  const auto seed_indices = select_variant_pattern_seed_indices(
      candidate_pool, DE_arguments, kVariantPatternTopK,
      kVariantPatternMinSeedDistanceNorm);
  aggregate.seeds_attempted = seed_indices.empty() ? 1 : seed_indices.size();
  aggregate.distinct_seed_basins = aggregate.seeds_attempted;

  const auto pattern_steps =
      variant_pattern_initial_steps(variant_stats, DE_arguments);

  if (seed_indices.empty()) {
    PullCandidateState local_best = best_candidate;
    PullPatternStats local_stats = refine_pull_candidate_pattern(
        local_best, rec, lg, workers, energy_ref, Arguments, DE_arguments,
        backend, geometry_context, eval_budget, executor, pattern_steps[0],
        pattern_steps[1], pattern_steps[2]);
    aggregate.sweeps = local_stats.sweeps;
    aggregate.accepted_moves = local_stats.accepted_moves;
    aggregate.evaluations_used = local_stats.evaluations_used;
    aggregate.stopped_by_eval_budget = local_stats.stopped_by_eval_budget;
    if (score_improves(local_best.score, best_candidate.score)) {
      best_candidate = local_best;
      aggregate.improved = true;
    }
  } else {
    for (const std::size_t seed_index : seed_indices) {
      if (eval_budget.max_evals != 0 &&
          total_pull_evaluations_used(workers) >= eval_budget.max_evals) {
        aggregate.stopped_by_eval_budget = true;
        break;
      }
      PullCandidateState local_best = candidate_pool[seed_index];
      PullPatternStats local_stats = refine_pull_candidate_pattern(
          local_best, rec, lg, workers, energy_ref, Arguments, DE_arguments,
          backend, geometry_context, eval_budget, executor, pattern_steps[0],
          pattern_steps[1], pattern_steps[2]);
      aggregate.sweeps += local_stats.sweeps;
      aggregate.accepted_moves += local_stats.accepted_moves;
      aggregate.evaluations_used += local_stats.evaluations_used;
      if (local_stats.improved &&
          score_improves(local_best.score, best_candidate.score)) {
        best_candidate = local_best;
        aggregate.improved = true;
      }
      if (local_stats.stopped_by_eval_budget) {
        aggregate.stopped_by_eval_budget = true;
        break;
      }
    }
  }

  aggregate.final_score = best_candidate.score;
  aggregate.wall_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    started_at)
          .count();
  return aggregate;
}

PullDirectStats run_pull_direct_direct(
    PullCandidateState &best_candidate,
    std::vector<PullCandidateState> &ranked_candidates, std::size_t &best_stage,
    bool &global_budget_stop, const Multi_MOL &rec, const Multi_MOL &lg,
    std::vector<PullWorkerContext> &workers, const cache_energy &energy_ref,
    const arguments &Arguments, const de_arg &DE_arguments,
    PullEnergyBackend backend, const PullGeometryContext &geometry_context,
    PullEvalBudget &eval_budget, std::size_t direct_phase_budget,
    TitanExecutor &executor) {
  PullDirectStats stats;
  stats.enabled = true;
  stats.effective_eval_budget = direct_phase_budget;
  stats.used_fallback_budget = (DE_arguments.pull_max_evals == 0);
  const auto started_at = std::chrono::steady_clock::now();
  const std::size_t evals_before = total_pull_evaluations_used(workers);
  const double local_threshold = direct_local_threshold_size(DE_arguments);
  const std::size_t ligand_atom_count = lg.m_mol[0].size();

  PullDirectCell root;
  root.center_norm = {0.5, 0.5, 0.5};
  root.half_width = {0.5, 0.5, 0.5};
  evaluate_pull_candidate(normalized_to_spherical(root.center_norm, DE_arguments),
                          root.state, rec, lg, workers.front(), energy_ref,
                          Arguments, DE_arguments, backend, geometry_context,
                          eval_budget);

  std::vector<PullDirectCell> frontier;
  frontier.push_back(root);
  if (root.state.valid) {
    best_candidate = root.state;
    ranked_candidates.push_back(root.state);
    stats.best_cell_size = direct_cell_size_metric(root);
    stats.best_half_width_norm = root.half_width;
  }

  std::size_t next_cell_id = 1;
  const std::size_t requested_workers = std::max<std::size_t>(1, Arguments.nt);
  const std::size_t planned_expansions =
      direct_phase_budget > 1 ? (direct_phase_budget - 1) / 2 : 0;
  const std::size_t max_batch_capacity =
      std::max<std::size_t>(1, workers.size() * 2);
  std::size_t expansions_since_best = 0;

  struct SplitPlan {
    std::size_t frontier_index{0};
    PullDirectCell middle{};
    std::array<PullDirectCell, 2> children{};
    bool active{false};
  };
  std::vector<SplitPlan> plans(max_batch_capacity);
  std::vector<PullCandidateState> child_states(max_batch_capacity * 2);

  while (!frontier.empty()) {
    const std::size_t used_total = total_pull_evaluations_used(workers);
    const std::size_t used_phase = used_total - evals_before;
    if (direct_phase_budget != 0 && used_phase >= direct_phase_budget) {
      stats.stopped_by_eval_budget = true;
      if (direct_phase_budget == eval_budget.max_evals)
        global_budget_stop = true;
      break;
    }

    const std::size_t chosen_preview = select_direct_cell_index(frontier);
    const double preview_size = direct_cell_size_metric(frontier[chosen_preview]);
    if (preview_size <= local_threshold) {
      stats.stopped_by_local_threshold = true;
      break;
    }
    if (expansions_since_best >= 32) {
      stats.stopped_by_stagnation = true;
      break;
    }

    if (planned_expansions > 0) {
      TitanUI::show_progress("Pulling Direction", stats.expansions,
                             planned_expansions, "(DIRECT)");
    }

    const std::size_t remaining_evals =
        direct_phase_budget > used_phase
            ? direct_phase_budget - used_phase
            : 0;
    std::size_t batch_cells =
        std::min(max_batch_capacity,
                 std::min(frontier.size(),
                          std::max<std::size_t>(1, remaining_evals / 2)));
    if (batch_cells == 0)
      break;

    const std::vector<std::size_t> selected_indices =
        select_direct_cell_batch_indices(frontier, batch_cells,
                                         best_candidate.score);
    if (selected_indices.empty())
      break;
    std::size_t active_plans = 0;
    for (const std::size_t frontier_index : selected_indices) {
      PullDirectCell chosen = frontier[frontier_index];
      const std::size_t split_dim = direct_split_dimension(chosen);
      const double split_half = chosen.half_width[split_dim] / 3.0;
      if (split_half < 1.0e-4)
        continue;

      SplitPlan plan;
      plan.frontier_index = frontier_index;
      plan.middle = chosen;
      plan.middle.depth = chosen.depth + 1;
      plan.middle.half_width[split_dim] = split_half;
      plan.middle.id = next_cell_id++;
      for (std::size_t child = 0; child < plan.children.size(); ++child) {
        plan.children[child] = plan.middle;
        plan.children[child].center_norm[split_dim] =
            chosen.center_norm[split_dim] +
            (child == 0 ? -split_half : split_half);
        plan.children[child].id = next_cell_id++;
      }
      plan.active = true;
      plans[active_plans++] = plan;
    }

    if (active_plans == 0)
      break;

    stats.max_batch_cells = std::max(stats.max_batch_cells, active_plans);

    executor.parallel_for(
        active_plans * 2, requested_workers,
        pull_parallel_grain(active_plans * 2, requested_workers,
                            ligand_atom_count),
        [&](std::size_t worker_id, std::size_t begin, std::size_t end) {
          auto &worker = workers[worker_id];
          for (std::size_t task = begin; task < end; ++task) {
            const std::size_t plan_index = task / 2;
            const std::size_t child_index = task % 2;
            evaluate_pull_candidate(
                normalized_to_spherical(plans[plan_index].children[child_index]
                                            .center_norm,
                                        DE_arguments),
                child_states[task], rec, lg, worker, energy_ref, Arguments,
                DE_arguments, backend, geometry_context, eval_budget);
          }
        });

    for (std::size_t plan_index = 0; plan_index < active_plans; ++plan_index) {
      frontier[plans[plan_index].frontier_index] = plans[plan_index].middle;
      for (std::size_t child_index = 0; child_index < 2; ++child_index) {
        PullDirectCell child = plans[plan_index].children[child_index];
        child.state = child_states[plan_index * 2 + child_index];
        frontier.push_back(child);
        if (!child.state.valid)
          continue;
        ranked_candidates.push_back(child.state);
        if (!best_candidate.valid ||
            score_improves(child.state.score, best_candidate.score)) {
          best_candidate = child.state;
          best_stage = stats.expansions + plan_index + 1;
          stats.best_expansion = best_stage;
          stats.best_cell_size = direct_cell_size_metric(child);
          stats.best_half_width_norm = child.half_width;
          expansions_since_best = 0;
        }
      }
    }

    expansions_since_best += active_plans;
    stats.expansions += active_plans;
    ++stats.batches;
  }

  stats.cells_remaining = frontier.size();
  stats.evaluated_candidates = ranked_candidates.size();
  stats.phase_evaluations_used =
      total_pull_evaluations_used(workers) - evals_before;
  if (best_candidate.valid && stats.best_expansion == 0)
    stats.best_expansion = best_stage;
  stats.wall_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    started_at)
          .count();
  return stats;
}

} // namespace

int find_pull_direct(Multi_MOL &rec, Multi_MOL &lg,
                     cache_sigma_epsilon &CachedData, de_arg &DE_arguments,
                     arguments &Arguments, info &output_info,
                     TitanExecutor &executor,
                     groups_cal &Groups_from_index_file) {
  if (Arguments.de.flag_pathway) {
    return run_pathway_pull_direction(rec, lg, CachedData, DE_arguments,
                                      Arguments, output_info, executor,
                                      Groups_from_index_file);
  }

  const PullOptimizerKind optimizer_kind =
      classify_pull_optimizer(DE_arguments.pull_opt);
  const bool use_de_family = optimizer_is_de_family(DE_arguments.pull_opt);
  const bool use_direct_family = optimizer_is_direct_family(DE_arguments.pull_opt);
  const bool use_legacy_de =
      (optimizer_kind == PullOptimizerKind::De ||
       optimizer_kind == PullOptimizerKind::DePattern);
  const bool use_variant_de =
      (optimizer_kind == PullOptimizerKind::DeShade ||
       optimizer_kind == PullOptimizerKind::DeLShade ||
       optimizer_kind == PullOptimizerKind::DeJso ||
       optimizer_kind == PullOptimizerKind::DeJsoPattern ||
       optimizer_kind == PullOptimizerKind::DeNlShadeLbc ||
       optimizer_kind == PullOptimizerKind::DeNlShadeLbcPattern);
  const bool use_jso_variant_pattern =
      (optimizer_kind == PullOptimizerKind::DeJsoPattern);
  const bool use_nl_shade_variant_pattern =
      (optimizer_kind == PullOptimizerKind::DeNlShadeLbcPattern);
  const bool use_variant_pattern =
      (use_jso_variant_pattern || use_nl_shade_variant_pattern);
  const bool use_direct_pattern = (DE_arguments.pull_opt == "direct+pattern");
  if (!use_de_family && !use_direct_family) {
    TitanUI::print_error("Pull-direction optimizer '" + DE_arguments.pull_opt +
                          "' is not available in this build.");
    return 1;
  }

  if (rec.m_mol[0].mindist_mol(lg.m_mol[0], Arguments, DE_arguments) == true) {
    TitanUI::print_info("Automatic termination is on!");
    return 1;
  }

  const auto started_at = std::chrono::steady_clock::now();
  const PullEnergyBackend backend = detect_pull_energy_backend(Arguments);
  const PullGeometryContext geometry_context =
      make_pull_geometry_context(DE_arguments);
  const std::size_t requested_workers = std::max<std::size_t>(1, Arguments.nt);
  const std::size_t worker_slots = std::max<std::size_t>(
      1, std::min(executor.capacity(), requested_workers));
  const std::size_t ligand_atom_count = lg.m_mol[0].size();
  PullEvalBudget eval_budget;
  eval_budget.max_evals = effective_total_eval_budget(DE_arguments);
  eval_budget.block_size =
      pull_eval_budget_block_size(DE_arguments.pop_num, worker_slots);

  const cache_energy energy_ref =
      pull_reference_energy(rec, lg, CachedData, Arguments, executor, backend);
  TitanUI::print_info(
      "  Energy ref: elec=" + std::to_string(energy_ref.elec_ener) +
      "  vdw=" + std::to_string(energy_ref.vdw_ener));

  std::vector<PullWorkerContext> workers(worker_slots);
  for (auto &worker : workers) {
    worker.scratch = lg.m_mol[0];
    worker.eval_cache = make_pull_eval_cache(CachedData);
  }

  std::vector<PullCandidateState> report_candidates;
  PullCandidateState best_candidate;
  std::size_t best_stage = 0;
  string_view best_stage_label = "Best iteration";
  string_view candidate_table_title = "Population Energy Table";
  std::size_t direct_phase_budget = 0;
  std::size_t variant_phase_budget = 0;
  std::size_t pattern_reserved_budget = 0;
  std::size_t hybrid_switch_stage = 0;
  int stagnation_count = 0;
  bool stopped_by_eval_budget = false;
  PullPatternStats pattern_stats;
  PullDirectStats direct_stats;
  PullDeVariantStats de_variant_stats;

  if (use_de_family) {
    const auto batch_evaluator =
        [&](std::span<const force> proposals,
            std::span<PullCandidateState> evaluated_candidates) {
          return evaluate_pull_candidate_batch(
              proposals, evaluated_candidates, rec, lg, workers, energy_ref,
              Arguments, DE_arguments, backend, geometry_context, eval_budget,
              executor, requested_workers, ligand_atom_count);
        };

    if (use_legacy_de) {
      std::vector<PullCandidateState> population(DE_arguments.pop_num);
      std::mt19937 init_rng(DE_arguments.pull_seed);
      for (auto &candidate : population)
        candidate.spherical =
            sample_initial_spherical_candidate(init_rng, DE_arguments);

      const std::size_t de_parallel_grain =
          pull_parallel_grain(population.size(), requested_workers,
                              ligand_atom_count);
      const std::size_t de_parallel_workers =
          pull_parallel_workers(population.size(), requested_workers,
                                de_parallel_grain, worker_slots);

      TitanUI::print_info("Generating pull-direction population...");
      executor.parallel_for(
          population.size(), requested_workers, de_parallel_grain,
          [&](std::size_t worker_id, std::size_t begin, std::size_t end) {
            auto &worker = workers[worker_id];
            for (std::size_t i = begin; i < end; ++i) {
              const force initial_spherical = population[i].spherical;
              evaluate_pull_candidate(initial_spherical, population[i], rec, lg,
                                      worker, energy_ref, Arguments,
                                      DE_arguments, backend, geometry_context,
                                      eval_budget);
            }
          });

      std::size_t best_index = best_candidate_index(population);
      if (!population[best_index].valid) {
        TitanUI::print_error(
            "Pull-direction DE: no valid initial candidate was evaluated. "
            "evals=" +
            std::to_string(total_pull_evaluations_used(workers)) +
            " first-score=" + std::to_string(population.front().score) +
            " first-rho=" + std::to_string(population.front().spherical.Fz));
        return 1;
      }

      double current_best_score = population[best_index].score;
      const std::size_t max_updates_per_worker = std::max<std::size_t>(
          1, (population.size() + de_parallel_workers - 1) / de_parallel_workers);
      std::vector<PullAcceptedBuffer> accepted_updates(de_parallel_workers);
      for (auto &updates : accepted_updates) {
        updates.entries.resize(max_updates_per_worker);
        updates.count = 0;
      }

      for (std::size_t iter = 0; iter < DE_arguments.ite_num; ++iter) {
        if (eval_budget.max_evals != 0 &&
            total_pull_evaluations_used(workers) >= eval_budget.max_evals) {
          stopped_by_eval_budget = true;
          break;
        }

        TitanUI::show_progress("Pulling Direction", iter, DE_arguments.ite_num,
                               "(Evolution)");
        for (auto &updates : accepted_updates)
          updates.count = 0;

        executor.parallel_for(
            population.size(), requested_workers, de_parallel_grain,
            [&](std::size_t worker_id, std::size_t begin, std::size_t end) {
              auto &worker = workers[worker_id];
              auto &updates = accepted_updates[worker_id];
              std::size_t accepted_count = 0;
              force trial_spherical{};
              PullCandidateState trial_candidate{};
              for (std::size_t i = begin; i < end; ++i) {
                if (!build_trial_spherical(population, i, iter, DE_arguments,
                                            trial_spherical)) {
                  continue;
                }
                if (!evaluate_pull_candidate(trial_spherical, trial_candidate, rec,
                                              lg, worker, energy_ref, Arguments,
                                              DE_arguments, backend,
                                              geometry_context, eval_budget)) {
                  continue;
                }
                if (score_improves(trial_candidate.score, population[i].score)) {
                  updates.entries[accepted_count].index = i;
                  updates.entries[accepted_count].state = trial_candidate;
                  ++accepted_count;
                }
              }
              updates.count = accepted_count;
            });

        bool iteration_improved = false;
        for (const auto &updates : accepted_updates) {
          for (std::size_t accepted_idx = 0; accepted_idx < updates.count;
               ++accepted_idx) {
            const auto &accepted = updates.entries[accepted_idx];
            population[accepted.index] = accepted.state;
            if (score_improves(accepted.state.score, current_best_score)) {
              current_best_score = accepted.state.score;
              best_index = accepted.index;
              iteration_improved = true;
            }
          }
        }

        if (iteration_improved) {
          best_stage = iter + 1;
          stagnation_count = 0;
        } else {
          ++stagnation_count;
        }

        if (DE_arguments.adaptive_F) {
          if (stagnation_count > 5) {
            DE_arguments.diff_weight =
                std::min(DE_arguments.F_max, DE_arguments.diff_weight * 1.05);
          } else if (stagnation_count == 0) {
            DE_arguments.diff_weight =
                std::max(DE_arguments.F_min, DE_arguments.diff_weight * 0.98);
          }
        }

        if (stagnation_count >= DE_arguments.stagnation_limit) {
          TitanUI::print_info("  DE converged early at iteration " +
                              std::to_string(iter) + " (stagnation=" +
                              std::to_string(DE_arguments.stagnation_limit) +
                              ")");
          break;
        }
      }

      best_candidate = population[best_index];
      if (!best_candidate.valid) {
        TitanUI::print_error(
            "Pull-direction DE: no valid candidate survived the search.");
        return 1;
      }

      report_candidates = population;
      if (DE_arguments.pull_opt == "de+pattern") {
        const double default_theta_step = kPatternInitAngleStepDeg * PI / 180.0;
        const double default_phi_step = kPatternInitAngleStepDeg * PI / 180.0;
        const double default_rho_step = std::max(
            0.1, 0.25 * (DE_arguments.scan_radius - DE_arguments.min_scan_radius));
        pattern_stats = refine_pull_candidate_pattern(
            best_candidate, rec, lg, workers, energy_ref, Arguments,
            DE_arguments, backend, geometry_context, eval_budget, executor,
            default_theta_step, default_phi_step, default_rho_step);
        if (pattern_stats.stopped_by_eval_budget)
          stopped_by_eval_budget = true;
        if (pattern_stats.improved) {
          report_candidates.push_back(best_candidate);
          candidate_table_title = "Ranked Candidate Table";
        }
      }
    } else if (use_variant_de) {
      best_stage_label = "Best generation";
      candidate_table_title = "Ranked Candidate Table";
      PullDeVariantOptions variant_options;
      variant_options.variant = de_variant_from_optimizer(optimizer_kind);
      variant_options.seed = DE_arguments.pull_seed;
      variant_options.initial_population = DE_arguments.pop_num;
      variant_options.minimum_population = DE_arguments.min_pop_num;
      variant_options.max_generations = DE_arguments.ite_num;
      const std::size_t variant_total_budget =
          DE_arguments.pull_max_evals != 0
              ? DE_arguments.pull_max_evals
              : DE_arguments.pop_num * (DE_arguments.ite_num + 1);
      if (use_nl_shade_variant_pattern) {
        pattern_reserved_budget = clamp_pattern_tail_budget(variant_total_budget);
        variant_phase_budget =
            variant_total_budget > pattern_reserved_budget
                ? variant_total_budget - pattern_reserved_budget
                : variant_total_budget;
      } else {
        variant_phase_budget = variant_total_budget;
      }
      variant_options.planned_evaluations = variant_total_budget;
      variant_options.memory_size = DE_arguments.memory_size;
      variant_options.p_best_rate = DE_arguments.p_best_rate;
      variant_options.archive_scale = DE_arguments.archive_scale;
      variant_options.stagnation_limit = DE_arguments.stagnation_limit;
      variant_options.scan_radius = DE_arguments.scan_radius;
      variant_options.min_scan_radius = DE_arguments.min_scan_radius;
      variant_options.has_old_vector = DE_arguments.flag_old_vec;
      variant_options.old_vector = DE_arguments.old_vector;
      variant_options.restrain_angle_deg = DE_arguments.restrain_angle;

      TitanUI::print_info("Running " +
                          std::string(pull_de_variant_name(variant_options.variant)) +
                          " pull-direction search...");
      const std::size_t saved_eval_budget = eval_budget.max_evals;
      if (use_nl_shade_variant_pattern)
        eval_budget.max_evals = variant_phase_budget;
      if (!run_pull_de_variant(variant_options, batch_evaluator,
                               report_candidates, best_candidate,
                               de_variant_stats)) {
        eval_budget.max_evals = saved_eval_budget;
        TitanUI::print_error(
            "Pull-direction DE variant: no valid candidate survived the search.");
        return 1;
      }
      if (use_nl_shade_variant_pattern)
        eval_budget.max_evals = variant_total_budget;
      else
        eval_budget.max_evals = saved_eval_budget;
      if (!use_nl_shade_variant_pattern)
        stopped_by_eval_budget = de_variant_stats.stopped_by_eval_budget;
      best_stage = de_variant_stats.best_generation;
      if (use_jso_variant_pattern) {
        const auto pattern_steps =
            variant_pattern_initial_steps(de_variant_stats, DE_arguments);
        pattern_stats = refine_pull_candidate_pattern(
            best_candidate, rec, lg, workers, energy_ref, Arguments,
            DE_arguments, backend, geometry_context, eval_budget, executor,
            pattern_steps[0], pattern_steps[1], pattern_steps[2]);
        pattern_stats.seeds_attempted = 1;
        pattern_stats.distinct_seed_basins = 1;
        if (pattern_stats.stopped_by_eval_budget)
          stopped_by_eval_budget = true;
        if (pattern_stats.improved)
          report_candidates.push_back(best_candidate);
      } else if (use_nl_shade_variant_pattern) {
        pattern_stats = refine_variant_multi_seed_pattern(
            best_candidate, report_candidates, rec, lg, workers, energy_ref,
            Arguments, DE_arguments, backend, geometry_context, eval_budget,
            executor, de_variant_stats);
        stopped_by_eval_budget =
            pattern_stats.stopped_by_eval_budget &&
            eval_budget.max_evals != 0 &&
            total_pull_evaluations_used(workers) >= eval_budget.max_evals;
        if (pattern_stats.improved)
          report_candidates.push_back(best_candidate);
      }
    }
  } else {
    best_stage_label = "Best expansion";
    candidate_table_title = "Ranked Candidate Table";
    const std::size_t total_budget =
        eval_budget.max_evals != 0 ? eval_budget.max_evals
                                   : effective_total_eval_budget(DE_arguments);
    pattern_reserved_budget =
        use_direct_pattern ? clamp_pattern_tail_budget(total_budget) : 0;
    direct_phase_budget = total_budget > pattern_reserved_budget
                              ? total_budget - pattern_reserved_budget
                              : total_budget;
    direct_stats = run_pull_direct_direct(
        best_candidate, report_candidates, best_stage, stopped_by_eval_budget,
        rec, lg, workers, energy_ref, Arguments, DE_arguments, backend,
        geometry_context, eval_budget, direct_phase_budget, executor);
    if (!best_candidate.valid) {
      TitanUI::print_error(
          "Pull-direction DIRECT: no valid candidate satisfied the search constraints.");
      return 1;
    }
    if (use_direct_pattern) {
      hybrid_switch_stage = direct_stats.expansions;
      candidate_table_title = "Ranked Candidate Table";
      const double theta_step =
          std::max(kPatternMinAngleStepDeg * PI / 180.0,
                   PI * direct_stats.best_half_width_norm[0]);
      const double phi_step =
          std::max(kPatternMinAngleStepDeg * PI / 180.0,
                   2.0 * PI * direct_stats.best_half_width_norm[1]);
      const double rho_step = std::max(
          0.02, (DE_arguments.scan_radius - DE_arguments.min_scan_radius) *
                    direct_stats.best_half_width_norm[2]);
      pattern_stats = refine_pull_candidate_pattern(
          best_candidate, rec, lg, workers, energy_ref, Arguments, DE_arguments,
          backend, geometry_context, eval_budget, executor, theta_step, phi_step,
          rho_step);
      if (pattern_stats.stopped_by_eval_budget)
        stopped_by_eval_budget = true;
      report_candidates.push_back(best_candidate);
    }
  }

  const auto finished_at = std::chrono::steady_clock::now();
  const double wall_seconds =
      std::chrono::duration<double>(finished_at - started_at).count();
  const std::size_t total_evaluations = total_pull_evaluations_used(workers);
  const double evals_per_second =
      wall_seconds > 0.0 ? static_cast<double>(total_evaluations) / wall_seconds
                         : 0.0;

  double rec_com[3]{};
  double lg_com[3]{};
  mass_weighted_com(rec.m_mol[0], rec.parameters, rec_com);
  mass_weighted_com(lg.m_mol[0], lg.parameters, lg_com);

  const double dX = best_candidate.translation.Fx;
  const double dY = best_candidate.translation.Fy;
  const double dZ = best_candidate.translation.Fz;
  const double best_com_x = lg_com[0] + dX;
  const double best_com_y = lg_com[1] + dY;
  const double best_com_z = lg_com[2] + dZ;
  const double best_angle =
      angle_to_previous_vector(best_candidate.translation, DE_arguments.old_vector);

  std::size_t nearest_rec_idx = 0;
  double nearest_rec_dist = 1.0e20;
  std::size_t target_rec_idx = 0;
  double target_rec_dist = 1.0e20;
  for (std::size_t i = 0; i < rec.m_mol[0].size(); ++i) {
    if (rec.info[i].get_atom_type().substr(0, 1) == "H")
      continue;

    const double dx0 = rec.m_mol[0].pos.x[i] - lg_com[0];
    const double dy0 = rec.m_mol[0].pos.y[i] - lg_com[1];
    const double dz0 = rec.m_mol[0].pos.z[i] - lg_com[2];
    const double d20 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0;
    if (d20 < nearest_rec_dist) {
      nearest_rec_dist = d20;
      nearest_rec_idx = i;
    }

    const double dx1 = rec.m_mol[0].pos.x[i] - best_com_x;
    const double dy1 = rec.m_mol[0].pos.y[i] - best_com_y;
    const double dz1 = rec.m_mol[0].pos.z[i] - best_com_z;
    const double d21 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
    if (d21 < target_rec_dist) {
      target_rec_dist = d21;
      target_rec_idx = i;
    }
  }
  nearest_rec_dist = std::sqrt(nearest_rec_dist);
  target_rec_dist = std::sqrt(target_rec_dist);

  std::vector<std::pair<double, std::size_t>> ranked(report_candidates.size());
  for (std::size_t i = 0; i < report_candidates.size(); ++i)
    ranked[i] = {report_candidates[i].score, i};
  std::sort(ranked.begin(), ranked.end());
  const std::size_t best_rank_index =
      ranked.empty() ? 0 : ranked.front().second;

  std::fstream de_output(output_info.get_DE_output_filename(), ios::out);
  std::fstream de_log(output_info.get_DE_log_filename(), ios::out | ios::app);
  std::fstream de_pdb_out(output_info.get_DE_pdb_vector_filename(),
                          ios::out | ios::app);
  std::fstream de_report("DE_report.dat", ios::out);
  std::fstream best_conf("best_conf.pdb", ios::out);

  de_report << std::fixed << std::setprecision(4);
  de_report << "==============================================================="
               "=================\n";
  de_report << "  TITAN Pull Direction Report\n";
  de_report << "==============================================================="
               "=================\n\n";

  de_report << "  Parameters\n";
  de_report << "  ----------\n";
  de_report << "  Optimizer          : " << DE_arguments.pull_opt << "\n";
  de_report << "  Optimizer family   : "
            << (use_de_family ? "DE" : "DIRECT") << "\n";
  de_report << "  Variant            : "
            << (use_variant_de
                    ? std::string(pull_de_variant_name(de_variant_stats.variant))
                    : DE_arguments.pull_opt)
            << "\n";
  de_report << "  Scan radius        : " << DE_arguments.scan_radius << " A\n";
  de_report << "  Min scan radius    : " << DE_arguments.min_scan_radius
            << " A\n";
  de_report << "  Restrain angle     : " << DE_arguments.restrain_angle
            << " deg\n";
  if (use_legacy_de) {
    de_report << "  Population size    : " << DE_arguments.pop_num << "\n";
    de_report << "  Max iterations     : " << DE_arguments.ite_num << "\n";
    de_report << "  Algorithm          : " << DE_arguments.algorithm << "\n";
    de_report << "  Diff weight (F)    : " << DE_arguments.diff_weight << "\n";
    de_report << "  Cross prob (CR)    : " << DE_arguments.cross_prob << "\n";
    de_report << "  Adaptive F         : "
              << (DE_arguments.adaptive_F ? "ON" : "OFF") << "\n";
    de_report << "  Stagnation limit   : " << DE_arguments.stagnation_limit
              << "\n";
    de_report << "  Seed               : " << DE_arguments.pull_seed << "\n";
  } else if (use_variant_de) {
    de_report << "  Population size    : " << de_variant_stats.initial_population
              << "\n";
    de_report << "  Min population     : " << DE_arguments.min_pop_num << "\n";
    de_report << "  Max generations    : " << DE_arguments.ite_num << "\n";
    de_report << "  Memory size        : " << de_variant_stats.memory_size
              << "\n";
    de_report << "  p-best rate        : " << de_variant_stats.p_best_rate
              << "\n";
    de_report << "  Archive scale      : " << de_variant_stats.archive_scale
              << "\n";
    de_report << "  Archive capacity   : " << de_variant_stats.archive_capacity
              << "\n";
    de_report << "  Stagnation limit   : " << DE_arguments.stagnation_limit
              << "\n";
    de_report << "  Seed               : " << DE_arguments.pull_seed << "\n";
  } else {
    de_report << "  Search mode        : DIRECT\n";
    de_report << "  Seed               : deterministic\n";
  }
  de_report << "  Score function     : " << Arguments.de.de_flag_score_function
            << "\n";
  de_report << "  Max evals          : ";
  if (use_direct_family) {
    de_report << eval_budget.max_evals;
    if (direct_stats.used_fallback_budget)
      de_report << " (derived fallback)";
    de_report << "\n";
  } else if (use_nl_shade_variant_pattern) {
    de_report << eval_budget.max_evals;
    if (DE_arguments.pull_max_evals == 0)
      de_report << " (derived fallback)";
    de_report << "\n";
  } else if (DE_arguments.pull_max_evals == 0) {
    de_report << "unlimited\n";
  } else {
    de_report << DE_arguments.pull_max_evals << "\n";
  }
  de_report << "  Worker threads     : " << worker_slots << "\n";
  de_report << "  Force field        : " << Arguments.force_f << "\n\n";

  if (DE_arguments.snapshot_metadata_valid) {
    de_report << "  Snapshot Selection\n";
    de_report << "  ------------------\n";
    de_report << "  Source file        : " << DE_arguments.snapshot_source_file
              << "\n";
    de_report << "  Input mode         : " << DE_arguments.snapshot_input_mode
              << "\n";
    de_report << "  Selection rule     : " << DE_arguments.snapshot_selection_rule
              << "\n";
    de_report << "  Snapshot time      : " << DE_arguments.snapshot_time_ps
              << " ps\n";
    de_report << "  Snapshot frame     : " << DE_arguments.snapshot_frame_index
              << "\n\n";
  }

  de_report << "  Benchmark\n";
  de_report << "  ---------\n";
  de_report << "  Total evaluations  : " << total_evaluations << "\n";
  de_report << "  Wall time          : " << wall_seconds << " s\n";
  de_report << "  Eval/s             : " << evals_per_second << "\n";
  de_report << "  " << best_stage_label << "     : " << best_stage << "\n";
  de_report << "  Eval budget stop   : "
            << (stopped_by_eval_budget ? "YES" : "NO") << "\n\n";

  if (use_variant_de) {
    de_report << "  DE Variant Stats\n";
    de_report << "  ----------------\n";
    de_report << "  Reduction mode   : "
              << (de_variant_stats.nonlinear_population_reduction ? "nonlinear"
                                                                  : "linear/fixed")
              << "\n";
    de_report << "  Initial pop       : " << de_variant_stats.initial_population
              << "\n";
    de_report << "  Final pop         : " << de_variant_stats.final_population
              << "\n";
    de_report << "  Reductions        : "
              << de_variant_stats.population_reductions << "\n";
    de_report << "  Generations done  : "
              << de_variant_stats.generations_completed << "\n";
    de_report << "  Best generation   : " << de_variant_stats.best_generation
              << "\n";
    de_report << "  Final p-best      : " << de_variant_stats.final_p_best_rate
              << "\n";
    if (de_variant_stats.selective_pressure > 0.0) {
      de_report << "  Selective press.  : "
                << de_variant_stats.selective_pressure << "\n";
    }
    if (de_variant_stats.nonlinear_population_reduction) {
      de_report << "  Bias power F      : " << de_variant_stats.bias_power_f
                << "\n";
      de_report << "  Bias power CR     : " << de_variant_stats.bias_power_cr
                << "\n";
    }
    de_report << "  Stagnation stop   : "
              << (de_variant_stats.stopped_by_stagnation ? "YES" : "NO")
              << "\n\n";
  }

  if (direct_stats.enabled) {
    de_report << "  DIRECT Search\n";
    de_report << "  -------------\n";
    de_report << "  Phase wall time    : " << direct_stats.wall_seconds << " s\n";
    de_report << "  Phase evals        : " << direct_stats.phase_evaluations_used
              << "\n";
    de_report << "  Direct budget      : " << direct_stats.effective_eval_budget
              << "\n";
    de_report << "  Expansions         : " << direct_stats.expansions << "\n";
    de_report << "  Batches            : " << direct_stats.batches << "\n";
    de_report << "  Max batch cells    : " << direct_stats.max_batch_cells
              << "\n";
    de_report << "  Cells remaining    : " << direct_stats.cells_remaining
              << "\n";
    de_report << "  Valid candidates   : " << direct_stats.evaluated_candidates
              << "\n";
    de_report << "  Best expansion     : " << direct_stats.best_expansion
              << "\n";
    de_report << "  Best cell size     : " << direct_stats.best_cell_size
              << "\n";
    de_report << "  Fallback budget    : "
              << (direct_stats.used_fallback_budget ? "YES" : "NO") << "\n";
    de_report << "  Budget stop direct : "
              << (direct_stats.stopped_by_eval_budget ? "YES" : "NO")
              << "\n";
    de_report << "  Local stop direct  : "
              << (direct_stats.stopped_by_local_threshold ? "YES" : "NO")
              << "\n";
    de_report << "  Stall stop direct  : "
              << (direct_stats.stopped_by_stagnation ? "YES" : "NO")
              << "\n\n";
  }

  if (pattern_stats.enabled && use_direct_pattern) {
    const std::size_t remaining_budget =
        eval_budget.max_evals > direct_stats.phase_evaluations_used
            ? eval_budget.max_evals - direct_stats.phase_evaluations_used
            : 0;
    de_report << "  Hybrid Refinement\n";
    de_report << "  -----------------\n";
    de_report << "  Switch stage      : " << hybrid_switch_stage << "\n";
    de_report << "  Direct budget     : " << direct_phase_budget << "\n";
    de_report << "  Pattern reserve   : " << pattern_reserved_budget << "\n";
    de_report << "  Pattern available : " << remaining_budget << "\n";
    de_report << "  Pattern wall time : " << pattern_stats.wall_seconds << " s\n";
    de_report << "  Pattern evals     : " << pattern_stats.evaluations_used
              << "\n";
    de_report << "  Score before      : " << pattern_stats.initial_score << "\n";
    de_report << "  Score after       : " << pattern_stats.final_score << "\n";
    de_report << "  Improved          : "
              << (pattern_stats.improved ? "YES" : "NO") << "\n";
    de_report << "  Budget stop       : "
              << (pattern_stats.stopped_by_eval_budget ? "YES" : "NO")
              << "\n\n";
  } else if (pattern_stats.enabled && use_variant_pattern) {
    const std::string variant_pattern_title =
        variant_pattern_section_title(de_variant_stats.variant);
    de_report << "  " << variant_pattern_title << "\n";
    de_report << "  "
              << std::string(variant_pattern_title.size(), '-') << "\n";
    de_report << "  Enabled            : YES\n";
    de_report << "  Improved           : "
              << (pattern_stats.improved ? "YES" : "NO") << "\n";
    if (use_nl_shade_variant_pattern) {
      de_report << "  Variant budget     : " << variant_phase_budget << "\n";
      de_report << "  Pattern reserve    : " << pattern_reserved_budget << "\n";
    }
    if (pattern_stats.seeds_attempted != 0) {
      de_report << "  Seeds tried        : " << pattern_stats.seeds_attempted
                << "\n";
      de_report << "  Distinct basins    : "
                << pattern_stats.distinct_seed_basins << "\n";
    }
    de_report << "  Sweeps             : " << pattern_stats.sweeps << "\n";
    de_report << "  Accepted moves     : " << pattern_stats.accepted_moves
              << "\n";
    de_report << "  Extra evaluations  : " << pattern_stats.evaluations_used
              << "\n";
    de_report << "  Pattern wall time  : " << pattern_stats.wall_seconds << " s\n";
    de_report << "  Score before local : " << pattern_stats.initial_score
              << "\n";
    de_report << "  Score after local  : " << pattern_stats.final_score
              << "\n";
    de_report << "  Budget stop local  : "
              << (pattern_stats.stopped_by_eval_budget ? "YES" : "NO")
              << "\n\n";
  } else if (pattern_stats.enabled) {
    de_report << "  Pattern Refinement\n";
    de_report << "  ------------------\n";
    de_report << "  Enabled            : YES\n";
    de_report << "  Improved           : "
              << (pattern_stats.improved ? "YES" : "NO") << "\n";
    de_report << "  Sweeps             : " << pattern_stats.sweeps << "\n";
    de_report << "  Accepted moves     : " << pattern_stats.accepted_moves
              << "\n";
    de_report << "  Extra evaluations  : " << pattern_stats.evaluations_used
              << "\n";
    de_report << "  Pattern wall time  : " << pattern_stats.wall_seconds << " s\n";
    de_report << "  Score before local : " << pattern_stats.initial_score
              << "\n";
    de_report << "  Score after local  : " << pattern_stats.final_score
              << "\n";
    de_report << "  Budget stop local  : "
              << (pattern_stats.stopped_by_eval_budget ? "YES" : "NO")
              << "\n\n";
  }

  de_report << "  Reference Energy\n";
  de_report << "  ----------------\n";
  de_report << "  Elec  : " << energy_ref.elec_ener << " kcal/mol\n";
  de_report << "  VdW   : " << energy_ref.vdw_ener << " kcal/mol\n";
  de_report << "  Total : " << energy_ref.elec_ener + energy_ref.vdw_ener
            << " kcal/mol\n\n";

  de_report << "  Ligand COM (original) : (" << lg_com[0] << ", " << lg_com[1]
            << ", " << lg_com[2] << ")\n";
  de_report << "  Receptor COM          : (" << rec_com[0] << ", " << rec_com[1]
            << ", " << rec_com[2] << ")\n";
  de_report << "  Ligand COM (best)     : (" << best_com_x << ", " << best_com_y
            << ", " << best_com_z << ")\n\n";

  de_report << "  Output Frames\n";
  de_report << "  -------------\n";
  de_report << "  Legacy outputs      : original input frame\n";
  de_report << "  Aligned snapshot    : "
            << output_info.get_pull_direction_aligned_pdb_filename()
            << " (+Z frame)\n";
  de_report << "  Aligned system info : "
            << output_info.get_pull_direction_system_info_filename() << "\n\n";

  de_report << "  Pull Direction Anatomy\n";
  de_report << "  ----------------------\n";
  de_report << "  Origin receptor atom  : "
            << rec.info[nearest_rec_idx].get_atom() << " "
            << rec.info[nearest_rec_idx].get_original_res_name()
            << rec.info[nearest_rec_idx].get_respos() << " (chain "
            << rec.info[nearest_rec_idx].get_chainnum()
            << ", dist=" << nearest_rec_dist << " A)\n";
  de_report << "  Target receptor atom  : "
            << rec.info[target_rec_idx].get_atom() << " "
            << rec.info[target_rec_idx].get_original_res_name()
            << rec.info[target_rec_idx].get_respos() << " (chain "
            << rec.info[target_rec_idx].get_chainnum()
            << ", dist=" << target_rec_dist << " A)\n";
  de_report << "  Pull vector (dX,dY,dZ): (" << dX << ", " << dY << ", " << dZ
            << ")\n";
  de_report << "  Pull distance         : " << best_candidate.translation.modul()
            << " A\n";
  de_report << "  Angle to prev vector  : " << best_angle << " deg\n\n";

  de_report << "  Best Candidate\n";
  de_report << "  --------------\n";
  de_report << "  Index  : " << best_rank_index << "\n";
  de_report << "  Score  : " << best_candidate.score << "\n";
  de_report << "  Elec   : " << best_candidate.energy.elec_ener
            << " kcal/mol\n";
  de_report << "  VdW    : " << best_candidate.energy.vdw_ener << " kcal/mol\n";
  de_report << "  dElec  : "
            << best_candidate.energy.elec_ener - energy_ref.elec_ener
            << " kcal/mol\n";
  de_report << "  dVdW   : " << best_candidate.energy.vdw_ener - energy_ref.vdw_ener
            << " kcal/mol\n\n";

  de_report << "  " << candidate_table_title << "\n";
  de_report << "  "
            << string(std::char_traits<char>::length(candidate_table_title.data()),
                      '-')
            << "\n";
  de_report << std::setw(8) << "Rank" << std::setw(8) << "Index"
            << std::setw(14) << "Score" << std::setw(14) << "Elec"
            << std::setw(14) << "VdW" << std::setw(14) << "dElec"
            << std::setw(14) << "dVdW" << std::setw(10) << "Rho(A)"
            << std::setw(12) << "Angle(deg)" << "\n";
  de_report << string(108, '-') << "\n";

  const std::size_t n_print = std::min<std::size_t>(50, ranked.size());
  for (std::size_t rank = 0; rank < n_print; ++rank) {
    const std::size_t idx = ranked[rank].second;
    const auto &candidate = report_candidates[idx];
    de_report << std::setw(8) << rank + 1 << std::setw(8) << idx
              << std::setw(14) << candidate.score << std::setw(14)
              << candidate.energy.elec_ener << std::setw(14)
              << candidate.energy.vdw_ener << std::setw(14)
              << candidate.energy.elec_ener - energy_ref.elec_ener
              << std::setw(14)
              << candidate.energy.vdw_ener - energy_ref.vdw_ener
              << std::setw(10) << candidate.spherical.Fz << std::setw(12)
              << candidate.angle_deg << "\n";
  }
  if (ranked.size() > n_print)
    de_report << "  ... (" << ranked.size() - n_print
              << " more candidates omitted)\n";
  de_report << "==============================================================="
               "=================\n";

  de_output << std::fixed << std::setprecision(4);
  de_output << dX << "  " << dY << "  " << dZ << "  Angle: " << best_angle
            << " -- Modul: " << best_candidate.translation.modul()
            << " Angstrom\n";
  de_output << "# Pull direction: from " << rec.info[nearest_rec_idx].get_atom()
            << " " << rec.info[nearest_rec_idx].get_original_res_name()
            << rec.info[nearest_rec_idx].get_respos() << " (chain "
            << rec.info[nearest_rec_idx].get_chainnum() << ") towards "
            << rec.info[target_rec_idx].get_atom() << " "
            << rec.info[target_rec_idx].get_original_res_name()
            << rec.info[target_rec_idx].get_respos() << " (chain "
            << rec.info[target_rec_idx].get_chainnum() << ")\n";
  de_output << "# Lig COM: (" << lg_com[0] << ", " << lg_com[1] << ", "
            << lg_com[2] << ") -> (" << best_com_x << ", " << best_com_y
            << ", " << best_com_z << ")\n";
  de_output << "# Best score: " << best_candidate.score
            << "  elec: " << best_candidate.energy.elec_ener
            << "  vdw: " << best_candidate.energy.vdw_ener << "\n";

  de_log << std::fixed << std::setprecision(4);
  de_log << dX << "  " << dY << "  " << dZ << "  Angle: " << best_angle
         << " -- Modul: " << best_candidate.translation.modul()
         << " Angstrom  Score: " << best_candidate.score
         << "  Eval: " << total_evaluations << "  Wall: " << wall_seconds
         << "s  Eval/s: " << evals_per_second
         << "\n--------------------------\n";

  translate_seed_to_scratch(lg.m_mol[0], workers.front().scratch, dX, dY, dZ);
  best_conf << std::fixed << std::setprecision(3);
  for (std::size_t i = 0; i < workers.front().scratch.size(); ++i) {
    best_conf << "ATOM" << std::setw(7) << i + 1 << std::setw(5)
              << lg.info[i].get_atom() << std::setw(4)
              << lg.info[i].get_original_res_name() << " A" << std::setw(4)
              << 1 << std::setw(12) << workers.front().scratch.pos.x[i]
              << std::setw(8) << workers.front().scratch.pos.y[i]
              << std::setw(8) << workers.front().scratch.pos.z[i]
              << std::setw(7) << 1.00 << std::setw(6) << 0.00 << '\n';
  }

  de_pdb_out << std::fixed << std::setprecision(3);
  de_pdb_out << "ATOM" << std::setw(7) << 1 << std::setw(5) << "CA"
             << std::setw(4) << "ALA" << " A" << std::setw(4) << 1
             << std::setw(12) << lg_com[0] << std::setw(8) << lg_com[1]
             << std::setw(8) << lg_com[2] << " " << std::setw(7) << 1.00
             << " " << std::setw(6) << 0.00 << '\n';
  de_pdb_out << "ATOM" << std::setw(7) << 2 << std::setw(5) << "CB"
             << std::setw(4) << "ALA" << " A" << std::setw(4) << 1
             << std::setw(12) << best_com_x << std::setw(8) << best_com_y
             << std::setw(8) << best_com_z << " " << std::setw(7) << 1.00
             << " " << std::setw(6) << 0.00 << '\n';
  de_pdb_out << "TER\n";

  const PullAlignedFrameData aligned_data =
      build_pull_aligned_frame_data(rec, lg, best_candidate.translation);
  if (!write_pull_direction_aligned_snapshot(aligned_data, rec, lg,
                                             output_info)) {
    TitanUI::print_error("Pull-direction: unable to write " +
                         output_info.get_pull_direction_aligned_pdb_filename());
    return 1;
  }
  if (!write_pull_direction_system_info(aligned_data, best_candidate.translation,
                                        output_info)) {
    TitanUI::print_error("Pull-direction: unable to write " +
                         output_info.get_pull_direction_system_info_filename());
    return 1;
  }
  return 0;
}
