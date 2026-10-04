#include "titan_dihedral_kernels.h"

#include "BoxMatrix.h"
#include "titan_executor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <immintrin.h>
#include <limits>

namespace titan_dihedral_kernel {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = 180.0 / kPi;
constexpr double kEps = 1.0e-12;

struct Vec3 {
  double x{};
  double y{};
  double z{};
};

double dot3(const Vec3 &a, const Vec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross3(const Vec3 &a, const Vec3 &b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}

double norm3(const Vec3 &a) { return std::sqrt(dot3(a, a)); }

Vec3 scale3(const Vec3 &a, double scale) {
  return {a.x * scale, a.y * scale, a.z * scale};
}

Vec3 sub3(const Vec3 &a, const Vec3 &b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

double compute_from_bonds(Vec3 v10, Vec3 v12, Vec3 v23) {
  const Vec3 b0 = scale3(v10, -1.0);
  const Vec3 b1 = v12;
  const Vec3 b2 = v23;
  const double b1_norm = norm3(b1);
  if (b1_norm <= kEps)
    return std::numeric_limits<double>::quiet_NaN();

  const Vec3 b1_unit = scale3(b1, 1.0 / b1_norm);
  const Vec3 v = sub3(b0, scale3(b1_unit, dot3(b0, b1_unit)));
  const Vec3 w = sub3(b2, scale3(b1_unit, dot3(b2, b1_unit)));
  if (norm3(v) <= kEps || norm3(w) <= kEps)
    return std::numeric_limits<double>::quiet_NaN();

  const double x = dot3(v, w);
  const double y = dot3(cross3(b1_unit, v), w);
  double angle = std::atan2(y, x) * kDeg;
  while (angle > 180.0)
    angle -= 360.0;
  while (angle <= -180.0)
    angle += 360.0;
  return angle;
}

bool valid_atom_index(int atom, const MOL &frame) {
  return atom >= 0 && static_cast<std::size_t>(atom) < frame.pos.size();
}

double compute_one(const TorsionPlanSoA &plan, const MOL &frame,
                   bool use_mic, const BoxMatrix *box_matrix,
                   std::size_t idx) {
  const int a0 = plan.atom0[idx];
  const int a1 = plan.atom1[idx];
  const int a2 = plan.atom2[idx];
  const int a3 = plan.atom3[idx];
  if (!valid_atom_index(a0, frame) || !valid_atom_index(a1, frame) ||
      !valid_atom_index(a2, frame) || !valid_atom_index(a3, frame))
    return std::numeric_limits<double>::quiet_NaN();

  Vec3 v10{frame.pos.x[a0] - frame.pos.x[a1],
           frame.pos.y[a0] - frame.pos.y[a1],
           frame.pos.z[a0] - frame.pos.z[a1]};
  Vec3 v12{frame.pos.x[a2] - frame.pos.x[a1],
           frame.pos.y[a2] - frame.pos.y[a1],
           frame.pos.z[a2] - frame.pos.z[a1]};
  Vec3 v23{frame.pos.x[a3] - frame.pos.x[a2],
           frame.pos.y[a3] - frame.pos.y[a2],
           frame.pos.z[a3] - frame.pos.z[a2]};

  if (use_mic && box_matrix != nullptr) {
    box_matrix->apply_mic(v10.x, v10.y, v10.z);
    box_matrix->apply_mic(v12.x, v12.y, v12.z);
    box_matrix->apply_mic(v23.x, v23.y, v23.z);
  }
  return compute_from_bonds(v10, v12, v23);
}

} // namespace

void TorsionPlanSoA::clear() {
  atom0.clear();
  atom1.clear();
  atom2.clear();
  atom3.clear();
  residue_index.clear();
  torsion_kind.clear();
}

void TorsionPlanSoA::reserve(std::size_t n) {
  atom0.reserve(n);
  atom1.reserve(n);
  atom2.reserve(n);
  atom3.reserve(n);
  residue_index.reserve(n);
  torsion_kind.reserve(n);
}

void TorsionPlanSoA::push(int a0, int a1, int a2, int a3,
                          std::size_t residue, TorsionKind kind) {
  atom0.push_back(a0);
  atom1.push_back(a1);
  atom2.push_back(a2);
  atom3.push_back(a3);
  residue_index.push_back(static_cast<std::int32_t>(residue));
  torsion_kind.push_back(static_cast<std::uint8_t>(kind));
}

double compute_dihedral_deg_from_points(double p0x, double p0y, double p0z,
                                        double p1x, double p1y, double p1z,
                                        double p2x, double p2y, double p2z,
                                        double p3x, double p3y, double p3z) {
  return compute_from_bonds(Vec3{p0x - p1x, p0y - p1y, p0z - p1z},
                            Vec3{p2x - p1x, p2y - p1y, p2z - p1z},
                            Vec3{p3x - p2x, p3y - p2y, p3z - p2z});
}

void compute_torsions_scalar(const TorsionPlanSoA &plan, const MOL &frame,
                             bool use_mic, const BoxMatrix *box_matrix,
                             std::size_t begin, std::size_t end,
                             AlignedDoubleVector &out_angle_deg) {
  const std::size_t stop = std::min(end, plan.size());
  if (out_angle_deg.size() < plan.size())
    out_angle_deg.assign(plan.size(), std::numeric_limits<double>::quiet_NaN());
  for (std::size_t i = begin; i < stop; ++i)
    out_angle_deg[i] = compute_one(plan, frame, use_mic, box_matrix, i);
}

void compute_torsions_optimized(const TorsionPlanSoA &plan, const MOL &frame,
                                bool use_mic, const BoxMatrix *box_matrix,
                                std::size_t begin, std::size_t end,
                                AlignedDoubleVector &out_angle_deg) {
  const std::size_t stop = std::min(end, plan.size());
  if (out_angle_deg.size() < plan.size())
    out_angle_deg.assign(plan.size(), std::numeric_limits<double>::quiet_NaN());

#if defined(__AVX2__)
  std::size_t i = begin;
  if (use_mic && box_matrix != nullptr) {
    alignas(32) std::array<double, 4> v10x{}, v10y{}, v10z{};
    alignas(32) std::array<double, 4> v12x{}, v12y{}, v12z{};
    alignas(32) std::array<double, 4> v23x{}, v23y{}, v23z{};
    for (; i + 4 <= stop; i += 4) {
      bool valid = true;
      for (std::size_t lane = 0; lane < 4; ++lane) {
        const std::size_t idx = i + lane;
        valid = valid && valid_atom_index(plan.atom0[idx], frame) &&
                valid_atom_index(plan.atom1[idx], frame) &&
                valid_atom_index(plan.atom2[idx], frame) &&
                valid_atom_index(plan.atom3[idx], frame);
      }
      if (!valid) {
        for (std::size_t lane = 0; lane < 4; ++lane)
          out_angle_deg[i + lane] =
              compute_one(plan, frame, use_mic, box_matrix, i + lane);
        continue;
      }
      for (std::size_t lane = 0; lane < 4; ++lane) {
        const std::size_t idx = i + lane;
        const int a0 = plan.atom0[idx];
        const int a1 = plan.atom1[idx];
        const int a2 = plan.atom2[idx];
        const int a3 = plan.atom3[idx];
        v10x[lane] = frame.pos.x[a0] - frame.pos.x[a1];
        v10y[lane] = frame.pos.y[a0] - frame.pos.y[a1];
        v10z[lane] = frame.pos.z[a0] - frame.pos.z[a1];
        v12x[lane] = frame.pos.x[a2] - frame.pos.x[a1];
        v12y[lane] = frame.pos.y[a2] - frame.pos.y[a1];
        v12z[lane] = frame.pos.z[a2] - frame.pos.z[a1];
        v23x[lane] = frame.pos.x[a3] - frame.pos.x[a2];
        v23y[lane] = frame.pos.y[a3] - frame.pos.y[a2];
        v23z[lane] = frame.pos.z[a3] - frame.pos.z[a2];
      }
      __m256d x10 = _mm256_load_pd(v10x.data());
      __m256d y10 = _mm256_load_pd(v10y.data());
      __m256d z10 = _mm256_load_pd(v10z.data());
      __m256d x12 = _mm256_load_pd(v12x.data());
      __m256d y12 = _mm256_load_pd(v12y.data());
      __m256d z12 = _mm256_load_pd(v12z.data());
      __m256d x23 = _mm256_load_pd(v23x.data());
      __m256d y23 = _mm256_load_pd(v23y.data());
      __m256d z23 = _mm256_load_pd(v23z.data());
      box_matrix->apply_mic_avx(x10, y10, z10);
      box_matrix->apply_mic_avx(x12, y12, z12);
      box_matrix->apply_mic_avx(x23, y23, z23);
      _mm256_store_pd(v10x.data(), x10);
      _mm256_store_pd(v10y.data(), y10);
      _mm256_store_pd(v10z.data(), z10);
      _mm256_store_pd(v12x.data(), x12);
      _mm256_store_pd(v12y.data(), y12);
      _mm256_store_pd(v12z.data(), z12);
      _mm256_store_pd(v23x.data(), x23);
      _mm256_store_pd(v23y.data(), y23);
      _mm256_store_pd(v23z.data(), z23);
      for (std::size_t lane = 0; lane < 4; ++lane) {
        out_angle_deg[i + lane] =
            compute_from_bonds(Vec3{v10x[lane], v10y[lane], v10z[lane]},
                               Vec3{v12x[lane], v12y[lane], v12z[lane]},
                               Vec3{v23x[lane], v23y[lane], v23z[lane]});
      }
    }
  }
  for (; i < stop; ++i)
    out_angle_deg[i] = compute_one(plan, frame, use_mic, box_matrix, i);
#else
  compute_torsions_scalar(plan, frame, use_mic, box_matrix, begin, stop,
                          out_angle_deg);
#endif
}

void compute_torsions_parallel(const TorsionPlanSoA &plan, const MOL &frame,
                               bool use_mic, const BoxMatrix *box_matrix,
                               TitanExecutor &executor,
                               TorsionFrameResult &out) {
  out.resize(plan.size());
  const std::size_t worker_count =
      std::max<std::size_t>(1, executor.capacity());
  executor.parallel_for(plan.size(), worker_count, 32,
                        [&](std::size_t, std::size_t begin,
                            std::size_t end) {
    compute_torsions_optimized(plan, frame, use_mic, box_matrix, begin, end,
                               out.angle_deg);
  });
}

} // namespace titan_dihedral_kernel
