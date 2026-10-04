#ifndef TITAN_DIHEDRAL_KERNELS_H
#define TITAN_DIHEDRAL_KERNELS_H

#include "MOL.h"
#include "aligned_allocator.h"

#include <cstddef>
#include <cstdint>
#include <vector>

class TitanExecutor;
struct BoxMatrix;

namespace titan_dihedral_kernel {

enum TorsionKind : std::uint8_t {
  Phi = 0,
  Psi = 1,
  Omega = 2,
  Chi1 = 3,
  Chi2 = 4
};

struct TorsionPlanSoA {
  AlignedIntVector atom0;
  AlignedIntVector atom1;
  AlignedIntVector atom2;
  AlignedIntVector atom3;
  AlignedIntVector residue_index;
  std::vector<std::uint8_t> torsion_kind;

  void clear();
  void reserve(std::size_t n);
  void push(int a0, int a1, int a2, int a3, std::size_t residue,
            TorsionKind kind);
  [[nodiscard]] std::size_t size() const noexcept { return atom0.size(); }
};

struct TorsionFrameResult {
  AlignedDoubleVector angle_deg;
  void resize(std::size_t n) { angle_deg.assign(n, 0.0); }
};

double compute_dihedral_deg_from_points(double p0x, double p0y, double p0z,
                                        double p1x, double p1y, double p1z,
                                        double p2x, double p2y, double p2z,
                                        double p3x, double p3y, double p3z);

void compute_torsions_scalar(const TorsionPlanSoA &plan, const MOL &frame,
                             bool use_mic, const BoxMatrix *box_matrix,
                             std::size_t begin, std::size_t end,
                             AlignedDoubleVector &out_angle_deg);

void compute_torsions_optimized(const TorsionPlanSoA &plan, const MOL &frame,
                                bool use_mic, const BoxMatrix *box_matrix,
                                std::size_t begin, std::size_t end,
                                AlignedDoubleVector &out_angle_deg);

void compute_torsions_parallel(const TorsionPlanSoA &plan, const MOL &frame,
                               bool use_mic, const BoxMatrix *box_matrix,
                               TitanExecutor &executor,
                               TorsionFrameResult &out);

} // namespace titan_dihedral_kernel

#endif // TITAN_DIHEDRAL_KERNELS_H
