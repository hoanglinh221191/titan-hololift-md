#ifndef TITAN_CONTACT_GEOMETRY_H
#define TITAN_CONTACT_GEOMETRY_H

#include "Multi_MOL.h"
#include "titan_contact_primitives.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace titan_contact_geometry {

struct ResidueSelection {
  titan_contact::ResidueKey residue;
  std::vector<std::size_t> atoms;
  std::vector<std::size_t> selected_atoms;
  std::vector<double> com_weights;
  double com_weight_sum{0.0};
};

struct ContactPoint {
  titan_contact::ResidueKey residue;
  double x{};
  double y{};
  double z{};
};

class ContactGeometryCache {
public:
  bool build(const Multi_MOL &mol, std::uint8_t molecule,
             titan_contact::ContactMode mode);

  bool build_points(const Multi_MOL &mol, std::size_t snap_idx,
                    std::vector<ContactPoint> &points) const;

  [[nodiscard]] const std::vector<ResidueSelection> &selections()
      const noexcept {
    return selections_;
  }

  [[nodiscard]] const std::vector<titan_contact::ResidueKey> &residues()
      const noexcept {
    return residues_;
  }

private:
  titan_contact::ContactMode mode_{titan_contact::ContactMode::SidechainCOM};
  std::vector<ResidueSelection> selections_;
  std::vector<titan_contact::ResidueKey> residues_;
};

} // namespace titan_contact_geometry

#endif // TITAN_CONTACT_GEOMETRY_H
