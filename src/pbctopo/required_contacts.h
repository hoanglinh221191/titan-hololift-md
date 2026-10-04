#ifndef TITAN_PBCTOPO_REQUIRED_CONTACTS_H
#define TITAN_PBCTOPO_REQUIRED_CONTACTS_H

#include "types.h"
#include "../box_type.h"

#include <cmath>
#include <cstddef>
#include <expected>
#include <string>
#include <vector>

namespace titan_pbctopo {

inline constexpr double pbctopo_required_contact_squared_tolerance = 1.0e-6;

inline bool pbctopo_required_contact_satisfied(double distance_squared,
                                               double maximum_squared) noexcept {
  return std::isfinite(distance_squared) && std::isfinite(maximum_squared) &&
         distance_squared <=
             maximum_squared + pbctopo_required_contact_squared_tolerance;
}

struct PbctopoRequiredContactConstraintGroup {
  std::size_t lhs_component = 0;
  std::size_t rhs_component = 0;
  std::vector<PbctopoRequiredContact> contacts;
};

struct PbctopoRequiredContactConstraintContext {
  std::size_t component_count = 0;
  std::vector<std::size_t> atom_component_index;
  std::vector<PbctopoRequiredContactConstraintGroup> groups;
  Vec3 v1{};
  Vec3 v2{};
  Vec3 v3{};
  std::size_t same_component_contacts = 0;
  bool valid = true;
  bool infeasible = false;
};

std::expected<std::vector<PbctopoRequiredContact>, std::string>
load_pbctopo_required_contacts(const std::string &path,
                               std::size_t selected_atom_count);

PbctopoRequiredContactConstraintContext
build_pbctopo_required_contact_constraint_context(
    const std::vector<PbctopoAtom> &atoms,
    const std::vector<std::vector<std::size_t>> &components, box Box,
    const std::vector<PbctopoRequiredContact> &contacts);

const PbctopoRequiredContactConstraintGroup *
find_pbctopo_required_contact_constraint_group(
    const PbctopoRequiredContactConstraintContext &context,
    std::size_t lhs_component, std::size_t rhs_component);

bool pbctopo_required_contact_relative_shift_allowed(
    const PbctopoRequiredContactConstraintContext &context,
    const std::vector<PbctopoAtom> &atoms, std::size_t lhs_component,
    std::size_t rhs_component, const Int3 &rhs_minus_lhs);

} // namespace titan_pbctopo

#endif
