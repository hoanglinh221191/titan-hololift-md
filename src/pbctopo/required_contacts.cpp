#include "required_contacts.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <tuple>
#include <unordered_map>

namespace titan_pbctopo {
namespace {

std::string strip_comment(std::string line) {
  const auto hash_pos = line.find('#');
  const auto semicolon_pos = line.find(';');
  const auto comment_pos = std::min(
      hash_pos == std::string::npos ? line.size() : hash_pos,
      semicolon_pos == std::string::npos ? line.size() : semicolon_pos);
  line.resize(comment_pos);
  return line;
}

} // namespace

std::expected<std::vector<PbctopoRequiredContact>, std::string>
load_pbctopo_required_contacts(const std::string &path,
                               std::size_t selected_atom_count) {
  std::ifstream input(path);
  if (!input)
    return std::unexpected("cannot open PBCTOPO required-contact file: " + path);
  if (selected_atom_count == 0) {
    return std::unexpected(
        "PBCTOPO required-contact file cannot be mapped to an empty selection");
  }

  std::vector<PbctopoRequiredContact> contacts;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    line = strip_comment(std::move(line));
    std::istringstream fields(line);
    std::size_t atom_a = 0;
    std::size_t atom_b = 0;
    double max_distance_A = 0.0;
    if (!(fields >> atom_a))
      continue;
    if (!(fields >> atom_b >> max_distance_A)) {
      return std::unexpected("invalid PBCTOPO required-contact record at " +
                             path + ":" + std::to_string(line_number));
    }
    double weight = 1.0;
    if (fields >> weight) {
      std::string trailing;
      if (fields >> trailing) {
        return std::unexpected(
            "too many fields in PBCTOPO required-contact record at " + path +
            ":" + std::to_string(line_number));
      }
    }
    if (atom_a == 0 || atom_b == 0 || atom_a > selected_atom_count ||
        atom_b > selected_atom_count || atom_a == atom_b) {
      return std::unexpected(
          "PBCTOPO required-contact atom indices must be distinct 1-based "
          "indices within the selected atom set at " +
          path + ":" + std::to_string(line_number));
    }
    if (!std::isfinite(max_distance_A) || max_distance_A <= 0.0 ||
        !std::isfinite(weight) || weight <= 0.0) {
      return std::unexpected(
          "PBCTOPO required-contact distance and weight must be finite and "
          "positive at " +
          path + ":" + std::to_string(line_number));
    }
    --atom_a;
    --atom_b;
    if (atom_b < atom_a)
      std::swap(atom_a, atom_b);
    contacts.push_back({atom_a, atom_b, max_distance_A, weight});
  }

  std::sort(contacts.begin(), contacts.end(),
            [](const PbctopoRequiredContact &lhs,
               const PbctopoRequiredContact &rhs) {
              return std::tie(lhs.a, lhs.b, lhs.max_distance_A, lhs.weight) <
                     std::tie(rhs.a, rhs.b, rhs.max_distance_A, rhs.weight);
            });
  std::vector<PbctopoRequiredContact> unique_contacts;
  unique_contacts.reserve(contacts.size());
  for (const auto &contact : contacts) {
    if (!unique_contacts.empty() && unique_contacts.back().a == contact.a &&
        unique_contacts.back().b == contact.b) {
      unique_contacts.back().max_distance_A =
          std::min(unique_contacts.back().max_distance_A,
                   contact.max_distance_A);
      unique_contacts.back().weight =
          std::max(unique_contacts.back().weight, contact.weight);
      continue;
    }
    unique_contacts.push_back(contact);
  }
  if (unique_contacts.empty()) {
    return std::unexpected("PBCTOPO required-contact file contains no contacts: " +
                           path);
  }
  return unique_contacts;
}

PbctopoRequiredContactConstraintContext
build_pbctopo_required_contact_constraint_context(
    const std::vector<PbctopoAtom> &atoms,
    const std::vector<std::vector<std::size_t>> &components, box Box,
    const std::vector<PbctopoRequiredContact> &contacts) {
  PbctopoRequiredContactConstraintContext context;
  context.component_count = components.size();
  context.atom_component_index.assign(
      atoms.size(), std::numeric_limits<std::size_t>::max());
  for (std::size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx) {
    for (std::size_t atom_idx : components[comp_idx]) {
      if (atom_idx >= atoms.size() ||
          context.atom_component_index[atom_idx] !=
              std::numeric_limits<std::size_t>::max()) {
        context.valid = false;
        continue;
      }
      context.atom_component_index[atom_idx] = comp_idx;
    }
  }

  double v1[3]{}, v2[3]{}, v3[3]{};
  Box.get_v1(v1);
  Box.get_v2(v2);
  Box.get_v3(v3);
  context.v1 = {v1[0], v1[1], v1[2]};
  context.v2 = {v2[0], v2[1], v2[2]};
  context.v3 = {v3[0], v3[1], v3[2]};

  struct ComponentPairHash {
    std::size_t operator()(
        const std::pair<std::size_t, std::size_t> &value) const noexcept {
      std::size_t seed = value.first;
      seed ^= value.second + 0x9e3779b9U + (seed << 6) + (seed >> 2);
      return seed;
    }
  };
  std::unordered_map<std::pair<std::size_t, std::size_t>, std::size_t,
                     ComponentPairHash>
      group_index;
  group_index.reserve(contacts.size());
  for (const PbctopoRequiredContact &contact : contacts) {
    if (contact.a >= atoms.size() || contact.b >= atoms.size() ||
        contact.a == contact.b) {
      context.valid = false;
      context.infeasible = true;
      continue;
    }
    const std::size_t comp_a = context.atom_component_index[contact.a];
    const std::size_t comp_b = context.atom_component_index[contact.b];
    if (comp_a >= components.size() || comp_b >= components.size()) {
      context.valid = false;
      context.infeasible = true;
      continue;
    }
    if (comp_a == comp_b) {
      ++context.same_component_contacts;
      const double dx = atoms[contact.b].x - atoms[contact.a].x;
      const double dy = atoms[contact.b].y - atoms[contact.a].y;
      const double dz = atoms[contact.b].z - atoms[contact.a].z;
      const double max_d2 = contact.max_distance_A * contact.max_distance_A;
      if (!pbctopo_required_contact_satisfied(
              dx * dx + dy * dy + dz * dz, max_d2))
        context.infeasible = true;
      continue;
    }

    const std::size_t lhs = std::min(comp_a, comp_b);
    const std::size_t rhs = std::max(comp_a, comp_b);
    const auto [position, inserted] =
        group_index.emplace(std::pair{lhs, rhs}, context.groups.size());
    if (inserted) {
      context.groups.push_back({lhs, rhs, {contact}});
    } else {
      context.groups[position->second].contacts.push_back(contact);
    }
  }
  std::sort(context.groups.begin(), context.groups.end(),
            [](const auto &lhs, const auto &rhs) {
              return std::tie(lhs.lhs_component, lhs.rhs_component) <
                     std::tie(rhs.lhs_component, rhs.rhs_component);
            });
  return context;
}

const PbctopoRequiredContactConstraintGroup *
find_pbctopo_required_contact_constraint_group(
    const PbctopoRequiredContactConstraintContext &context,
    std::size_t lhs_component, std::size_t rhs_component) {
  if (rhs_component < lhs_component)
    std::swap(lhs_component, rhs_component);
  const auto found = std::lower_bound(
      context.groups.begin(), context.groups.end(),
      std::pair{lhs_component, rhs_component},
      [](const PbctopoRequiredContactConstraintGroup &group,
         const std::pair<std::size_t, std::size_t> &key) {
        return std::pair{group.lhs_component, group.rhs_component} < key;
      });
  return found != context.groups.end() &&
                 found->lhs_component == lhs_component &&
                 found->rhs_component == rhs_component
             ? &*found
             : nullptr;
}

bool pbctopo_required_contact_relative_shift_allowed(
    const PbctopoRequiredContactConstraintContext &context,
    const std::vector<PbctopoAtom> &atoms, std::size_t lhs_component,
    std::size_t rhs_component, const Int3 &rhs_minus_lhs) {
  if (!context.valid || context.infeasible)
    return false;
  if (rhs_component < lhs_component) {
    std::swap(lhs_component, rhs_component);
    Int3 reversed{};
    if (!checked_int3_negate(rhs_minus_lhs, reversed))
      return false;
    return pbctopo_required_contact_relative_shift_allowed(
        context, atoms, lhs_component, rhs_component, reversed);
  }
  const auto *group = find_pbctopo_required_contact_constraint_group(
      context, lhs_component, rhs_component);
  if (group == nullptr)
    return true;

  const Vec3 rhs_translation{
      rhs_minus_lhs.x * context.v1.x + rhs_minus_lhs.y * context.v2.x +
          rhs_minus_lhs.z * context.v3.x,
      rhs_minus_lhs.x * context.v1.y + rhs_minus_lhs.y * context.v2.y +
          rhs_minus_lhs.z * context.v3.y,
      rhs_minus_lhs.x * context.v1.z + rhs_minus_lhs.y * context.v2.z +
          rhs_minus_lhs.z * context.v3.z};
  for (const PbctopoRequiredContact &contact : group->contacts) {
    const std::size_t comp_a = context.atom_component_index[contact.a];
    const std::size_t comp_b = context.atom_component_index[contact.b];
    const Vec3 shift_a = comp_a == rhs_component ? rhs_translation : Vec3{};
    const Vec3 shift_b = comp_b == rhs_component ? rhs_translation : Vec3{};
    const double dx = atoms[contact.b].x + shift_b.x - atoms[contact.a].x -
                      shift_a.x;
    const double dy = atoms[contact.b].y + shift_b.y - atoms[contact.a].y -
                      shift_a.y;
    const double dz = atoms[contact.b].z + shift_b.z - atoms[contact.a].z -
                      shift_a.z;
    if (!pbctopo_required_contact_satisfied(
            dx * dx + dy * dy + dz * dz,
            contact.max_distance_A * contact.max_distance_A)) {
      return false;
    }
  }
  return true;
}

} // namespace titan_pbctopo
