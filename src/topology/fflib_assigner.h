#ifndef TITAN_TOPOLOGY_FFLIB_ASSIGNER_H
#define TITAN_TOPOLOGY_FFLIB_ASSIGNER_H

#include "Multi_MOL.h"
#include "amino_a.h"
#include "arguments.h"
#include "c6c12.h"

#include <string>
#include <string_view>
#include <vector>

class TitanExecutor;

namespace TitanFfLib {

int assign_from_library(Multi_MOL &mol, const amino_acid &amine,
                        const para &parameter, const arguments &Arguments,
                        std::string_view force_f, TitanExecutor *executor,
                        std::string_view molecule_label);

// Copies sigma/epsilon/radius of the registered definition of `atom_type`
// (para::atom_type_to_id, the last definition read: topology atom types
// override the lib_ff table) and sets the type id. False when the type is
// unknown; the parameters are then left as they were and the id is -1.
bool fill_vdw_from_atom_type(Multi_MOL_parameters &parameters,
                             std::size_t atom_index, const para &parameter,
                             std::string_view atom_type);

// ENE-08 guard: a type name defined more than once whose registered (active)
// definition has sigma = epsilon = 0 while another definition is non-zero
// means a pair/NBFIX row or a malformed line was read as an atom type and
// would silently remove the LJ parameters of every atom of that type.
struct NonbondedTableCheck {
  std::size_t entries{0};
  std::size_t duplicate_types{0};
  std::size_t shadowed_types{0};
  std::vector<std::string> shadowed_examples;
};

[[nodiscard]] NonbondedTableCheck validate_nonbonded_table(const para &parameter);

} // namespace TitanFfLib

#endif // TITAN_TOPOLOGY_FFLIB_ASSIGNER_H
