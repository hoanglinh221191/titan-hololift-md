#ifndef TITAN_FFLIB_ASSIGNER_H
#define TITAN_FFLIB_ASSIGNER_H

#include "Multi_MOL.h"
#include "amino_a.h"
#include "arguments.h"
#include "c6c12.h"

#include <string_view>

class TitanExecutor;

namespace TitanFfLib {

int assign_from_library(Multi_MOL &mol, const amino_acid &amine,
                        const para &parameter, const arguments &Arguments,
                        std::string_view force_f, TitanExecutor *executor,
                        std::string_view molecule_label);

bool fill_vdw_from_atom_type(Multi_MOL_parameters &parameters,
                             std::size_t atom_index, const para &parameter,
                             std::string_view atom_type);

} // namespace TitanFfLib

#endif // TITAN_FFLIB_ASSIGNER_H
