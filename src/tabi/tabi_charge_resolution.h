#ifndef TITAN_TABI_CHARGE_RESOLUTION_H
#define TITAN_TABI_CHARGE_RESOLUTION_H
#include "tabi_types.h"
namespace TitanPBE::Tabi {
ChargeResolutionDiagnostic diagnose_charge_resolution(
    const std::vector<Atom> &atoms, const std::vector<Panel> &panels,
    const Options &options);
// Exposed for independent exhaustive checks of the accelerated diagnostic.
double charge_to_spherical_panel_distance(const Atom &charge, const Atom &sphere,
                                          const Panel &panel);
}
#endif
