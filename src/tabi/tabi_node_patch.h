#ifndef TITAN_PBE_TABI_NODE_PATCH_H
#define TITAN_PBE_TABI_NODE_PATCH_H

#include "tabi_types.h"

#include <string>
#include <vector>

namespace TitanPBE::Tabi {

// Converts a conforming triangle-panel surface into barycentric vertex
// patches. The returned panel count equals the number of mesh vertices and
// the patch areas sum to the original triangle area.
std::vector<Panel>
build_node_patch_panels(const std::vector<Panel> &triangle_panels,
                        const std::string &diagonal_mode = "solid-angle");

} // namespace TitanPBE::Tabi

#endif // TITAN_PBE_TABI_NODE_PATCH_H
