#include "Multi_MOL.h"
#include "titan_ui.h"
#include "topology/martini_topology.h"

#include <fstream>
#include <string>

int Multi_MOL::parse_martini_topology(const arguments &Arguments,
                                      std::fstream &log_file,
                                      cache_sigma_epsilon *martini_cache) {
  if (info.empty())
    return 0;

  TitanUI::print_info(
      "Initializing Object-Oriented MARTINI Topology Parser...");
  TitanUI::log_section(log_file, "MARTINI Topology Parser");

  titan_martini::MartiniTopology parser;

  // Step 2: Parse Topology Structure
  parser.parse_files(Arguments.topo_name, log_file);
  TitanUI::log_kv(log_file, "Files Parsed", std::to_string(parser.files_read));

  // Step 3: Flatten Templates to Linear Sequence
  parser.build_flat_system();

  // Step 4: Map Properties and Assign Respos/Chains
  parser.map_to_titan(info, parameters, log_file, martini_cache);

  // Auto-detect the primary molecule copies (e.g., molecule_0 or generic
  // Protein) Assuming the primary molecule type is the first one in the [
  // molecules ] section if not explicit
  if (!parser.system_molecules.empty()) {
    int auto_copies = parser.system_molecules.front().second;
    TitanUI::print_info("MARTINI Parser: Auto-detected " +
                        std::to_string(auto_copies) +
                        " copies of primary molecule '" +
                        parser.system_molecules.front().first + "'.");

    // We can't modify const arguments, but we can store this if needed
    // elsewhere, or we handle the copies generation right here based on
    // auto_copies if the design requires it. For now, Titan relies on
    // `Arguments.num_martini_copies` in `generate_martini_copies`. We will need
    // to update the caller or remove the const constraint if we must overwrite
    // it.
  }

  // Re-use `generate_index()` to build the system index using the PDB's native
  // chain/resnr
  // this->generate_index(); // REMOVED because PDB parser already does it!
  return 0;
}
