#ifndef TITAN_TOPOLOGY_MARTINI_TOPOLOGY_H
#define TITAN_TOPOLOGY_MARTINI_TOPOLOGY_H

// Reader of a MARTINI GROMACS topology (Multi_MOL::parse_martini_topology):
//   martini_topology_read.cpp    #include walk, sections, pair table
//   martini_topology_map.cpp     bead types, charges, LJ and radii per atom
//   martini_topology_chains.cpp  chain number of each bead

#include "Multi_MOL.h"

#include <cstddef>
#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace titan_martini {

struct LJParam {
  double sigma;
  double epsilon;
};
struct PairRule {
  std::string type_i;
  std::string type_j;
  LJParam lj;
};
struct AtomDef {
  std::string type;
  int resnr;
  std::string resname;
  std::string atomname;
  double charge;
};
struct MolTemplate {
  std::vector<AtomDef> atoms;
};

class MartiniTopology {
public:
  std::vector<std::pair<std::string, int>> system_molecules;
  int files_read = 0;

  void parse_files(const std::string &top_file, std::fstream &log_file);
  void build_flat_system();
  void map_to_titan(std::vector<mol_info> &info,
                    Multi_MOL_parameters &parameters, std::fstream &log_file,
                    cache_sigma_epsilon *martini_cache);
  std::size_t get_flat_size() const { return flat_system.size(); }
  int get_molecule_count(const std::string &target_mol) const;

private:
  std::unordered_map<std::string, LJParam> atomtype_params;
  std::vector<PairRule> nonbond_rules;
  std::unordered_map<std::string, MolTemplate> molecule_types;
  std::vector<AtomDef> flat_system;
  std::unordered_map<std::string, int> type_to_id;
  std::vector<double> pair_sigma_table;
  std::vector<double> pair_epsilon_table;
  std::size_t explicit_pair_rules{0};
  std::size_t fallback_pair_rules{0};

  void finalize_pair_table();
  std::string get_directive(const std::string &line);
  void assign_chain_numbers(std::vector<mol_info> &info,
                            std::fstream &log_file);
};

} // namespace titan_martini

#endif // TITAN_TOPOLOGY_MARTINI_TOPOLOGY_H
