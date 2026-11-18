#ifndef TITAN_PTOPO_GRAPH_BUILDER_H
#define TITAN_PTOPO_GRAPH_BUILDER_H

// Adapter (Backbone VI.6 gate A): GROMACS topology + one trajectory frame ->
// shift-labelled periodic graph.
//
// Hard edges come from the topology (never from geometry); each gets the
// unique nearest-image shift, and a tie fails closed. Contacts are every
// periodic image within the cutoff (a pair touching two copies gives two
// edges), typed by species, with activation = distance so one graph at the
// largest cutoff carries the whole cutoff filtration. Node IDs are 0-based
// system atom indices; edge IDs are per-frame labels, not temporal
// identities.

#include "ptopo_geometry.h"
#include "ptopo_gromacs_topology.h"

#include <optional>
#include <string_view>

class TitanExecutor;

namespace titan_ptopo {

struct HardLink {
  std::uint32_t a = 0; // global atom indices, a < b
  std::uint32_t b = 0;
  TopologyLinkKind kind = TopologyLinkKind::Bond;
  int function = 0;
};

struct SystemModel {
  std::size_t atom_count = 0;
  // Per atom.
  std::vector<std::uint32_t> molecule_instance;
  std::vector<std::uint32_t> molecule_type; // index into molecule_type_names
  std::vector<Species> species;
  std::vector<std::uint32_t> residue_name_id; // index into residue_names
  std::vector<std::int64_t> residue_number;
  std::vector<std::string> residue_names;
  std::vector<std::string> molecule_type_names;
  // Species of each molecule type, as resolved.
  std::vector<Species> molecule_type_species;
  // Unique atom pairs, sorted by (a, b). A pair listed twice (a bond and a
  // constraint, or both FLEXIBLE branches in another reader) is one edge.
  std::vector<HardLink> hard_links;
  std::size_t duplicate_links_merged = 0;
};

struct SpeciesPolicy {
  // Exact molecule-type name -> species, applied first.
  std::vector<std::pair<std::string, Species>> molecule_types;
  // Otherwise by residue names: amino acids -> Protein, ribonucleotides ->
  // RNA. A molecule type with mixed or unknown residues is Other.
  bool residue_name_heuristic = true;
};

struct HardLinkPolicy {
  bool bonds = true;
  bool constraints = true;
  bool settles = true;
  bool virtual_sites = true;
  // [bonds] function types left out of the hard graph, e.g. 6 for an
  // elastic network that joins chains of one molecule type.
  std::vector<int> excluded_bond_functions;
};

std::optional<Species> parse_species(std::string_view name);

Result<SystemModel> build_system_model(const GromacsTopology &topology,
                                       const SpeciesPolicy &species = {},
                                       const HardLinkPolicy &links = {});

struct AtomSelection {
  bool protein = true;
  bool rna = true;
  bool other = false;
  std::vector<std::string> include_molecule_types; // regardless of species
  std::vector<std::string> exclude_molecule_types;
};

// Sorted global atom indices.
std::vector<std::uint32_t> select_atoms(const SystemModel &model,
                                        const AtomSelection &selection);

enum class IntramolecularContacts : std::uint8_t {
  Keep,
  // Drop contacts that close a contractible cycle with the hard graph of
  // their molecule (internal contacts); keep self-image contacts. Fails
  // closed when a hard component itself winds.
  DropContractible,
  Drop,
};

struct GraphBuildPolicy {
  double contact_cutoff = 0.0; // coordinate units (nm for XTC)
  GeometryPolicy geometry;
  // Drop a contact that repeats a hard path of <= depth links with the same
  // image (1 = the bonded pair itself; 0 keeps every contact).
  std::size_t bonded_exclusion_depth = 1;
  IntramolecularContacts intramolecular = IntramolecularContacts::Keep;
  // A hard edge longer than this fails closed (0 disables). Catches a
  // trajectory/topology atom-order mismatch.
  double maximum_hard_edge_length = 0.0;
  std::size_t maximum_contacts = 50000000;
  std::string identity_epoch;
  std::string policy_id;
  std::string reference_basis_id;
};

struct GraphBuildStats {
  std::size_t nodes = 0;
  std::size_t hard_edges = 0;
  std::size_t hard_links_cut_by_selection = 0;
  std::size_t hard_components = 0;
  std::size_t hard_winding_rank = 0; // aggregate rank of the hard graph
  std::size_t contacts_found = 0;
  std::size_t contacts_dropped_bonded = 0;
  std::size_t contacts_dropped_intramolecular = 0;
  std::size_t contact_edges = 0;
  ContactSearchStats search;
};

// `positions` holds every system atom (Cartesian, same units as the box);
// `selection` comes from select_atoms. Hard edges get IDs 1..H in (a, b)
// order, contacts H+1.. in (i, j, shift) order.
Result<Graph> build_periodic_graph(const SystemModel &model,
                                   std::span<const std::uint32_t> selection,
                                   const std::array<Vec3, 3> &box_rows,
                                   std::span<const Vec3> positions,
                                   const GraphBuildPolicy &policy,
                                   TitanExecutor *executor = nullptr,
                                   std::size_t workers = 1,
                                   GraphBuildStats *stats = nullptr);

} // namespace titan_ptopo

#endif // TITAN_PTOPO_GRAPH_BUILDER_H
