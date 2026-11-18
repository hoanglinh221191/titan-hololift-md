#include "ptopo_graph_builder.h"

#include "ptopo_integer_lattice.h"
#include "ptopo_potential_dsu.h"

#include "../titan_executor.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace titan_ptopo {
namespace {

constexpr std::uint32_t kNone = std::numeric_limits<std::uint32_t>::max();

std::string upper(std::string_view s) {
  std::string out(s);
  for (char &c : out)
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return out;
}

template <std::size_t N>
bool listed(const std::array<std::string_view, N> &names, std::string_view s) {
  return std::find(names.begin(), names.end(), s) != names.end();
}

// Residue names of the common force fields (CHARMM, AMBER, Martini,
// OPLS). Phosphorylated residues are protein.
constexpr std::array<std::string_view, 50> kAminoAcids{
    "ALA", "ARG", "ASN", "ASP", "ASH", "ASX", "CYS", "CYX", "CYM", "GLN",
    "GLU", "GLH", "GLX", "GLY", "HIS", "HSD", "HSE", "HSP", "HID", "HIE",
    "HIP", "HISD", "HISE", "HISH", "HISA", "HISB", "ILE", "LEU", "LYS", "LYN",
    "LSN", "MET", "MSE", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "SEP", "TPO", "PTR", "ACE", "NME", "NMA", "NHE", "NH2", "CT3", "HYP"};

// Ribonucleotides. CHARMM's ADE/GUA/CYT are shared with DNA; a molecule with
// THY or a D-prefixed residue is DNA and so not RNA (Other).
constexpr std::array<std::string_view, 30> kRibonucleotides{
    "A",   "U",   "G",   "C",   "RA",  "RU",  "RG",  "RC",  "RA3", "RA5",
    "RU3", "RU5", "RG3", "RG5", "RC3", "RC5", "A3",  "A5",  "U3",  "U5",
    "G3",  "G5",  "C3",  "C5",  "ADE", "URA", "GUA", "CYT", "RAN", "RUN"};

bool is_amino_acid(std::string_view r) {
  if (listed(kAminoAcids, r))
    return true;
  // AMBER terminal variants: NALA, CALA, ...
  return r.size() == 4 && (r[0] == 'N' || r[0] == 'C') &&
         listed(kAminoAcids, r.substr(1));
}

bool is_ribonucleotide(std::string_view r) { return listed(kRibonucleotides, r); }

Species classify_residues(const TopologyMoleculeType &type) {
  if (type.atoms.empty())
    return Species::Other;
  bool protein = true, rna = true;
  for (const TopologyAtom &atom : type.atoms) {
    const std::string r = upper(atom.residue_name);
    protein = protein && is_amino_acid(r);
    rna = rna && is_ribonucleotide(r);
    if (!protein && !rna)
      return Species::Other;
  }
  if (protein == rna) // cannot be both; both false handled above
    return Species::Other;
  return protein ? Species::Protein : Species::RNA;
}

bool link_kept(const TopologyBond &bond, const HardLinkPolicy &p) {
  switch (bond.kind) {
  case TopologyLinkKind::Bond:
    return p.bonds && std::find(p.excluded_bond_functions.begin(),
                                p.excluded_bond_functions.end(),
                                bond.function) == p.excluded_bond_functions.end();
  case TopologyLinkKind::Constraint:
    return p.constraints;
  case TopologyLinkKind::Settle:
    return p.settles;
  case TopologyLinkKind::VirtualSite:
    return p.virtual_sites;
  }
  return false;
}

EdgeKind contact_kind(Species a, Species b) {
  if (a == Species::Protein && b == Species::Protein)
    return EdgeKind::ProteinProtein;
  if (a == Species::RNA && b == Species::RNA)
    return EdgeKind::RNAInteraction;
  if ((a == Species::Protein && b == Species::RNA) ||
      (a == Species::RNA && b == Species::Protein))
    return EdgeKind::ProteinRNA;
  return EdgeKind::Generic;
}

// A contact repeating a short hard path with the same image.
struct PathKey {
  std::uint32_t u = 0, v = 0; // u < v
  Z3 shift{};
  friend auto operator<=>(const PathKey &, const PathKey &) = default;
};

struct HardEdge {
  std::uint32_t u = 0, v = 0;
  Z3 shift{};
};

Result<std::vector<PathKey>>
bonded_paths(std::size_t n, const std::vector<HardEdge> &hard,
             std::size_t depth) {
  std::vector<PathKey> keys;
  if (depth == 0 || hard.empty())
    return keys;
  std::vector<std::uint32_t> offsets(n + 1, 0);
  for (const HardEdge &e : hard) {
    ++offsets[e.u + 1];
    ++offsets[e.v + 1];
  }
  for (std::size_t i = 1; i <= n; ++i)
    offsets[i] += offsets[i - 1];
  struct Arc {
    std::uint32_t to = 0;
    Z3 shift{};
  };
  std::vector<Arc> arcs(offsets[n]);
  std::vector<std::uint32_t> cursor(offsets.begin(), offsets.end() - 1);
  for (const HardEdge &e : hard) {
    Z3 back{};
    if (!checked::neg3(e.shift, back))
      return overflow_error("bonded exclusion paths");
    arcs[cursor[e.u]++] = {e.v, e.shift};
    arcs[cursor[e.v]++] = {e.u, back};
  }
  std::vector<std::pair<std::uint32_t, Z3>> frontier, next;
  for (std::uint32_t u = 0; u < n; ++u) {
    frontier.assign(1, {u, Z3{}});
    for (std::size_t level = 0; level < depth && !frontier.empty(); ++level) {
      next.clear();
      for (const auto &[x, path] : frontier)
        for (std::uint32_t k = offsets[x]; k < offsets[x + 1]; ++k) {
          Z3 reached{};
          if (!checked::add3(path, arcs[k].shift, reached))
            return overflow_error("bonded exclusion paths");
          if (arcs[k].to > u)
            keys.push_back({u, arcs[k].to, reached});
          next.emplace_back(arcs[k].to, reached);
        }
      frontier.swap(next);
    }
  }
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  return keys;
}

} // namespace

std::optional<Species> parse_species(std::string_view name) {
  const std::string s = upper(name);
  if (s == "PROTEIN")
    return Species::Protein;
  if (s == "RNA")
    return Species::RNA;
  if (s == "OTHER")
    return Species::Other;
  return std::nullopt;
}

Result<SystemModel> build_system_model(const GromacsTopology &topology,
                                       const SpeciesPolicy &species_policy,
                                       const HardLinkPolicy &links) {
  SystemModel model;
  std::unordered_map<std::string, std::uint32_t> type_index;
  for (std::uint32_t t = 0; t < topology.molecule_types.size(); ++t) {
    const TopologyMoleculeType &type = topology.molecule_types[t];
    type_index.emplace(type.name, t);
    model.molecule_type_names.push_back(type.name);
    Species s = species_policy.residue_name_heuristic ? classify_residues(type)
                                                      : Species::Other;
    for (const auto &[name, forced] : species_policy.molecule_types)
      if (name == type.name)
        s = forced;
    model.molecule_type_species.push_back(s);
  }
  for (const auto &[name, forced] : species_policy.molecule_types)
    if (!type_index.contains(name))
      return make_error(ErrorCode::InvalidInput,
                        "species override names unknown molecule type '" +
                            name + "'");
  std::uint64_t total = 0;
  for (const TopologyMoleculeBlock &block : topology.molecules) {
    const auto it = type_index.find(block.name);
    if (it == type_index.end())
      return make_error(ErrorCode::InvalidInput,
                        "unknown molecule type '" + block.name + "'");
    const std::uint64_t atoms = topology.molecule_types[it->second].atoms.size();
    if (block.count != 0 && atoms > (kNone - 1 - total) / block.count)
      return make_error(ErrorCode::ResourceLimit,
                        "system exceeds 32-bit atom indexing");
    total += atoms * block.count;
  }
  model.atom_count = static_cast<std::size_t>(total);
  model.molecule_instance.reserve(model.atom_count);
  model.molecule_type.reserve(model.atom_count);
  model.species.reserve(model.atom_count);
  model.residue_name_id.reserve(model.atom_count);
  model.residue_number.reserve(model.atom_count);
  std::unordered_map<std::string, std::uint32_t> residue_index;
  std::uint32_t offset = 0, instance = 0;
  for (const TopologyMoleculeBlock &block : topology.molecules) {
    const std::uint32_t t = type_index.at(block.name);
    const TopologyMoleculeType &type = topology.molecule_types[t];
    std::vector<std::uint32_t> residue_ids(type.atoms.size());
    for (std::size_t a = 0; a < type.atoms.size(); ++a) {
      const auto [it, fresh] = residue_index.emplace(
          type.atoms[a].residue_name,
          static_cast<std::uint32_t>(model.residue_names.size()));
      if (fresh)
        model.residue_names.push_back(type.atoms[a].residue_name);
      residue_ids[a] = it->second;
    }
    for (std::uint64_t copy = 0; copy < block.count; ++copy, ++instance) {
      for (std::size_t a = 0; a < type.atoms.size(); ++a) {
        model.molecule_instance.push_back(instance);
        model.molecule_type.push_back(t);
        model.species.push_back(model.molecule_type_species[t]);
        model.residue_name_id.push_back(residue_ids[a]);
        model.residue_number.push_back(type.atoms[a].residue_number);
      }
      for (const TopologyBond &bond : type.bonds) {
        if (!link_kept(bond, links))
          continue;
        const std::uint32_t a = offset + bond.a, b = offset + bond.b;
        model.hard_links.push_back(
            {std::min(a, b), std::max(a, b), bond.kind, bond.function});
      }
      offset += static_cast<std::uint32_t>(type.atoms.size());
    }
  }
  std::sort(model.hard_links.begin(), model.hard_links.end(),
            [](const HardLink &x, const HardLink &y) {
              if (x.a != y.a)
                return x.a < y.a;
              if (x.b != y.b)
                return x.b < y.b;
              if (x.kind != y.kind)
                return x.kind < y.kind;
              return x.function < y.function;
            });
  const std::size_t before = model.hard_links.size();
  model.hard_links.erase(
      std::unique(model.hard_links.begin(), model.hard_links.end(),
                  [](const HardLink &x, const HardLink &y) {
                    return x.a == y.a && x.b == y.b;
                  }),
      model.hard_links.end());
  model.duplicate_links_merged = before - model.hard_links.size();
  return model;
}

std::vector<std::uint32_t> select_atoms(const SystemModel &model,
                                        const AtomSelection &selection) {
  std::vector<unsigned char> type_state(model.molecule_type_names.size(), 0);
  for (std::size_t t = 0; t < model.molecule_type_names.size(); ++t) {
    const Species s = model.molecule_type_species[t];
    bool keep = (s == Species::Protein && selection.protein) ||
                (s == Species::RNA && selection.rna) ||
                (s == Species::Other && selection.other);
    const std::string &name = model.molecule_type_names[t];
    if (std::find(selection.include_molecule_types.begin(),
                  selection.include_molecule_types.end(),
                  name) != selection.include_molecule_types.end())
      keep = true;
    if (std::find(selection.exclude_molecule_types.begin(),
                  selection.exclude_molecule_types.end(),
                  name) != selection.exclude_molecule_types.end())
      keep = false;
    type_state[t] = keep ? 1 : 0;
  }
  std::vector<std::uint32_t> atoms;
  for (std::uint32_t i = 0; i < model.atom_count; ++i)
    if (type_state[model.molecule_type[i]])
      atoms.push_back(i);
  return atoms;
}

Result<Graph> build_periodic_graph(const SystemModel &model,
                                   std::span<const std::uint32_t> selection,
                                   const std::array<Vec3, 3> &box_rows,
                                   std::span<const Vec3> positions,
                                   const GraphBuildPolicy &policy,
                                   TitanExecutor *executor, std::size_t workers,
                                   GraphBuildStats *stats) {
  if (positions.size() != model.atom_count)
    return make_error(ErrorCode::InvalidInput,
                      "frame atom count " + std::to_string(positions.size()) +
                          " differs from topology atom count " +
                          std::to_string(model.atom_count));
  if (policy.identity_epoch.empty() || policy.policy_id.empty() ||
      policy.reference_basis_id.empty())
    return make_error(ErrorCode::InvalidInput,
                      "identity_epoch, policy_id and reference_basis_id are "
                      "required");
  if (!(policy.maximum_hard_edge_length >= 0.0) ||
      !std::isfinite(policy.maximum_hard_edge_length))
    return make_error(ErrorCode::InvalidInput,
                      "maximum hard-edge length must be finite and >= 0");
  for (std::size_t k = 0; k < selection.size(); ++k)
    if (selection[k] >= model.atom_count ||
        (k > 0 && selection[k] <= selection[k - 1]))
      return make_error(ErrorCode::InvalidInput,
                        "selection must be strictly increasing atom indices");
  const Cell cell = cell_from_box_rows(box_rows);
  auto metrics = analyze_cell(cell, policy.geometry);
  if (!metrics)
    return std::unexpected(metrics.error());

  const std::size_t n = selection.size();
  std::vector<std::uint32_t> node_of(model.atom_count, kNone);
  std::vector<Vec3> fractional(n);
  for (std::uint32_t k = 0; k < n; ++k) {
    const Vec3 &x = positions[selection[k]];
    if (!std::isfinite(x[0]) || !std::isfinite(x[1]) || !std::isfinite(x[2]))
      return make_error(ErrorCode::InvalidInput,
                        "nonfinite coordinate for atom " +
                            std::to_string(selection[k]));
    node_of[selection[k]] = k;
    fractional[k] = to_fractional(*metrics, x);
  }

  GraphBuildStats local;
  local.nodes = n;

  // Hard edges: unique nearest-image shift per topology link.
  std::vector<std::uint32_t> link_index;
  for (std::uint32_t l = 0; l < model.hard_links.size(); ++l) {
    const HardLink &link = model.hard_links[l];
    const bool a_in = node_of[link.a] != kNone, b_in = node_of[link.b] != kNone;
    if (a_in && b_in)
      link_index.push_back(l);
    else if (a_in || b_in)
      ++local.hard_links_cut_by_selection;
  }
  std::vector<HardEdge> hard(link_index.size());
  struct LinkFailure {
    bool failed = false;
    std::size_t index = 0;
    Error error;
  };
  const std::size_t slots =
      executor != nullptr ? std::max<std::size_t>(1, executor->capacity()) : 1;
  std::vector<LinkFailure> failures(slots);
  const auto shift_body = [&](std::size_t worker, std::size_t begin,
                              std::size_t end) {
    LinkFailure &failure = failures[worker];
    for (std::size_t k = begin; k < end; ++k) {
      const HardLink &link = model.hard_links[link_index[k]];
      HardEdge &edge = hard[k];
      edge.u = node_of[link.a];
      edge.v = node_of[link.b];
      const Vec3 &su = fractional[edge.u];
      const Vec3 &sv = fractional[edge.v];
      const Vec3 delta{sv[0] - su[0], sv[1] - su[1], sv[2] - su[2]};
      const auto record = [&](Error error) {
        if (!failure.failed || k < failure.index) {
          failure.failed = true;
          failure.index = k;
          failure.error = std::move(error);
        }
      };
      auto shift = unique_nearest_shift(*metrics, delta, policy.geometry);
      if (!shift) {
        Error error = shift.error();
        error.message = "hard edge " + std::to_string(link.a) + "-" +
                        std::to_string(link.b) + ": " + error.message;
        record(std::move(error));
        continue;
      }
      edge.shift = *shift;
      if (policy.maximum_hard_edge_length > 0.0) {
        const Vec3 x = cartesian(
            cell, {delta[0] + static_cast<double>(edge.shift[0]),
                   delta[1] + static_cast<double>(edge.shift[1]),
                   delta[2] + static_cast<double>(edge.shift[2])});
        const double length = std::hypot(x[0], x[1], x[2]);
        if (length > policy.maximum_hard_edge_length)
          record(Error{ErrorCode::InvalidInput,
                       "hard edge " + std::to_string(link.a) + "-" +
                           std::to_string(link.b) + " is " +
                           std::to_string(length) +
                           " long, above the declared maximum (atom order "
                           "of trajectory and topology?)"});
      }
    }
  };
  if (executor != nullptr && workers > 1 && link_index.size() >= 8192)
    executor->parallel_for(link_index.size(), workers, 4096, shift_body);
  else if (!link_index.empty())
    shift_body(0, 0, link_index.size());
  const LinkFailure *first_failure = nullptr;
  for (const LinkFailure &f : failures)
    if (f.failed && (first_failure == nullptr || f.index < first_failure->index))
      first_failure = &f;
  if (first_failure != nullptr)
    return std::unexpected(first_failure->error);
  local.hard_edges = hard.size();

  // Hard-graph potentials: components, hard winding, contractibility.
  detail::PotentialDsu dsu(n);
  Hnf3 hard_winding;
  for (const HardEdge &e : hard) {
    detail::DsuInsert step;
    if (!dsu.insert(e.u, e.v, e.shift, step))
      return overflow_error("hard-graph potentials");
    if (!step.merged && !is_zero(step.residual))
      if (auto ok = hnf_insert(hard_winding, step.residual); !ok)
        return std::unexpected(ok.error());
  }
  std::vector<std::uint32_t> root(n);
  std::vector<Z3> potential(n);
  std::size_t components = 0;
  for (std::uint32_t k = 0; k < n; ++k) {
    if (!dsu.find(k, root[k], potential[k]))
      return overflow_error("hard-graph potentials");
    if (root[k] == k)
      ++components;
  }
  local.hard_components = components;
  local.hard_winding_rank = hard_winding.rank;
  if (policy.intramolecular == IntramolecularContacts::DropContractible &&
      hard_winding.rank > 0)
    return make_error(ErrorCode::InvalidInput,
                      "intramolecular=drop-contractible needs a non-winding "
                      "hard graph; a hard component winds around the cell");

  // Contacts over every periodic image.
  ContactSearchOptions search;
  search.cutoff = policy.contact_cutoff;
  search.policy = policy.geometry;
  search.maximum_contacts = policy.maximum_contacts;
  search.workers = workers;
  auto contacts = find_periodic_contacts(cell, fractional, search, executor,
                                         &local.search);
  if (!contacts)
    return std::unexpected(contacts.error());
  local.contacts_found = contacts->size();

  auto bonded = bonded_paths(n, hard, policy.bonded_exclusion_depth);
  if (!bonded)
    return std::unexpected(bonded.error());

  Graph g;
  g.identity_epoch = policy.identity_epoch;
  g.policy_id = policy.policy_id;
  g.reference_basis_id = policy.reference_basis_id;
  g.nodes.resize(n);
  for (std::uint32_t k = 0; k < n; ++k) {
    const std::uint32_t atom = selection[k];
    g.nodes[k] = {atom, model.molecule_instance[atom], model.species[atom]};
  }
  g.edges.reserve(hard.size() + contacts->size());
  std::uint64_t next_id = 1;
  for (const HardEdge &e : hard)
    g.edges.push_back({next_id++, e.u, e.v, e.shift, EdgeKind::Hard, 0.0});
  for (const PeriodicContact &c : *contacts) {
    if (c.i != c.j &&
        std::binary_search(bonded->begin(), bonded->end(),
                           PathKey{c.i, c.j, c.shift})) {
      ++local.contacts_dropped_bonded;
      continue;
    }
    if (policy.intramolecular != IntramolecularContacts::Keep &&
        root[c.i] == root[c.j]) {
      bool drop = policy.intramolecular == IntramolecularContacts::Drop;
      if (!drop) {
        // Residual of the cycle the contact closes with the hard graph.
        Z3 t{}, w{};
        if (!checked::add3(potential[c.i], c.shift, t) ||
            !checked::sub3(t, potential[c.j], w))
          return overflow_error("intramolecular contact residual");
        drop = is_zero(w);
      }
      if (drop) {
        ++local.contacts_dropped_intramolecular;
        continue;
      }
    }
    g.edges.push_back({next_id++, c.i, c.j, c.shift,
                       contact_kind(g.nodes[c.i].species, g.nodes[c.j].species),
                       c.distance});
  }
  local.contact_edges = g.edges.size() - hard.size();
  if (stats != nullptr)
    *stats = local;
  return g;
}

} // namespace titan_ptopo
