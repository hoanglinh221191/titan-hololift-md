// Hydrogens of the residue templates that the structure lacks: where
// they are reported, and where they are inserted (-fflib-hydrogens
// reference).
#include "topology/fflib_detail.h"

#include "titan_executor.h"

#include <cctype>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace TitanFfLib::detail {
namespace {

std::tuple<double, double, double> hydrogen_position(const MOL &frame,
                                                     std::size_t anchor,
                                                     std::size_t salt) {
  const double x = anchor < frame.pos.x.size() ? frame.pos.x[anchor] : 0.0;
  const double y = anchor < frame.pos.y.size() ? frame.pos.y[anchor] : 0.0;
  const double z = anchor < frame.pos.z.size() ? frame.pos.z[anchor] : 0.0;
  const double sx = (salt % 3 == 0) ? 1.0 : (salt % 3 == 1 ? -0.5 : 0.5);
  const double sy = (salt % 5 == 0) ? 0.0 : (salt % 5 < 3 ? 0.866 : -0.866);
  const double sz = (salt % 7 == 0) ? 0.707 : -0.707;
  return {x + kHydrogenOffset * sx, y + kHydrogenOffset * sy,
          z + kHydrogenOffset * sz};
}

void restore_index_types(Multi_MOL &mol, const std::vector<ResidueSpan> &spans) {
  if (mol.Index.size() <= 1)
    return;

  std::size_t span_pos = 0;
  for (std::size_t c = 0; c + 1 < mol.Index.size(); ++c) {
    std::unordered_set<std::string> chain_types;
    for (std::size_t r = 0; r < mol.Index[c].get_size_v(); ++r) {
      const std::string type =
          span_pos < spans.size() ? spans[span_pos].type : "unknown";
      mol.Index[c].set_type_base(r, type);
      chain_types.insert(type);
      ++span_pos;
    }
    if (chain_types.size() == 1)
      mol.Index[c].set_type(*chain_types.begin());
    else if (!chain_types.empty())
      mol.Index[c].set_type("complex");
  }
}

} // namespace

std::size_t find_anchor_atom(const Multi_MOL &mol, const ResidueSpan &span,
                             const std::string &hydrogen_key) {
  std::vector<std::string> candidates;
  if (hydrogen_key == "H" || hydrogen_key == "HN" ||
      hydrogen_key == "H1" || hydrogen_key == "H2" ||
      hydrogen_key == "H3") {
    candidates.emplace_back("N");
  }
  if (hydrogen_key.size() >= 2) {
    const char region = hydrogen_key[1];
    if (region == 'A')
      candidates.emplace_back("CA");
    else if (region == 'B')
      candidates.emplace_back("CB");
    else if (region == 'G')
      candidates.emplace_back("CG");
    else if (region == 'D')
      candidates.emplace_back("CD");
    else if (region == 'E')
      candidates.emplace_back("CE");
    else if (region == 'Z')
      candidates.emplace_back("CZ");
    else if (std::isdigit(static_cast<unsigned char>(region))) {
      candidates.emplace_back("C" + std::string(1, region));
      candidates.emplace_back("N" + std::string(1, region));
    }
  }
  if (hydrogen_key.find("'") != std::string::npos &&
      hydrogen_key.size() >= 2) {
    candidates.emplace_back("C" + std::string(1, hydrogen_key[1]) + "'");
    candidates.emplace_back("O" + std::string(1, hydrogen_key[1]) + "'");
  }
  candidates.emplace_back("N");
  candidates.emplace_back("CA");
  candidates.emplace_back("C");
  candidates.emplace_back("P");

  for (const auto &candidate : candidates) {
    for (std::size_t i = span.begin; i < span.end; ++i) {
      if (normalize_atom_key(mol.info[i].get_atom()) == candidate)
        return i;
    }
  }
  for (std::size_t i = span.begin; i < span.end; ++i) {
    if (!is_hydrogen_name(normalize_atom_key(mol.info[i].get_atom())))
      return i;
  }
  return span.begin;
}

bool should_report_hydrogens(const arguments &Arguments) {
  return Arguments.fflib_hydrogens == "reference" && !Arguments.flag_xdr;
}

bool should_add_hydrogens(const arguments &Arguments, const Multi_MOL &mol) {
  if (!should_report_hydrogens(Arguments))
    return false;
  if (Arguments.flag_1_pdb &&
      (!Arguments.flag_nocal || Arguments.cmap.flag_cmap) &&
      mol.m_mol.size() <= 1)
    return false;
  return true;
}

std::string hydrogen_skip_policy_message(const arguments &Arguments,
                                         const Multi_MOL &mol) {
  if (Arguments.flag_xdr)
    return "hydrogen not added for XDR trajectory atom-count stability";
  if (Arguments.flag_1_pdb &&
      (!Arguments.flag_nocal || Arguments.cmap.flag_cmap) &&
      mol.m_mol.size() <= 1)
    return "hydrogen not added because PDB trajectory frames keep original atom count";
  return "no heavy atom anchor found";
}

void insert_planned_hydrogens(Multi_MOL &mol,
                              const std::vector<ResidueSpan> &spans,
                              const std::vector<ResiduePlan> &plans,
                              const para &parameter,
                              const arguments &Arguments,
                              TitanExecutor *executor) {
  std::size_t extra_atoms = 0;
  for (const auto &plan : plans) {
    for (const auto &h : plan.hydrogens) {
      if (h.can_place)
        ++extra_atoms;
    }
  }

  if (extra_atoms > 0) {
    std::vector<std::size_t> output_offsets(spans.size() + 1, 0);
    for (std::size_t s = 0; s < spans.size(); ++s) {
      std::size_t span_extra = 0;
      for (const auto &h : plans[s].hydrogens) {
        if (h.can_place)
          ++span_extra;
      }
      output_offsets[s + 1] =
          output_offsets[s] + (spans[s].end - spans[s].begin) + span_extra;
    }

    const std::size_t final_atoms = output_offsets.back();
    std::vector<mol_info> new_info(final_atoms);
    Multi_MOL_parameters new_parameters(final_atoms);
    new_parameters.ensure_type_ids();
    std::vector<AlignedVector3D> new_positions(mol.m_mol.size());
    for (auto &pos : new_positions)
      pos.resize(final_atoms);

    auto rebuild_span = [&](std::size_t worker, std::size_t begin,
                            std::size_t end) {
      (void)worker;
      for (std::size_t s = begin; s < end; ++s) {
        const auto &span = spans[s];
        std::size_t out = output_offsets[s];
        for (std::size_t atom = span.begin; atom < span.end; ++atom, ++out) {
          new_info[out] = mol.info[atom];
          new_parameters[out].set_data(mol.parameters[atom].get_charge(),
                                       mol.parameters[atom].get_radius(),
                                       mol.parameters[atom].get_sigma(),
                                       mol.parameters[atom].get_epsilon(),
                                       mol.parameters[atom].get_mass());
          new_parameters[out].set_type_id(mol.parameters[atom].get_type_id());
          for (std::size_t f = 0; f < mol.m_mol.size(); ++f) {
            if (atom < mol.m_mol[f].pos.x.size()) {
              new_positions[f].x[out] = mol.m_mol[f].pos.x[atom];
              new_positions[f].y[out] = mol.m_mol[f].pos.y[atom];
              new_positions[f].z[out] = mol.m_mol[f].pos.z[atom];
            }
          }
        }
        for (const auto &h : plans[s].hydrogens) {
          if (!h.can_place)
            continue;
          mol_info h_info(span.resnr, h.atom.atom_type, h.atom.atom, span.chain,
                          span.resname);
          h_info.set_side_chain(true);
          new_info[out] = std::move(h_info);
          new_parameters[out].set_data(h.atom.charge, h.atom.radius,
                                       h.atom.sigma, h.atom.epsilon,
                                       h.atom.mass);
          fill_vdw_from_atom_type(new_parameters, out, parameter,
                                  h.atom.atom_type);
          for (std::size_t f = 0; f < mol.m_mol.size(); ++f) {
            const auto [x, y, z] =
                hydrogen_position(mol.m_mol[f], h.anchor, out + f);
            new_positions[f].x[out] = x;
            new_positions[f].y[out] = y;
            new_positions[f].z[out] = z;
          }
          ++out;
        }
      }
    };

    if (executor != nullptr && spans.size() > 1) {
      executor->parallel_for(spans.size(), Arguments.nt, 16, rebuild_span);
    } else {
      rebuild_span(0, 0, spans.size());
    }

    mol.info = std::move(new_info);
    mol.parameters = std::move(new_parameters);
    for (std::size_t f = 0; f < mol.m_mol.size(); ++f) {
      mol.m_mol[f].pos = std::move(new_positions[f]);
      mol.m_mol[f].force_vec.reset();
      mol.m_mol[f].thread_local_forces.clear();
      mol.m_mol[f].Com.clear();
    }
    mol.Index.clear();
    mol.generate_index();
    restore_index_types(mol, spans);
    mol.set_chainnum();
    if (Arguments.flag_sidechain)
      mol.check_side_chain(Arguments);
  }
}

} // namespace TitanFfLib::detail
