#include "nucleic_profiler.h"
#include "BoxMatrix.h"
#include "polymer_classification.h"
#include "titan_executor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <optional>

namespace PC = PolymerClassification;

namespace {

std::pair<size_t, size_t>
residue_atom_range(const std::vector<class index> &indices, size_t chain_pos,
                   size_t res_idx, size_t total_atoms) {
  if (chain_pos >= indices.size())
    return {0, 0};
  const auto &chn = indices[chain_pos];
  if (res_idx >= chn.get_size_v())
    return {0, 0};

  const size_t start_idx = chn.get_res_pos_e(res_idx) - 1;
  size_t end_idx = total_atoms;
  if (res_idx + 1 < chn.get_size_v()) {
    end_idx = chn.get_res_pos_e(res_idx + 1) - 1;
  } else if (chain_pos + 1 < indices.size()) {
    end_idx = indices[chain_pos + 1].get_res_pos_e(0) - 1;
  }
  return {start_idx, std::min(end_idx, total_atoms)};
}

bool atom_has_nucleic_signature(const std::string &atom_name) {
  const std::string atom = PC::normalize_nucleic_atom_name(atom_name);
  if (atom == "BB1" || atom == "BB2" || atom == "BB3" || atom == "SC1" ||
      atom == "SC2" || atom == "SC3" || atom == "SC4" || atom == "SC5")
    return true;
  return atom == "P" || atom == "OP1" || atom == "OP2" || atom == "OP3" ||
         atom == "O1P" || atom == "O2P" || atom == "O3P" ||
         atom == "C1'" || atom == "C2'" || atom == "O2'" || atom == "C3'" ||
         atom == "O3'" || atom == "C4'" || atom == "O4'" || atom == "C5'" ||
         atom == "O5'" || atom == "N9" || atom == "C8" || atom == "N7" ||
         atom == "O6" || atom == "N6" || atom == "N2" || atom == "O2" ||
         atom == "O4" || atom == "N4";
}

bool is_martini_nucleic_bead(const std::string &atom_name) {
  const std::string atom = PC::normalize_nucleic_atom_name(atom_name);
  return atom == "BB1" || atom == "BB2" || atom == "BB3" || atom == "SC1" ||
         atom == "SC2" || atom == "SC3" || atom == "SC4" || atom == "SC5";
}

int wrapped_cell_index(double coordinate, double box_length, double cell_length,
                       int cell_count) {
  if (!(box_length > 0.0) || !(cell_length > 0.0) || cell_count <= 1)
    return 0;
  const double wrapped =
      coordinate - std::floor(coordinate / box_length) * box_length;
  return std::clamp(static_cast<int>(wrapped / cell_length), 0,
                    cell_count - 1);
}

int wrap_grid_index(int index, int cell_count) {
  const int wrapped = index % cell_count;
  return wrapped < 0 ? wrapped + cell_count : wrapped;
}

bool residue_is_protein_candidate(const class index &chn, size_t res_idx) {
  const auto kind = PC::classify_residue_polymer(chn.get_res_name_e(res_idx));
  if (kind == PC::PolymerKind::Protein)
    return true;

  const std::string type = PC::uppercase_copy(chn.get_type());
  return kind == PC::PolymerKind::Other &&
         type.find("PROTEIN") != std::string::npos;
}

bool residue_is_nucleic_candidate(const std::vector<class index> &indices,
                                  size_t chain_pos, size_t res_idx,
                                  const std::vector<mol_info> &info) {
  if (chain_pos >= indices.size())
    return false;
  const auto &chn = indices[chain_pos];
  const auto kind = PC::classify_residue_polymer(chn.get_res_name_e(res_idx));
  if (kind == PC::PolymerKind::Rna || kind == PC::PolymerKind::Dna)
    return true;
  if (kind == PC::PolymerKind::Protein)
    return false;

  const auto [start_idx, end_idx] =
      residue_atom_range(indices, chain_pos, res_idx, info.size());
  if (start_idx >= end_idx || start_idx >= info.size())
    return false;

  const std::string chain_type = PC::uppercase_copy(chn.get_type());
  const bool chain_hints_nucleic =
      PC::is_nucleic_type(chain_type) || chain_type == "COMPLEX";
  for (size_t i = start_idx; i < end_idx && i < info.size(); ++i) {
    if (atom_has_nucleic_signature(info[i].get_atom()))
      return true;
  }
  return chain_hints_nucleic;
}

} // namespace

void NucleicProfiler::reset() {
  interaction_timeline.clear();
  protein_slots_.clear();
  protein_residues_.clear();
  nucleic_residues_.clear();
  global_recognition_profile_.clear();
  protein_atom_buffer_.clear();
  worker_base_hits_.clear();
  worker_backbone_hits_.clear();
  protein_grid_head_.clear();
  protein_grid_next_.clear();
  initialized_ = false;
  coarse_grained_nucleic_ = false;
  protein_grid_active_ = false;
  contact_cutoff_a_ = 3.5;
  total_frames_processed = 0;
}

void NucleicProfiler::initialize(
    const std::vector<class index> &pro_index,
    const std::vector<class index> &na_index,
    const std::vector<mol_info> &pro_info,
    const std::vector<mol_info> &na_info) {
  protein_slots_.clear();
  protein_residues_.clear();
  nucleic_residues_.clear();
  global_recognition_profile_.clear();
  protein_atom_buffer_.clear();
  worker_base_hits_.clear();
  worker_backbone_hits_.clear();
  protein_grid_head_.clear();
  protein_grid_next_.clear();

  coarse_grained_nucleic_ =
      std::any_of(na_info.begin(), na_info.end(), [](const mol_info &atom) {
        return is_martini_nucleic_bead(atom.get_atom());
      });
  contact_cutoff_a_ = coarse_grained_nucleic_ ? 7.0 : 3.5;

  for (size_t chain_pos = 0; chain_pos < pro_index.size(); ++chain_pos) {
    const auto &chn = pro_index[chain_pos];
    for (size_t r = 0; r < chn.get_size_v(); ++r) {
      if (!residue_is_protein_candidate(chn, r))
        continue;
      const auto [start_idx, end_idx] =
          residue_atom_range(pro_index, chain_pos, r, pro_info.size());
      if (start_idx >= end_idx)
        continue;

      const size_t slot_index = protein_slots_.size();
      ProteinResidueSlot slot;
      slot.resnr = static_cast<int>(chn.get_res_num_e(r));
      slot.resname = chn.get_res_name_e(r);
      protein_slots_.push_back(std::move(slot));

      ResidueRangeRef ref;
      ref.chain_pos = chain_pos;
      ref.res_idx = r;
      ref.start_idx = start_idx;
      ref.end_idx = end_idx;
      ref.resnr = static_cast<int>(chn.get_res_num_e(r));
      ref.resname = chn.get_res_name_e(r);
      ref.slot_index = slot_index;
      protein_residues_.push_back(std::move(ref));
    }
  }

  for (size_t chain_pos = 0; chain_pos < na_index.size(); ++chain_pos) {
    const auto &chn = na_index[chain_pos];
    for (size_t r = 0; r < chn.get_size_v(); ++r) {
      if (!residue_is_nucleic_candidate(na_index, chain_pos, r, na_info))
        continue;
      const auto [start_idx, end_idx] =
          residue_atom_range(na_index, chain_pos, r, na_info.size());
      if (start_idx >= end_idx)
        continue;

      ResidueRangeRef ref;
      ref.chain_pos = chain_pos;
      ref.res_idx = r;
      ref.start_idx = start_idx;
      ref.end_idx = end_idx;
      ref.resnr = static_cast<int>(chn.get_res_num_e(r));
      ref.resname = chn.get_res_name_e(r);
      nucleic_residues_.push_back(std::move(ref));
    }
  }

  global_recognition_profile_.assign(protein_slots_.size(), RecognitionCount{});
  protein_atom_buffer_.reserve(pro_info.size());
  initialized_ = true;

  std::cout << "[RNA-Profile] mode="
            << (coarse_grained_nucleic_ ? "martini-cg" : "all-atom")
            << ", cutoff=" << contact_cutoff_a_ << " A"
            << ", protein_residues=" << protein_slots_.size()
            << ", nucleic_residues=" << nucleic_residues_.size() << '\n';
}

void NucleicProfiler::initialize_if_needed(
    const std::vector<class index> &pro_index,
    const std::vector<class index> &na_index,
    const std::vector<mol_info> &pro_info,
    const std::vector<mol_info> &na_info) {
  if (!initialized_) {
    initialize(pro_index, na_index, pro_info, na_info);
  }
}

void NucleicProfiler::ensure_worker_storage(size_t worker_capacity) {
  const size_t slot_count = protein_slots_.size();
  if (worker_base_hits_.size() < worker_capacity) {
    worker_base_hits_.resize(worker_capacity);
    worker_backbone_hits_.resize(worker_capacity);
  }
  for (size_t i = 0; i < worker_capacity; ++i) {
    if (worker_base_hits_[i].size() != slot_count)
      worker_base_hits_[i].assign(slot_count, 0);
    else
      std::fill(worker_base_hits_[i].begin(), worker_base_hits_[i].end(), 0);

    if (worker_backbone_hits_[i].size() != slot_count)
      worker_backbone_hits_[i].assign(slot_count, 0);
    else
      std::fill(worker_backbone_hits_[i].begin(), worker_backbone_hits_[i].end(),
                0);
  }
}

void NucleicProfiler::build_protein_grid(const MOL &pro_frame) {
  protein_grid_active_ = false;
  protein_grid_head_.clear();
  protein_grid_next_.assign(protein_atom_buffer_.size(), -1);

  const box &frame_box = pro_frame.Box;
  constexpr double angle_tolerance = 1.0e-6;
  if (!BoxMatrix::valid_box(frame_box) ||
      std::abs(frame_box.get_alpha() - 90.0) > angle_tolerance ||
      std::abs(frame_box.get_beta() - 90.0) > angle_tolerance ||
      std::abs(frame_box.get_gamma() - 90.0) > angle_tolerance ||
      !(contact_cutoff_a_ > 0.0)) {
    return;
  }

  protein_grid_len_x_ = frame_box.get_a();
  protein_grid_len_y_ = frame_box.get_b();
  protein_grid_len_z_ = frame_box.get_c();
  protein_grid_nx_ =
      std::max(1, static_cast<int>(protein_grid_len_x_ / contact_cutoff_a_));
  protein_grid_ny_ =
      std::max(1, static_cast<int>(protein_grid_len_y_ / contact_cutoff_a_));
  protein_grid_nz_ =
      std::max(1, static_cast<int>(protein_grid_len_z_ / contact_cutoff_a_));

  const std::size_t max_cells =
      std::max<std::size_t>(1024, protein_atom_buffer_.size() * 16);
  auto grid_cell_count = [&]() {
    return static_cast<std::size_t>(protein_grid_nx_) *
           static_cast<std::size_t>(protein_grid_ny_) *
           static_cast<std::size_t>(protein_grid_nz_);
  };
  while (grid_cell_count() > max_cells) {
    if (protein_grid_nx_ >= protein_grid_ny_ &&
        protein_grid_nx_ >= protein_grid_nz_ && protein_grid_nx_ > 1) {
      protein_grid_nx_ = (protein_grid_nx_ + 1) / 2;
    } else if (protein_grid_ny_ >= protein_grid_nz_ &&
               protein_grid_ny_ > 1) {
      protein_grid_ny_ = (protein_grid_ny_ + 1) / 2;
    } else if (protein_grid_nz_ > 1) {
      protein_grid_nz_ = (protein_grid_nz_ + 1) / 2;
    } else {
      break;
    }
  }

  protein_grid_cell_x_ =
      protein_grid_len_x_ / static_cast<double>(protein_grid_nx_);
  protein_grid_cell_y_ =
      protein_grid_len_y_ / static_cast<double>(protein_grid_ny_);
  protein_grid_cell_z_ =
      protein_grid_len_z_ / static_cast<double>(protein_grid_nz_);
  protein_grid_head_.assign(grid_cell_count(), -1);

  for (std::size_t atom_idx = 0; atom_idx < protein_atom_buffer_.size();
       ++atom_idx) {
    const ProteinAtomInfo &atom = protein_atom_buffer_[atom_idx];
    const int ix = wrapped_cell_index(atom.x, protein_grid_len_x_,
                                      protein_grid_cell_x_, protein_grid_nx_);
    const int iy = wrapped_cell_index(atom.y, protein_grid_len_y_,
                                      protein_grid_cell_y_, protein_grid_ny_);
    const int iz = wrapped_cell_index(atom.z, protein_grid_len_z_,
                                      protein_grid_cell_z_, protein_grid_nz_);
    const std::size_t cell =
        (static_cast<std::size_t>(ix) * protein_grid_ny_ + iy) *
            protein_grid_nz_ +
        iz;
    protein_grid_next_[atom_idx] = protein_grid_head_[cell];
    protein_grid_head_[cell] = static_cast<int>(atom_idx);
  }
  protein_grid_active_ = true;
}

std::vector<double> NucleicProfiler::calc_residue_center(
    const MOL &frame, const std::vector<mol_info> &info,
    const class index &chain_idx, size_t res_idx_in_chain,
    const std::string &center_type) {
  std::vector<double> center = {0.0, 0.0, 0.0};
  int count = 0;

  size_t start_idx = (res_idx_in_chain == 0)
                         ? chain_idx.get_res_pos_e(0) - 1
                         : chain_idx.get_res_pos_e(res_idx_in_chain - 1);
  size_t end_idx = chain_idx.get_res_pos_e(res_idx_in_chain) - 1;

  for (size_t i = start_idx; i <= end_idx && i < info.size(); ++i) {
    std::string atom_name = info[i].get_atom();
    bool match = false;

    if (center_type == "AromaticRing") {
      if (atom_name == "CG" || atom_name == "CD1" || atom_name == "CD2" ||
          atom_name == "CE1" || atom_name == "CE2" || atom_name == "CE3" ||
          atom_name == "CZ" || atom_name == "CZ2" || atom_name == "CZ3" ||
          atom_name == "CH2") {
        match = true;
      }
    } else if (center_type == "CationicGroup") {
      if (atom_name == "NH1" || atom_name == "NH2" || atom_name == "NZ" ||
          atom_name == "ND1" || atom_name == "NE2") {
        match = true;
      }
    } else if (center_type == "Phosphate") {
      if (atom_name == "P" || atom_name == "OP1" || atom_name == "OP2" ||
          atom_name == "O1P" || atom_name == "O2P") {
        match = true;
      }
    } else if (center_type == "Nucleobase") {
      if (atom_name == "N1" || atom_name == "C2" || atom_name == "N3" ||
          atom_name == "C4" || atom_name == "C5" || atom_name == "C6" ||
          atom_name == "N7" || atom_name == "C8" || atom_name == "N9") {
        match = true;
      }
    }

    if (match) {
      center[0] += frame.pos.x[i];
      center[1] += frame.pos.y[i];
      center[2] += frame.pos.z[i];
      count++;
    }
  }

  if (count > 0) {
    center[0] /= count;
    center[1] /= count;
    center[2] /= count;
  }
  return center;
}

void NucleicProfiler::process_frame(const MOL &pro_frame, const MOL &na_frame,
                                    const std::vector<mol_info> &pro_info,
                                    const std::vector<mol_info> &na_info,
                                    const std::vector<class index> &pro_index,
                                    const std::vector<class index> &na_index,
                                    size_t num_snap, double time_ps,
                                    TitanExecutor &executor,
                                    size_t requested_workers) {
  (void)num_snap;
  (void)time_ps;
  initialize_if_needed(pro_index, na_index, pro_info, na_info);
  total_frames_processed++;
  if (protein_slots_.empty() || nucleic_residues_.empty())
    return;

  protein_atom_buffer_.clear();
  for (const auto &ref : protein_residues_) {
    const size_t end_idx =
        std::min(ref.end_idx, std::min(pro_info.size(), pro_frame.pos.x.size()));
    for (size_t i = ref.start_idx; i < end_idx; ++i) {
      protein_atom_buffer_.push_back(
          {ref.slot_index, pro_frame.pos.x[i], pro_frame.pos.y[i],
           pro_frame.pos.z[i]});
    }
  }

  if (protein_atom_buffer_.empty())
    return;

  build_protein_grid(pro_frame);
  const box &frame_box = BoxMatrix::valid_box(pro_frame.Box)
                             ? pro_frame.Box
                             : na_frame.Box;
  const std::optional<BoxMatrix> box_matrix =
      BoxMatrix::valid_box(frame_box)
          ? std::optional<BoxMatrix>{frame_box}
          : std::nullopt;
  const double cutoff_sq = contact_cutoff_a_ * contact_cutoff_a_;

  const size_t total_residues = nucleic_residues_.size();
  const size_t requested = std::max<size_t>(1, requested_workers);
  const size_t grain_limited = (total_residues + 7) / 8;
  const size_t active_workers =
      std::max<size_t>(1, std::min(executor.capacity(),
                                   std::min(requested, grain_limited)));
  ensure_worker_storage(active_workers);

  executor.parallel_for(
      total_residues, requested_workers, 8,
      [&](size_t worker_id, size_t begin, size_t end) {
        auto &base_hits = worker_base_hits_[worker_id];
        auto &backbone_hits = worker_backbone_hits_[worker_id];
        for (size_t residue_idx = begin; residue_idx < end; ++residue_idx) {
          const auto &ref = nucleic_residues_[residue_idx];
          const size_t end_idx = std::min(
              ref.end_idx, std::min(na_info.size(), na_frame.pos.x.size()));
          for (size_t i = ref.start_idx; i < end_idx; ++i) {
            const std::string atom_name =
                PC::normalize_nucleic_atom_name(na_info[i].get_atom());
            const bool is_bb = is_backbone(atom_name);
            const bool is_b = is_base(atom_name);
            if (!is_bb && !is_b)
              continue;

            const double na_x = na_frame.pos.x[i];
            const double na_y = na_frame.pos.y[i];
            const double na_z = na_frame.pos.z[i];

            const auto test_protein_atom = [&](std::size_t protein_atom_idx) {
              const ProteinAtomInfo &p_atom =
                  protein_atom_buffer_[protein_atom_idx];
              double dx = na_x - p_atom.x;
              double dy = na_y - p_atom.y;
              double dz = na_z - p_atom.z;
              if (box_matrix)
                box_matrix->apply_mic(dx, dy, dz);
              const double dist_sq = dx * dx + dy * dy + dz * dz;
              if (dist_sq <= cutoff_sq) {
                if (is_bb)
                  backbone_hits[p_atom.slot_index] = 1;
                if (is_b)
                  base_hits[p_atom.slot_index] = 1;
              }
            };

            if (!protein_grid_active_) {
              for (std::size_t protein_atom_idx = 0;
                   protein_atom_idx < protein_atom_buffer_.size();
                   ++protein_atom_idx) {
                test_protein_atom(protein_atom_idx);
              }
              continue;
            }

            const int center_x = wrapped_cell_index(
                na_x, protein_grid_len_x_, protein_grid_cell_x_,
                protein_grid_nx_);
            const int center_y = wrapped_cell_index(
                na_y, protein_grid_len_y_, protein_grid_cell_y_,
                protein_grid_ny_);
            const int center_z = wrapped_cell_index(
                na_z, protein_grid_len_z_, protein_grid_cell_z_,
                protein_grid_nz_);
            std::array<std::size_t, 27> visited_cells{};
            std::size_t visited_count = 0;
            for (int ox = -1; ox <= 1; ++ox) {
              for (int oy = -1; oy <= 1; ++oy) {
                for (int oz = -1; oz <= 1; ++oz) {
                  const int ix = wrap_grid_index(center_x + ox,
                                                 protein_grid_nx_);
                  const int iy = wrap_grid_index(center_y + oy,
                                                 protein_grid_ny_);
                  const int iz = wrap_grid_index(center_z + oz,
                                                 protein_grid_nz_);
                  const std::size_t cell =
                      (static_cast<std::size_t>(ix) * protein_grid_ny_ + iy) *
                          protein_grid_nz_ +
                      iz;
                  if (std::find(visited_cells.begin(),
                                visited_cells.begin() + visited_count,
                                cell) != visited_cells.begin() + visited_count)
                    continue;
                  visited_cells[visited_count++] = cell;
                  for (int protein_atom_idx = protein_grid_head_[cell];
                       protein_atom_idx != -1;
                       protein_atom_idx = protein_grid_next_[
                           static_cast<std::size_t>(protein_atom_idx)]) {
                    test_protein_atom(
                        static_cast<std::size_t>(protein_atom_idx));
                  }
                }
              }
            }
          }
        }
      });

  for (size_t slot_idx = 0; slot_idx < protein_slots_.size(); ++slot_idx) {
    bool base_hit = false;
    bool backbone_hit = false;
    for (size_t worker_id = 0; worker_id < active_workers; ++worker_id) {
      base_hit = base_hit || (worker_base_hits_[worker_id][slot_idx] != 0);
      backbone_hit =
          backbone_hit || (worker_backbone_hits_[worker_id][slot_idx] != 0);
      if (base_hit && backbone_hit)
        break;
    }
    if (base_hit)
      global_recognition_profile_[slot_idx].base_contacts++;
    if (backbone_hit)
      global_recognition_profile_[slot_idx].backbone_contacts++;
  }
}

void NucleicProfiler::export_results(const std::string &csv_filename) {
  if (total_frames_processed == 0)
    return;

  std::ofstream out(csv_filename);
  if (!out)
    return;

  out << "Protein_ResNum,ResName,Base_Contact_Frames,Backbone_Contact_Frames,"
         "Base_Occupancy_%,Backbone_Occupancy_%,Recognition_Type\n";

  std::size_t rows_written = 0;
  for (size_t slot_idx = 0; slot_idx < protein_slots_.size(); ++slot_idx) {
    const auto &slot = protein_slots_[slot_idx];
    const auto &counts = global_recognition_profile_[slot_idx];
    const int base_contacts = counts.base_contacts;
    const int backbone_contacts = counts.backbone_contacts;

    double base_occ =
        (static_cast<double>(base_contacts) / total_frames_processed) * 100.0;
    double back_occ =
        (static_cast<double>(backbone_contacts) / total_frames_processed) *
        100.0;

    if (base_occ > 5.0 || back_occ > 5.0) {
      std::string type = "Mixed_Binding";
      if (base_contacts > backbone_contacts * 1.5) {
        type = "Highly_Specific";
      } else if (backbone_contacts > base_contacts * 1.5) {
        type = "Non_Specific_Backbone";
      }

      out << slot.resnr << "," << slot.resname << "," << base_contacts << ","
          << backbone_contacts << "," << base_occ << "," << back_occ << ","
          << type << "\n";
      ++rows_written;
    }
  }
  if (rows_written == 0) {
    std::cerr << "[WARNING] RNA-Profile: no protein residue exceeded the 5% "
                 "contact-occupancy threshold (mode="
              << (coarse_grained_nucleic_ ? "martini-cg" : "all-atom")
              << ", cutoff=" << contact_cutoff_a_
              << " A, frames=" << total_frames_processed << ").\n";
  }
}
