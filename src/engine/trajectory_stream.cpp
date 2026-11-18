#include "engine/engine_run.h"

namespace engine_detail {

std::optional<int> EngineTrajectory::stream_chunks() {
  while (is_streaming) {
    start_input_stream = std::chrono::steady_clock::now();
    // With prefetch active this measures the reader stall, not the decode.
    expected_frames = take_next_chunk();
    time_input_stream += std::chrono::duration_cast<milliseconds>(
        std::chrono::steady_clock::now() - start_input_stream);
    if (!expected_frames) {
      TitanUI::print_error(expected_frames.error());
      return 1;
    }

    chunk_num_snaps = expected_frames.value();
    if (chunk_num_snaps == 0) {
      is_streaming = false;
      break;
    }

    NUM = chunk_num_snaps;

    // Decode chunk N+1 while chunk N is analysed below.
    start_chunk_prefetch();

    // --- PBC & WATER SHIFT LOGIC (Applied Per-Chunk) ---
    old_cogs = std::vector<shift_vector>(pro_in.m_mol.size(), {0.0, 0.0, 0.0});
    if (Arguments.n_wat >= 0) {
      for (size_t snap = 0; snap < pro_in.m_mol.size(); ++snap) {
        double old_cx = 0.0, old_cy = 0.0, old_cz = 0.0;
        size_t p_size = pro_in.m_mol[snap].pos.x.size();
        if (p_size > 0) {
          for (size_t i = 0; i < p_size; ++i) {
            old_cx += pro_in.m_mol[snap].pos.x[i];
            old_cy += pro_in.m_mol[snap].pos.y[i];
            old_cz += pro_in.m_mol[snap].pos.z[i];
          }
          old_cx /= p_size;
          old_cy /= p_size;
          old_cz /= p_size;
        }
        old_cogs[snap] = {old_cx, old_cy, old_cz};
      }
    }

    start_pbc = std::chrono::steady_clock::now();
    have_mol2_frames =
        !pro_in.m_mol.empty() && lig.m_mol.size() >= pro_in.m_mol.size() &&
        std::all_of(pro_in.m_mol.begin(), pro_in.m_mol.end(),
                    [&](const MOL &frame) {
                      const size_t index =
                          static_cast<size_t>(&frame - pro_in.m_mol.data());
                      return lig.m_mol[index].size() != 0;
                    });
    if (Arguments.pbc.solv_pbctopo) {
      pro_in.solve_pbc_pbctopo(
          lig, Arguments, &executor, tot_snap_allfiles + global_frame_counter);
    } else if (Arguments.pbc.solv_pbcn) {
      pro_in.solv_pbc_pbcn(lig, Arguments, &executor);
      align_loaded_complex();
    } else if (Arguments.pbc.solv_pbc_int == 1) {
      if (Arguments.pbc.chain_ref == -2) {
        executor.parallel_do(num_threads, [&](size_t worker_id) {
          pro_in.solve_pbc_int(worker_id, num_threads);
        });
      }
      if (Arguments.pbc.chain_ref > -2) {
        pro_in.solve_pbc_new(lig, Arguments, amine);
        align_loaded_complex();
      }
    }
    if (Arguments.pbc.solv_pbc == 1) {
      executor.parallel_do(num_threads, [&](size_t worker_id) {
        pro_in.pbc_prob(lig, worker_id, num_threads);
      });
      align_loaded_complex();
    }
    time_solve_pbc += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_pbc);

    // ── NOCAL MODE: PBC only → write PDB/PQR → skip energy ──────────
    if (Arguments.flag_nocal) {
      // Keep a loaded two-molecule system together. Single-molecule nocal
      // workflows still center mol1 independently.
      // STR-28 (22/10/2026): -msd needs the molecule's own translation;
      // re-centring it in the box every frame made a lone molecule's MSD 0.
      // The MSD module unwraps frame to frame with minimum image, so the
      // whole-but-uncentred coordinates are what it needs.
      // STR-30 (22/10/2026): the same holds for -membrane: the lipid frames
      // are not moved with mol1, and a non-lattice shift of mol1 cannot be
      // undone by the minimum image of the protein-lipid contacts.
      if (Arguments.pbc.solv_pbcn || Arguments.pbc.solv_pbctopo ||
          Arguments.pbc.solv_pbc_int)
        if (!have_mol2_frames && !Arguments.msd.flag_msd &&
            !membrane_analyzer.enabled())
          pro_in.align_to_box();

      // PBC clustering (non-int mode)
      if (Arguments.pbc.solv_pbc_clust && !Arguments.pbc.solv_pbc_int)
        pro_in.solve_pbc_cluster(Arguments, info_output);

      TitanUI::log_kv(
          log, "Nocal Chunk",
          "Writing frames " + std::to_string(global_frame_counter + 1) +
              " to " +
              std::to_string(global_frame_counter + chunk_num_snaps));

      for (size_t chunk_snap_idx = 0; chunk_snap_idx < chunk_num_snaps;
           ++chunk_snap_idx) {
             if (const std::optional<int> exit_code = process_nocal_frame(chunk_snap_idx))
               return exit_code;
      }

      global_frame_counter += chunk_num_snaps;
      continue; // skip energy calculations, read next chunk
    }
    // ── END NOCAL ───────────────────────────────────────────────────────

    if (Arguments.entropy_quasi) {
      if (Arguments.pbc.solv_pbc == 0 && Arguments.pbc.chain_ref == -1)
        align_complex(pro_in, lig, 0, 0);
      cal_entropy(pro_in, lig, Groups_from_index_file, Arguments, paths,
                  static_cast<size_t>(input_name), info_output, &executor);
    }

    if (Arguments.n_wat >= 0 && !water_traj.m_mol.empty()) {
      size_t limit = std::min(pro_in.m_mol.size(), water_traj.m_mol.size());
      for (size_t snap = 0; snap < limit; ++snap) {
        size_t p_size = pro_in.m_mol[snap].pos.x.size();
        if (p_size > 0) {
          double new_cx = 0.0, new_cy = 0.0, new_cz = 0.0;
          for (size_t i = 0; i < p_size; ++i) {
            new_cx += pro_in.m_mol[snap].pos.x[i];
            new_cy += pro_in.m_mol[snap].pos.y[i];
            new_cz += pro_in.m_mol[snap].pos.z[i];
          }
          new_cx /= p_size;
          new_cy /= p_size;
          new_cz /= p_size;
          double shift_x = new_cx - old_cogs[snap].x;
          double shift_y = new_cy - old_cogs[snap].y;
          double shift_z = new_cz - old_cogs[snap].z;
          size_t w_size = water_traj.m_mol[snap].pos.x.size();
          for (size_t i = 0; i < w_size; ++i) {
            water_traj.m_mol[snap].pos.x[i] += shift_x;
            water_traj.m_mol[snap].pos.y[i] += shift_y;
            water_traj.m_mol[snap].pos.z[i] += shift_z;
          }
        }
      }
    }

    TitanUI::log_kv(
        log, "Processing Chunk",
        "Frames " + std::to_string(global_frame_counter + 1) + " to " +
            std::to_string(global_frame_counter + chunk_num_snaps));

    // --- INNER FRAME LOOP ---
    for (size_t chunk_snap_idx = 0; chunk_snap_idx < chunk_num_snaps;
         ++chunk_snap_idx) {
           if (const std::optional<int> exit_code = process_frame(chunk_snap_idx))
             return exit_code;
    } // End of inner chunk loop

    // We update the sizes for the next iteration
    // INITIATE THREAD-LOCAL FORCES FOR OMP LOOP
    for (size_t f = 0; f < pro_in.m_mol.size(); ++f) {
      if (pro_in.compute_forces)
        pro_in.m_mol[f].initialize_forces(num_threads);
    }
    for (size_t f = 0; f < lig.m_mol.size(); ++f) {
      if (lig.compute_forces)
        lig.m_mol[f].initialize_forces(num_threads);
    }

    global_frame_counter += chunk_num_snaps;

  } // End of outer chunk while loop
  return std::nullopt;
}

void EngineTrajectory::start_chunk_prefetch() {
  if (!chunk_prefetch_enabled || chunk_prefetch_worker)
    return;
  chunk_prefetch_worker.emplace([&]() {
    prefetched_chunk = (*streamer).read_next_chunk(
        CHUNK_SIZE, pro_next, lig_next, water_next, membrane_next);
  });
}

std::expected<size_t, std::string> EngineTrajectory::take_next_chunk() {
  if (!chunk_prefetch_worker)
    return (*streamer).read_next_chunk(CHUNK_SIZE, pro_in, lig, water_traj,
                                    membrane_traj);
  chunk_prefetch_worker->join();
  chunk_prefetch_worker.reset();
  auto result = std::move(*prefetched_chunk);
  prefetched_chunk.reset();
  // read_next_chunk() clears its destination frame vectors on entry, so the
  // back buffers hold exactly what a synchronous read would have produced.
  std::swap(pro_in.m_mol, pro_next.m_mol);
  std::swap(lig.m_mol, lig_next.m_mol);
  std::swap(water_traj.m_mol, water_next.m_mol);
  std::swap(membrane_traj.m_mol, membrane_next.m_mol);
  return result;
}

void EngineTrajectory::align_loaded_complex() {
  if (have_mol2_frames)
    align_complex(pro_in, lig, 0, 0);
}

} // namespace engine_detail
