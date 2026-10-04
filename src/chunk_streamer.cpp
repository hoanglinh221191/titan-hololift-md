#include "chunk_streamer.h"
#include "titan_executor.h"
#include "titan_ui.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <immintrin.h> // AVX2 intrinsics
#include <iostream>
#include <thread>
#include <vector>

// XDR headers included HERE (not in .h) to avoid `matrix` typedef clash with
// Eigen
extern "C" {
#include "xdrfile.h"
#include "xdrfile_trr.h"
#include "xdrfile_xtc.h"
}

namespace {

void initialize_stream_frame(MOL &frame, const Multi_MOL &source_topology,
                             const box &frame_box, double time_ps,
                             size_t atom_count) {
  frame.force_f = source_topology.force_f;
  frame.mol_type = std::string(source_topology.get_mol_type());
  frame.Box = frame_box;
  frame.time_of_snap = time_ps;
  frame.flag_set_time = true;
  frame.compute_forces = false;
  frame.pos.resize(atom_count);
}

void initialize_stream_water_frame(MOL &frame, std::string_view force_field,
                                   const box &frame_box, double time_ps,
                                   size_t atom_count) {
  frame.force_f = std::string(force_field);
  frame.mol_type = "water";
  frame.Box = frame_box;
  frame.time_of_snap = time_ps;
  frame.flag_set_time = true;
  frame.compute_forces = false;
  frame.pos.resize(atom_count);
}

struct XdrFrameProbe {
  int64_t frame_start = -1;
  double time_ps = 0.0;
  int step = 0;
  int natoms = 0;
  bool is_trr = false;
  t_trnheader trr_header{};
};

constexpr double kXdrTimeTolerancePs = 1e-4;
constexpr double kOrthorhombicBoxToleranceNm = 1e-7;

std::string xdr_status_message(int rc) {
  if (rc >= 0 && rc < exdrNR && exdr_message[rc] != nullptr) {
    return std::string(exdr_message[rc]);
  }
  return "unknown XDR status";
}

template <typename T>
box build_frame_box_from_xdr_matrix(const T (&box_nm)[DIM][DIM]) {
  constexpr double kNmToAng = 10.0;
  constexpr double kRadToDeg = 180.0 / 3.14159265358979;

  const bool orthorhombic =
      std::fabs(static_cast<double>(box_nm[0][1])) <=
          kOrthorhombicBoxToleranceNm &&
      std::fabs(static_cast<double>(box_nm[0][2])) <=
          kOrthorhombicBoxToleranceNm &&
      std::fabs(static_cast<double>(box_nm[1][0])) <=
          kOrthorhombicBoxToleranceNm &&
      std::fabs(static_cast<double>(box_nm[1][2])) <=
          kOrthorhombicBoxToleranceNm &&
      std::fabs(static_cast<double>(box_nm[2][0])) <=
          kOrthorhombicBoxToleranceNm &&
      std::fabs(static_cast<double>(box_nm[2][1])) <=
          kOrthorhombicBoxToleranceNm;

  if (orthorhombic) {
    return box(std::abs(static_cast<double>(box_nm[0][0])) * kNmToAng,
               std::abs(static_cast<double>(box_nm[1][1])) * kNmToAng,
               std::abs(static_cast<double>(box_nm[2][2])) * kNmToAng, 90.0,
               90.0, 90.0);
  }

  const double v1x = static_cast<double>(box_nm[0][0]) * kNmToAng;
  const double v1y = static_cast<double>(box_nm[0][1]) * kNmToAng;
  const double v1z = static_cast<double>(box_nm[0][2]) * kNmToAng;
  const double v2x = static_cast<double>(box_nm[1][0]) * kNmToAng;
  const double v2y = static_cast<double>(box_nm[1][1]) * kNmToAng;
  const double v2z = static_cast<double>(box_nm[1][2]) * kNmToAng;
  const double v3x = static_cast<double>(box_nm[2][0]) * kNmToAng;
  const double v3y = static_cast<double>(box_nm[2][1]) * kNmToAng;
  const double v3z = static_cast<double>(box_nm[2][2]) * kNmToAng;

  const double la = std::sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
  const double lb = std::sqrt(v2x * v2x + v2y * v2y + v2z * v2z);
  const double lc = std::sqrt(v3x * v3x + v3y * v3y + v3z * v3z);

  double cos_alpha = (lb > 0.0 && lc > 0.0)
                         ? (v2x * v3x + v2y * v3y + v2z * v3z) / (lb * lc)
                         : 0.0;
  double cos_beta = (la > 0.0 && lc > 0.0)
                        ? (v1x * v3x + v1y * v3y + v1z * v3z) / (la * lc)
                        : 0.0;
  double cos_gamma = (la > 0.0 && lb > 0.0)
                         ? (v1x * v2x + v1y * v2y + v1z * v2z) / (la * lb)
                         : 0.0;

  cos_alpha = std::clamp(cos_alpha, -1.0, 1.0);
  cos_beta = std::clamp(cos_beta, -1.0, 1.0);
  cos_gamma = std::clamp(cos_gamma, -1.0, 1.0);

  box frame_box;
  frame_box.set_a(la);
  frame_box.set_b(lb);
  frame_box.set_c(lc);
  frame_box.set_alpha(std::acos(cos_alpha) * kRadToDeg);
  frame_box.set_beta(std::acos(cos_beta) * kRadToDeg);
  frame_box.set_gamma(std::acos(cos_gamma) * kRadToDeg);
  return frame_box;
}

int read_xtc_frame_payload_after_header(XDRFILE *xdr_handle, int natoms,
                                        float (&box_out)[DIM][DIM],
                                        rvec *pos_out, float *precision,
                                        std::string &error) {
  error.clear();

  if (xdrfile_read_float(box_out[0], DIM * DIM, xdr_handle) != DIM * DIM) {
    error = "XDR: failed to read XTC box block.";
    return exdrFLOAT;
  }

  int coord_count = natoms;
  const int rc = xdrfile_decompress_coord_float(
      reinterpret_cast<float *>(pos_out), &coord_count, precision, xdr_handle);
  if (rc != natoms || coord_count != natoms) {
    error = "XDR: failed to decode XTC coordinates after header (decoder_rc=" +
            std::to_string(rc) + ", coord_count=" +
            std::to_string(coord_count) + ", expected=" +
            std::to_string(natoms) + ", detail=" +
            xdrfile_last_coord_error() + ").";
    return exdr3DX;
  }

  return exdrOK;
}

bool trr_selective_fast_path_supported(const t_trnheader &header,
                                       std::string &reason) {
  reason.clear();

  const int coord_elem_bytes = header.bDouble ? sizeof(double) : sizeof(float);
  const int expected_coord_bytes = header.natoms * DIM * coord_elem_bytes;
  const int expected_box_bytes = DIM * DIM * coord_elem_bytes;

  if (header.ir_size != 0 || header.e_size != 0 || header.top_size != 0 ||
      header.sym_size != 0) {
    reason = "TRR header carries unsupported legacy/topology payload blocks.";
    return false;
  }
  if (header.x_size == 0) {
    reason = "TRR frame does not contain coordinates.";
    return false;
  }
  if (header.x_size != expected_coord_bytes) {
    reason = "TRR coordinate block size does not match natoms.";
    return false;
  }
  if (header.box_size != 0 && header.box_size != expected_box_bytes) {
    reason = "TRR box block size is not a standard 3x3 matrix.";
    return false;
  }
  if (header.vir_size != 0 && header.vir_size != expected_box_bytes) {
    reason = "TRR virial block has unexpected size.";
    return false;
  }
  if (header.pres_size != 0 && header.pres_size != expected_box_bytes) {
    reason = "TRR pressure block has unexpected size.";
    return false;
  }

  return true;
}

int read_trr_box_and_skip_prefix_after_header(XDRFILE *xdr_handle,
                                              const t_trnheader &header,
                                              float (&box_out)[DIM][DIM],
                                              std::string &error) {
  error.clear();

  for (int i = 0; i < DIM; ++i) {
    for (int j = 0; j < DIM; ++j) {
      box_out[i][j] = 0.0f;
    }
  }

  if (header.box_size != 0) {
    if (header.bDouble) {
      double box_tmp[DIM * DIM]{};
      if (xdrfile_read_double(box_tmp, DIM * DIM, xdr_handle) != DIM * DIM) {
        error = "XDR: failed to read TRR box block.";
        return exdrDOUBLE;
      }
      for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
          box_out[i][j] = static_cast<float>(box_tmp[i * DIM + j]);
        }
      }
    } else if (xdrfile_read_float(box_out[0], DIM * DIM, xdr_handle) !=
               DIM * DIM) {
      error = "XDR: failed to read TRR box block.";
      return exdrFLOAT;
    }
  }

  if (header.vir_size != 0) {
    const int rc = xdr_seek(xdr_handle, header.vir_size, SEEK_CUR);
    if (rc != exdrOK) {
      error = "XDR: failed to skip TRR virial block (rc=" +
              std::to_string(rc) + ", " + xdr_status_message(rc) + ")";
      return rc;
    }
  }

  if (header.pres_size != 0) {
    const int rc = xdr_seek(xdr_handle, header.pres_size, SEEK_CUR);
    if (rc != exdrOK) {
      error = "XDR: failed to skip TRR pressure block (rc=" +
              std::to_string(rc) + ", " + xdr_status_message(rc) + ")";
      return rc;
    }
  }

  return exdrOK;
}

int probe_xdr_frame(XDRFILE *xdr_handle, const std::string &xdr_type,
                    XdrFrameProbe &probe, std::string &error) {
  probe = XdrFrameProbe{};
  error.clear();

  const int64_t frame_start = xdr_tell(xdr_handle);
  if (frame_start < 0) {
    error = "XDR: failed to query current frame offset.";
    return exdrHEADER;
  }
  probe.frame_start = frame_start;

  if (xdr_type == "xtc") {
    int natoms = 0;
    int step = 0;
    float time_ps = 0.0f;
    const int rc = xtc_header(xdr_handle, &natoms, &step, &time_ps, 1);
    if (rc != exdrOK) {
      error = "XDR: failed to read XTC frame header (rc=" + std::to_string(rc) +
              ", " + xdr_status_message(rc) + ")";
      return rc;
    }
    probe.time_ps = static_cast<double>(time_ps);
    probe.step = step;
    probe.natoms = natoms;
    return exdrOK;
  }

  t_trnheader trr_header{};
  const int rc = do_trnheader(xdr_handle, 1, &trr_header);
  if (rc != exdrOK) {
    error = "XDR: failed to read TRR frame header (rc=" + std::to_string(rc) +
            ", " + xdr_status_message(rc) + ")";
    return rc;
  }
  probe.time_ps = static_cast<double>(trr_header.td);
  probe.step = trr_header.step;
  probe.natoms = trr_header.natoms;
  probe.is_trr = true;
  probe.trr_header = trr_header;
  return exdrOK;
}

int read_xdr_opaque_discard(XDRFILE *xdr_handle, int64_t byte_count,
                            std::string &error) {
  if (byte_count < 0) {
    error = "XDR: invalid negative opaque byte count while skipping.";
    return exdrINT;
  }

  static thread_local std::vector<char> scratch;
  constexpr int64_t kChunkBytes = 1024LL * 1024LL;
  scratch.resize(static_cast<size_t>(kChunkBytes));

  int64_t remaining = byte_count;
  while (remaining > 0) {
    const int chunk =
        static_cast<int>(std::min<int64_t>(remaining, kChunkBytes));
    if (xdrfile_read_opaque(scratch.data(), chunk, xdr_handle) != chunk) {
      error = "XDR: failed to read compressed XTC payload while skipping.";
      return exdr3DX;
    }
    remaining -= chunk;
  }
  return exdrOK;
}

int skip_xtc_frame_payload_after_header(XDRFILE *xdr_handle,
                                        int expected_natoms,
                                        std::string &error) {
  error.clear();

  float dummy_box[DIM * DIM]{};
  if (xdrfile_read_float(dummy_box, DIM * DIM, xdr_handle) != DIM * DIM) {
    error = "XDR: failed to skip XTC box block.";
    return exdrFLOAT;
  }

  int coord_count = 0;
  if (xdrfile_read_int(&coord_count, 1, xdr_handle) != 1) {
    error = "XDR: failed to read XTC coordinate count while skipping.";
    return exdrINT;
  }
  if (coord_count < 0) {
    error = "XDR: invalid negative XTC coordinate count while skipping.";
    return exdrINT;
  }
  if (coord_count != expected_natoms) {
    error = "XDR: XTC coordinate count changed while skipping (expected " +
            std::to_string(expected_natoms) + ", got " +
            std::to_string(coord_count) + ").";
    return exdr3DX;
  }

  if (coord_count <= 9) {
    float raw_coords[9 * DIM]{};
    const int raw_value_count = coord_count * DIM;
    if (xdrfile_read_float(raw_coords, raw_value_count, xdr_handle) !=
        raw_value_count) {
      error = "XDR: failed to read uncompressed XTC coordinates while skipping.";
      return exdrFLOAT;
    }
    return exdrOK;
  }

  float precision = 0.0f;
  int minint[DIM]{};
  int maxint[DIM]{};
  int smallidx = 0;
  int64_t opaque_bytes = 0;

  if (xdrfile_read_float(&precision, 1, xdr_handle) != 1) {
    error = "XDR: failed to read XTC precision while skipping.";
    return exdrFLOAT;
  }
  if (xdrfile_read_int(minint, DIM, xdr_handle) != DIM ||
      xdrfile_read_int(maxint, DIM, xdr_handle) != DIM ||
      xdrfile_read_int(&smallidx, 1, xdr_handle) != 1) {
    error = "XDR: failed to read XTC compressed block header while skipping.";
    return exdrINT;
  }
  if (xdrfile_get_xtc_magic(xdr_handle) == XTC_NEW_MAGIC) {
    if (xdrfile_read_int64(&opaque_bytes, 1, xdr_handle) != 1) {
      error =
          "XDR: failed to read 64-bit XTC opaque byte count while skipping.";
      return exdrINT;
    }
  } else {
    int legacy_opaque_bytes = 0;
    if (xdrfile_read_int(&legacy_opaque_bytes, 1, xdr_handle) != 1) {
      error = "XDR: failed to read XTC opaque byte count while skipping.";
      return exdrINT;
    }
    opaque_bytes = legacy_opaque_bytes;
  }
  if (opaque_bytes < 0) {
    error = "XDR: invalid negative XTC opaque byte count while skipping.";
    return exdrINT;
  }

  const int64_t raw_coord_bytes =
      static_cast<int64_t>(expected_natoms) * DIM * sizeof(float);
  const int64_t max_reasonable_opaque_bytes =
      std::max<int64_t>(128LL * 1024LL * 1024LL, raw_coord_bytes * 8LL);
  if (opaque_bytes > max_reasonable_opaque_bytes) {
    error = "XDR: unreasonable XTC opaque byte count while skipping (" +
            std::to_string(opaque_bytes) + " bytes).";
    return exdr3DX;
  }

  return read_xdr_opaque_discard(xdr_handle, opaque_bytes, error);
}

int skip_trr_frame_payload_after_header(XDRFILE *xdr_handle,
                                        const t_trnheader &trr_header,
                                        std::string &error) {
  error.clear();

  const int64_t payload_bytes =
      static_cast<int64_t>(trr_header.ir_size) +
      static_cast<int64_t>(trr_header.e_size) +
      static_cast<int64_t>(trr_header.box_size) +
      static_cast<int64_t>(trr_header.vir_size) +
      static_cast<int64_t>(trr_header.pres_size) +
      static_cast<int64_t>(trr_header.top_size) +
      static_cast<int64_t>(trr_header.sym_size) +
      static_cast<int64_t>(trr_header.x_size) +
      static_cast<int64_t>(trr_header.v_size) +
      static_cast<int64_t>(trr_header.f_size);

  if (payload_bytes < 0) {
    error = "XDR: invalid negative TRR payload size while skipping.";
    return exdrHEADER;
  }

  const int rc = xdr_seek(xdr_handle, payload_bytes, SEEK_CUR);
  if (rc != exdrOK) {
    error = "XDR: failed to seek across TRR payload (rc=" +
            std::to_string(rc) + ", " + xdr_status_message(rc) + ")";
  }
  return rc;
}

int skip_xdr_frame_payload_after_probe(XDRFILE *xdr_handle,
                                       const std::string &xdr_type,
                                       const XdrFrameProbe &probe,
                                       std::string &error) {
  if (xdr_type == "xtc") {
    return skip_xtc_frame_payload_after_header(xdr_handle, probe.natoms, error);
  }
  return skip_trr_frame_payload_after_header(xdr_handle, probe.trr_header,
                                             error);
}

} // namespace

ChunkStreamer::ChunkStreamer(arguments &args, info &output_info,
                             groups_cal &groups, std::string_view force_field,
                             TitanExecutor &executor, size_t num_threads)
    : args_ref(args), info_ref(output_info), groups_ref(groups),
      force_f(force_field), executor_(&executor), nt(num_threads) {}

ChunkStreamer::~ChunkStreamer() {
  // XDR cleanup
  if (xdr_handle_) {
    xdrfile_close(xdr_handle_);
    xdr_handle_ = nullptr;
  }
  if (xdr_pos_buf_) {
    delete[] xdr_pos_buf_;
    xdr_pos_buf_ = nullptr;
  }
  // PDB cleanup
  if (stream_pro.is_open())
    stream_pro.close();
  if (stream_lig.is_open())
    stream_lig.close();
}

std::expected<void, std::string> ChunkStreamer::open_streams(
    const std::string &name_pro, const std::string &name_lig,
    const Multi_MOL &init_pro, const Multi_MOL &init_lig,
    const Multi_MOL &init_membrane) {

  // Store topology references for later use in parallel_split_coordinates
  init_pro_ = &init_pro;
  init_lig_ = &init_lig;
  init_membrane_ = &init_membrane;
  pdb_route_plan_ready_ = false;
  pdb_warned_direct_route_fallback_ = false;
  pdb_next_target_time_ = -1e18;
  pro_total_bytes_ = 0;
  lig_total_bytes_ = 0;
  pro_data_offset_bytes_ = 0;
  lig_data_offset_bytes_ = 0;

  auto file_size_bytes = [](const std::string &path) -> size_t {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    return ec ? 0 : static_cast<size_t>(size);
  };
  auto stream_offset = [](std::ifstream &stream, size_t fallback_total) -> size_t {
    const std::streampos pos = stream.tellg();
    if (pos == std::streampos(-1))
      return fallback_total;
    const auto as_off = static_cast<std::streamoff>(pos);
    return as_off < 0 ? 0 : static_cast<size_t>(as_off);
  };

  // ── XDR PATH (XTC/TRR) ──────────────────────────────────────────────
  if (args_ref.flag_xdr) {
    namespace fs = std::filesystem;
    const fs::path traj_path = fs::absolute(fs::path(args_ref.traj_file));
    const std::string tf = traj_path.string();
    std::error_code ec;
    const bool use_lig_group = args_ref.needs_mol2_coordinates();

    if (!fs::exists(traj_path, ec) || ec) {
      return std::unexpected("XDR: trajectory file not found: " + tf);
    }

    const auto traj_size = fs::file_size(traj_path, ec);
    if (ec) {
      return std::unexpected("XDR: cannot stat trajectory file: " + tf);
    }
    if (traj_size == 0) {
      return std::unexpected("XDR: trajectory file is empty: " + tf);
    }

    // 1. Get natoms from XDR file header
    int natoms_check = 0;
    int rc;
    if (args_ref.xdr_type == "xtc")
      rc = read_xtc_natoms(const_cast<char *>(tf.c_str()), &natoms_check);
    else
      rc = read_trr_natoms(const_cast<char *>(tf.c_str()), &natoms_check);

    if (rc != exdrOK)
      return std::unexpected("XDR: Cannot read natoms from " + tf +
                             " (size=" + std::to_string(traj_size) +
                             " bytes, rc=" + std::to_string(rc) + ")");

    // 2. Validate natoms: XDR must have >= atoms referenced by index groups
    size_t topology_natoms =
        groups_ref.groups_atom_number(groups_ref.group_1_index());
    if (use_lig_group)
      topology_natoms += groups_ref.groups_atom_number(groups_ref.group_2_index());
    if (args_ref.n_wat >= 0)
      topology_natoms += groups_ref.groups_atom_number(args_ref.n_wat);

    if (natoms_check < static_cast<int>(topology_natoms))
      return std::unexpected("XDR: natoms too small — XDR has " +
                             std::to_string(natoms_check) +
                             " but index groups reference " +
                             std::to_string(topology_natoms) + " atoms");
    if (natoms_check != static_cast<int>(topology_natoms)) {
      TitanUI::print_info("XDR will stream " + std::to_string(natoms_check) +
                          " atoms/frame and subset " +
                          std::to_string(topology_natoms) +
                          " selected atoms on-the-fly.");
    }
    xdr_natoms_ = natoms_check;

    auto cache_group_indices = [&](size_t group_index,
                                   std::vector<size_t> &out_indices,
                                   size_t &out_size)
        -> std::expected<void, std::string> {
      out_size = groups_ref.groups_atom_number(group_index);
      out_indices.clear();
      out_indices.reserve(out_size);
      for (size_t i = 0; i < out_size; ++i) {
        const size_t atom_index_1based =
            groups_ref.atom_index(group_index, i);
        if (atom_index_1based == 0) {
          return std::unexpected("XDR: invalid atom index in selected group.");
        }
        out_indices.push_back(atom_index_1based - 1);
      }
      return {};
    };

    auto pro_cache =
        cache_group_indices(groups_ref.group_1_index(), pro_atom_indices0_,
                            pro_size_);
    if (!pro_cache)
      return std::unexpected(pro_cache.error());

    if (use_lig_group) {
      auto lig_cache =
          cache_group_indices(groups_ref.group_2_index(), lig_atom_indices0_,
                              lig_size_);
      if (!lig_cache)
        return std::unexpected(lig_cache.error());
    } else {
      lig_atom_indices0_.clear();
      lig_size_ = 0;
    }

    if (args_ref.n_wat >= 0) {
      auto wat_cache =
          cache_group_indices(args_ref.n_wat, wat_atom_indices0_, wat_size_);
      if (!wat_cache)
        return std::unexpected(wat_cache.error());
    } else {
      wat_atom_indices0_.clear();
      wat_size_ = 0;
    }

    if (args_ref.needs_membrane_coordinates()) {
      if (args_ref.membrane.resolved_group < 0)
        return std::unexpected("XDR: membrane group has not been resolved.");
      auto membrane_cache = cache_group_indices(
          static_cast<size_t>(args_ref.membrane.resolved_group),
          membrane_atom_indices0_, membrane_size_);
      if (!membrane_cache)
        return std::unexpected(membrane_cache.error());
    } else {
      membrane_atom_indices0_.clear();
      membrane_size_ = 0;
    }

    xdr_extract_runs_.clear();
    xdr_max_run_length_ = 0;
    xdr_has_overlapping_runs_ = false;
    xdr_warned_xtc_fallback_ = false;
    xdr_warned_trr_fallback_ = false;

    auto append_runs = [&](const std::vector<size_t> &indices,
                           ChunkStreamer::XdrRunTarget target) {
      if (indices.empty()) {
        return;
      }

      size_t run_source_begin = indices.front();
      size_t run_output_begin = 0;
      size_t run_length = 1;

      auto flush_run = [&]() {
        xdr_extract_runs_.push_back(
            {run_source_begin, run_output_begin, run_length, target});
        xdr_max_run_length_ = std::max(xdr_max_run_length_, run_length);
      };

      for (size_t out_idx = 1; out_idx < indices.size(); ++out_idx) {
        if (indices[out_idx] == indices[out_idx - 1] + 1) {
          ++run_length;
          continue;
        }
        flush_run();
        run_source_begin = indices[out_idx];
        run_output_begin = out_idx;
        run_length = 1;
      }

      flush_run();
    };

    append_runs(pro_atom_indices0_, ChunkStreamer::XdrRunTarget::Pro);
    append_runs(lig_atom_indices0_, ChunkStreamer::XdrRunTarget::Lig);
    append_runs(wat_atom_indices0_, ChunkStreamer::XdrRunTarget::Wat);
    append_runs(membrane_atom_indices0_,
                ChunkStreamer::XdrRunTarget::Membrane);

    std::stable_sort(xdr_extract_runs_.begin(), xdr_extract_runs_.end(),
                     [](const auto &lhs, const auto &rhs) {
                       return lhs.source_begin < rhs.source_begin;
                     });

    for (size_t run_idx = 1; run_idx < xdr_extract_runs_.size(); ++run_idx) {
      const auto &prev_run = xdr_extract_runs_[run_idx - 1];
      const auto &curr_run = xdr_extract_runs_[run_idx];
      if (curr_run.source_begin < prev_run.source_begin + prev_run.length) {
        xdr_has_overlapping_runs_ = true;
        break;
      }
    }

    if (xdr_max_run_length_ > 0) {
      xdr_trr_float_scratch_.resize(xdr_max_run_length_ * DIM);
      xdr_trr_double_scratch_.resize(xdr_max_run_length_ * DIM);
    } else {
      xdr_trr_float_scratch_.clear();
      xdr_trr_double_scratch_.clear();
    }

    size_t max_selected_atom_index = 0;
    auto update_max_index = [&](const std::vector<size_t> &indices) {
      if (!indices.empty()) {
        max_selected_atom_index =
            std::max(max_selected_atom_index,
                     *std::max_element(indices.begin(), indices.end()));
      }
    };
    update_max_index(pro_atom_indices0_);
    if (use_lig_group)
      update_max_index(lig_atom_indices0_);
    update_max_index(wat_atom_indices0_);
    update_max_index(membrane_atom_indices0_);
    if (natoms_check <= static_cast<int>(max_selected_atom_index)) {
      return std::unexpected(
          "XDR: natoms too small for selected atom indices - XDR has " +
          std::to_string(natoms_check) + " atoms but needs at least " +
          std::to_string(max_selected_atom_index + 1));
    }

    // 3. Open XDR stream
    xdr_path_ = tf;
    xdr_handle_ = xdrfile_open(tf.c_str(), "r");
    if (!xdr_handle_)
      return std::unexpected("XDR: Cannot open " + tf);

    // 4. Pre-allocate position buffer ONCE
    xdr_pos_buf_ = new rvec[xdr_natoms_];
    constexpr int64_t kXtcReserveBytes = 1024LL * 1024LL;
    if (args_ref.xdr_type == "xtc" &&
        xdrfile_reserve_coord_buffers(xdr_handle_, xdr_natoms_,
                                      kXtcReserveBytes) != exdrOK) {
      xdrfile_close(xdr_handle_);
      xdr_handle_ = nullptr;
      delete[] xdr_pos_buf_;
      xdr_pos_buf_ = nullptr;
      return std::unexpected("XDR: cannot reserve XTC decoder buffers for " +
                             tf);
    }
    xdr_eof_ = false;
    xdr_next_target_time_ = -1e18;

    return {}; // success — don't fall through to PDB
  }

  // ── PDB PATH (unchanged) ────────────────────────────────────────────
  if (args_ref.flag_1_pdb) {
    pro_total_bytes_ = file_size_bytes(args_ref.pdb_name);
    if (auto route_plan = build_pdb_route_plan(groups_ref, args_ref); route_plan) {
      pdb_route_plan_ = std::move(route_plan.value());
      pdb_route_plan_ready_ = true;
    }

    stream_pro.open(args_ref.pdb_name, std::ios::in);
    if (!stream_pro.is_open())
      return std::unexpected("Failed to open stream: " + args_ref.pdb_name);

    // Frame 0 provides topology/reference coordinates during initialization,
    // but it is also a real trajectory sample. Re-read it through the normal
    // streaming path so every enabled analysis sees the complete trajectory.
    pro_data_offset_bytes_ = stream_offset(stream_pro, pro_total_bytes_);
    if (args_ref.dt > 0.0 && !init_pro.m_mol.empty()) {
      pdb_next_target_time_ =
          args_ref.flag_begin_set ? args_ref.begin_time
                                  : init_pro.m_mol[0].time_of_snap;
    }
  } else {
    pro_total_bytes_ = file_size_bytes(name_pro);
    lig_total_bytes_ = file_size_bytes(name_lig);
    stream_pro.open(name_pro, std::ios::in);
    stream_lig.open(name_lig, std::ios::in);

    if (!stream_pro.is_open())
      return std::unexpected("Failed to open stream: " + name_pro);
    if (!stream_lig.is_open())
      return std::unexpected("Failed to open stream: " + name_lig);

    // As in the combined-PDB path, the initialization/reference snapshot is
    // still part of both trajectories and must be streamed for analysis.
    pro_data_offset_bytes_ = stream_offset(stream_pro, pro_total_bytes_);
    lig_data_offset_bytes_ = stream_offset(stream_lig, lig_total_bytes_);
  }
  return {};
}

// ── FIX 1: parallel_split_coordinates with topology copy ──────────────
void ChunkStreamer::parallel_split_coordinates(
    const Multi_MOL &source_chunk, const Multi_MOL &init_pro,
    const Multi_MOL &init_lig, const Multi_MOL &init_membrane,
    Multi_MOL &pro_out, Multi_MOL &lig_out, Multi_MOL &wat_out,
    Multi_MOL &membrane_out, size_t num_frames) {
  // Resize outputs
  pro_out.m_mol.resize(num_frames);
  lig_out.m_mol.resize(num_frames);
  if (args_ref.n_wat >= 0)
    wat_out.m_mol.resize(num_frames);
  if (args_ref.needs_membrane_coordinates())
    membrane_out.m_mol.resize(num_frames);

  // Pre-calculate target sizes to avoid redundant lookups
  size_t pro_size = groups_ref.groups_atom_number(groups_ref.group_1_index());
  size_t lig_size = args_ref.needs_mol2_coordinates()
                        ? groups_ref.groups_atom_number(groups_ref.group_2_index())
                        : 0;
  size_t wat_size =
      (args_ref.n_wat >= 0) ? groups_ref.groups_atom_number(args_ref.n_wat) : 0;
  size_t membrane_size =
      args_ref.needs_membrane_coordinates()
          ? groups_ref.groups_atom_number(
                static_cast<size_t>(args_ref.membrane.resolved_group))
          : 0;

  // Lambda to process a range of frames
  auto process_frames = [&](size_t start_f, size_t end_f) {
    for (size_t f = start_f; f < end_f; ++f) {
      // Lightweight frame init: keep stream metadata, avoid full MOL copies.
      initialize_stream_frame(pro_out.m_mol[f], init_pro,
                              source_chunk.m_mol[f].Box,
                              source_chunk.m_mol[f].time_of_snap, pro_size);
      initialize_stream_frame(lig_out.m_mol[f], init_lig,
                              source_chunk.m_mol[f].Box,
                              source_chunk.m_mol[f].time_of_snap, lig_size);

      // Overwrite ONLY coordinates from source_chunk
      pro_out.m_mol[f].pos.x.resize(pro_size);
      pro_out.m_mol[f].pos.y.resize(pro_size);
      pro_out.m_mol[f].pos.z.resize(pro_size);

      for (size_t j = 0; j < pro_size; ++j) {
        size_t atom_idx =
            groups_ref.atom_index(groups_ref.group_1_index(), j) - 1;
        if (atom_idx < source_chunk.m_mol[f].pos.x.size()) {
          pro_out.m_mol[f].pos.x[j] = source_chunk.m_mol[f].pos.x[atom_idx];
          pro_out.m_mol[f].pos.y[j] = source_chunk.m_mol[f].pos.y[atom_idx];
          pro_out.m_mol[f].pos.z[j] = source_chunk.m_mol[f].pos.z[atom_idx];
        }
      }

      // 2. Ligand: overwrite coords only
      lig_out.m_mol[f].pos.x.resize(lig_size);
      lig_out.m_mol[f].pos.y.resize(lig_size);
      lig_out.m_mol[f].pos.z.resize(lig_size);

      for (size_t j = 0; j < lig_size; ++j) {
        size_t atom_idx =
            groups_ref.atom_index(groups_ref.group_2_index(), j) - 1;
        if (atom_idx < source_chunk.m_mol[f].pos.x.size()) {
          lig_out.m_mol[f].pos.x[j] = source_chunk.m_mol[f].pos.x[atom_idx];
          lig_out.m_mol[f].pos.y[j] = source_chunk.m_mol[f].pos.y[atom_idx];
          lig_out.m_mol[f].pos.z[j] = source_chunk.m_mol[f].pos.z[atom_idx];
        }
      }

      // 3. Setup Water (no topology copy needed for water)
      if (wat_size > 0) {
        wat_out.m_mol[f].Box = source_chunk.m_mol[f].Box;
        wat_out.m_mol[f].time_of_snap = source_chunk.m_mol[f].time_of_snap;
        wat_out.m_mol[f].pos.x.resize(wat_size);
        wat_out.m_mol[f].pos.y.resize(wat_size);
        wat_out.m_mol[f].pos.z.resize(wat_size);

        for (size_t j = 0; j < wat_size; ++j) {
          size_t atom_idx = groups_ref.atom_index(args_ref.n_wat, j) - 1;
          if (atom_idx < source_chunk.m_mol[f].pos.x.size()) {
            wat_out.m_mol[f].pos.x[j] = source_chunk.m_mol[f].pos.x[atom_idx];
            wat_out.m_mol[f].pos.y[j] = source_chunk.m_mol[f].pos.y[atom_idx];
            wat_out.m_mol[f].pos.z[j] = source_chunk.m_mol[f].pos.z[atom_idx];
          }
        }
      }

      if (membrane_size > 0) {
        initialize_stream_frame(membrane_out.m_mol[f], init_membrane,
                                source_chunk.m_mol[f].Box,
                                source_chunk.m_mol[f].time_of_snap,
                                membrane_size);
        for (size_t j = 0; j < membrane_size; ++j) {
          const size_t atom_idx =
              groups_ref.atom_index(
                  static_cast<size_t>(args_ref.membrane.resolved_group), j) -
              1;
          if (atom_idx < source_chunk.m_mol[f].pos.x.size()) {
            membrane_out.m_mol[f].pos.x[j] =
                source_chunk.m_mol[f].pos.x[atom_idx];
            membrane_out.m_mol[f].pos.y[j] =
                source_chunk.m_mol[f].pos.y[atom_idx];
            membrane_out.m_mol[f].pos.z[j] =
                source_chunk.m_mol[f].pos.z[atom_idx];
          }
        }
      }
    }
  };

  executor_->parallel_for(
      num_frames, nt, 1,
      [&](size_t /*worker_id*/, size_t start, size_t end) {
        process_frames(start, end);
      });
}

std::expected<size_t, std::string>
ChunkStreamer::read_next_chunk(size_t chunk_size, Multi_MOL &pro_in,
                               Multi_MOL &lig, Multi_MOL &water_traj,
                               Multi_MOL &membrane_traj) {
  pro_in.m_mol.clear();
  lig.m_mol.clear();
  if (args_ref.n_wat >= 0)
    water_traj.m_mol.clear();
  if (args_ref.needs_membrane_coordinates())
    membrane_traj.m_mol.clear();

  auto current_stream_offset = [](std::ifstream &stream, size_t fallback_total) {
    const std::streampos pos = stream.tellg();
    if (pos == std::streampos(-1))
      return fallback_total;
    const auto as_off = static_cast<std::streamoff>(pos);
    return as_off < 0 ? size_t(0) : static_cast<size_t>(as_off);
  };
  auto progress_bytes = [](size_t current, size_t begin_offset, size_t total) {
    if (current <= begin_offset)
      return size_t(0);
    return std::min(current - begin_offset, total > begin_offset ? total - begin_offset
                                                                 : size_t(0));
  };
  auto show_pdb_loading_progress = [&](const std::string &label, size_t current,
                                       size_t total) {
    if (total == 0)
      return;
    TitanUI::show_progress_exact(label, current, total, "Loading PDB");
  };

  // ── XDR PATH (XTC/TRR) ──────────────────────────────────────────────
  if (args_ref.flag_xdr) {
    if (xdr_eof_ || !xdr_handle_)
      return 0;

    constexpr double NM_TO_ANG = 10.0;

    const size_t pro_size = pro_size_;
    const size_t lig_size = lig_size_;
    const size_t wat_size = wat_size_;
    const size_t membrane_size = membrane_size_;

    // Reserve frame slots only. Per-frame topology is shared from frame 0.
    pro_in.m_mol.reserve(chunk_size);
    lig.m_mol.reserve(chunk_size);
    if (wat_size > 0)
      water_traj.m_mol.reserve(chunk_size);
    if (membrane_size > 0)
      membrane_traj.m_mol.reserve(chunk_size);

    size_t frames_read = 0;
    size_t skip_count = 0;

    auto position_at_next_selected_frame =
        [&](XdrFrameProbe &selected_probe) -> std::expected<bool, std::string> {
      while (true) {
        XdrFrameProbe probe;
        std::string probe_error;
        const int probe_rc =
            probe_xdr_frame(xdr_handle_, args_ref.xdr_type, probe, probe_error);

        if (probe_rc == exdrENDOFFILE) {
          xdr_eof_ = true;
          return false;
        }
        if (probe_rc != exdrOK) {
          xdr_eof_ = true;
          return std::unexpected(probe_error);
        }
        if (probe.natoms != xdr_natoms_) {
          xdr_eof_ = true;
          return std::unexpected(
              "XDR: frame atom count changed from " +
              std::to_string(xdr_natoms_) + " to " +
              std::to_string(probe.natoms) + " while streaming.");
        }

        const double time_val = probe.time_ps;
        if (args_ref.flag_end_set && time_val > args_ref.end_time) {
          xdr_eof_ = true;
          return false;
        }

        const bool before_begin =
            args_ref.flag_begin_set && time_val < args_ref.begin_time;

        if (args_ref.dt > 0.0 && !before_begin &&
            xdr_next_target_time_ < -1e17) {
          xdr_next_target_time_ =
              args_ref.flag_begin_set ? args_ref.begin_time : time_val;
        }

        const bool before_dt_target =
            args_ref.dt > 0.0 &&
            time_val < xdr_next_target_time_ - kXdrTimeTolerancePs;

        if (before_begin || before_dt_target) {
          std::string skip_error;
          const int skip_rc = skip_xdr_frame_payload_after_probe(
              xdr_handle_, args_ref.xdr_type, probe, skip_error);
          if (skip_rc != exdrOK) {
            xdr_eof_ = true;
            return std::unexpected(skip_error);
          }

          if (before_begin) {
            ++skip_count;
            if (skip_count % 50 == 0) {
              TitanUI::show_continuous_progress(
                  "Skipping", skip_count,
                  std::to_string(static_cast<int>(time_val)) +
                      " ps (before -b=" +
                      std::to_string(static_cast<int>(args_ref.begin_time)) +
                      ")");
            }
          }
          continue;
        }

        if (skip_count > 0) {
          std::cout << "\n";
          TitanUI::print_info("[TITAN] Skipped " + std::to_string(skip_count) +
                              " frames. Starting analysis at t=" +
                              std::to_string(static_cast<int>(time_val)) +
                              " ps");
          skip_count = 0;
        }

        if (args_ref.dt > 0.0) {
          while (xdr_next_target_time_ <= time_val + kXdrTimeTolerancePs) {
            xdr_next_target_time_ += args_ref.dt;
          }
        }

        selected_probe = probe;
        return true;
      }
    };

    auto select_target_positions =
        [&](XdrRunTarget target, MOL &pro_frame, MOL &lig_frame,
            MOL *water_frame, MOL *membrane_frame) -> AlignedVector3D * {
      switch (target) {
      case XdrRunTarget::Pro:
        return &pro_frame.pos;
      case XdrRunTarget::Lig:
        return &lig_frame.pos;
      case XdrRunTarget::Wat:
        return water_frame ? &water_frame->pos : nullptr;
      case XdrRunTarget::Membrane:
        return membrane_frame ? &membrane_frame->pos : nullptr;
      }
      return nullptr;
    };

    auto materialize_run_from_float =
        [&](const XdrAtomRun &run, const float *src_xyz, MOL &pro_frame,
            MOL &lig_frame, MOL *water_frame, MOL *membrane_frame)
        -> std::expected<void, std::string> {
      auto *target_pos =
          select_target_positions(run.target, pro_frame, lig_frame, water_frame,
                                  membrane_frame);
      if (!target_pos) {
        return std::unexpected(
            "XDR: water atom run requested without an initialized water frame.");
      }

      for (size_t atom_offset = 0; atom_offset < run.length; ++atom_offset) {
        const size_t source_offset = atom_offset * DIM;
        const size_t output_index = run.output_begin + atom_offset;
        target_pos->x[output_index] =
            static_cast<double>(src_xyz[source_offset]) * NM_TO_ANG;
        target_pos->y[output_index] =
            static_cast<double>(src_xyz[source_offset + 1]) * NM_TO_ANG;
        target_pos->z[output_index] =
            static_cast<double>(src_xyz[source_offset + 2]) * NM_TO_ANG;
      }

      return {};
    };

    auto materialize_run_from_double =
        [&](const XdrAtomRun &run, const double *src_xyz, MOL &pro_frame,
            MOL &lig_frame, MOL *water_frame, MOL *membrane_frame)
        -> std::expected<void, std::string> {
      auto *target_pos =
          select_target_positions(run.target, pro_frame, lig_frame, water_frame,
                                  membrane_frame);
      if (!target_pos) {
        return std::unexpected(
            "XDR: water atom run requested without an initialized water frame.");
      }

      for (size_t atom_offset = 0; atom_offset < run.length; ++atom_offset) {
        const size_t source_offset = atom_offset * DIM;
        const size_t output_index = run.output_begin + atom_offset;
        target_pos->x[output_index] = src_xyz[source_offset] * NM_TO_ANG;
        target_pos->y[output_index] = src_xyz[source_offset + 1] * NM_TO_ANG;
        target_pos->z[output_index] = src_xyz[source_offset + 2] * NM_TO_ANG;
      }

      return {};
    };

    auto warn_xtc_fallback_once = [&](const std::string &reason) {
      if (!xdr_warned_xtc_fallback_) {
        TitanUI::print_warning(
            "XDR: streaming XTC fast payload path fell back to read_xtc(). " +
            reason);
        xdr_warned_xtc_fallback_ = true;
      }
    };

    auto warn_trr_fallback_once = [&](const std::string &reason) {
      if (!xdr_warned_trr_fallback_) {
        TitanUI::print_warning(
            "XDR: streaming TRR selective read fell back to read_trr(). " +
            reason);
        xdr_warned_trr_fallback_ = true;
      }
    };

    // Read frames until chunk is filled or EOF/end-time
    while (frames_read < chunk_size) {
      XdrFrameProbe selected_probe;
      auto positioned = position_at_next_selected_frame(selected_probe);
      if (!positioned)
        return std::unexpected(positioned.error());
      if (!positioned.value())
        break;

      {
        MOL pro_frame;
        MOL lig_frame;
        MOL water_frame;
        MOL membrane_frame;
        MOL *water_frame_ptr = nullptr;
        MOL *membrane_frame_ptr = nullptr;

        auto initialize_output_frames = [&](const box &frame_box,
                                            double time_val) {
          pro_frame = MOL{};
          lig_frame = MOL{};
          water_frame = MOL{};
          membrane_frame = MOL{};
          initialize_stream_frame(pro_frame, *init_pro_, frame_box, time_val,
                                  pro_size);
          initialize_stream_frame(lig_frame, *init_lig_, frame_box, time_val,
                                  lig_size);
          if (wat_size > 0) {
            initialize_stream_water_frame(water_frame, force_f, frame_box,
                                          time_val, wat_size);
            water_frame_ptr = &water_frame;
          } else {
            water_frame_ptr = nullptr;
          }
          if (membrane_size > 0) {
            initialize_stream_frame(membrane_frame, *init_membrane_, frame_box,
                                    time_val, membrane_size);
            membrane_frame_ptr = &membrane_frame;
          } else {
            membrane_frame_ptr = nullptr;
          }
        };

        if (args_ref.xdr_type == "xtc") {
          float time_ps = static_cast<float>(selected_probe.time_ps);
          std::string xtc_error;
          int rc = read_xtc_frame_payload_after_header(
              xdr_handle_, xdr_natoms_, xdr_box_, xdr_pos_buf_, &xdr_prec_,
              xtc_error);

          if (rc != exdrOK) {
            warn_xtc_fallback_once(xtc_error);
            // The failed fast decoder has consumed part of the frame. Recover
            // from a fresh XDR/stdio state instead of rewinding that handle.
            constexpr int kRecoveryAttempts = 3;
            std::string recovery_detail;
            for (int attempt = 1; attempt <= kRecoveryAttempts; ++attempt) {
              XDRFILE *recovery_handle = xdrfile_open(xdr_path_.c_str(), "r");
              if (!recovery_handle) {
                recovery_detail = "failed to open fresh handle";
                continue;
              }
              if (xdrfile_reserve_coord_buffers(recovery_handle, xdr_natoms_,
                                                1024LL * 1024LL) != exdrOK) {
                recovery_detail = "failed to reserve fresh decoder buffers";
                xdrfile_close(recovery_handle);
                continue;
              }
              const int seek_rc = xdr_seek(recovery_handle,
                                           selected_probe.frame_start, SEEK_SET);
              if (seek_rc != exdrOK) {
                recovery_detail = "failed to seek fresh handle (rc=" +
                                  std::to_string(seek_rc) + ")";
                xdrfile_close(recovery_handle);
                continue;
              }

              int step = 0;
              rc = read_xtc(recovery_handle, xdr_natoms_, &step, &time_ps,
                            xdr_box_, xdr_pos_buf_, &xdr_prec_);
              if (rc == exdrOK) {
                xdrfile_close(xdr_handle_);
                xdr_handle_ = recovery_handle;
                break;
              }
              recovery_detail = xdrfile_last_coord_error();
              xdrfile_close(recovery_handle);
              if (attempt < kRecoveryAttempts)
                std::this_thread::sleep_for(std::chrono::milliseconds(25 * attempt));
            }
            if (rc == exdrENDOFFILE) {
              xdr_eof_ = true;
              break;
            }
            if (rc != exdrOK) {
              xdr_eof_ = true;
              return std::unexpected(
                  "XDR: failed to recover selected XTC frame from a fresh "
                  "handle (offset=" +
                  std::to_string(selected_probe.frame_start) + ", time_ps=" +
                  std::to_string(selected_probe.time_ps) + ", rc=" +
                  std::to_string(rc) + ", " + xdr_status_message(rc) +
                  "; first pass: " + xtc_error + "; recovery detail: " +
                  recovery_detail + ")");
            }
          }

          initialize_output_frames(build_frame_box_from_xdr_matrix(xdr_box_),
                                   static_cast<double>(time_ps));

          for (const auto &run : xdr_extract_runs_) {
            auto materialized = materialize_run_from_float(
                run, &xdr_pos_buf_[run.source_begin][0], pro_frame, lig_frame,
                water_frame_ptr, membrane_frame_ptr);
            if (!materialized) {
              return std::unexpected(materialized.error());
            }
          }
        } else {
          float time_ps = static_cast<float>(selected_probe.time_ps);
          bool use_full_trr_fallback = false;
          std::string trr_reason;

          if (xdr_has_overlapping_runs_) {
            trr_reason =
                "Overlapping receptor/ligand/water atom selections require the "
                "full TRR reader for safety.";
            use_full_trr_fallback = true;
          } else if (!trr_selective_fast_path_supported(
                         selected_probe.trr_header, trr_reason)) {
            use_full_trr_fallback = true;
          }

          if (!use_full_trr_fallback) {
            std::string trr_error;
            int rc = read_trr_box_and_skip_prefix_after_header(
                xdr_handle_, selected_probe.trr_header, xdr_box_, trr_error);

            if (rc == exdrOK) {
              initialize_output_frames(build_frame_box_from_xdr_matrix(xdr_box_),
                                       static_cast<double>(time_ps));

              size_t cursor_atom = 0;
              const auto coord_elem_bytes =
                  static_cast<int64_t>(selected_probe.trr_header.bDouble
                                           ? sizeof(double)
                                           : sizeof(float));

              auto seek_trr_bytes =
                  [&](int64_t byte_count, const std::string &context)
                  -> std::expected<void, std::string> {
                if (byte_count == 0) {
                  return {};
                }

                const int seek_rc = xdr_seek(xdr_handle_, byte_count, SEEK_CUR);
                if (seek_rc != exdrOK) {
                  return std::unexpected("XDR: failed to skip " + context +
                                         " in TRR frame (rc=" +
                                         std::to_string(seek_rc) + ", " +
                                         xdr_status_message(seek_rc) + ")");
                }
                return {};
              };

              for (const auto &run : xdr_extract_runs_) {
                if (run.source_begin < cursor_atom) {
                  rc = exdrHEADER;
                  trr_error =
                      "XDR: overlapping atom runs reached TRR selective path.";
                  break;
                }

                const size_t gap_atoms = run.source_begin - cursor_atom;
                auto skipped_gap = seek_trr_bytes(
                    static_cast<int64_t>(gap_atoms) * DIM * coord_elem_bytes,
                    "a coordinate gap");
                if (!skipped_gap) {
                  rc = exdrHEADER;
                  trr_error = skipped_gap.error();
                  break;
                }

                const int value_count = static_cast<int>(run.length * DIM);
                if (selected_probe.trr_header.bDouble) {
                  if (xdrfile_read_double(xdr_trr_double_scratch_.data(),
                                          value_count, xdr_handle_) !=
                      value_count) {
                    rc = exdrDOUBLE;
                    trr_error =
                        "XDR: failed to read a selected TRR coordinate span.";
                    break;
                  }

                  auto materialized = materialize_run_from_double(
                      run, xdr_trr_double_scratch_.data(), pro_frame, lig_frame,
                      water_frame_ptr, membrane_frame_ptr);
                  if (!materialized) {
                    rc = exdrHEADER;
                    trr_error = materialized.error();
                    break;
                  }
                } else {
                  if (xdrfile_read_float(xdr_trr_float_scratch_.data(),
                                         value_count, xdr_handle_) !=
                      value_count) {
                    rc = exdrFLOAT;
                    trr_error =
                        "XDR: failed to read a selected TRR coordinate span.";
                    break;
                  }

                  auto materialized = materialize_run_from_float(
                      run, xdr_trr_float_scratch_.data(), pro_frame, lig_frame,
                      water_frame_ptr, membrane_frame_ptr);
                  if (!materialized) {
                    rc = exdrHEADER;
                    trr_error = materialized.error();
                    break;
                  }
                }

                cursor_atom = run.source_begin + run.length;
              }

              if (rc == exdrOK) {
                const size_t tail_atoms =
                    static_cast<size_t>(selected_probe.trr_header.natoms) -
                    cursor_atom;
                auto skipped_tail = seek_trr_bytes(
                    static_cast<int64_t>(tail_atoms) * DIM * coord_elem_bytes,
                    "the trailing coordinate block");
                if (!skipped_tail) {
                  rc = exdrHEADER;
                  trr_error = skipped_tail.error();
                }
              }

              if (rc == exdrOK) {
                auto skipped_velocity = seek_trr_bytes(
                    selected_probe.trr_header.v_size, "the velocity block");
                if (!skipped_velocity) {
                  rc = exdrHEADER;
                  trr_error = skipped_velocity.error();
                }
              }

              if (rc == exdrOK) {
                auto skipped_force =
                    seek_trr_bytes(selected_probe.trr_header.f_size,
                                   "the force block");
                if (!skipped_force) {
                  rc = exdrHEADER;
                  trr_error = skipped_force.error();
                }
              }
            }

            if (rc != exdrOK) {
              trr_reason = trr_error;
              use_full_trr_fallback = true;
            }
          }

          if (use_full_trr_fallback) {
            warn_trr_fallback_once(trr_reason);
            const int seek_rc =
                xdr_seek(xdr_handle_, selected_probe.frame_start, SEEK_SET);
            if (seek_rc != exdrOK) {
              xdr_eof_ = true;
              return std::unexpected(
                  "XDR: failed to rewind to selected TRR frame (rc=" +
                  std::to_string(seek_rc) + ", " +
                  xdr_status_message(seek_rc) + ")");
            }

            int has_prop = 0;
            int step = 0;
            float time_ps_fallback = time_ps;
            float lambda = 0.0f;
            const int rc = read_trr(xdr_handle_, xdr_natoms_, &step,
                                    &time_ps_fallback, &lambda, xdr_box_,
                                    xdr_pos_buf_, nullptr, nullptr, &has_prop);
            if (rc == exdrENDOFFILE) {
              xdr_eof_ = true;
              break;
            }
            if (rc != exdrOK) {
              xdr_eof_ = true;
              return std::unexpected(
                  "XDR: failed to read selected TRR frame (rc=" +
                  std::to_string(rc) + ", " + xdr_status_message(rc) + ")");
            }
            if ((has_prop & HASX) == 0) {
              xdr_eof_ = true;
              return std::unexpected(
                  "XDR: selected TRR frame does not contain coordinates.");
            }

            time_ps = time_ps_fallback;
            initialize_output_frames(build_frame_box_from_xdr_matrix(xdr_box_),
                                     static_cast<double>(time_ps));

            for (const auto &run : xdr_extract_runs_) {
              auto materialized = materialize_run_from_float(
                  run, &xdr_pos_buf_[run.source_begin][0], pro_frame, lig_frame,
                  water_frame_ptr, membrane_frame_ptr);
              if (!materialized) {
                return std::unexpected(materialized.error());
              }
            }
          }
        }

        pro_in.m_mol.push_back(std::move(pro_frame));
        lig.m_mol.push_back(std::move(lig_frame));
        if (water_frame_ptr) {
          water_traj.m_mol.push_back(std::move(water_frame));
        }
        if (membrane_frame_ptr) {
          membrane_traj.m_mol.push_back(std::move(membrane_frame));
        }

        ++frames_read;
        continue;
      }

      int step;
      float time_ps, lambda = 0.0f;
      int rc;

      if (args_ref.xdr_type == "xtc") {
        rc = read_xtc(xdr_handle_, xdr_natoms_, &step, &time_ps, xdr_box_,
                      xdr_pos_buf_, &xdr_prec_);
      } else {
        int has_prop = 0;
        rc = read_trr(xdr_handle_, xdr_natoms_, &step, &time_ps, &lambda,
                      xdr_box_, xdr_pos_buf_, nullptr, nullptr, &has_prop);
      }

      if (rc != exdrOK) {
        xdr_eof_ = true;
        break;
      }

      double time_val = static_cast<double>(time_ps);
      goto xdr_selected_frame;
#if 0

      // ── A. END FILTER — hard stop ──
      if (args_ref.flag_end_set && time_val > args_ref.end_time) {
        xdr_eof_ = true;
        break;
      }

      // ── B. BEGIN FILTER — skip frames before -b ──
      if (args_ref.flag_begin_set && time_val < args_ref.begin_time) {
        ++skip_count;
        if (skip_count % 50 == 0) {
          TitanUI::show_continuous_progress(
              "Skipping", skip_count,
              std::to_string(static_cast<int>(time_val)) + " ps (before -b=" +
                  std::to_string(static_cast<int>(args_ref.begin_time)) + ")");
        }
        continue;
      }
      if (skip_count > 0) {
        std::cout << "\n";
        TitanUI::print_info("[TITAN] Skipped " + std::to_string(skip_count) +
                            " frames. Starting analysis at t=" +
                            std::to_string(static_cast<int>(time_val)) + " ps");
        skip_count = 0;
      }

      // ── C. DT FILTER — stride ──
      if (args_ref.dt > 0.0) {
        if (xdr_next_target_time_ < -1e17) {
          // Initialize on first accepted-by-begin frame
          xdr_next_target_time_ =
              args_ref.flag_begin_set ? args_ref.begin_time : time_val;
        }
        if (time_val < xdr_next_target_time_ - 1e-4) {
          continue; // skip this frame (doesn't hit target time)
        }
        // Advance target: keep stepping until strictly past time_val
        while (xdr_next_target_time_ <= time_val + 1e-4)
          xdr_next_target_time_ += args_ref.dt;
      }

      // ── Frame passed all filters — store at frames_read index ──

      // ── Compute box from GROMACS matrix ──
#endif
    xdr_selected_frame:
      // Only frames that pass begin/end/dt filters are fully decoded here.
      double v1x = static_cast<double>(xdr_box_[0][0]) * NM_TO_ANG;
      double v1y = static_cast<double>(xdr_box_[0][1]) * NM_TO_ANG;
      double v1z = static_cast<double>(xdr_box_[0][2]) * NM_TO_ANG;
      double v2x = static_cast<double>(xdr_box_[1][0]) * NM_TO_ANG;
      double v2y = static_cast<double>(xdr_box_[1][1]) * NM_TO_ANG;
      double v2z = static_cast<double>(xdr_box_[1][2]) * NM_TO_ANG;
      double v3x = static_cast<double>(xdr_box_[2][0]) * NM_TO_ANG;
      double v3y = static_cast<double>(xdr_box_[2][1]) * NM_TO_ANG;
      double v3z = static_cast<double>(xdr_box_[2][2]) * NM_TO_ANG;

      double la = std::sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
      double lb = std::sqrt(v2x * v2x + v2y * v2y + v2z * v2z);
      double lc = std::sqrt(v3x * v3x + v3y * v3y + v3z * v3z);

      constexpr double RAD_TO_DEG = 180.0 / 3.14159265358979;
      double cos_alpha = (lb > 0 && lc > 0)
                             ? (v2x * v3x + v2y * v3y + v2z * v3z) / (lb * lc)
                             : 0.0;
      double cos_beta = (la > 0 && lc > 0)
                            ? (v1x * v3x + v1y * v3y + v1z * v3z) / (la * lc)
                            : 0.0;
      double cos_gamma = (la > 0 && lb > 0)
                             ? (v1x * v2x + v1y * v2y + v1z * v2z) / (la * lb)
                             : 0.0;
      cos_alpha = std::max(-1.0, std::min(1.0, cos_alpha));
      cos_beta = std::max(-1.0, std::min(1.0, cos_beta));
      cos_gamma = std::max(-1.0, std::min(1.0, cos_gamma));

      box frame_box;
      frame_box.set_a(la);
      frame_box.set_b(lb);
      frame_box.set_c(lc);
      frame_box.set_alpha(std::acos(cos_alpha) * RAD_TO_DEG);
      frame_box.set_beta(std::acos(cos_beta) * RAD_TO_DEG);
      frame_box.set_gamma(std::acos(cos_gamma) * RAD_TO_DEG);
      frame_box.setv();

      // ── Protein: topology copy + extract coords from rvec ──
      pro_in.m_mol.emplace_back();
      MOL &pro_frame = pro_in.m_mol.back();
      initialize_stream_frame(pro_frame, *init_pro_, frame_box, time_val,
                              pro_size);
      for (size_t j = 0; j < pro_size; ++j) {
        const size_t ai = pro_atom_indices0_[j];
        pro_frame.pos.x[j] = static_cast<double>(xdr_pos_buf_[ai][0]) * NM_TO_ANG;
        pro_frame.pos.y[j] = static_cast<double>(xdr_pos_buf_[ai][1]) * NM_TO_ANG;
        pro_frame.pos.z[j] = static_cast<double>(xdr_pos_buf_[ai][2]) * NM_TO_ANG;
      }

      // ── Ligand: topology copy + extract coords from rvec ──
      lig.m_mol.emplace_back();
      MOL &lig_frame = lig.m_mol.back();
      initialize_stream_frame(lig_frame, *init_lig_, frame_box, time_val,
                              lig_size);
      for (size_t j = 0; j < lig_size; ++j) {
        const size_t ai = lig_atom_indices0_[j];
        lig_frame.pos.x[j] = static_cast<double>(xdr_pos_buf_[ai][0]) * NM_TO_ANG;
        lig_frame.pos.y[j] = static_cast<double>(xdr_pos_buf_[ai][1]) * NM_TO_ANG;
        lig_frame.pos.z[j] = static_cast<double>(xdr_pos_buf_[ai][2]) * NM_TO_ANG;
      }

      // ── Water: extract coords only (no topology copy needed) ──
      if (wat_size > 0) {
        water_traj.m_mol.emplace_back();
        MOL &water_frame = water_traj.m_mol.back();
        initialize_stream_water_frame(water_frame, force_f, frame_box, time_val,
                                      wat_size);
        for (size_t j = 0; j < wat_size; ++j) {
          const size_t ai = wat_atom_indices0_[j];
          water_frame.pos.x[j] =
              static_cast<double>(xdr_pos_buf_[ai][0]) * NM_TO_ANG;
          water_frame.pos.y[j] =
              static_cast<double>(xdr_pos_buf_[ai][1]) * NM_TO_ANG;
          water_frame.pos.z[j] =
              static_cast<double>(xdr_pos_buf_[ai][2]) * NM_TO_ANG;
        }
      }

      if (membrane_size > 0) {
        membrane_traj.m_mol.emplace_back();
        MOL &membrane_frame = membrane_traj.m_mol.back();
        initialize_stream_frame(membrane_frame, *init_membrane_, frame_box,
                                time_val, membrane_size);
        for (size_t j = 0; j < membrane_size; ++j) {
          const size_t ai = membrane_atom_indices0_[j];
          membrane_frame.pos.x[j] =
              static_cast<double>(xdr_pos_buf_[ai][0]) * NM_TO_ANG;
          membrane_frame.pos.y[j] =
              static_cast<double>(xdr_pos_buf_[ai][1]) * NM_TO_ANG;
          membrane_frame.pos.z[j] =
              static_cast<double>(xdr_pos_buf_[ai][2]) * NM_TO_ANG;
        }
      }

      ++frames_read;
    }

    if (frames_read == 0)
      return 0;

    return frames_read;
  }

  // ── PDB PATH (unchanged) ────────────────────────────────────────────
  if (args_ref.flag_1_pdb == false) {
    const std::string load_label = "rec+lig";
    const size_t total_load_bytes =
        (pro_total_bytes_ > pro_data_offset_bytes_
             ? pro_total_bytes_ - pro_data_offset_bytes_
             : 0) +
        (lig_total_bytes_ > lig_data_offset_bytes_
             ? lig_total_bytes_ - lig_data_offset_bytes_
             : 0);
    const size_t load_before =
        progress_bytes(current_stream_offset(stream_pro, pro_total_bytes_),
                       pro_data_offset_bytes_, pro_total_bytes_) +
        progress_bytes(current_stream_offset(stream_lig, lig_total_bytes_),
                       lig_data_offset_bytes_, lig_total_bytes_);
    show_pdb_loading_progress(load_label, load_before, total_load_bytes);

    size_t pro_frames = 0;
    size_t lig_frames = 0;

    // REQUIREMENT 2: Read only if stream is still alive and hasn't failed
    if (!stream_pro.eof() && !stream_pro.fail()) {
      auto expected_pro = pro_in.read_pdb_chunk(stream_pro, chunk_size,
                                                args_ref, info_ref, force_f);
      if (!expected_pro)
        return std::unexpected("Receptor read failed: " + expected_pro.error());
      pro_frames = expected_pro.value();
      if (pro_frames > 0)
        last_pro_frame = pro_in.m_mol.back(); // Track last known frame
    }

    if (!stream_lig.eof() && !stream_lig.fail()) {
      auto expected_lig = lig.read_pdb_chunk(stream_lig, chunk_size, args_ref,
                                             info_ref, force_f);
      if (!expected_lig)
        return std::unexpected("Ligand read failed: " + expected_lig.error());
      lig_frames = expected_lig.value();
      if (lig_frames > 0)
        last_lig_frame = lig.m_mol.back(); // Track last known frame
    }

    const size_t load_after =
        progress_bytes(current_stream_offset(stream_pro, pro_total_bytes_),
                       pro_data_offset_bytes_, pro_total_bytes_) +
        progress_bytes(current_stream_offset(stream_lig, lig_total_bytes_),
                       lig_data_offset_bytes_, lig_total_bytes_);
    show_pdb_loading_progress(load_label, load_after, total_load_bytes);

    size_t actual_frames = std::max(pro_frames, lig_frames);

    // Stop if BOTH trajectories are entirely exhausted
    if (actual_frames == 0)
      return 0;

    // Sync shorter trajectory by holding its last known frame
    if (pro_frames < actual_frames) {
      if (pro_frames == 0)
        TitanUI::print_warning(
            "Receptor trajectory ended early. Freezing at last frame.");
      pro_in.m_mol.resize(actual_frames, last_pro_frame);
    }
    if (lig_frames < actual_frames) {
      if (lig_frames == 0)
        TitanUI::print_warning(
            "Ligand trajectory ended early. Freezing at last frame.");
      lig.m_mol.resize(actual_frames, last_lig_frame);
    }

    return actual_frames;
  } else {
    const std::string load_label =
        std::filesystem::path(args_ref.pdb_name).filename().string();
    const size_t total_load_bytes =
        pro_total_bytes_ > pro_data_offset_bytes_
            ? pro_total_bytes_ - pro_data_offset_bytes_
            : 0;
    const size_t load_before =
        progress_bytes(current_stream_offset(stream_pro, pro_total_bytes_),
                       pro_data_offset_bytes_, pro_total_bytes_);
    show_pdb_loading_progress(load_label, load_before, total_load_bytes);

    std::expected<size_t, std::string> expected_frames =
        std::unexpected("Chunk read failed.");
    const std::streampos chunk_start_pos = stream_pro.tellg();

    if (pdb_route_plan_ready_) {
      expected_frames = read_routed_pdb_chunk(
          stream_pro, args_ref.pdb_name, chunk_size, force_f, args_ref, info_ref,
          pro_in, lig, water_traj, membrane_traj, pdb_route_plan_,
          pdb_next_target_time_, false);
    }

    if (!pdb_route_plan_ready_ || !expected_frames) {
      if (pdb_route_plan_ready_ && !pdb_warned_direct_route_fallback_) {
        TitanUI::print_warning(
            "PDB direct routed streaming fell back to chunk split path: " +
            expected_frames.error());
        pdb_warned_direct_route_fallback_ = true;
      }
      if (chunk_start_pos != std::streampos(-1)) {
        stream_pro.clear();
        stream_pro.seekg(chunk_start_pos);
      }

      pro_in.m_mol.clear();
      lig.m_mol.clear();
      water_traj.m_mol.clear();
      membrane_traj.m_mol.clear();

      Multi_MOL chunk_pdb;
      expected_frames = chunk_pdb.read_pdb_chunk(stream_pro, chunk_size, args_ref,
                                                 info_ref, force_f);
      if (!expected_frames)
        return std::unexpected("Chunk read failed: " + expected_frames.error());
      if (expected_frames.value() == 0) {
        const size_t load_after =
            progress_bytes(current_stream_offset(stream_pro, pro_total_bytes_),
                           pro_data_offset_bytes_, pro_total_bytes_);
        show_pdb_loading_progress(load_label, load_after, total_load_bytes);
        return 0;
      }

      parallel_split_coordinates(chunk_pdb, *init_pro_, *init_lig_,
                                 *init_membrane_, pro_in, lig, water_traj,
                                 membrane_traj, expected_frames.value());
    }

    const size_t load_after =
        progress_bytes(current_stream_offset(stream_pro, pro_total_bytes_),
                       pro_data_offset_bytes_, pro_total_bytes_);
    show_pdb_loading_progress(load_label, load_after, total_load_bytes);

    if (expected_frames.value() == 0)
      return 0;
    return expected_frames.value();
  }
}
