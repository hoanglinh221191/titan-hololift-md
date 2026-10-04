#ifndef CHUNK_STREAMER_H
#define CHUNK_STREAMER_H

#include "Multi_MOL.h"
#include "arguments.h"
#include "groups.h"
#include "info.h"
#include <cmath> // acos, sqrt for triclinic box conversion
#include <cstdint>
#include <expected>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

class TitanExecutor;

// Forward-declare XDR types to avoid including xdrfile.h
// (which defines `typedef float matrix[3][3]` conflicting with Eigen)
struct XDRFILE;

class ChunkStreamer {
public:
  ChunkStreamer(arguments &args, info &output_info, groups_cal &groups,
                std::string_view force_f, TitanExecutor &executor,
                size_t num_threads);
  ~ChunkStreamer();

  std::expected<void, std::string> open_streams(const std::string &name_pro,
                                                const std::string &name_lig,
                                                const Multi_MOL &init_pro,
                                                const Multi_MOL &init_lig,
                                                const Multi_MOL &init_membrane);

  std::expected<size_t, std::string> read_next_chunk(size_t chunk_size,
                                                     Multi_MOL &pro_in,
                                                     Multi_MOL &lig,
                                                     Multi_MOL &water_traj,
                                                     Multi_MOL &membrane_traj);

private:
  enum class XdrRunTarget : std::uint8_t {
    Pro = 0,
    Lig = 1,
    Wat = 2,
    Membrane = 3
  };

  struct XdrAtomRun {
    size_t source_begin = 0;
    size_t output_begin = 0;
    size_t length = 0;
    XdrRunTarget target = XdrRunTarget::Pro;
  };

  arguments &args_ref;
  info &info_ref;
  groups_cal &groups_ref;
  std::string force_f;
  TitanExecutor *executor_;
  size_t nt;

  std::ifstream stream_pro;
  std::ifstream stream_lig;

  MOL last_pro_frame;
  MOL last_lig_frame;
  PdbRoutePlan pdb_route_plan_;
  bool pdb_route_plan_ready_ = false;
  bool pdb_warned_direct_route_fallback_ = false;
  size_t pro_total_bytes_ = 0;
  size_t lig_total_bytes_ = 0;
  size_t pro_data_offset_bytes_ = 0;
  size_t lig_data_offset_bytes_ = 0;
  double pdb_next_target_time_ = -1e18;

  void parallel_split_coordinates(const Multi_MOL &source_chunk,
                                  const Multi_MOL &init_pro,
                                  const Multi_MOL &init_lig,
                                  const Multi_MOL &init_membrane,
                                  Multi_MOL &pro_out, Multi_MOL &lig_out,
                                  Multi_MOL &wat_out,
                                  Multi_MOL &membrane_out, size_t num_frames);

  // ── XDR state ──
  // Using opaque pointer + float arrays to avoid xdrfile.h in header
  XDRFILE *xdr_handle_ = nullptr;
  std::string xdr_path_;
  int xdr_natoms_ = 0;
  float (*xdr_pos_buf_)[3] = nullptr; // rvec = float[3], allocated as array
  float xdr_box_[3][3]{};             // matrix = float[3][3]
  float xdr_prec_ = 1000.0f;
  bool xdr_eof_ = false;
  size_t xdr_file_idx_ = 0;
  size_t pro_size_ = 0;
  size_t lig_size_ = 0;
  size_t wat_size_ = 0;
  size_t membrane_size_ = 0;
  std::vector<size_t> pro_atom_indices0_;
  std::vector<size_t> lig_atom_indices0_;
  std::vector<size_t> wat_atom_indices0_;
  std::vector<size_t> membrane_atom_indices0_;
  std::vector<XdrAtomRun> xdr_extract_runs_;
  size_t xdr_max_run_length_ = 0;
  bool xdr_has_overlapping_runs_ = false;
  std::vector<float> xdr_trr_float_scratch_;
  std::vector<double> xdr_trr_double_scratch_;
  bool xdr_warned_xtc_fallback_ = false;
  bool xdr_warned_trr_fallback_ = false;

  const Multi_MOL *init_pro_ = nullptr;
  const Multi_MOL *init_lig_ = nullptr;
  const Multi_MOL *init_membrane_ = nullptr;

  // ── XDR time filter state (persists across chunks) ──
  double xdr_next_target_time_ = -1e18; // for -dt stride
};

#endif
