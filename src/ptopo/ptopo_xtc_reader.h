#ifndef TITAN_PTOPO_XTC_READER_H
#define TITAN_PTOPO_XTC_READER_H

// Sequential XTC frames for the periodic-graph adapter, over TITAN's bundled
// xdrfile (src/gmxtraj). A clean end of file ends the stream; a truncated or
// corrupt frame is an error, never a silent end (chunk_streamer treats both
// as EOF).

#include "ptopo_types.h"

#include <filesystem>
#include <memory>

namespace titan_ptopo {

struct XtcFrame {
  std::int64_t index = 0; // 0-based position in the file
  std::int64_t step = 0;
  double time = 0.0;      // ps
  // GROMACS layout: box_rows[k] is lattice vector k, nm.
  std::array<Vec3, 3> box_rows{};
  std::vector<Vec3> positions; // nm
  double precision = 0.0;
};

class XtcReader {
public:
  static Result<XtcReader> open(const std::filesystem::path &path);

  XtcReader(XtcReader &&other) noexcept;
  XtcReader &operator=(XtcReader &&other) noexcept;
  XtcReader(const XtcReader &) = delete;
  XtcReader &operator=(const XtcReader &) = delete;
  ~XtcReader();

  [[nodiscard]] std::size_t atom_count() const noexcept {
    return static_cast<std::size_t>(natoms_);
  }
  // true: `frame` holds the next frame; false: clean end of file.
  Result<bool> next(XtcFrame &frame);

private:
  XtcReader() = default;
  void close() noexcept;

  void *file_ = nullptr; // XDRFILE*, kept opaque to avoid the C header here
  int natoms_ = 0;
  std::int64_t frames_read_ = 0;
  std::unique_ptr<float[]> buffer_;
  std::string path_;
};

} // namespace titan_ptopo

#endif // TITAN_PTOPO_XTC_READER_H
