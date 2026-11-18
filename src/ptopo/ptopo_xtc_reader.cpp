#include "ptopo_xtc_reader.h"

#include "../gmxtraj/include/xdrfile.h"
#include "../gmxtraj/include/xdrfile_xtc.h"

#include <utility>

namespace titan_ptopo {

Result<XtcReader> XtcReader::open(const std::filesystem::path &path) {
  XtcReader reader;
  reader.path_ = path.string();
  std::string mutable_path = reader.path_; // read_xtc_natoms takes char*
  int natoms = 0;
  if (read_xtc_natoms(mutable_path.data(), &natoms) != exdrOK || natoms <= 0)
    return make_error(ErrorCode::InvalidInput,
                      "cannot read the atom count of XTC " + reader.path_);
  XDRFILE *file = xdrfile_open(reader.path_.c_str(), "r");
  if (file == nullptr)
    return make_error(ErrorCode::InvalidInput,
                      "cannot open XTC " + reader.path_);
  reader.file_ = file;
  reader.natoms_ = natoms;
  reader.buffer_ =
      std::make_unique<float[]>(static_cast<std::size_t>(natoms) * 3);
  return reader;
}

XtcReader::XtcReader(XtcReader &&other) noexcept
    : file_(std::exchange(other.file_, nullptr)),
      natoms_(std::exchange(other.natoms_, 0)),
      frames_read_(std::exchange(other.frames_read_, 0)),
      buffer_(std::move(other.buffer_)), path_(std::move(other.path_)) {}

XtcReader &XtcReader::operator=(XtcReader &&other) noexcept {
  if (this != &other) {
    close();
    file_ = std::exchange(other.file_, nullptr);
    natoms_ = std::exchange(other.natoms_, 0);
    frames_read_ = std::exchange(other.frames_read_, 0);
    buffer_ = std::move(other.buffer_);
    path_ = std::move(other.path_);
  }
  return *this;
}

XtcReader::~XtcReader() { close(); }

void XtcReader::close() noexcept {
  if (file_ != nullptr) {
    xdrfile_close(static_cast<XDRFILE *>(file_));
    file_ = nullptr;
  }
}

Result<bool> XtcReader::next(XtcFrame &frame) {
  if (file_ == nullptr)
    return make_error(ErrorCode::InvalidInput, "XTC reader is not open");
  int step = 0;
  float time = 0.0f, precision = 0.0f;
  matrix box{};
  auto *coords = reinterpret_cast<rvec *>(buffer_.get());
  const int rc = read_xtc(static_cast<XDRFILE *>(file_), natoms_, &step, &time,
                          box, coords, &precision);
  if (rc == exdrENDOFFILE)
    return false;
  if (rc != exdrOK)
    return make_error(ErrorCode::InvalidInput,
                      "truncated or corrupt XTC frame " +
                          std::to_string(frames_read_) + " in " + path_ +
                          " (xdrfile code " + std::to_string(rc) + ")");
  frame.index = frames_read_++;
  frame.step = step;
  frame.time = static_cast<double>(time);
  frame.precision = static_cast<double>(precision);
  for (std::size_t r = 0; r < 3; ++r)
    for (std::size_t k = 0; k < 3; ++k)
      frame.box_rows[r][k] = static_cast<double>(box[r][k]);
  const std::size_t n = static_cast<std::size_t>(natoms_);
  frame.positions.resize(n);
  const float *raw = buffer_.get();
  for (std::size_t i = 0; i < n; ++i)
    frame.positions[i] = {static_cast<double>(raw[3 * i]),
                          static_cast<double>(raw[3 * i + 1]),
                          static_cast<double>(raw[3 * i + 2])};
  return true;
}

} // namespace titan_ptopo
