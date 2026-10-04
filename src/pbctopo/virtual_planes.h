#ifndef TITAN_PBCTOPO_VIRTUAL_PLANES_H
#define TITAN_PBCTOPO_VIRTUAL_PLANES_H

#include "types.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace titan_pbctopo {

struct PbctopoVirtualPlaneBuildOptions {
  std::size_t histogram_bins = 64;
  std::size_t max_sparse_planes_per_axis = 24;
};

struct PbctopoVirtualPlaneSet {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::size_t dense_equivalent_x = 0;
  std::size_t dense_equivalent_y = 0;
  std::size_t dense_equivalent_z = 0;
  bool used_dense_fallback = false;
};

std::vector<double>
build_dense_virtual_box_event_planes(const std::vector<PbctopoAtom> &atoms,
                                     char axis);

std::vector<double>
build_density_virtual_box_event_planes(const std::vector<PbctopoAtom> &atoms,
                                       char axis,
                                       std::size_t histogram_bins = 64,
                                       std::size_t max_planes = 24);

PbctopoVirtualPlaneSet build_density_cut_plane_shortlist_or_dense_fallback(
    const std::vector<PbctopoAtom> &atoms,
    PbctopoVirtualPlaneBuildOptions options = {});

class PbctopoVirtualStateTracker {
public:
  void reserve(std::size_t expected_states);
  void reset(const std::vector<double> &x_planes,
             const std::vector<double> &y_planes,
             const std::vector<double> &z_planes);
  void clear();

  bool try_insert(double cut_x, double cut_y, double cut_z);
  bool try_insert(double cut_x, double cut_y, double cut_z,
                  std::uint64_t shift_signature);

  std::size_t collision_count() const noexcept { return collisions_; }
  std::size_t inserted_count() const noexcept { return inserted_; }

private:
  struct CenterKey {
    std::int64_t x = 0;
    std::int64_t y = 0;
    std::int64_t z = 0;
    std::uint64_t shift_signature = 0;
    bool has_shift_signature = false;

    bool operator==(const CenterKey &rhs) const noexcept {
      return x == rhs.x && y == rhs.y && z == rhs.z;
    }
  };

  std::uint64_t sparse_state_key(double cut_x, double cut_y,
                                 double cut_z) const;
  static CenterKey quantize_center(double cut_x, double cut_y, double cut_z);

  std::vector<double> x_planes_;
  std::vector<double> y_planes_;
  std::vector<double> z_planes_;
  std::unordered_map<std::uint64_t, std::vector<CenterKey>> buckets_;
  std::size_t collisions_ = 0;
  std::size_t inserted_ = 0;
};

class PbctopoVirtualAssignmentSignatureTable {
public:
  void build(const std::vector<PbctopoAtom> &atoms);
  void clear();

  bool empty() const noexcept {
    return x_coords_.empty() && y_coords_.empty() && z_coords_.empty();
  }

  std::uint64_t state_signature(double cut_x, double cut_y,
                                double cut_z) const;

private:
  static void append_axis_coords(const std::vector<PbctopoAtom> &atoms,
                                 char axis, std::vector<double> &coords);
  static std::uint64_t axis_interval_signature(
      const std::vector<double> &unique_coords, double center);

  std::vector<double> x_coords_;
  std::vector<double> y_coords_;
  std::vector<double> z_coords_;
};

} // namespace titan_pbctopo

#endif // TITAN_PBCTOPO_VIRTUAL_PLANES_H
