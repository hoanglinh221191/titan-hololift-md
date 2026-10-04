#ifndef TITAN_ROTATIONAL_DIFFUSION_H
#define TITAN_ROTATIONAL_DIFFUSION_H

#include "MOL.h"
#include "aligned_allocator.h"
#include "titan_selection_utils.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <span>
#include <string>
#include <vector>

class TitanExecutor;

namespace titan_rotational_diffusion {

struct Point3 {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

struct Quaternion {
  double w{1.0};
  double x{0.0};
  double y{0.0};
  double z{0.0};

  [[nodiscard]] static Quaternion identity() noexcept { return {}; }
};

struct Covariance3 {
  // xx, yy, zz, xy, xz, yz
  std::array<double, 6> values{};
};

struct OrientationFit {
  bool valid{false};
  Quaternion orientation{};
  double rmsd_a{0.0};
};

struct LagResult {
  std::size_t lag_index{0};
  double lag_time_ps{0.0};
  double rotational_msd_rad2{0.0};
  double c1{0.0};
  double c2{0.0};
  double orientation_correlation{0.0};
  std::uint64_t samples{0};
  Covariance3 covariance{};
  Covariance3 covariance_fit{};
  std::array<double, 3> instantaneous_d_per_ps{};
  std::array<double, 9> instantaneous_axes{};
  bool instantaneous_valid{false};
};

struct TensorFit {
  bool ok{false};
  std::string status{"insufficient_points"};
  std::string window_policy{"none"};
  double tmin_ps{0.0};
  double tmax_ps{0.0};
  std::size_t points{0};
  std::array<double, 3> d_per_ps{};
  std::array<double, 9> axes{};
  double residual{0.0};
};

[[nodiscard]] Quaternion normalized(Quaternion q);
[[nodiscard]] Quaternion canonicalize_hemisphere(Quaternion q,
                                                 const Quaternion &previous);
[[nodiscard]] Quaternion conjugate(const Quaternion &q) noexcept;
[[nodiscard]] Quaternion multiply(const Quaternion &lhs,
                                  const Quaternion &rhs) noexcept;
[[nodiscard]] double angular_distance_rad(const Quaternion &lhs,
                                          const Quaternion &rhs);

[[nodiscard]] OrientationFit
fit_orientation(std::span<const Point3> reference,
                std::span<const Point3> mobile,
                std::span<const double> masses = {},
                const box *frame_box = nullptr);

template <std::size_t N>
[[nodiscard]] OrientationFit
fit_orientation_from_triad(const std::array<Point3, N> &reference,
                           const std::array<Point3, N> &mobile,
                           const box *frame_box) {
  return fit_orientation(reference, mobile, {}, frame_box);
}

[[nodiscard]] Covariance3 covariance_scalar(const Point3 *reference,
                                            const Point3 *frame,
                                            std::size_t count);
[[nodiscard]] Covariance3 covariance_optimized(const Point3 *reference,
                                               const Point3 *frame,
                                               std::size_t count);
[[nodiscard]] Covariance3
favro_covariance_model(double lag_time_ps, const std::array<double, 3> &d,
                       const std::array<double, 9> &axes);
void update_instantaneous_tensor(LagResult &row);

[[nodiscard]] std::vector<LagResult>
compute_lag_results(std::span<const Quaternion> orientations,
                    std::span<const double> time_ps, std::size_t maxlag,
                    TitanExecutor &executor);
[[nodiscard]] std::vector<LagResult>
compute_lag_results_scalar(std::span<const Quaternion> orientations,
                           std::span<const double> time_ps,
                           std::size_t maxlag);

[[nodiscard]] TensorFit fit_tensor(std::vector<LagResult> &results,
                                   const std::string &fit_window);

class Analyzer {
public:
  void configure(bool enabled, std::string fit_window, std::string maxlag,
                 std::string pbc, std::string selection);
  void init(const std::vector<titan_selection::SelectedAtom> &selected_atoms,
            std::span<const double> selected_masses, std::fstream &log);
  void append_frame(const MOL &frame);
  void finalize(const std::string &msd_filename,
                const std::string &quaternion_filename,
                const std::string &covariance_filename,
                const std::string &summary_filename, TitanExecutor &executor,
                std::fstream &log);
  void reset_trajectory_state();

  [[nodiscard]] bool enabled() const noexcept {
    return requested_ && initialized_;
  }

private:
  [[nodiscard]] bool valid_box(const box &frame_box) const noexcept;
  [[nodiscard]] std::vector<Point3> reconstruct_body(const MOL &frame) const;
  void export_msd(const std::string &filename,
                  const std::vector<LagResult> &results) const;
  void export_quaternions(const std::string &filename) const;
  void export_covariance(const std::string &filename,
                         const std::vector<LagResult> &results) const;
  void export_summary(const std::string &filename, const TensorFit &fit,
                      const std::string &effective_pbc,
                      const std::string &pbc_warning) const;

  bool requested_{false};
  bool initialized_{false};
  std::string fit_window_{"auto"};
  std::string maxlag_{"auto"};
  std::string pbc_{"auto"};
  std::string selection_{"mol1"};
  std::vector<titan_selection::SelectedAtom> selected_atoms_;
  AlignedDoubleVector selected_masses_;
  std::vector<Point3> reference_;
  std::vector<std::vector<Point3>> bodies_;
  std::vector<box> boxes_;
  std::vector<double> time_ps_;
  std::vector<bool> box_valid_;
  AlignedDoubleVector qw_;
  AlignedDoubleVector qx_;
  AlignedDoubleVector qy_;
  AlignedDoubleVector qz_;
  AlignedDoubleVector rmsd_a_;
};

} // namespace titan_rotational_diffusion

#endif // TITAN_ROTATIONAL_DIFFUSION_H
