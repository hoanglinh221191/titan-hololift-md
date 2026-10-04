#ifndef TITAN_ALIGNMENT_H
#define TITAN_ALIGNMENT_H

#include "algebra/mat3.h"
#include "algebra/matrix_view.h"
#include "algebra/vec3.h"

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace TitanAlignment {

struct AlignResult {
  titan::algebra::Mat3 R;
  titan::algebra::Vec3 ref_centroid;
  titan::algebra::Vec3 mob_centroid;
};

titan::algebra::Mat3
kabsch_rotation(titan::algebra::Matrix3ConstView ref_centered,
                titan::algebra::Matrix3ConstView mobile_centered);

AlignResult align_frame(titan::algebra::Matrix3ConstView ref_centered,
                        titan::algebra::Matrix3View mobile);

void align_frame_inplace(titan::algebra::Matrix3ConstView ref_centered,
                         titan::algebra::Matrix3View mobile,
                         titan::algebra::Mat3 &R_out);

namespace detail {

template <typename T>
using remove_cvref_t = std::remove_cvref_t<T>;

template <typename T>
concept Matrix3ConstViewLike =
    std::same_as<remove_cvref_t<T>, titan::algebra::Matrix3ConstView>;

template <typename T>
concept Matrix3ViewLike =
    std::same_as<remove_cvref_t<T>, titan::algebra::Matrix3View>;

template <typename Matrix>
concept StridedMatrixLike = requires(Matrix &m, const Matrix &cm) {
  typename remove_cvref_t<Matrix>::Index;
  { remove_cvref_t<Matrix>::IsRowMajor } -> std::convertible_to<int>;
  { cm.data() };
  { m.data() };
  { cm.rows() };
  { cm.innerStride() };
  { cm.outerStride() };
  { m(static_cast<typename remove_cvref_t<Matrix>::Index>(0),
      static_cast<typename remove_cvref_t<Matrix>::Index>(0)) };
};

template <typename Matrix>
  requires StridedMatrixLike<Matrix>
titan::algebra::Matrix3ConstView matrix3_const_view(const Matrix &m) {
  using M = remove_cvref_t<Matrix>;
  const auto row_stride = M::IsRowMajor ? m.outerStride() : m.innerStride();
  const auto col_stride = M::IsRowMajor ? m.innerStride() : m.outerStride();
  return {m.data(), static_cast<std::size_t>(m.rows()),
          static_cast<std::ptrdiff_t>(row_stride),
          static_cast<std::ptrdiff_t>(col_stride)};
}

template <typename Matrix>
  requires StridedMatrixLike<Matrix>
titan::algebra::Matrix3View matrix3_view(Matrix &m) {
  using M = remove_cvref_t<Matrix>;
  const auto row_stride = M::IsRowMajor ? m.outerStride() : m.innerStride();
  const auto col_stride = M::IsRowMajor ? m.innerStride() : m.outerStride();
  return {m.data(), static_cast<std::size_t>(m.rows()),
          static_cast<std::ptrdiff_t>(row_stride),
          static_cast<std::ptrdiff_t>(col_stride)};
}

template <typename Matrix>
  requires StridedMatrixLike<Matrix>
void assign_mat3(Matrix &out, const titan::algebra::Mat3 &in) {
  for (std::size_t r = 0; r < 3; ++r)
    for (std::size_t c = 0; c < 3; ++c)
      out(static_cast<typename remove_cvref_t<Matrix>::Index>(r),
          static_cast<typename remove_cvref_t<Matrix>::Index>(c)) = in(r, c);
}

} // namespace detail

template <typename RefMatrix, typename MobileMatrix>
  requires detail::StridedMatrixLike<RefMatrix> &&
           detail::StridedMatrixLike<MobileMatrix>
titan::algebra::Mat3 kabsch_rotation(const RefMatrix &ref_centered,
                                     const MobileMatrix &mobile_centered) {
  return kabsch_rotation(detail::matrix3_const_view(ref_centered),
                         detail::matrix3_const_view(mobile_centered));
}

template <typename RefMatrix, typename MobileMatrix>
  requires detail::StridedMatrixLike<RefMatrix> &&
           detail::StridedMatrixLike<MobileMatrix>
AlignResult align_frame(const RefMatrix &ref_centered, MobileMatrix &mobile) {
  return align_frame(detail::matrix3_const_view(ref_centered),
                     detail::matrix3_view(mobile));
}

template <typename RefMatrix, typename MobileMatrix, typename RotationMatrix>
  requires detail::StridedMatrixLike<RefMatrix> &&
           detail::StridedMatrixLike<MobileMatrix> &&
           detail::StridedMatrixLike<RotationMatrix>
void align_frame_inplace(const RefMatrix &ref_centered, MobileMatrix &mobile,
                         RotationMatrix &R_out) {
  titan::algebra::Mat3 R{};
  align_frame_inplace(detail::matrix3_const_view(ref_centered),
                      detail::matrix3_view(mobile), R);
  detail::assign_mat3(R_out, R);
}

} // namespace TitanAlignment

#endif // TITAN_ALIGNMENT_H
