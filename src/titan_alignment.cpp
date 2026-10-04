#include "titan_alignment.h"

#include "algebra/kabsch3.h"

#include <cassert>

namespace TitanAlignment {

titan::algebra::Mat3
kabsch_rotation(titan::algebra::Matrix3ConstView ref_centered,
                titan::algebra::Matrix3ConstView mobile_centered) {
  assert(ref_centered.rows == mobile_centered.rows);
  return titan::algebra::kabsch_rotation(ref_centered, mobile_centered);
}

AlignResult align_frame(titan::algebra::Matrix3ConstView ref_centered,
                        titan::algebra::Matrix3View mobile) {
  assert(ref_centered.rows == mobile.rows);

  AlignResult result{};

  for (std::size_t r = 0; r < mobile.rows; ++r) {
    result.mob_centroid.x += mobile(r, 0);
    result.mob_centroid.y += mobile(r, 1);
    result.mob_centroid.z += mobile(r, 2);
  }
  result.mob_centroid = result.mob_centroid / static_cast<double>(mobile.rows);

  for (std::size_t r = 0; r < mobile.rows; ++r) {
    mobile(r, 0) -= result.mob_centroid.x;
    mobile(r, 1) -= result.mob_centroid.y;
    mobile(r, 2) -= result.mob_centroid.z;
  }

  result.R = TitanAlignment::kabsch_rotation(
      ref_centered,
      titan::algebra::Matrix3ConstView{mobile.data, mobile.rows,
                                       mobile.row_stride,
                                       mobile.col_stride});

  for (std::size_t r = 0; r < mobile.rows; ++r) {
    const titan::algebra::Vec3 rotated =
        titan::algebra::mul_row({mobile(r, 0), mobile(r, 1), mobile(r, 2)},
                                result.R);
    mobile(r, 0) = rotated.x;
    mobile(r, 1) = rotated.y;
    mobile(r, 2) = rotated.z;
  }

  return result;
}

void align_frame_inplace(titan::algebra::Matrix3ConstView ref_centered,
                         titan::algebra::Matrix3View mobile,
                         titan::algebra::Mat3 &R_out) {
  const AlignResult result = align_frame(ref_centered, mobile);
  R_out = result.R;
}

} // namespace TitanAlignment
