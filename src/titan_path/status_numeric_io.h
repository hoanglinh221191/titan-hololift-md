#pragma once

#include <array>
#include <iomanip>
#include <limits>
#include <locale>
#include <ostream>

namespace titan_path {
// Machine-readable control data must survive a double -> text -> double
// round trip. Presentation files may retain their separate compact format.
inline void configure_status_numeric_output(std::ostream &out) {
  out.imbue(std::locale::classic());
  out << std::defaultfloat
      << std::setprecision(std::numeric_limits<double>::max_digits10);
}

inline void write_next_vector_status(std::ostream &out,
                                     const std::array<double, 3> &vector,
                                     double displacement, double angle_deg) {
  configure_status_numeric_output(out);
  out << "NUMERIC_SERIALIZATION=FLOAT64_MAX_DIGITS10\n";
  out << "NEXT_VECTOR_X_A=" << vector[0] << '\n';
  out << "NEXT_VECTOR_Y_A=" << vector[1] << '\n';
  out << "NEXT_VECTOR_Z_A=" << vector[2] << '\n';
  out << "NEXT_REFERENCE_DISPLACEMENT_A=" << displacement << '\n';
  out << "NEXT_VECTOR_ANGLE_DEG=" << angle_deg << '\n';
}
} // namespace titan_path
