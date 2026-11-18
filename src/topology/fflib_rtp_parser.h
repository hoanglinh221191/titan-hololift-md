#ifndef TITAN_FFLIB_RTP_PARSER_H
#define TITAN_FFLIB_RTP_PARSER_H

#include "amino_a.h"

#include <cstddef>
#include <expected>
#include <istream>
#include <string>

namespace TitanFfLib {

struct RtpParseSummary {
  std::size_t residue_count{0};
  std::size_t atom_count{0};
};

// Parse only [ atoms ] records from a GROMACS RTP file. Bond, angle,
// improper, CMAP, and other subsection records must never enter the residue
// atom table because they can reuse atom names and silently overwrite atom
// types.
[[nodiscard]] std::expected<RtpParseSummary, std::string>
parse_gromacs_rtp_atoms(std::istream &input, amino_acid &amine);

} // namespace TitanFfLib

#endif
