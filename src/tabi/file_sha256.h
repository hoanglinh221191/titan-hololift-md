#ifndef TITAN_PBE_FILE_SHA256_H
#define TITAN_PBE_FILE_SHA256_H

#include <string>

namespace TitanPBE {

[[nodiscard]] std::string file_sha256_hex(const std::string &path);

} // namespace TitanPBE

#endif
