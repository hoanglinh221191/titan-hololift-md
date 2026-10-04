#ifndef TITAN_PBCTOPO_LOGGING_H
#define TITAN_PBCTOPO_LOGGING_H

#include <string>

namespace titan_pbctopo {

void reset_log_file();
void append_log_line(const std::string &message);

} // namespace titan_pbctopo

#endif
