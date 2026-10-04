#ifndef TITAN_CONFIG_SCHEMA_H
#define TITAN_CONFIG_SCHEMA_H

#include "arguments.h"

#include <fstream>
#include <string>

namespace titan_config_schema {

void export_schema_json(const arguments &args, const std::string &filename,
                        std::fstream *log = nullptr);

} // namespace titan_config_schema

#endif // TITAN_CONFIG_SCHEMA_H
