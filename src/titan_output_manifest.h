#ifndef TITAN_OUTPUT_MANIFEST_H
#define TITAN_OUTPUT_MANIFEST_H

#include "arguments.h"
#include "info.h"

#include <fstream>
#include <string>
#include <vector>

namespace titan_output_manifest {

struct OutputManifestRow {
  std::string module;
  std::string filename;
  std::string trajectory;
  std::string units;
  std::string averaging_policy;
  std::string semantic_note;
};

std::vector<OutputManifestRow> build_output_manifest(const arguments &args,
                                                     const info &out_info);

void export_output_manifest(const std::vector<OutputManifestRow> &rows,
                            const std::string &filename,
                            std::fstream *log = nullptr);

void export_run_config_summary(const arguments &args,
                               const std::vector<OutputManifestRow> &rows,
                               const std::string &filename,
                               std::fstream *log = nullptr);

} // namespace titan_output_manifest

#endif // TITAN_OUTPUT_MANIFEST_H
