#ifndef TITAN_HOTSPOT_REPORT_H
#define TITAN_HOTSPOT_REPORT_H

#include <cstddef>
#include <filesystem>
#include <string>

struct arguments;
class info;

namespace titan_hotspot {

struct ReportConfig {
  std::string source{"both"};
  std::string rank_by{"combined"};
  std::string edecomp_source{"mm"};
  std::string cas_receptor_file{"cas_receptor.dat"};
  std::string cas_ligand_file{"cas_ligand.dat"};
  std::string cas_pdb_template{"cas_hotspots_b_factor.pdb"};
  std::string ranking_file{"hotspot_residue_ranking.csv"};
  std::string summary_file{"hotspot_summary.dat"};
  std::string bfactor_pdb_file{"hotspot_bfactor.pdb"};
};

struct ReportResult {
  std::size_t cas_rows{0};
  std::size_t edecomp_rows{0};
  std::size_t rows_written{0};
  std::size_t matched_rows{0};
  std::size_t cas_only_rows{0};
  std::size_t edecomp_only_rows{0};
  std::string selected_edecomp_source;
  bool bfactor_written{false};
};

ReportConfig make_config_from_arguments(const arguments &args,
                                        const info &info_output);

ReportResult export_report(const std::filesystem::path &base_dir,
                           ReportConfig config);

} // namespace titan_hotspot

#endif // TITAN_HOTSPOT_REPORT_H
