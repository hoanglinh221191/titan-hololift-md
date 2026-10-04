#include "titan_hotspot_report.h"

#include "arguments.h"
#include "info.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace titan_hotspot {
namespace {

struct ResidueKey {
  std::string role;
  std::string chain;
  int resnr{0};
  std::string resname;

  std::string text() const {
    return role + "\x1f" + chain + "\x1f" + std::to_string(resnr) + "\x1f" +
           resname;
  }
};

struct CasRow {
  ResidueKey key;
  std::string chainnum;
  std::string mutation_status;
  double occupancy{std::numeric_limits<double>::quiet_NaN()};
  double ddg_total_mean{std::numeric_limits<double>::quiet_NaN()};
  double ddg_total_bound_mean{std::numeric_limits<double>::quiet_NaN()};
  double ddg_elec_mean{std::numeric_limits<double>::quiet_NaN()};
  double ddg_vdw_mean{std::numeric_limits<double>::quiet_NaN()};
  double ddg_polar_mean{std::numeric_limits<double>::quiet_NaN()};
  double ddg_nonpolar_mean{std::numeric_limits<double>::quiet_NaN()};
  double avg_distance{std::numeric_limits<double>::quiet_NaN()};
};

struct EDecompRow {
  ResidueKey key;
  std::string source;
  double abs_mean{std::numeric_limits<double>::quiet_NaN()};
  double mean_abs{std::numeric_limits<double>::quiet_NaN()};
  double total_mean{std::numeric_limits<double>::quiet_NaN()};
  double elec_mean{std::numeric_limits<double>::quiet_NaN()};
  double vdw_mean{std::numeric_limits<double>::quiet_NaN()};
  double polar_mean{std::numeric_limits<double>::quiet_NaN()};
  double nonpolar_mean{std::numeric_limits<double>::quiet_NaN()};
};

struct MergedRow {
  ResidueKey key;
  std::string chainnum;
  std::optional<CasRow> cas;
  std::optional<EDecompRow> edecomp;
  double cas_score{std::numeric_limits<double>::quiet_NaN()};
  double edecomp_score{std::numeric_limits<double>::quiet_NaN()};
  double combined_score{std::numeric_limits<double>::quiet_NaN()};
};

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  return value;
}

std::string csv_escape(const std::string &value) {
  if (value.find_first_of(",\"\n\r") == std::string::npos)
    return value;
  std::string out = "\"";
  for (char c : value) {
    if (c == '"')
      out += "\"\"";
    else
      out += c;
  }
  out += '"';
  return out;
}

std::vector<std::string> parse_csv_line(const std::string &line) {
  std::vector<std::string> fields;
  std::string field;
  bool in_quotes = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (in_quotes) {
      if (c == '"' && i + 1 < line.size() && line[i + 1] == '"') {
        field.push_back('"');
        ++i;
      } else if (c == '"') {
        in_quotes = false;
      } else {
        field.push_back(c);
      }
    } else if (c == '"') {
      in_quotes = true;
    } else if (c == ',') {
      fields.push_back(field);
      field.clear();
    } else {
      field.push_back(c);
    }
  }
  fields.push_back(field);
  return fields;
}

std::unordered_map<std::string, std::size_t>
header_map(const std::vector<std::string> &header) {
  std::unordered_map<std::string, std::size_t> out;
  for (std::size_t i = 0; i < header.size(); ++i)
    out[header[i]] = i;
  return out;
}

std::string field(const std::vector<std::string> &row,
                  const std::unordered_map<std::string, std::size_t> &header,
                  const std::string &name) {
  const auto it = header.find(name);
  if (it == header.end() || it->second >= row.size())
    return {};
  return row[it->second];
}

double parse_double(const std::string &value) {
  if (value.empty())
    return std::numeric_limits<double>::quiet_NaN();
  try {
    std::size_t parsed = 0;
    const double out = std::stod(value, &parsed);
    if (parsed != value.size())
      return std::numeric_limits<double>::quiet_NaN();
    return out;
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

int parse_int_or_zero(const std::string &value) {
  try {
    return std::stoi(value);
  } catch (...) {
    return 0;
  }
}

std::string format_double(double value) {
  if (!std::isfinite(value))
    return "nan";
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(10) << value;
  return ss.str();
}

bool source_has_cas(const std::string &source) {
  return source == "cas" || source == "both";
}

bool source_has_edecomp(const std::string &source) {
  return source == "edecomp" || source == "both";
}

bool cas_mutatable(const CasRow &row) {
  return row.mutation_status == "valid_mutation" &&
         std::isfinite(row.ddg_total_bound_mean);
}

std::optional<ResidueKey> parse_edecomp_label(const std::string &role,
                                              const std::string &label) {
  const std::size_t bar = label.find('|');
  const std::string body = bar == std::string::npos ? label : label.substr(bar + 1);
  const std::size_t first = body.find(':');
  const std::size_t second =
      first == std::string::npos ? std::string::npos : body.find(':', first + 1);
  if (first == std::string::npos || second == std::string::npos)
    return std::nullopt;
  ResidueKey key;
  key.role = role.empty() && bar != std::string::npos ? label.substr(0, bar) : role;
  key.chain = body.substr(0, first);
  key.resnr = parse_int_or_zero(body.substr(first + 1, second - first - 1));
  key.resname = body.substr(second + 1);
  return key;
}

std::vector<CasRow> read_cas_file(const std::filesystem::path &path) {
  std::ifstream in(path);
  if (!in.is_open())
    return {};
  std::string line;
  if (!std::getline(in, line))
    return {};
  const auto header = header_map(parse_csv_line(line));
  std::vector<CasRow> rows;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    const auto fields = parse_csv_line(line);
    CasRow row;
    row.key.role = field(fields, header, "role");
    row.chainnum = field(fields, header, "chainnum");
    row.key.chain = field(fields, header, "chain");
    row.key.resnr = parse_int_or_zero(field(fields, header, "resnr"));
    row.key.resname = field(fields, header, "resname");
    row.mutation_status = field(fields, header, "mutation_status");
    row.occupancy = parse_double(field(fields, header, "occupancy_fraction"));
    row.ddg_total_mean = parse_double(field(fields, header, "ddG_total_mean"));
    row.ddg_total_bound_mean =
        parse_double(field(fields, header, "ddG_total_bound_mean"));
    row.ddg_elec_mean = parse_double(field(fields, header, "ddG_elec_mean"));
    row.ddg_vdw_mean = parse_double(field(fields, header, "ddG_vdw_mean"));
    row.ddg_polar_mean = parse_double(field(fields, header, "ddG_polar_mean"));
    row.ddg_nonpolar_mean =
        parse_double(field(fields, header, "ddG_nonpolar_mean"));
    row.avg_distance = parse_double(field(fields, header, "avg_distance_A"));
    if (!row.key.role.empty() && !row.key.chain.empty() && !row.key.resname.empty())
      rows.push_back(row);
  }
  return rows;
}

std::filesystem::path edecomp_filename(const std::filesystem::path &base_dir,
                                       const std::string &source) {
  if (source == "mm")
    return base_dir / "edecomp_mm_residue_ranking.csv";
  if (source == "mmgbsa")
    return base_dir / "edecomp_mmgbsa_residue_ranking.csv";
  return base_dir / "edecomp_mmpbsa_residue_ranking.csv";
}

std::string select_edecomp_source(const std::filesystem::path &base_dir,
                                  const std::string &requested) {
  if (requested == "gbsa")
    return "mmgbsa";
  if (requested == "pbsa")
    return "mmpbsa";
  if (requested == "mmgbsa" || requested == "mmpbsa" || requested == "mm")
    return requested;
  if (requested == "all") {
    for (const std::string &candidate : {"mmpbsa", "mmgbsa", "mm"}) {
      if (std::filesystem::exists(edecomp_filename(base_dir, candidate)))
        return candidate;
    }
  }
  return "mm";
}

std::vector<EDecompRow> read_edecomp_file(const std::filesystem::path &path,
                                          const std::string &source) {
  std::ifstream in(path);
  if (!in.is_open())
    return {};
  std::string line;
  if (!std::getline(in, line))
    return {};
  const auto header = header_map(parse_csv_line(line));
  std::vector<EDecompRow> rows;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    const auto fields = parse_csv_line(line);
    const std::string role = field(fields, header, "molecule_role");
    const std::string label = field(fields, header, "entity_label");
    const auto parsed_key = parse_edecomp_label(role, label);
    if (!parsed_key)
      continue;
    EDecompRow row;
    row.key = *parsed_key;
    row.source = source;
    row.abs_mean = parse_double(field(fields, header, "abs_mean"));
    row.mean_abs = parse_double(field(fields, header, "mean_abs"));
    if (source == "mm") {
      row.total_mean = parse_double(field(fields, header, "signed_mean"));
      row.elec_mean = parse_double(field(fields, header, "elec_signed_mean"));
      row.vdw_mean = parse_double(field(fields, header, "vdw_signed_mean"));
    } else {
      row.total_mean = parse_double(field(fields, header, "DeltaG_no_entropy"));
      row.elec_mean = parse_double(field(fields, header, "E_elec"));
      row.vdw_mean = parse_double(field(fields, header, "E_vdw"));
      row.polar_mean = source == "mmpbsa"
                           ? parse_double(field(fields, header, "DeltaDeltaG_solv_PB"))
                           : parse_double(field(fields, header, "G_polar_GB"));
      row.nonpolar_mean = source == "mmpbsa"
                              ? parse_double(field(fields, header,
                                                   "DeltaG_nonpolar_SASA"))
                              : parse_double(field(fields, header,
                                                   "G_nonpolar_SASA"));
    }
    rows.push_back(row);
  }
  return rows;
}

template <typename Row, typename ValueFn, typename SetFn>
void assign_percentile_scores(const std::vector<Row *> &rows, ValueFn value_fn,
                              SetFn set_fn) {
  std::vector<Row *> valid;
  for (Row *row : rows) {
    const double value = value_fn(*row);
    if (std::isfinite(value))
      valid.push_back(row);
  }
  std::stable_sort(valid.begin(), valid.end(), [&](const Row *lhs, const Row *rhs) {
    const double lv = value_fn(*lhs);
    const double rv = value_fn(*rhs);
    if (lv != rv)
      return lv > rv;
    return lhs->key.text() < rhs->key.text();
  });
  if (valid.empty())
    return;
  for (std::size_t i = 0; i < valid.size(); ++i) {
    const double score =
        valid.size() == 1
            ? 1.0
            : 1.0 - static_cast<double>(i) / static_cast<double>(valid.size() - 1);
    set_fn(*valid[i], score);
  }
}

std::string coverage(const MergedRow &row) {
  if (row.cas && row.edecomp)
    return "matched";
  if (row.cas)
    return "cas_only";
  return "edecomp_only";
}

double selected_score(const MergedRow &row, const std::string &rank_by) {
  if (rank_by == "cas-ddg")
    return row.cas_score;
  if (rank_by == "edecomp-abs-mean")
    return row.edecomp_score;
  return row.combined_score;
}

std::string notes_for_row(const MergedRow &row) {
  std::string note = "dimensionless_rank_score";
  if (row.cas && row.cas->mutation_status != "valid_mutation")
    note += ";cas_" + row.cas->mutation_status;
  if (row.cas && !row.edecomp)
    note += ";missing_edecomp";
  if (row.edecomp && !row.cas)
    note += ";missing_cas";
  return note;
}

std::vector<MergedRow> merge_rows(const std::vector<CasRow> &cas_rows,
                                  const std::vector<EDecompRow> &edecomp_rows) {
  std::map<std::string, MergedRow> by_key;
  for (const CasRow &row : cas_rows) {
    auto &merged = by_key[row.key.text()];
    merged.key = row.key;
    merged.chainnum = row.chainnum;
    merged.cas = row;
  }
  for (const EDecompRow &row : edecomp_rows) {
    auto &merged = by_key[row.key.text()];
    merged.key = row.key;
    merged.edecomp = row;
  }
  std::vector<MergedRow> rows;
  rows.reserve(by_key.size());
  for (auto &[_, row] : by_key)
    rows.push_back(std::move(row));

  std::vector<MergedRow *> cas_scored;
  std::vector<MergedRow *> edecomp_scored;
  for (MergedRow &row : rows) {
    if (row.cas && cas_mutatable(*row.cas))
      cas_scored.push_back(&row);
    if (row.edecomp && std::isfinite(row.edecomp->abs_mean))
      edecomp_scored.push_back(&row);
  }
  assign_percentile_scores(
      cas_scored,
      [](const MergedRow &row) { return std::fabs(row.cas->ddg_total_bound_mean); },
      [](MergedRow &row, double score) { row.cas_score = score; });
  assign_percentile_scores(
      edecomp_scored, [](const MergedRow &row) { return row.edecomp->abs_mean; },
      [](MergedRow &row, double score) { row.edecomp_score = score; });

  for (MergedRow &row : rows) {
    double sum = 0.0;
    int count = 0;
    if (std::isfinite(row.cas_score)) {
      sum += row.cas_score;
      ++count;
    }
    if (std::isfinite(row.edecomp_score)) {
      sum += row.edecomp_score;
      ++count;
    }
    if (count > 0)
      row.combined_score = sum / static_cast<double>(count);
  }
  return rows;
}

void sort_rows(std::vector<MergedRow> &rows, const std::string &rank_by) {
  std::stable_sort(rows.begin(), rows.end(), [&](const auto &lhs,
                                                 const auto &rhs) {
    const double lv = selected_score(lhs, rank_by);
    const double rv = selected_score(rhs, rank_by);
    const bool lf = std::isfinite(lv);
    const bool rf = std::isfinite(rv);
    if (lf != rf)
      return lf;
    if (lf && lv != rv)
      return lv > rv;
    return lhs.key.text() < rhs.key.text();
  });
}

void write_ranking(const std::filesystem::path &path,
                   const std::vector<MergedRow> &rows,
                   const std::string &rank_by) {
  std::ofstream out(path);
  out << "rank,role,chainnum,chain,resnr,resname,source_coverage,"
         "combined_score,cas_score,edecomp_score,rank_by,cas_ddG_total_mean,"
         "cas_ddG_total_bound_mean,cas_ddG_elec_mean,cas_ddG_vdw_mean,"
         "cas_ddG_polar_mean,cas_ddG_nonpolar_mean,cas_occupancy_fraction,"
         "cas_avg_distance_A,mutation_status,edecomp_source,"
         "edecomp_total_mean,edecomp_abs_mean,edecomp_mean_abs,"
         "edecomp_elec_mean,edecomp_vdw_mean,edecomp_polar_mean,"
         "edecomp_nonpolar_mean,notes\n";
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const MergedRow &row = rows[i];
    out << (i + 1) << ',' << row.key.role << ',' << csv_escape(row.chainnum)
        << ',' << csv_escape(row.key.chain) << ',' << row.key.resnr << ','
        << csv_escape(row.key.resname) << ',' << coverage(row) << ','
        << format_double(row.combined_score) << ','
        << format_double(row.cas_score) << ','
        << format_double(row.edecomp_score) << ',' << rank_by << ',';
    if (row.cas) {
      out << format_double(row.cas->ddg_total_mean) << ','
          << format_double(row.cas->ddg_total_bound_mean) << ','
          << format_double(row.cas->ddg_elec_mean) << ','
          << format_double(row.cas->ddg_vdw_mean) << ','
          << format_double(row.cas->ddg_polar_mean) << ','
          << format_double(row.cas->ddg_nonpolar_mean) << ','
          << format_double(row.cas->occupancy) << ','
          << format_double(row.cas->avg_distance) << ','
          << row.cas->mutation_status << ',';
    } else {
      out << "nan,nan,nan,nan,nan,nan,nan,nan,,";
    }
    if (row.edecomp) {
      out << row.edecomp->source << ',' << format_double(row.edecomp->total_mean)
          << ',' << format_double(row.edecomp->abs_mean) << ','
          << format_double(row.edecomp->mean_abs) << ','
          << format_double(row.edecomp->elec_mean) << ','
          << format_double(row.edecomp->vdw_mean) << ','
          << format_double(row.edecomp->polar_mean) << ','
          << format_double(row.edecomp->nonpolar_mean) << ',';
    } else {
      out << ",nan,nan,nan,nan,nan,nan,nan,";
    }
    out << csv_escape(notes_for_row(row)) << '\n';
  }
}

struct PdbKey {
  std::string chain;
  int resnr{0};
  std::string resname;
};

std::string pdb_key_text(const PdbKey &key) {
  return key.chain + "\x1f" + std::to_string(key.resnr) + "\x1f" + key.resname;
}

std::string trim_copy(std::string value) {
  const auto begin = value.find_first_not_of(' ');
  if (begin == std::string::npos)
    return {};
  const auto end = value.find_last_not_of(' ');
  return value.substr(begin, end - begin + 1);
}

std::optional<PdbKey> parse_pdb_key(const std::string &line) {
  if (line.size() < 27 ||
      !(line.rfind("ATOM", 0) == 0 || line.rfind("HETATM", 0) == 0))
    return std::nullopt;
  PdbKey key;
  key.resname = trim_copy(line.substr(17, 3));
  key.chain = trim_copy(line.substr(21, 1));
  key.resnr = parse_int_or_zero(trim_copy(line.substr(22, 4)));
  if (key.chain.empty() || key.resname.empty() || key.resnr == 0)
    return std::nullopt;
  return key;
}

bool write_bfactor_pdb(const std::filesystem::path &template_path,
                       const std::filesystem::path &out_path,
                       const std::vector<MergedRow> &rows,
                       const std::string &rank_by) {
  std::ifstream in(template_path);
  if (!in.is_open())
    return false;
  std::map<std::string, double> score_by_pdb_key;
  for (const MergedRow &row : rows) {
    const double score = selected_score(row, rank_by);
    if (!std::isfinite(score))
      continue;
    PdbKey key{row.key.chain, row.key.resnr, row.key.resname};
    const std::string text = pdb_key_text(key);
    auto it = score_by_pdb_key.find(text);
    if (it == score_by_pdb_key.end() || score > it->second)
      score_by_pdb_key[text] = score;
  }

  std::ofstream out(out_path);
  out << "REMARK TITAN hotspot B-factor stores dimensionless hotspot score "
         "in [0,1], not energy.\n";
  out << "REMARK Scores are rank-consensus diagnostics; inspect "
         "hotspot_residue_ranking.csv for CAS and EDecomp components.\n";
  std::string line;
  while (std::getline(in, line)) {
    if (const auto key = parse_pdb_key(line)) {
      auto it = score_by_pdb_key.find(pdb_key_text(*key));
      if (it != score_by_pdb_key.end()) {
        if (line.size() < 66)
          line.resize(66, ' ');
        std::ostringstream bf;
        bf << std::setw(6) << std::fixed << std::setprecision(2) << it->second;
        line.replace(60, 6, bf.str().substr(0, 6));
      }
    }
    out << line << '\n';
  }
  return true;
}

void write_summary(const std::filesystem::path &path, const ReportConfig &config,
                   const ReportResult &result, bool bfactor_template_exists) {
  std::ofstream out(path);
  out << "method: CAS + EDecomp hotspot report\n";
  out << "source: " << config.source << '\n';
  out << "rank_by: " << config.rank_by << '\n';
  out << "selected_edecomp_source: " << result.selected_edecomp_source << '\n';
  out << "rows_written: " << result.rows_written << '\n';
  out << "cas_rows: " << result.cas_rows << '\n';
  out << "edecomp_rows: " << result.edecomp_rows << '\n';
  out << "matched_rows: " << result.matched_rows << '\n';
  out << "cas_only_rows: " << result.cas_only_rows << '\n';
  out << "edecomp_only_rows: " << result.edecomp_only_rows << '\n';
  out << "coverage_warning: "
      << ((result.cas_only_rows || result.edecomp_only_rows)
              ? "CAS and EDecomp sources cover different residue sets."
              : "none")
      << '\n';
  out << "CAS method: GBSA alanine-like side-chain neutralization scan; no "
         "relaxation, repacking, or explicit mutant topology regeneration.\n";
  out << "EDecomp note: endpoint entity decomposition for MM/GBSA and MM/PBSA "
         "terms; solvation maps are not residue-pair matrices.\n";
  out << "score_note: combined_score is a dimensionless rank-consensus score; "
         "raw CAS and EDecomp components remain in hotspot_residue_ranking.csv.\n";
  out << "bfactor_status: "
      << (result.bfactor_written
              ? "written"
              : (bfactor_template_exists ? "not_written" : "template_unavailable"))
      << '\n';
}

} // namespace

ReportConfig make_config_from_arguments(const arguments &args,
                                        const info &info_output) {
  ReportConfig config;
  config.source = args.hotspot.source;
  config.rank_by = args.hotspot.rank_by;
  config.edecomp_source = args.edecomp.source;
  config.cas_receptor_file = info_output.get_cas_receptor_filename();
  config.cas_ligand_file = info_output.get_cas_ligand_filename();
  config.cas_pdb_template = info_output.get_cas_hotspot_pdb_filename();
  config.ranking_file = info_output.get_hotspot_residue_ranking_filename();
  config.summary_file = info_output.get_hotspot_summary_filename();
  config.bfactor_pdb_file = info_output.get_hotspot_bfactor_pdb_filename();
  return config;
}

ReportResult export_report(const std::filesystem::path &base_dir,
                           ReportConfig config) {
  config.source = lower_copy(config.source);
  config.rank_by = lower_copy(config.rank_by);
  config.edecomp_source = lower_copy(config.edecomp_source);

  std::vector<CasRow> cas_rows;
  if (source_has_cas(config.source)) {
    auto receptor = read_cas_file(base_dir / config.cas_receptor_file);
    auto ligand = read_cas_file(base_dir / config.cas_ligand_file);
    cas_rows.reserve(receptor.size() + ligand.size());
    cas_rows.insert(cas_rows.end(), receptor.begin(), receptor.end());
    cas_rows.insert(cas_rows.end(), ligand.begin(), ligand.end());
  }

  std::vector<EDecompRow> edecomp_rows;
  std::string selected_source;
  if (source_has_edecomp(config.source)) {
    selected_source = select_edecomp_source(base_dir, config.edecomp_source);
    edecomp_rows = read_edecomp_file(edecomp_filename(base_dir, selected_source),
                                     selected_source);
  }

  std::vector<MergedRow> rows = merge_rows(cas_rows, edecomp_rows);
  sort_rows(rows, config.rank_by);

  ReportResult result;
  result.cas_rows = cas_rows.size();
  result.edecomp_rows = edecomp_rows.size();
  result.rows_written = rows.size();
  result.selected_edecomp_source = selected_source;
  for (const MergedRow &row : rows) {
    if (row.cas && row.edecomp)
      ++result.matched_rows;
    else if (row.cas)
      ++result.cas_only_rows;
    else
      ++result.edecomp_only_rows;
  }

  std::filesystem::create_directories(base_dir);
  write_ranking(base_dir / config.ranking_file, rows, config.rank_by);
  const bool bfactor_template_exists =
      std::filesystem::exists(base_dir / config.cas_pdb_template);
  result.bfactor_written =
      write_bfactor_pdb(base_dir / config.cas_pdb_template,
                        base_dir / config.bfactor_pdb_file, rows,
                        config.rank_by);
  write_summary(base_dir / config.summary_file, config, result,
                bfactor_template_exists);
  return result;
}

} // namespace titan_hotspot
