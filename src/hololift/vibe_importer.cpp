#include "vibe_importer.h"

#include "identity.h"
#include "frame_binding.h"
#include "validation.h"
#include "vibe_adapter.h"
#include "vibe_contract.h"

#include "../pbctopo/observation_layout.h"
#include "../pbctopo/certification.h"
#include "../pbctopo/evidence_audit.h"
#include "../pbctopo/observation_schema.h"
#include "../pbctopo/report.h"
#include "../pbctopo/temporal_path.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace titan_hololift {
namespace {

using ImportError = HoloLiftVibeImportError;

ImportError make_error(HoloLiftVibeImportCode code,
                       const std::filesystem::path &artifact, std::size_t line,
                       std::string message) {
  return {code, artifact, line, std::move(message)};
}

std::expected<std::vector<std::string>, std::string>
parse_csv_fields(std::string_view line) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  bool quote_closed = false;
  for (std::size_t idx = 0; idx < line.size(); ++idx) {
    const char ch = line[idx];
    if (quoted) {
      if (ch == '"') {
        if (idx + 1 < line.size() && line[idx + 1] == '"') {
          field.push_back('"');
          ++idx;
        } else {
          quoted = false;
          quote_closed = true;
        }
      } else {
        field.push_back(ch);
      }
      continue;
    }
    if (quote_closed) {
      if (ch == ',') {
        fields.push_back(std::move(field));
        field.clear();
        quote_closed = false;
        continue;
      }
      return std::unexpected("characters follow a closing CSV quote");
    }
    if (ch == ',') {
      fields.push_back(std::move(field));
      field.clear();
    } else if (ch == '"') {
      if (!field.empty())
        return std::unexpected("CSV quote begins inside an unquoted field");
      quoted = true;
    } else {
      field.push_back(ch);
    }
  }
  if (quoted)
    return std::unexpected("unterminated CSV quote");
  fields.push_back(std::move(field));
  return fields;
}

struct CsvRow {
  std::vector<std::string> fields;
  std::size_t line = 0;
};

class CsvReader {
public:
  CsvReader(CsvReader &&) noexcept = default;
  CsvReader &operator=(CsvReader &&) noexcept = default;
  CsvReader(const CsvReader &) = delete;
  CsvReader &operator=(const CsvReader &) = delete;

  static std::expected<CsvReader, ImportError>
  open(const std::filesystem::path &path,
       std::span<const std::string_view> required_columns) {
    std::error_code exists_error;
    if (!std::filesystem::is_regular_file(path, exists_error)) {
      return std::unexpected(
          make_error(HoloLiftVibeImportCode::MissingArtifact, path, 0,
                     exists_error ? "cannot inspect required VIBE artifact: " +
                                        exists_error.message()
                                  : "missing required VIBE artifact"));
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
      return std::unexpected(make_error(HoloLiftVibeImportCode::IoError, path,
                                        0, "cannot open VIBE artifact"));
    }
    std::string header_line;
    if (!std::getline(stream, header_line)) {
      return std::unexpected(make_error(HoloLiftVibeImportCode::CsvSyntax, path,
                                        1, "missing CSV header"));
    }
    if (!header_line.empty() && header_line.back() == '\r')
      header_line.pop_back();
    if (header_line.size() >= 3 &&
        static_cast<unsigned char>(header_line[0]) == 0xef &&
        static_cast<unsigned char>(header_line[1]) == 0xbb &&
        static_cast<unsigned char>(header_line[2]) == 0xbf) {
      header_line.erase(0, 3);
    }
    auto header = parse_csv_fields(header_line);
    if (!header) {
      return std::unexpected(
          make_error(HoloLiftVibeImportCode::CsvSyntax, path, 1,
                     "invalid CSV header: " + header.error()));
    }
    CsvReader reader(path, std::move(stream), std::move(*header));
    for (std::size_t idx = 0; idx < reader.header_.size(); ++idx) {
      if (reader.header_[idx].empty()) {
        return std::unexpected(
            make_error(HoloLiftVibeImportCode::CsvSyntax, path, 1,
                       "CSV header contains an empty column name"));
      }
      if (!reader.column_by_name_.emplace(reader.header_[idx], idx).second) {
        return std::unexpected(
            make_error(HoloLiftVibeImportCode::DuplicateColumn, path, 1,
                       "duplicate CSV column: " + reader.header_[idx]));
      }
    }
    for (const std::string_view required : required_columns) {
      if (!reader.column_by_name_.contains(std::string(required))) {
        return std::unexpected(make_error(
            HoloLiftVibeImportCode::MissingColumn, path, 1,
            "missing required CSV column: " + std::string(required)));
      }
    }
    return reader;
  }

  std::expected<bool, ImportError> next(CsvRow &row) {
    std::string line;
    if (!std::getline(stream_, line)) {
      if (stream_.bad()) {
        return std::unexpected(make_error(HoloLiftVibeImportCode::IoError,
                                          path_, line_,
                                          "I/O error while reading CSV"));
      }
      return false;
    }
    ++line_;
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (line.empty()) {
      return std::unexpected(make_error(HoloLiftVibeImportCode::CsvSyntax,
                                        path_, line_,
                                        "blank CSV records are not allowed"));
    }
    auto fields = parse_csv_fields(line);
    if (!fields) {
      return std::unexpected(
          make_error(HoloLiftVibeImportCode::CsvSyntax, path_, line_,
                     "invalid CSV record: " + fields.error()));
    }
    if (fields->size() != header_.size()) {
      return std::unexpected(make_error(
          HoloLiftVibeImportCode::CsvSyntax, path_, line_,
          fields->size() > header_.size() ? "CSV record has extra fields"
                                          : "CSV record has missing fields"));
    }
    row.fields = std::move(*fields);
    row.line = line_;
    return true;
  }

  [[nodiscard]] std::string_view get(const CsvRow &row,
                                     std::string_view column) const {
    return row.fields[column_by_name_.at(std::string(column))];
  }

  [[nodiscard]] bool has_column(std::string_view column) const {
    return column_by_name_.contains(std::string(column));
  }

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return path_;
  }

private:
  CsvReader(std::filesystem::path path, std::ifstream stream,
            std::vector<std::string> header)
      : path_(std::move(path)), stream_(std::move(stream)),
        header_(std::move(header)) {}

  std::filesystem::path path_;
  std::ifstream stream_;
  std::vector<std::string> header_;
  std::unordered_map<std::string, std::size_t> column_by_name_;
  std::size_t line_ = 1;
};

constexpr std::array<std::string_view, 13> kCommonColumns{
    "schema_version",
    "producer_version",
    "coordinate_unit",
    "time_unit",
    "endianness",
    "floating_format",
    "identity_hash_algorithm",
    "binding_hash_algorithm",
    "payload_checksum_algorithm",
    "binding_mode",
    "negative_zero_policy",
    "nonfinite_policy",
    "observation_scope"};

std::vector<std::string_view>
with_common(std::initializer_list<std::string_view> extra) {
  std::vector<std::string_view> columns(kCommonColumns.begin(),
                                        kCommonColumns.end());
  columns.insert(columns.end(), extra.begin(), extra.end());
  return columns;
}

class ContractTracker {
public:
  std::expected<void, ImportError> observe(const CsvReader &reader,
                                           const CsvRow &row,
                                           std::string_view expected_scope) {
    auto version = parse_integer<std::uint32_t>(reader, row, "schema_version");
    if (!version)
      return std::unexpected(version.error());
    if (*version != titan_pbctopo::VIBE_OBSERVATION_SCHEMA_VERSION) {
      return std::unexpected(make_error(
          HoloLiftVibeImportCode::ContractMismatch, reader.path(), row.line,
          "unsupported VIBE observation schema version " +
              std::to_string(*version) + "; supported version is " +
              std::to_string(
                  titan_pbctopo::VIBE_OBSERVATION_SCHEMA_VERSION)));
    }
    const std::string producer(reader.get(row, "producer_version"));
    if (producer.empty()) {
      return std::unexpected(
          make_error(HoloLiftVibeImportCode::ContractMismatch, reader.path(),
                     row.line, "VIBE producer_version is empty"));
    }
    if (!producer_version_.empty() && producer != producer_version_) {
      return std::unexpected(
          make_error(HoloLiftVibeImportCode::ContractMismatch, reader.path(),
                     row.line, "mixed producer versions in one VIBE package"));
    }
    if (producer_version_.empty())
      producer_version_ = producer;
    if (reader.get(row, "coordinate_unit") != "angstrom" ||
        reader.get(row, "time_unit") != "ps" ||
        reader.get(row, "endianness") != "not_applicable_text") {
      return std::unexpected(
          make_error(HoloLiftVibeImportCode::ContractMismatch, reader.path(),
                     row.line, "unsupported VIBE units or text endianness"));
    }
    if (reader.get(row, "floating_format") != "IEEE754-binary64" ||
        reader.get(row, "identity_hash_algorithm") != "sha256-128" ||
        reader.get(row, "binding_hash_algorithm") != "sha256-128" ||
        reader.get(row, "payload_checksum_algorithm") != "sha256-256" ||
        reader.get(row, "binding_mode") != "bit_exact" ||
        reader.get(row, "negative_zero_policy") !=
            "canonicalized_to_positive_zero" ||
        reader.get(row, "nonfinite_policy") != "forbidden") {
      return std::unexpected(make_error(
          HoloLiftVibeImportCode::ContractMismatch, reader.path(), row.line,
          "unsupported VIBE floating-point binding contract"));
    }
    if (reader.get(row, "observation_scope") != expected_scope) {
      return std::unexpected(
          make_error(HoloLiftVibeImportCode::ContractMismatch, reader.path(),
                     row.line, "unexpected VIBE observation_scope"));
    }
    return {};
  }

  [[nodiscard]] const std::string &producer_version() const noexcept {
    return producer_version_;
  }

private:
  template <typename Integer>
  static std::expected<Integer, ImportError>
  parse_integer(const CsvReader &reader, const CsvRow &row,
                std::string_view column) {
    const std::string_view text = reader.get(row, column);
    Integer value{};
    const auto parsed =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
      return std::unexpected(make_error(
          HoloLiftVibeImportCode::InvalidValue, reader.path(), row.line,
          "invalid integer in column " + std::string(column)));
    }
    return value;
  }

  std::string producer_version_;
};

template <typename Integer>
std::expected<Integer, ImportError> parse_integer(const CsvReader &reader,
                                                  const CsvRow &row,
                                                  std::string_view column) {
  const std::string_view text = reader.get(row, column);
  Integer value{};
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::InvalidValue, reader.path(), row.line,
        "invalid integer in column " + std::string(column)));
  }
  return value;
}

std::expected<std::uint64_t, ImportError>
parse_optional_u64(const CsvReader &reader, const CsvRow &row,
                   std::string_view column) {
  const std::string_view text = reader.get(row, column);
  if (text.empty() || text == "na")
    return std::uint64_t{0};
  return parse_integer<std::uint64_t>(reader, row, column);
}

std::expected<bool, ImportError> parse_bool(const CsvReader &reader,
                                            const CsvRow &row,
                                            std::string_view column) {
  const std::string_view text = reader.get(row, column);
  if (text == "0")
    return false;
  if (text == "1")
    return true;
  return std::unexpected(
      make_error(HoloLiftVibeImportCode::InvalidValue, reader.path(), row.line,
                 "expected 0 or 1 in column " + std::string(column)));
}

std::expected<std::optional<bool>, ImportError>
parse_optional_bool(const CsvReader &reader, const CsvRow &row,
                    std::string_view column) {
  const std::string_view text = reader.get(row, column);
  if (text == "na" || text.empty())
    return std::optional<bool>{};
  auto value = parse_bool(reader, row, column);
  if (!value)
    return std::unexpected(value.error());
  return std::optional<bool>{*value};
}

std::expected<double, ImportError> parse_double(const CsvReader &reader,
                                                const CsvRow &row,
                                                std::string_view column,
                                                bool allow_na = false) {
  const std::string_view text = reader.get(row, column);
  if (allow_na && (text == "na" || text.empty()))
    return std::numeric_limits<double>::quiet_NaN();
  double value = 0.0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(),
                                      value, std::chars_format::general);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      !std::isfinite(value)) {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::InvalidValue, reader.path(), row.line,
        "invalid floating-point value in column " + std::string(column)));
  }
  return value;
}

std::vector<std::string_view> split_tokens(std::string_view value,
                                           char delimiter) {
  std::vector<std::string_view> tokens;
  std::size_t begin = 0;
  while (begin <= value.size()) {
    const std::size_t end = value.find(delimiter, begin);
    tokens.push_back(value.substr(begin, end == std::string_view::npos
                                             ? value.size() - begin
                                             : end - begin));
    if (end == std::string_view::npos)
      break;
    begin = end + 1;
  }
  return tokens;
}

template <typename Integer>
std::expected<Integer, std::string> parse_token_integer(std::string_view text) {
  Integer value{};
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
    return std::unexpected("invalid integer token");
  }
  return value;
}

std::expected<std::vector<HoloLiftSourceAtomKey>, ImportError>
parse_exact_membership(const CsvReader &reader, const CsvRow &row) {
  const std::string_view serialized = reader.get(row, "exact_atom_membership");
  std::vector<HoloLiftSourceAtomKey> atoms;
  for (const std::string_view token : split_tokens(serialized, '|')) {
    const auto fields = split_tokens(token, ':');
    if (fields.size() != 2 || fields[0].empty() || fields[1].empty()) {
      return std::unexpected(make_error(HoloLiftVibeImportCode::InvalidValue,
                                        reader.path(), row.line,
                                        "invalid exact_atom_membership"));
    }
    auto owner = parse_token_integer<int>(fields[0]);
    auto source_id = parse_token_integer<std::uint64_t>(fields[1]);
    if (!owner || !source_id) {
      return std::unexpected(
          make_error(HoloLiftVibeImportCode::InvalidValue, reader.path(),
                     row.line, "invalid exact_atom_membership integer"));
    }
    atoms.push_back({*owner, *source_id});
  }
  if (atoms.empty() ||
      !std::is_sorted(atoms.begin(), atoms.end(),
                      [](const auto &lhs, const auto &rhs) {
                        if (lhs.owner != rhs.owner)
                          return lhs.owner < rhs.owner;
                        return lhs.source_atom_id < rhs.source_atom_id;
                      }) ||
      std::adjacent_find(atoms.begin(), atoms.end()) != atoms.end()) {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::InvalidValue, reader.path(), row.line,
        "exact_atom_membership is not canonical and unique"));
  }
  return atoms;
}

std::expected<std::vector<std::uint64_t>, ImportError>
parse_source_atom_ids(const CsvReader &reader, const CsvRow &row) {
  const std::string_view serialized = reader.get(row, "sorted_source_atom_ids");
  std::vector<std::uint64_t> ids;
  for (const std::string_view token : split_tokens(serialized, '|')) {
    auto id = parse_token_integer<std::uint64_t>(token);
    if (!id) {
      return std::unexpected(make_error(HoloLiftVibeImportCode::InvalidValue,
                                        reader.path(), row.line,
                                        "invalid sorted_source_atom_ids"));
    }
    ids.push_back(*id);
  }
  if (ids.empty() || !std::is_sorted(ids.begin(), ids.end())) {
    return std::unexpected(make_error(HoloLiftVibeImportCode::InvalidValue,
                                      reader.path(), row.line,
                                      "sorted_source_atom_ids is not sorted"));
  }
  return ids;
}

std::expected<std::vector<double>, ImportError>
parse_exact_atom_masses(const CsvReader &reader, const CsvRow &row) {
  if (!reader.has_column("exact_atom_masses"))
    return std::vector<double>{};
  const std::string_view serialized = reader.get(row, "exact_atom_masses");
  std::vector<double> masses;
  for (const std::string_view token : split_tokens(serialized, '|')) {
    double mass = 0.0;
    const auto parsed = std::from_chars(
        token.data(), token.data() + token.size(), mass,
        std::chars_format::general);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != token.data() + token.size() ||
        !std::isfinite(mass) || mass < 0.0) {
      return std::unexpected(make_error(
          HoloLiftVibeImportCode::InvalidValue, reader.path(), row.line,
          "invalid exact_atom_masses"));
    }
    masses.push_back(mass);
  }
  if (masses.empty()) {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::InvalidValue, reader.path(), row.line,
        "exact_atom_masses is empty"));
  }
  return masses;
}

std::expected<titan_pbctopo::PbctopoHardGraphSource, ImportError>
parse_hard_graph_source_value(const CsvReader &reader, const CsvRow &row,
                              std::string_view column) {
  const std::string_view value = reader.get(row, column);
  if (value == "none")
    return titan_pbctopo::PbctopoHardGraphSource::None;
  if (value == "explicit_topology")
    return titan_pbctopo::PbctopoHardGraphSource::ExplicitTopology;
  if (value == "validated_metadata")
    return titan_pbctopo::PbctopoHardGraphSource::ValidatedMetadata;
  if (value == "mixed_hard")
    return titan_pbctopo::PbctopoHardGraphSource::MixedHard;
  return std::unexpected(make_error(HoloLiftVibeImportCode::InvalidValue,
                                    reader.path(), row.line,
                                    "unknown hard_graph_source"));
}

struct ParsedLayoutRow {
  std::uint64_t topology_epoch_id = 0;
  std::uint64_t layout_signature = 0;
  std::uint64_t component_id = 0;
  std::size_t component_index = 0;
  std::string owner;
  std::uint64_t root_atom_id = 0;
  std::size_t component_atom_count = 0;
  std::vector<std::uint64_t> source_atom_ids;
  std::vector<HoloLiftSourceAtomKey> exact_membership;
  std::vector<double> exact_atom_masses;
  HoloLiftHash128 canonical_atom_universe_hash;
  titan_pbctopo::PbctopoHardGraphSource hard_graph_source =
      titan_pbctopo::PbctopoHardGraphSource::None;
  std::size_t line = 0;
};

std::expected<ParsedLayoutRow, ImportError>
parse_layout_row(const CsvReader &reader, const CsvRow &row) {
  ParsedLayoutRow parsed;
  auto epoch = parse_integer<std::uint64_t>(reader, row, "topology_epoch_id");
  auto layout = parse_integer<std::uint64_t>(reader, row, "layout_signature");
  auto component = parse_integer<std::uint64_t>(reader, row, "component_id");
  auto index = parse_integer<std::size_t>(reader, row, "component_index");
  auto root = parse_integer<std::uint64_t>(reader, row, "root_atom_id");
  auto count = parse_integer<std::size_t>(reader, row, "component_atom_count");
  auto source_ids = parse_source_atom_ids(reader, row);
  auto membership = parse_exact_membership(reader, row);
  auto masses = parse_exact_atom_masses(reader, row);
  auto canonical_atom_universe_hash = parse_hololift_hash128(
      reader.get(row, "canonical_atom_universe_hash"));
  auto hard_graph_source =
      parse_hard_graph_source_value(reader, row, "hard_graph_source");
  if (!epoch)
    return std::unexpected(epoch.error());
  if (!layout)
    return std::unexpected(layout.error());
  if (!component)
    return std::unexpected(component.error());
  if (!index)
    return std::unexpected(index.error());
  if (!root)
    return std::unexpected(root.error());
  if (!count)
    return std::unexpected(count.error());
  if (!source_ids)
    return std::unexpected(source_ids.error());
  if (!membership)
    return std::unexpected(membership.error());
  if (!masses)
    return std::unexpected(masses.error());
  if (!canonical_atom_universe_hash) {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::InvalidValue, reader.path(), row.line,
        canonical_atom_universe_hash.error()));
  }
  if (!hard_graph_source)
    return std::unexpected(hard_graph_source.error());
  parsed.topology_epoch_id = *epoch;
  parsed.layout_signature = *layout;
  parsed.component_id = *component;
  parsed.component_index = *index;
  parsed.owner = std::string(reader.get(row, "owner"));
  parsed.root_atom_id = *root;
  parsed.component_atom_count = *count;
  parsed.source_atom_ids = std::move(*source_ids);
  parsed.exact_membership = std::move(*membership);
  parsed.exact_atom_masses = std::move(*masses);
  parsed.canonical_atom_universe_hash = *canonical_atom_universe_hash;
  parsed.hard_graph_source = *hard_graph_source;
  parsed.line = row.line;
  if (parsed.topology_epoch_id == 0 || parsed.layout_signature == 0 ||
      parsed.component_id == 0 ||
      parsed.canonical_atom_universe_hash.empty()) {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::InvalidValue, reader.path(), row.line,
        "component layout identity metadata is incomplete"));
  }
  return parsed;
}

struct EpochLookup {
  std::uint32_t dense_index = HOLOLIFT_NO_DENSE_INDEX;
  std::uint64_t layout_signature = 0;
  titan_pbctopo::PbctopoHardGraphSource hard_graph_source =
      titan_pbctopo::PbctopoHardGraphSource::None;
  HoloLiftHash128 canonical_atom_universe_hash;
};

struct ParsedHardEdgeRow {
  std::uint64_t topology_epoch_id = 0;
  std::uint64_t layout_signature = 0;
  std::size_t edge_index = 0;
  titan_pbctopo::PbctopoObservationHardEdge edge;
  titan_pbctopo::PbctopoHardGraphSource hard_graph_source =
      titan_pbctopo::PbctopoHardGraphSource::None;
  std::size_t line = 0;
};

std::expected<ParsedHardEdgeRow, ImportError>
parse_hard_edge_row(const CsvReader &reader, const CsvRow &row) {
  ParsedHardEdgeRow parsed;
  auto epoch = parse_integer<std::uint64_t>(reader, row, "topology_epoch_id");
  auto layout = parse_integer<std::uint64_t>(reader, row, "layout_signature");
  auto edge_index = parse_integer<std::size_t>(reader, row, "edge_index");
  auto atom_a = parse_integer<std::uint32_t>(reader, row, "atom_a");
  auto atom_b = parse_integer<std::uint32_t>(reader, row, "atom_b");
  auto hard_graph_source =
      parse_hard_graph_source_value(reader, row, "hard_graph_source");
  if (!epoch)
    return std::unexpected(epoch.error());
  if (!layout)
    return std::unexpected(layout.error());
  if (!edge_index)
    return std::unexpected(edge_index.error());
  if (!atom_a)
    return std::unexpected(atom_a.error());
  if (!atom_b)
    return std::unexpected(atom_b.error());
  if (!hard_graph_source)
    return std::unexpected(hard_graph_source.error());
  const std::string_view kind = reader.get(row, "edge_kind");
  if (kind == "topology_bond") {
    parsed.edge.kind =
        titan_pbctopo::PbctopoObservationHardEdgeKind::TopologyBond;
  } else if (kind == "validated_metadata") {
    parsed.edge.kind =
        titan_pbctopo::PbctopoObservationHardEdgeKind::ValidatedMetadata;
  } else {
    return std::unexpected(make_error(HoloLiftVibeImportCode::InvalidValue,
                                      reader.path(), row.line,
                                      "unknown hard edge kind"));
  }
  parsed.topology_epoch_id = *epoch;
  parsed.layout_signature = *layout;
  parsed.edge_index = *edge_index;
  parsed.edge.atom_a = *atom_a;
  parsed.edge.atom_b = *atom_b;
  parsed.hard_graph_source = *hard_graph_source;
  parsed.line = row.line;
  if (parsed.topology_epoch_id == 0 || parsed.layout_signature == 0 ||
      parsed.edge.atom_a >= parsed.edge.atom_b) {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::InvalidValue, reader.path(), row.line,
        "hard edge identity or canonical orientation is invalid"));
  }
  return parsed;
}

std::expected<std::unordered_map<std::uint64_t, EpochLookup>, ImportError>
import_layout(const HoloLiftVibeImportOptions &options,
              ContractTracker &contract,
              HoloLiftObservationStoreBuilder &builder) {
  const auto path = options.package_directory / "pbctopo_component_layout.csv";
  const auto required = with_common(
      {"topology_epoch_id", "layout_signature", "component_id",
        "component_index", "owner", "root_atom_id", "component_atom_count",
        "sorted_source_atom_ids", "exact_atom_membership",
        "canonical_atom_universe_hash", "hard_graph_source"});
  auto opened = CsvReader::open(path, required);
  if (!opened)
    return std::unexpected(opened.error());
  CsvReader reader = std::move(*opened);
  std::map<std::uint64_t, std::vector<ParsedLayoutRow>> grouped;
  std::vector<std::uint64_t> epoch_order;
  CsvRow row;
  while (true) {
    auto next = reader.next(row);
    if (!next)
      return std::unexpected(next.error());
    if (!*next)
      break;
    const auto observed =
        contract.observe(reader, row, "static_component_membership");
    if (!observed)
      return std::unexpected(observed.error());
    auto parsed = parse_layout_row(reader, row);
    if (!parsed)
      return std::unexpected(parsed.error());
    if (!grouped.contains(parsed->topology_epoch_id))
      epoch_order.push_back(parsed->topology_epoch_id);
    grouped[parsed->topology_epoch_id].push_back(std::move(*parsed));
  }
  if (grouped.empty()) {
    return std::unexpected(
        make_error(HoloLiftVibeImportCode::UnsupportedObservation, path, 0,
                   "component layout contains no topology epochs"));
  }
  if (contract.producer_version().empty()) {
    return std::unexpected(
        make_error(HoloLiftVibeImportCode::ContractMismatch, path, 0,
                   "component layout contains no producer metadata"));
  }
  builder.source_contract = make_hololift_vibe_contract(
      contract.producer_version(), options.package_id, options.trajectory_id);

  const auto hard_edge_path =
      options.package_directory / "pbctopo_hard_edges.csv";
  const auto hard_edge_required =
      with_common({"topology_epoch_id", "layout_signature", "edge_index",
                   "atom_a", "atom_b", "edge_kind",
                   "hard_graph_source"});
  auto opened_hard_edges = CsvReader::open(hard_edge_path, hard_edge_required);
  if (!opened_hard_edges)
    return std::unexpected(opened_hard_edges.error());
  CsvReader hard_edge_reader = std::move(*opened_hard_edges);
  std::map<std::uint64_t, std::vector<ParsedHardEdgeRow>> grouped_hard_edges;
  CsvRow hard_edge_row;
  while (true) {
    auto next = hard_edge_reader.next(hard_edge_row);
    if (!next)
      return std::unexpected(next.error());
    if (!*next)
      break;
    const auto observed =
        contract.observe(hard_edge_reader, hard_edge_row, "static_hard_edges");
    if (!observed)
      return std::unexpected(observed.error());
    auto parsed = parse_hard_edge_row(hard_edge_reader, hard_edge_row);
    if (!parsed)
      return std::unexpected(parsed.error());
    grouped_hard_edges[parsed->topology_epoch_id].push_back(
        std::move(*parsed));
  }
  for (const auto &[epoch_id, _] : grouped_hard_edges) {
    if (!grouped.contains(epoch_id)) {
      return std::unexpected(make_error(
          HoloLiftVibeImportCode::ReferentialIntegrity, hard_edge_path, 0,
          "hard edge references a missing topology epoch"));
    }
  }

  std::unordered_map<std::uint64_t, EpochLookup> epochs;
  epochs.reserve(grouped.size());
  for (const std::uint64_t epoch_id : epoch_order) {
    auto &rows = grouped.at(epoch_id);
    std::sort(rows.begin(), rows.end(), [](const auto &lhs, const auto &rhs) {
      return lhs.component_index < rhs.component_index;
    });
    titan_pbctopo::PbctopoComponentLayoutObservation layout;
    layout.topology_epoch_id = epoch_id;
    layout.layout_signature = rows.front().layout_signature;
    layout.hard_graph_source = rows.front().hard_graph_source;
    layout.canonical_atom_universe_hash = {
        rows.front().canonical_atom_universe_hash.lo,
        rows.front().canonical_atom_universe_hash.hi};
    layout.components.reserve(rows.size());
    for (std::size_t idx = 0; idx < rows.size(); ++idx) {
      const auto &source = rows[idx];
      if (source.component_index != idx ||
          source.layout_signature != layout.layout_signature ||
          source.hard_graph_source != layout.hard_graph_source ||
          source.canonical_atom_universe_hash !=
              rows.front().canonical_atom_universe_hash) {
        return std::unexpected(make_error(
            HoloLiftVibeImportCode::ReferentialIntegrity, path, source.line,
            "component layout epoch rows are inconsistent or non-contiguous"));
      }
      if (idx != 0 && rows[idx - 1].component_id >= source.component_id) {
        return std::unexpected(make_error(
            HoloLiftVibeImportCode::ReferentialIntegrity, path, source.line,
            "component layout is not in canonical component-id order"));
      }
      if (source.component_atom_count != source.exact_membership.size()) {
        return std::unexpected(make_error(
            HoloLiftVibeImportCode::ReferentialIntegrity, path, source.line,
            "component_atom_count does not match membership"));
      }
      if (!source.exact_atom_masses.empty() &&
          source.exact_atom_masses.size() !=
              source.exact_membership.size()) {
        return std::unexpected(make_error(
            HoloLiftVibeImportCode::ReferentialIntegrity, path, source.line,
            "exact_atom_masses does not match membership"));
      }
      std::vector<std::uint64_t> expected_source_ids;
      expected_source_ids.reserve(source.exact_membership.size());
      for (const auto &atom : source.exact_membership)
        expected_source_ids.push_back(atom.source_atom_id);
      std::sort(expected_source_ids.begin(), expected_source_ids.end());
      if (expected_source_ids != source.source_atom_ids) {
        return std::unexpected(make_error(
            HoloLiftVibeImportCode::ReferentialIntegrity, path, source.line,
            "sorted_source_atom_ids does not match exact membership"));
      }
      const bool mixed_owner = std::any_of(
          source.exact_membership.begin(), source.exact_membership.end(),
          [&](const auto &atom) {
            return atom.owner != source.exact_membership.front().owner;
          });
      const std::string expected_owner =
          mixed_owner ? "mixed"
                      : std::to_string(source.exact_membership.front().owner);
      if (source.owner != expected_owner ||
          source.root_atom_id !=
              source.exact_membership.front().source_atom_id) {
        return std::unexpected(make_error(
            HoloLiftVibeImportCode::ReferentialIntegrity, path, source.line,
            "component root or owner is not canonical"));
      }
      auto expected_component_id =
          hololift_component_id(source.exact_membership);
      if (!expected_component_id ||
          *expected_component_id != source.component_id) {
        return std::unexpected(make_error(
            HoloLiftVibeImportCode::ReferentialIntegrity, path, source.line,
            "component identity does not match membership"));
      }

      titan_pbctopo::PbctopoComponentMembership component;
      component.component_id = source.component_id;
      component.owner = source.exact_membership.front().owner;
      component.mixed_owner = mixed_owner;
      component.root_atom_id = static_cast<std::size_t>(source.root_atom_id);
      for (const auto id : source.source_atom_ids) {
        if (id > std::numeric_limits<std::size_t>::max()) {
          return std::unexpected(make_error(
              HoloLiftVibeImportCode::InvalidValue, path, source.line,
              "source atom id exceeds this build's size_t range"));
        }
        component.source_atom_ids.push_back(static_cast<std::size_t>(id));
      }
      for (const auto &atom : source.exact_membership) {
        if (atom.source_atom_id > std::numeric_limits<std::size_t>::max()) {
          return std::unexpected(make_error(
              HoloLiftVibeImportCode::InvalidValue, path, source.line,
              "exact source atom id exceeds this build's size_t range"));
        }
        component.exact_atom_membership.push_back(
            {atom.owner, static_cast<std::size_t>(atom.source_atom_id)});
      }
      component.exact_atom_masses = source.exact_atom_masses;
      layout.components.push_back(std::move(component));
    }
    auto edge_rows_it = grouped_hard_edges.find(epoch_id);
    if (edge_rows_it != grouped_hard_edges.end()) {
      auto &edge_rows = edge_rows_it->second;
      std::sort(edge_rows.begin(), edge_rows.end(), [](const auto &lhs,
                                                       const auto &rhs) {
        return lhs.edge_index < rhs.edge_index;
      });
      layout.hard_edges.reserve(edge_rows.size());
      for (std::size_t idx = 0; idx < edge_rows.size(); ++idx) {
        const auto &source = edge_rows[idx];
        if (source.edge_index != idx ||
            source.layout_signature != layout.layout_signature ||
            source.hard_graph_source != layout.hard_graph_source) {
          return std::unexpected(make_error(
              HoloLiftVibeImportCode::ReferentialIntegrity, hard_edge_path,
              source.line,
              "hard-edge epoch rows are inconsistent or non-contiguous"));
        }
        if (idx != 0) {
          const auto &previous = edge_rows[idx - 1].edge;
          const auto previous_key = std::tuple{
              previous.atom_a, previous.atom_b,
              static_cast<std::uint8_t>(previous.kind)};
          const auto current_key = std::tuple{
              source.edge.atom_a, source.edge.atom_b,
              static_cast<std::uint8_t>(source.edge.kind)};
          if (!(previous_key < current_key)) {
            return std::unexpected(make_error(
                HoloLiftVibeImportCode::ReferentialIntegrity,
                hard_edge_path, source.line,
                "hard edges are not in strict canonical order"));
          }
        }
        layout.hard_edges.push_back(source.edge);
      }
    }
    layout.valid = true;
    auto appended = append_hololift_vibe_topology_epoch(builder, layout);
    if (!appended) {
      return std::unexpected(
          make_error(HoloLiftVibeImportCode::ReferentialIntegrity, path,
                     rows.front().line, appended.error()));
    }
    epochs.emplace(
        epoch_id,
        EpochLookup{*appended, layout.layout_signature,
                    layout.hard_graph_source,
                    rows.front().canonical_atom_universe_hash});
  }
  return epochs;
}

std::expected<titan_pbctopo::PbctopoFrameStatus, ImportError>
parse_status(const CsvReader &reader, const CsvRow &row) {
  const std::string_view value = reader.get(row, "status");
  if (value == "certified")
    return titan_pbctopo::PbctopoFrameStatus::Certified;
  if (value == "weak_observation")
    return titan_pbctopo::PbctopoFrameStatus::WeakObservation;
  if (value == "local_unwrap")
    return titan_pbctopo::PbctopoFrameStatus::LocalUnwrap;
  if (value == "rescued")
    return titan_pbctopo::PbctopoFrameStatus::Rescued;
  if (value == "fallback")
    return titan_pbctopo::PbctopoFrameStatus::Fallback;
  if (value == "contradicted")
    return titan_pbctopo::PbctopoFrameStatus::Contradicted;
  return std::unexpected(make_error(HoloLiftVibeImportCode::InvalidValue,
                                    reader.path(), row.line,
                                    "unknown frame status"));
}

std::expected<titan_pbctopo::PbctopoEvidenceState, ImportError>
parse_evidence_state(const CsvReader &reader, const CsvRow &row) {
  const std::string_view value = reader.get(row, "evidence_state");
  if (value == "not_evaluated")
    return titan_pbctopo::PbctopoEvidenceState::NotEvaluated;
  if (value == "not_applicable")
    return titan_pbctopo::PbctopoEvidenceState::NotApplicable;
  if (value == "supported_connected")
    return titan_pbctopo::PbctopoEvidenceState::SupportedConnected;
  if (value == "weak_disconnected")
    return titan_pbctopo::PbctopoEvidenceState::WeakDisconnected;
  if (value == "contradicted")
    return titan_pbctopo::PbctopoEvidenceState::Contradicted;
  if (value == "weak_ambiguous")
    return titan_pbctopo::PbctopoEvidenceState::WeakAmbiguous;
  return std::unexpected(make_error(HoloLiftVibeImportCode::InvalidValue,
                                    reader.path(), row.line,
                                    "unknown evidence_state"));
}

std::expected<titan_pbctopo::PbctopoCertificateScope, ImportError>
parse_certificate_scope(const CsvReader &reader, const CsvRow &row) {
  const std::string_view value = reader.get(row, "certificate_scope");
  if (value == "none")
    return titan_pbctopo::PbctopoCertificateScope::None;
  if (value == "local_unwrap")
    return titan_pbctopo::PbctopoCertificateScope::LocalUnwrap;
  if (value == "image_intercomponent_steric")
    return titan_pbctopo::PbctopoCertificateScope::ImageIntercomponentSteric;
  if (value == "image_intercomponent_steric_interface")
    return titan_pbctopo::PbctopoCertificateScope::
        ImageIntercomponentStericInterface;
  return std::unexpected(make_error(HoloLiftVibeImportCode::InvalidValue,
                                    reader.path(), row.line,
                                    "unknown certificate_scope"));
}

std::expected<titan_pbctopo::PbctopoCertificateGraphSource, ImportError>
parse_certificate_graph_source(const CsvReader &reader, const CsvRow &row) {
  const std::string_view value = reader.get(row, "certificate_graph_source");
  if (value == "none")
    return titan_pbctopo::PbctopoCertificateGraphSource::None;
  if (value == "gromacs_topology")
    return titan_pbctopo::PbctopoCertificateGraphSource::GromacsTopology;
  if (value == "metadata_chain_residue")
    return titan_pbctopo::PbctopoCertificateGraphSource::MetadataChainResidue;
  if (value == "geometry_cutoff")
    return titan_pbctopo::PbctopoCertificateGraphSource::GeometryCutoff;
  if (value == "mixed")
    return titan_pbctopo::PbctopoCertificateGraphSource::Mixed;
  return std::unexpected(make_error(HoloLiftVibeImportCode::InvalidValue,
                                    reader.path(), row.line,
                                    "unknown certificate_graph_source"));
}

std::expected<titan_pbctopo::PbctopoHardGraphSource, ImportError>
parse_hard_graph_source(const CsvReader &reader, const CsvRow &row) {
  return parse_hard_graph_source_value(reader, row, "hard_graph_source");
}

std::expected<titan_pbctopo::PbctopoSearchEvidenceSource, ImportError>
parse_search_evidence_source(const CsvReader &reader, const CsvRow &row) {
  const std::string_view value = reader.get(row, "search_evidence_source");
  if (value == "none")
    return titan_pbctopo::PbctopoSearchEvidenceSource::None;
  if (value == "geometry_cutoff")
    return titan_pbctopo::PbctopoSearchEvidenceSource::GeometryCutoff;
  if (value == "scored_contacts")
    return titan_pbctopo::PbctopoSearchEvidenceSource::ScoredContacts;
  if (value == "explicit_contacts")
    return titan_pbctopo::PbctopoSearchEvidenceSource::ExplicitContacts;
  if (value == "mixed")
    return titan_pbctopo::PbctopoSearchEvidenceSource::Mixed;
  return std::unexpected(make_error(HoloLiftVibeImportCode::InvalidValue,
                                    reader.path(), row.line,
                                    "unknown search_evidence_source"));
}

std::expected<HoloLiftFeasibleSetSemantics, ImportError>
parse_feasible_set_semantics(const CsvReader &reader, const CsvRow &row) {
  const std::string_view value =
      reader.get(row, "feasible_assignment_set_semantics");
  if (value == "not_enumerated")
    return HoloLiftFeasibleSetSemantics::NotEnumerated;
  if (value == "hard_constraints_within_search_domain") {
    return HoloLiftFeasibleSetSemantics::HardConstraintsWithinSearchDomain;
  }
  return std::unexpected(make_error(
      HoloLiftVibeImportCode::InvalidValue, reader.path(), row.line,
      "unknown feasible_assignment_set_semantics"));
}

std::expected<HoloLiftDomainCompleteness, ImportError>
parse_domain_completeness(const CsvReader &reader, const CsvRow &row) {
  const std::string_view value = reader.get(row, "domain_completeness");
  if (value == "unknown")
    return HoloLiftDomainCompleteness::Unknown;
  if (value == "bounded_exhaustive")
    return HoloLiftDomainCompleteness::BoundedExhaustive;
  if (value == "global_certified") {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::UnsupportedObservation, reader.path(),
        row.line,
        "global_certified is unsupported without an outside-domain "
        "infeasibility proof"));
  }
  return std::unexpected(make_error(
      HoloLiftVibeImportCode::InvalidValue, reader.path(), row.line,
      "unknown domain_completeness"));
}

struct ParsedFrameRow {
  titan_pbctopo::PbctopoFrameReport report;
  std::size_t committed_candidate_count = 0;
  std::uint64_t layout_signature = 0;
  titan_pbctopo::PbctopoHardGraphSource hard_graph_source =
      titan_pbctopo::PbctopoHardGraphSource::None;
  std::size_t trajectory_frame_index = 0;
  std::array<double, 9> box_matrix{};
  HoloLiftHash128 box_hash;
  HoloLiftHash128 atom_selection_hash;
  HoloLiftHash128 canonical_atom_universe_hash;
  HoloLiftHash128 wrapped_coordinate_hash;
  bool frame_binding_valid = false;
  std::uint32_t spatial_lift_policy_version = 0;
  HoloLiftHash128 spatial_lift_identity;
  std::size_t spatial_lift_ambiguous_hard_edges = 0;
  std::size_t spatial_lift_cycle_residuals = 0;
  bool spatial_lift_valid = false;
  HoloLiftFrameEvidence evidence;
  std::size_t line = 0;
};

std::expected<ParsedFrameRow, ImportError>
parse_frame_row(const CsvReader &reader, const CsvRow &row) {
  ParsedFrameRow parsed;
  auto frame = parse_integer<std::size_t>(reader, row, "frame");
  auto time = parse_double(reader, row, "time_ps");
  auto trajectory_frame_index =
      parse_integer<std::size_t>(reader, row, "trajectory_frame_index");
  auto frame_binding_valid = parse_bool(reader, row, "frame_binding_valid");
  auto spatial_lift_policy_version = parse_integer<std::uint32_t>(
      reader, row, "spatial_lift_policy_version");
  auto spatial_lift_ambiguous_hard_edges = parse_integer<std::size_t>(
      reader, row, "spatial_lift_ambiguous_hard_edges");
  auto spatial_lift_cycle_residuals = parse_integer<std::size_t>(
      reader, row, "spatial_lift_cycle_residuals");
  auto spatial_lift_valid = parse_bool(reader, row, "spatial_lift_valid");
  auto status = parse_status(reader, row);
  auto scope = parse_certificate_scope(reader, row);
  auto certificate_graph = parse_certificate_graph_source(reader, row);
  auto hard_graph = parse_hard_graph_source(reader, row);
  auto search_evidence = parse_search_evidence_source(reader, row);
  auto hard_feasible = parse_bool(reader, row, "hard_feasible");
  auto hard_evaluated = parse_bool(reader, row, "hard_evaluated");
  auto hard_certified = parse_bool(reader, row, "hard_certified");
  auto topology_edges_available = parse_integer<std::size_t>(
      reader, row, "topology_edges_available");
  auto metadata_edges_available = parse_integer<std::size_t>(
      reader, row, "metadata_edges_available");
  auto topology_edges_checked = parse_integer<std::size_t>(
      reader, row, "hard_topology_edges_checked");
  auto metadata_edges_checked = parse_integer<std::size_t>(
      reader, row, "hard_metadata_edges_checked");
  auto hard_edge_residuals =
      parse_integer<std::size_t>(reader, row, "hard_edge_residuals");
  auto atom_image_residuals = parse_integer<std::size_t>(
      reader, row, "hard_atom_image_residuals");
  auto assignment_point_residuals = parse_integer<std::size_t>(
      reader, row, "hard_assignment_point_residuals");
  auto topology_cycle_residuals = parse_integer<std::size_t>(
      reader, row, "hard_topology_cycle_residuals");
  auto unsupported_winding_rank = parse_integer<std::size_t>(
      reader, row, "hard_unsupported_winding_rank");
  auto steric_audit_available =
      parse_bool(reader, row, "hard_steric_audit_available");
  auto steric_decision_complete =
      parse_bool(reader, row, "hard_steric_decision_complete");
  auto steric_pair_enumeration_complete =
      parse_bool(reader, row, "hard_steric_pair_enumeration_complete");
  auto steric_scan_complete =
      parse_bool(reader, row, "hard_steric_scan_complete");
  auto hard_clashes =
      parse_integer<std::size_t>(reader, row, "hard_clashes");
  auto metadata_internal_clashes = parse_integer<std::size_t>(
      reader, row, "hard_metadata_internal_clashes");
  auto required_contacts_lost = parse_integer<std::size_t>(
      reader, row, "hard_required_contacts_lost");
  auto evidence_state = parse_evidence_state(reader, row);
  auto evidence_evaluated =
      parse_bool(reader, row, "evidence_consistency_evaluated");
  auto evidence_consistent = parse_bool(reader, row, "evidence_consistent");
  auto evidence_graph_connected =
      parse_bool(reader, row, "evidence_graph_connected");
  auto observed_contacts = parse_integer<std::size_t>(
      reader, row, "soft_observed_contacts_checked");
  auto lost_contacts = parse_integer<std::size_t>(
      reader, row, "soft_observed_contacts_lost");
  auto hypothesis_count = parse_integer<std::size_t>(
      reader, row, "soft_observed_hypothesis_count");
  auto relation_count = parse_integer<std::size_t>(
      reader, row, "soft_observed_component_relation_count");
  auto ambiguous_relation_count = parse_integer<std::size_t>(
      reader, row, "soft_observed_ambiguous_relation_count");
  auto compatible_hypothesis_count = parse_integer<std::size_t>(
      reader, row, "soft_observed_compatible_hypothesis_count");
  auto compatible_contact_count = parse_integer<std::size_t>(
      reader, row, "soft_observed_selected_compatible_contact_count");
  auto alternative_contact_count = parse_integer<std::size_t>(
      reader, row, "soft_observed_alternative_hypothesis_contact_count");
  auto compatible_loss_count = parse_integer<std::size_t>(
      reader, row, "soft_observed_selected_compatible_contact_loss_count");
  auto no_support_relation_count = parse_integer<std::size_t>(
      reader, row, "soft_observed_no_support_relation_count");
  auto selected_relation_has_support = parse_bool(
      reader, row, "soft_observed_selected_relation_has_support");
  auto hypothesis_construction_attempted = parse_bool(
      reader, row, "soft_observed_hypothesis_construction_attempted");
  auto hypothesis_construction_complete = parse_bool(
      reader, row, "soft_observed_hypothesis_construction_complete");
  auto hypothesis_mic_failures = parse_integer<std::size_t>(
      reader, row, "soft_observed_hypothesis_mic_ambiguity_failures");
  auto hypothesis_mapping_failures = parse_integer<std::size_t>(
      reader, row, "soft_observed_hypothesis_component_mapping_failures");
  auto hypothesis_internal_edges = parse_integer<std::size_t>(
      reader, row, "soft_observed_hypothesis_internal_edges_ignored");
  std::expected<double, ImportError> contact_cutoff =
      reader.get(row, "soft_observed_contact_cutoff_A") == "na"
          ? std::expected<double, ImportError>(
                std::numeric_limits<double>::quiet_NaN())
          : parse_double(reader, row, "soft_observed_contact_cutoff_A");
  auto audit_policy_version = parse_integer<std::uint32_t>(
      reader, row, "soft_observed_contact_audit_policy_version");
  auto max_pairs_per_unit_pair = parse_integer<std::size_t>(
      reader, row, "soft_observed_contact_max_pairs_per_unit_pair");
  auto selection_tie_tolerance = parse_double(
      reader, row, "soft_observed_contact_selection_tie_tolerance_d2_A2");
  auto loss_tolerance = parse_double(
      reader, row, "soft_observed_contact_loss_tolerance_d2_A2");
  auto contact_construction_attempted = parse_bool(
      reader, row, "soft_observed_contact_construction_attempted");
  auto contact_construction_complete = parse_bool(
      reader, row, "soft_observed_contact_construction_complete");
  auto contact_cell_setup_failures = parse_integer<std::size_t>(
      reader, row, "soft_observed_contact_cell_setup_failures");
  auto contact_atom_index_failures = parse_integer<std::size_t>(
      reader, row, "soft_observed_contact_atom_index_failures");
  auto contact_mic_query_failures = parse_integer<std::size_t>(
      reader, row, "soft_observed_contact_mic_query_failures");
  auto identified = parse_bool(reader, row, "identified");
  auto uniqueness_exhaustive =
      parse_bool(reader, row, "uniqueness_search_exhaustive");
  auto objective_known = parse_bool(reader, row, "objective_uniqueness_known");
  auto objective_unique =
      parse_bool(reader, row, "objective_unique_within_search_domain");
  auto evidence_known = parse_bool(reader, row, "evidence_uniqueness_known");
  auto evidence_unique =
      parse_bool(reader, row, "evidence_unique_within_search_domain");
  auto feasible_enumerated =
      parse_bool(reader, row, "feasible_assignment_set_enumerated");
  auto feasible_semantics = parse_feasible_set_semantics(reader, row);
  auto feasible_total = parse_integer<std::size_t>(
      reader, row, "feasible_assignments_within_domain");
  auto equivalent_total = parse_integer<std::size_t>(
      reader, row, "equivalent_best_assignments_within_domain");
  auto committed_count =
      parse_integer<std::size_t>(reader, row, "committed_candidate_count");
  auto soft_total = parse_integer<std::size_t>(
      reader, row, "component_soft_score_valid_assignments_within_domain");
  auto epoch = parse_optional_u64(reader, row, "topology_epoch_id");
  auto layout = parse_optional_u64(reader, row, "component_layout_signature");
  auto assignment = parse_optional_u64(reader, row, "assignment_signature");
  auto component_count =
      parse_integer<std::size_t>(reader, row, "assignment_component_count");
  auto temporal_selected = parse_bool(reader, row, "temporal_selected");
  auto temporal_changed =
      parse_bool(reader, row, "temporal_changed_from_greedy");
  auto temporal_assignment =
      parse_optional_u64(reader, row, "temporal_assignment_signature");
  if (!frame)
    return std::unexpected(frame.error());
  if (!time)
    return std::unexpected(time.error());
  if (!trajectory_frame_index)
    return std::unexpected(trajectory_frame_index.error());
  if (!frame_binding_valid)
    return std::unexpected(frame_binding_valid.error());
  if (!spatial_lift_policy_version)
    return std::unexpected(spatial_lift_policy_version.error());
  if (!spatial_lift_ambiguous_hard_edges)
    return std::unexpected(spatial_lift_ambiguous_hard_edges.error());
  if (!spatial_lift_cycle_residuals)
    return std::unexpected(spatial_lift_cycle_residuals.error());
  if (!spatial_lift_valid)
    return std::unexpected(spatial_lift_valid.error());
  if (!status)
    return std::unexpected(status.error());
  if (!scope)
    return std::unexpected(scope.error());
  if (!certificate_graph)
    return std::unexpected(certificate_graph.error());
  if (!hard_graph)
    return std::unexpected(hard_graph.error());
  if (!search_evidence)
    return std::unexpected(search_evidence.error());
  if (!hard_feasible)
    return std::unexpected(hard_feasible.error());
  if (!hard_evaluated)
    return std::unexpected(hard_evaluated.error());
  if (!hard_certified)
    return std::unexpected(hard_certified.error());
  if (!topology_edges_available)
    return std::unexpected(topology_edges_available.error());
  if (!metadata_edges_available)
    return std::unexpected(metadata_edges_available.error());
  if (!topology_edges_checked)
    return std::unexpected(topology_edges_checked.error());
  if (!metadata_edges_checked)
    return std::unexpected(metadata_edges_checked.error());
  if (!hard_edge_residuals)
    return std::unexpected(hard_edge_residuals.error());
  if (!atom_image_residuals)
    return std::unexpected(atom_image_residuals.error());
  if (!assignment_point_residuals)
    return std::unexpected(assignment_point_residuals.error());
  if (!topology_cycle_residuals)
    return std::unexpected(topology_cycle_residuals.error());
  if (!unsupported_winding_rank)
    return std::unexpected(unsupported_winding_rank.error());
  if (!steric_audit_available)
    return std::unexpected(steric_audit_available.error());
  if (!steric_decision_complete)
    return std::unexpected(steric_decision_complete.error());
  if (!steric_pair_enumeration_complete)
    return std::unexpected(steric_pair_enumeration_complete.error());
  if (!steric_scan_complete)
    return std::unexpected(steric_scan_complete.error());
  if (!hard_clashes)
    return std::unexpected(hard_clashes.error());
  if (!metadata_internal_clashes)
    return std::unexpected(metadata_internal_clashes.error());
  if (!required_contacts_lost)
    return std::unexpected(required_contacts_lost.error());
  if (!evidence_state)
    return std::unexpected(evidence_state.error());
  if (!evidence_evaluated)
    return std::unexpected(evidence_evaluated.error());
  if (!evidence_consistent)
    return std::unexpected(evidence_consistent.error());
  if (!evidence_graph_connected)
    return std::unexpected(evidence_graph_connected.error());
  if (!observed_contacts)
    return std::unexpected(observed_contacts.error());
  if (!lost_contacts)
    return std::unexpected(lost_contacts.error());
  if (!contact_cutoff)
    return std::unexpected(contact_cutoff.error());
  if (!audit_policy_version)
    return std::unexpected(audit_policy_version.error());
  if (!max_pairs_per_unit_pair)
    return std::unexpected(max_pairs_per_unit_pair.error());
  if (!selection_tie_tolerance)
    return std::unexpected(selection_tie_tolerance.error());
  if (!loss_tolerance)
    return std::unexpected(loss_tolerance.error());
  if (!contact_construction_attempted)
    return std::unexpected(contact_construction_attempted.error());
  if (!contact_construction_complete)
    return std::unexpected(contact_construction_complete.error());
  if (!contact_cell_setup_failures)
    return std::unexpected(contact_cell_setup_failures.error());
  if (!contact_atom_index_failures)
    return std::unexpected(contact_atom_index_failures.error());
  if (!contact_mic_query_failures)
    return std::unexpected(contact_mic_query_failures.error());
  if (reader.get(row, "soft_observed_contact_hash_algorithm") !=
          titan_pbctopo::PBCTOPO_EOBS_PAIR_HASH_ALGORITHM ||
      reader.get(row, "soft_observed_contact_hash_domain") !=
          titan_pbctopo::PBCTOPO_EOBS_PAIR_HASH_DOMAIN ||
      reader.get(row, "soft_observed_contact_cutoff_policy") !=
          titan_pbctopo::PBCTOPO_EOBS_CUTOFF_POLICY ||
      reader.get(row, "soft_observed_hypothesis_hash_algorithm") !=
          titan_pbctopo::PBCTOPO_EOBS_PAIR_HASH_ALGORITHM ||
      reader.get(row, "soft_observed_hypothesis_hash_domain") !=
          titan_pbctopo::PBCTOPO_EOBS_HYPOTHESIS_HASH_DOMAIN ||
      *audit_policy_version !=
          titan_pbctopo::PBCTOPO_EOBS_AUDIT_POLICY_VERSION ||
      *max_pairs_per_unit_pair !=
          titan_pbctopo::PBCTOPO_EOBS_MAX_PAIRS_PER_UNIT_PAIR ||
      *selection_tie_tolerance !=
          titan_pbctopo::PBCTOPO_EOBS_SELECTION_TIE_TOLERANCE_D2_A2 ||
      *loss_tolerance !=
          titan_pbctopo::PBCTOPO_EOBS_LOSS_TOLERANCE_D2_A2) {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::UnsupportedObservation, reader.path(),
        row.line, "unsupported selected-evidence hash or cutoff policy"));
  }
  if (!identified)
    return std::unexpected(identified.error());
  if (!uniqueness_exhaustive)
    return std::unexpected(uniqueness_exhaustive.error());
  if (!objective_known)
    return std::unexpected(objective_known.error());
  if (!objective_unique)
    return std::unexpected(objective_unique.error());
  if (!evidence_known)
    return std::unexpected(evidence_known.error());
  if (!evidence_unique)
    return std::unexpected(evidence_unique.error());
  if (!feasible_enumerated)
    return std::unexpected(feasible_enumerated.error());
  if (!feasible_semantics)
    return std::unexpected(feasible_semantics.error());
  if (!feasible_total)
    return std::unexpected(feasible_total.error());
  if (!equivalent_total)
    return std::unexpected(equivalent_total.error());
  if (!committed_count)
    return std::unexpected(committed_count.error());
  if (!soft_total)
    return std::unexpected(soft_total.error());
  if (!epoch)
    return std::unexpected(epoch.error());
  if (!layout)
    return std::unexpected(layout.error());
  if (!assignment)
    return std::unexpected(assignment.error());
  if (!component_count)
    return std::unexpected(component_count.error());
  if (!temporal_selected)
    return std::unexpected(temporal_selected.error());
  if (!temporal_changed)
    return std::unexpected(temporal_changed.error());
  if (!temporal_assignment)
    return std::unexpected(temporal_assignment.error());

  const std::string_view policy = reader.get(row, "temporal_policy_status");
  if (policy != "production_framewise_greedy" &&
      policy != "experimental_viterbi_diagnostic") {
    return std::unexpected(make_error(HoloLiftVibeImportCode::InvalidValue,
                                      reader.path(), row.line,
                                      "unknown temporal_policy_status"));
  }
  parsed.report.frame = *frame;
  parsed.report.time_ps = *time;
  parsed.report.status = *status;
  parsed.report.certificate_scope = *scope;
  parsed.report.certificate_graph_source = *certificate_graph;
  parsed.report.hard_graph_source = *hard_graph;
  parsed.report.search_evidence_source = *search_evidence;
  parsed.report.hard_feasible = *hard_feasible;
  parsed.report.evidence_state = *evidence_state;
  parsed.report.evidence_consistency_evaluated = *evidence_evaluated;
  parsed.report.evidence_consistent = *evidence_consistent;
  parsed.report.evidence_graph_connected = *evidence_graph_connected;
  if (*identified != *evidence_graph_connected) {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::ReferentialIntegrity, reader.path(), row.line,
        "legacy identified alias disagrees with evidence_graph_connected"));
  }
  parsed.report.soft_observed_contacts_checked = *observed_contacts;
  parsed.report.soft_observed_contacts_lost = *lost_contacts;
  if (!hypothesis_count) return std::unexpected(hypothesis_count.error());
  if (!relation_count) return std::unexpected(relation_count.error());
  if (!ambiguous_relation_count)
    return std::unexpected(ambiguous_relation_count.error());
  if (!compatible_hypothesis_count)
    return std::unexpected(compatible_hypothesis_count.error());
  if (!compatible_contact_count)
    return std::unexpected(compatible_contact_count.error());
  if (!alternative_contact_count)
    return std::unexpected(alternative_contact_count.error());
  if (!compatible_loss_count)
    return std::unexpected(compatible_loss_count.error());
  if (!no_support_relation_count)
    return std::unexpected(no_support_relation_count.error());
  if (!selected_relation_has_support)
    return std::unexpected(selected_relation_has_support.error());
  if (!hypothesis_construction_attempted)
    return std::unexpected(hypothesis_construction_attempted.error());
  if (!hypothesis_construction_complete)
    return std::unexpected(hypothesis_construction_complete.error());
  if (!hypothesis_mic_failures)
    return std::unexpected(hypothesis_mic_failures.error());
  if (!hypothesis_mapping_failures)
    return std::unexpected(hypothesis_mapping_failures.error());
  if (!hypothesis_internal_edges)
    return std::unexpected(hypothesis_internal_edges.error());
  parsed.report.soft_observed_hypothesis_count = *hypothesis_count;
  parsed.report.soft_observed_component_relation_count = *relation_count;
  parsed.report.soft_observed_ambiguous_relation_count =
      *ambiguous_relation_count;
  parsed.report.soft_observed_compatible_hypothesis_count =
      *compatible_hypothesis_count;
  parsed.report.soft_observed_selected_compatible_contact_count =
      *compatible_contact_count;
  parsed.report.soft_observed_alternative_hypothesis_contact_count =
      *alternative_contact_count;
  parsed.report.soft_observed_selected_compatible_contact_loss_count =
      *compatible_loss_count;
  parsed.report.soft_observed_no_support_relation_count =
      *no_support_relation_count;
  parsed.report.soft_observed_selected_relation_has_support =
      *selected_relation_has_support;
  parsed.report.soft_observed_hypothesis_construction_attempted =
      *hypothesis_construction_attempted;
  parsed.report.soft_observed_hypothesis_construction_complete =
      *hypothesis_construction_complete;
  parsed.report.soft_observed_hypothesis_mic_ambiguity_failures =
      *hypothesis_mic_failures;
  parsed.report.soft_observed_hypothesis_component_mapping_failures =
      *hypothesis_mapping_failures;
  parsed.report.soft_observed_hypothesis_internal_edges_ignored =
      *hypothesis_internal_edges;
  parsed.report.soft_observed_contact_cutoff_A = *contact_cutoff;
  parsed.report.soft_observed_contact_construction_attempted =
      *contact_construction_attempted;
  parsed.report.soft_observed_contact_construction_complete =
      *contact_construction_complete;
  parsed.report.soft_observed_contact_cell_setup_failures =
      *contact_cell_setup_failures;
  parsed.report.soft_observed_contact_atom_index_failures =
      *contact_atom_index_failures;
  parsed.report.soft_observed_contact_mic_query_failures =
      *contact_mic_query_failures;
  const bool evaluated_evidence =
      *evidence_state == titan_pbctopo::PbctopoEvidenceState::SupportedConnected ||
      *evidence_state == titan_pbctopo::PbctopoEvidenceState::WeakDisconnected ||
      *evidence_state == titan_pbctopo::PbctopoEvidenceState::WeakAmbiguous ||
      *evidence_state == titan_pbctopo::PbctopoEvidenceState::Contradicted;
  if (evaluated_evidence &&
      (!*contact_construction_attempted || !*contact_construction_complete ||
       *contact_cell_setup_failures != 0 || *contact_atom_index_failures != 0 ||
       *contact_mic_query_failures != 0)) {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::ReferentialIntegrity, reader.path(), row.line,
        "evaluated evidence depends on incomplete E_obs construction"));
  }
  parsed.report.topology_epoch_id = *epoch;
  parsed.report.component_layout_signature = *layout;
  parsed.report.assignment_signature = *assignment;
  parsed.report.assignment_component_count = *component_count;
  auto &hard = parsed.report.hard_feasibility;
  hard.evaluated = *hard_evaluated;
  hard.assignment_available = *assignment != 0;
  hard.topology_edges_available = *topology_edges_available;
  hard.metadata_edges_available = *metadata_edges_available;
  hard.topology_edges_checked = *topology_edges_checked;
  hard.metadata_edges_checked = *metadata_edges_checked;
  hard.hard_edge_residuals = *hard_edge_residuals;
  hard.atom_image_residuals = *atom_image_residuals;
  hard.assignment_point_residuals = *assignment_point_residuals;
  hard.topology_cycle_residuals = *topology_cycle_residuals;
  hard.unsupported_winding_rank = *unsupported_winding_rank;
  hard.steric_audit_available = *steric_audit_available;
  hard.steric_decision_complete = *steric_decision_complete;
  hard.steric_pair_enumeration_complete =
      *steric_pair_enumeration_complete;
  const auto termination_reason =
      reader.get(row, "hard_steric_termination_reason");
  if (termination_reason == "completed") {
    hard.steric_termination_reason = static_cast<std::uint8_t>(
        titan_pbctopo::PbctopoStericTerminationReason::Completed);
  } else if (termination_reason == "first_hard_clash") {
    hard.steric_termination_reason = static_cast<std::uint8_t>(
        titan_pbctopo::PbctopoStericTerminationReason::FirstHardClash);
  } else if (termination_reason == "invalid_input") {
    hard.steric_termination_reason = static_cast<std::uint8_t>(
        titan_pbctopo::PbctopoStericTerminationReason::InvalidInput);
  } else if (termination_reason == "numerical_failure") {
    hard.steric_termination_reason = static_cast<std::uint8_t>(
        titan_pbctopo::PbctopoStericTerminationReason::NumericalFailure);
  } else {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::InvalidValue, reader.path(), row.line,
        "unknown hard_steric_termination_reason"));
  }
  hard.steric_scan_complete = *steric_scan_complete;
  hard.hard_clashes = *hard_clashes;
  hard.metadata_internal_clashes = *metadata_internal_clashes;
  hard.required_contacts_lost = *required_contacts_lost;
  if (hard.certified() != *hard_certified ||
      *hard_feasible != *hard_certified) {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::ReferentialIntegrity, reader.path(), row.line,
        "hard_feasible, hard_certified, and replayed hard audit disagree"));
  }
  parsed.report.temporal_selected = *temporal_selected;
  parsed.report.temporal_changed_from_greedy = *temporal_changed;
  parsed.report.temporal_assignment_signature = *temporal_assignment;
  parsed.report.temporal_inference_experimental =
      policy == "experimental_viterbi_diagnostic";
  parsed.layout_signature = *layout;
  parsed.hard_graph_source = *hard_graph;
  parsed.trajectory_frame_index = *trajectory_frame_index;
  parsed.frame_binding_valid = *frame_binding_valid;
  parsed.spatial_lift_policy_version = *spatial_lift_policy_version;
  parsed.spatial_lift_ambiguous_hard_edges =
      *spatial_lift_ambiguous_hard_edges;
  parsed.spatial_lift_cycle_residuals = *spatial_lift_cycle_residuals;
  parsed.spatial_lift_valid = *spatial_lift_valid;
  constexpr std::array<std::string_view, 9> box_columns{
      "box_b00", "box_b01", "box_b02", "box_b10", "box_b11",
      "box_b12", "box_b20", "box_b21", "box_b22"};
  for (std::size_t idx = 0; idx < box_columns.size(); ++idx) {
    auto value = parse_double(reader, row, box_columns[idx]);
    if (!value)
      return std::unexpected(value.error());
    parsed.box_matrix[idx] = *value;
  }
  const auto parse_binding_hash = [&](std::string_view column)
      -> std::expected<HoloLiftHash128, std::string> {
    const std::string_view value = reader.get(row, column);
    if (!parsed.frame_binding_valid &&
        value == "00000000000000000000000000000000")
      return HoloLiftHash128{};
    return parse_hololift_hash128(value);
  };
  auto box_hash = parse_binding_hash("box_hash");
  auto selection_hash = parse_binding_hash("atom_selection_hash");
  auto atom_universe_hash =
      parse_binding_hash("canonical_atom_universe_hash");
  auto coordinate_hash = parse_binding_hash("wrapped_coordinate_hash");
  if (!box_hash || !selection_hash || !atom_universe_hash ||
      !coordinate_hash) {
    const std::string message =
        !box_hash       ? box_hash.error()
        : !selection_hash ? selection_hash.error()
        : !atom_universe_hash ? atom_universe_hash.error()
                              : coordinate_hash.error();
    return std::unexpected(make_error(HoloLiftVibeImportCode::InvalidValue,
                                      reader.path(), row.line, message));
  }
  parsed.box_hash = *box_hash;
  parsed.atom_selection_hash = *selection_hash;
  parsed.canonical_atom_universe_hash = *atom_universe_hash;
  parsed.wrapped_coordinate_hash = *coordinate_hash;
  const auto parse_evidence_hash = [&](std::string_view column)
      -> std::expected<HoloLiftHash128, std::string> {
    const std::string_view value = reader.get(row, column);
    if (value == "00000000000000000000000000000000" &&
        (*evidence_state == titan_pbctopo::PbctopoEvidenceState::NotEvaluated ||
         *evidence_state == titan_pbctopo::PbctopoEvidenceState::NotApplicable ||
         ((column == "soft_observed_selected_hypothesis_hash" ||
           column == "soft_observed_selected_compatible_pair_hash") &&
          *compatible_hypothesis_count == 0))) {
      return HoloLiftHash128{};
    }
    return parse_hololift_hash128(value);
  };
  const auto observed_pair_hash =
      parse_evidence_hash("soft_observed_contact_pair_hash");
  const auto lost_pair_hash =
      parse_evidence_hash("soft_observed_lost_pair_hash");
  const auto all_hypotheses_hash =
      parse_evidence_hash("soft_observed_all_hypotheses_hash");
  const auto selected_hypothesis_hash =
      parse_evidence_hash("soft_observed_selected_hypothesis_hash");
  const auto selected_compatible_pair_hash =
      parse_evidence_hash("soft_observed_selected_compatible_pair_hash");
  if (!observed_pair_hash || !lost_pair_hash || !all_hypotheses_hash ||
      !selected_hypothesis_hash || !selected_compatible_pair_hash) {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::InvalidValue, reader.path(), row.line,
        !observed_pair_hash ? observed_pair_hash.error()
        : !lost_pair_hash ? lost_pair_hash.error()
        : !all_hypotheses_hash ? all_hypotheses_hash.error()
        : !selected_hypothesis_hash ? selected_hypothesis_hash.error()
                                    : selected_compatible_pair_hash.error()));
  }
  parsed.report.soft_observed_contact_pair_hash =
      {observed_pair_hash->lo, observed_pair_hash->hi};
  parsed.report.soft_observed_lost_pair_hash =
      {lost_pair_hash->lo, lost_pair_hash->hi};
  parsed.report.soft_observed_all_hypotheses_hash =
      {all_hypotheses_hash->lo, all_hypotheses_hash->hi};
  parsed.report.soft_observed_selected_hypothesis_hash =
      {selected_hypothesis_hash->lo, selected_hypothesis_hash->hi};
  parsed.report.soft_observed_selected_compatible_pair_hash =
      {selected_compatible_pair_hash->lo, selected_compatible_pair_hash->hi};
  const auto expected_evidence_state = [&]() {
    using titan_pbctopo::PbctopoEvidenceState;
    using titan_pbctopo::PbctopoFrameStatus;
    switch (*status) {
    case PbctopoFrameStatus::Certified:
    case PbctopoFrameStatus::Rescued:
      return PbctopoEvidenceState::SupportedConnected;
    case PbctopoFrameStatus::WeakObservation:
      return *evidence_state;
    case PbctopoFrameStatus::Contradicted:
      return PbctopoEvidenceState::Contradicted;
    case PbctopoFrameStatus::Fallback:
      return PbctopoEvidenceState::NotEvaluated;
    case PbctopoFrameStatus::LocalUnwrap:
      return PbctopoEvidenceState::NotApplicable;
    }
    return PbctopoEvidenceState::NotEvaluated;
  }();
  if (*evidence_state != expected_evidence_state ||
      (*status == titan_pbctopo::PbctopoFrameStatus::WeakObservation &&
       *evidence_state != titan_pbctopo::PbctopoEvidenceState::SupportedConnected &&
       *evidence_state != titan_pbctopo::PbctopoEvidenceState::WeakDisconnected &&
       *evidence_state != titan_pbctopo::PbctopoEvidenceState::WeakAmbiguous)) {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::ReferentialIntegrity, reader.path(), row.line,
        "frame status and evidence state disagree"));
  }
  if (evaluated_evidence) {
    const bool graph_state_consistent =
        (*evidence_state !=
             titan_pbctopo::PbctopoEvidenceState::SupportedConnected ||
         *evidence_graph_connected) &&
        (*evidence_state !=
             titan_pbctopo::PbctopoEvidenceState::WeakDisconnected ||
         !*evidence_graph_connected) &&
        (*evidence_state !=
             titan_pbctopo::PbctopoEvidenceState::WeakAmbiguous ||
         *ambiguous_relation_count > 0 || *no_support_relation_count > 0);
    if (!*evidence_evaluated || observed_pair_hash->empty() ||
        lost_pair_hash->empty() || all_hypotheses_hash->empty() ||
        (*compatible_hypothesis_count == 0) !=
            selected_hypothesis_hash->empty() ||
        (*compatible_hypothesis_count == 0) !=
            selected_compatible_pair_hash->empty() ||
        *selected_relation_has_support != (*compatible_hypothesis_count > 0) ||
        *compatible_contact_count > *observed_contacts ||
        *ambiguous_relation_count > *relation_count ||
        !std::isfinite(*contact_cutoff) ||
        *contact_cutoff <= 0.0 || *lost_contacts > *observed_contacts ||
        *evidence_consistent != (*lost_contacts == 0) ||
        (*evidence_state == titan_pbctopo::PbctopoEvidenceState::Contradicted) !=
            (*lost_contacts > 0) ||
        !graph_state_consistent) {
      return std::unexpected(make_error(
          HoloLiftVibeImportCode::ReferentialIntegrity, reader.path(), row.line,
          "evaluated evidence audit fields are incoherent"));
    }
  } else {
    const bool selected_payload_empty =
        !*evidence_evaluated && !*evidence_consistent &&
        lost_pair_hash->empty() && selected_hypothesis_hash->empty() &&
        selected_compatible_pair_hash->empty() && *hypothesis_count == 0 &&
        *relation_count == 0 && *ambiguous_relation_count == 0 &&
        *compatible_hypothesis_count == 0 && *compatible_contact_count == 0 &&
        *alternative_contact_count == 0 && *compatible_loss_count == 0 &&
        *no_support_relation_count == 0 && !*selected_relation_has_support &&
        *lost_contacts == 0;
    const bool raw_diagnostics_coherent =
        (*observed_contacts == 0 || !observed_pair_hash->empty()) &&
        (observed_pair_hash->empty() || *contact_construction_attempted) &&
        (!std::isfinite(*contact_cutoff) || *contact_construction_attempted);
    const bool local_unwrap_payload_empty =
        *observed_contacts == 0 && observed_pair_hash->empty() &&
        all_hypotheses_hash->empty() && !*contact_construction_attempted &&
        !*contact_construction_complete &&
        !*hypothesis_construction_attempted &&
        !*hypothesis_construction_complete &&
        !std::isfinite(*contact_cutoff);
    if (!selected_payload_empty || !raw_diagnostics_coherent ||
        (*evidence_state ==
             titan_pbctopo::PbctopoEvidenceState::NotApplicable &&
         !local_unwrap_payload_empty)) {
      return std::unexpected(make_error(
          HoloLiftVibeImportCode::ReferentialIntegrity, reader.path(), row.line,
          "non-evaluated evidence carries incoherent selected or raw audit data"));
    }
  }
  const std::string_view spatial_lift_hash_text =
      reader.get(row, "spatial_lift_identity");
  if (!parsed.spatial_lift_valid &&
      spatial_lift_hash_text == "00000000000000000000000000000000") {
    parsed.spatial_lift_identity = {};
  } else {
    auto spatial_lift_hash =
        parse_hololift_hash128(spatial_lift_hash_text);
    if (!spatial_lift_hash) {
      return std::unexpected(make_error(HoloLiftVibeImportCode::InvalidValue,
                                        reader.path(), row.line,
                                        spatial_lift_hash.error()));
    }
    parsed.spatial_lift_identity = *spatial_lift_hash;
  }
  parsed.evidence.identified = *identified;
  parsed.evidence.uniqueness_search_exhaustive = *uniqueness_exhaustive;
  parsed.evidence.objective_uniqueness_known = *objective_known;
  parsed.evidence.objective_unique_within_search_domain = *objective_unique;
  parsed.evidence.evidence_uniqueness_known = *evidence_known;
  parsed.evidence.evidence_unique_within_search_domain = *evidence_unique;
  parsed.evidence.feasible_assignment_set_enumerated = *feasible_enumerated;
  parsed.evidence.feasible_set_semantics = *feasible_semantics;
  if ((*feasible_enumerated) !=
      (*feasible_semantics ==
       HoloLiftFeasibleSetSemantics::HardConstraintsWithinSearchDomain)) {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::ReferentialIntegrity, reader.path(), row.line,
        "feasible-set enumeration flag and semantics disagree"));
  }
  parsed.evidence.feasible_assignments_within_domain = *feasible_total;
  parsed.evidence.equivalent_best_assignments_within_domain = *equivalent_total;
  parsed.committed_candidate_count = *committed_count;
  parsed.evidence.soft_score_valid_assignments_within_domain = *soft_total;
  parsed.line = row.line;
  return parsed;
}

std::expected<std::vector<HoloLiftComponentImage>, ImportError>
parse_component_assignment(const CsvReader &reader, const CsvRow &row) {
  const std::string_view serialized = reader.get(row, "component_assignment");
  if (serialized.empty() || serialized == "na") {
    return std::unexpected(
        make_error(HoloLiftVibeImportCode::InvalidValue, reader.path(),
                   row.line, "retained candidate has no component_assignment"));
  }
  std::vector<HoloLiftComponentImage> images;
  for (const std::string_view token : split_tokens(serialized, '|')) {
    const auto fields = split_tokens(token, ':');
    if (fields.size() != 4) {
      return std::unexpected(make_error(HoloLiftVibeImportCode::InvalidValue,
                                        reader.path(), row.line,
                                        "invalid component_assignment"));
    }
    auto component_id = parse_token_integer<std::uint64_t>(fields[0]);
    auto x = parse_token_integer<std::int64_t>(fields[1]);
    auto y = parse_token_integer<std::int64_t>(fields[2]);
    auto z = parse_token_integer<std::int64_t>(fields[3]);
    if (!component_id || !x || !y || !z || *component_id == 0) {
      return std::unexpected(
          make_error(HoloLiftVibeImportCode::InvalidValue, reader.path(),
                     row.line, "invalid component_assignment integer"));
    }
    if (!images.empty() && images.back().component_id >= *component_id) {
      return std::unexpected(make_error(
          HoloLiftVibeImportCode::InvalidValue, reader.path(), row.line,
          "component_assignment is not in canonical component order"));
    }
    images.push_back({*component_id, {*x, *y, *z}});
  }
  return images;
}

struct ParsedCandidate {
  std::size_t frame = 0;
  double time_ps = 0.0;
  std::uint64_t topology_epoch_id = 0;
  std::size_t component_count = 0;
  std::string candidate_set_scope;
  std::array<double, 6> normalized_gram{};
  HoloLiftCandidateRecord record;
  HoloLiftFrameEvidence frame_evidence;
  HoloLiftFrameSearchDomain frame_search_domain;
  std::vector<HoloLiftComponentImage> raw_images;
  std::size_t line = 0;
};

std::expected<ParsedCandidate, ImportError>
parse_candidate_row(const CsvReader &reader, const CsvRow &row) {
  ParsedCandidate parsed;
  auto frame = parse_integer<std::size_t>(reader, row, "frame");
  auto time = parse_double(reader, row, "time_ps");
  auto epoch = parse_integer<std::uint64_t>(reader, row, "topology_epoch_id");
  auto rank = parse_integer<std::size_t>(reader, row, "candidate_rank");
  auto selected = parse_bool(reader, row, "selected_framewise");
  auto observation_class = parse_hololift_vibe_observation_class(
      reader.get(row, "observation_class"));
  auto equivalence_known = parse_bool(reader, row, "search_equivalence_known");
  auto equivalent =
      parse_optional_bool(reader, row, "equivalent_to_search_best");
  auto equivalent_output =
      parse_bool(reader, row, "equivalent_under_output_order");
  auto hard_feasible = parse_bool(reader, row, "hard_feasible");
  auto layout = parse_integer<std::uint64_t>(reader, row, "layout_signature");
  auto assignment =
      parse_integer<std::uint64_t>(reader, row, "assignment_signature");
  auto relation_signature = parse_integer<std::uint64_t>(
      reader, row, "evidence_relative_relation_signature");
  const auto parse_candidate_hash = [&](std::string_view column)
      -> std::expected<HoloLiftHash128, ImportError> {
    const auto value = reader.get(row, column);
    if (value == "00000000000000000000000000000000")
      return HoloLiftHash128{};
    const auto parsed_hash = parse_hololift_hash128(value);
    if (!parsed_hash) {
      return std::unexpected(make_error(HoloLiftVibeImportCode::InvalidValue,
                                        reader.path(), row.line,
                                        parsed_hash.error()));
    }
    return *parsed_hash;
  };
  auto compatible_hypothesis_hash =
      parse_candidate_hash("evidence_compatible_hypothesis_hash");
  auto compatible_pair_hash =
      parse_candidate_hash("evidence_compatible_pair_hash");
  auto compatible_hypothesis_count = parse_integer<std::size_t>(
      reader, row, "evidence_compatible_hypothesis_count");
  auto supported_relation_count = parse_integer<std::size_t>(
      reader, row, "evidence_supported_relation_count");
  auto compatible_contact_count = parse_integer<std::size_t>(
      reader, row, "evidence_compatible_contact_count");
  auto no_support_relation_count = parse_integer<std::size_t>(
      reader, row, "evidence_no_support_relation_count");
  auto component_count =
      parse_integer<std::size_t>(reader, row, "component_count");
  auto images = parse_component_assignment(reader, row);
  if (!frame)
    return std::unexpected(frame.error());
  if (!time)
    return std::unexpected(time.error());
  if (!epoch)
    return std::unexpected(epoch.error());
  if (!rank)
    return std::unexpected(rank.error());
  if (!selected)
    return std::unexpected(selected.error());
  if (!observation_class) {
    return std::unexpected(make_error(HoloLiftVibeImportCode::InvalidValue,
                                      reader.path(), row.line,
                                      observation_class.error()));
  }
  if (!equivalence_known)
    return std::unexpected(equivalence_known.error());
  if (!equivalent)
    return std::unexpected(equivalent.error());
  if (!equivalent_output)
    return std::unexpected(equivalent_output.error());
  if (!hard_feasible)
    return std::unexpected(hard_feasible.error());
  if (!layout)
    return std::unexpected(layout.error());
  if (!assignment)
    return std::unexpected(assignment.error());
  if (!relation_signature)
    return std::unexpected(relation_signature.error());
  if (!compatible_hypothesis_hash)
    return std::unexpected(compatible_hypothesis_hash.error());
  if (!compatible_pair_hash)
    return std::unexpected(compatible_pair_hash.error());
  if (!compatible_hypothesis_count)
    return std::unexpected(compatible_hypothesis_count.error());
  if (!supported_relation_count)
    return std::unexpected(supported_relation_count.error());
  if (!compatible_contact_count)
    return std::unexpected(compatible_contact_count.error());
  if (!no_support_relation_count)
    return std::unexpected(no_support_relation_count.error());
  if (!component_count)
    return std::unexpected(component_count.error());
  if (!images)
    return std::unexpected(images.error());
  if (*equivalence_known != equivalent->has_value()) {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::ReferentialIntegrity, reader.path(), row.line,
        "equivalent_to_search_best known/na state is inconsistent"));
  }

  parsed.frame = *frame;
  parsed.time_ps = *time;
  parsed.topology_epoch_id = *epoch;
  parsed.component_count = *component_count;
  parsed.candidate_set_scope =
      std::string(reader.get(row, "candidate_set_scope"));
  parsed.record.rank = *rank;
  parsed.record.layout_signature = *layout;
  parsed.record.assignment_signature = *assignment;
  parsed.record.evidence_relative_relation_signature = *relation_signature;
  parsed.record.evidence_compatible_hypothesis_hash =
      *compatible_hypothesis_hash;
  parsed.record.evidence_compatible_pair_hash = *compatible_pair_hash;
  parsed.record.evidence_compatible_hypothesis_count =
      *compatible_hypothesis_count;
  parsed.record.evidence_supported_relation_count =
      *supported_relation_count;
  parsed.record.evidence_compatible_contact_count = *compatible_contact_count;
  parsed.record.evidence_no_support_relation_count =
      *no_support_relation_count;
  parsed.record.observation_class = *observation_class;
  parsed.record.evidence.selected_framewise = *selected;
  parsed.record.evidence.hard_feasible = *hard_feasible;
  parsed.record.evidence.search_equivalence_known = *equivalence_known;
  parsed.record.evidence.equivalent_to_search_best =
      equivalent->value_or(false);
  parsed.record.evidence.equivalent_under_output_order = *equivalent_output;
  parsed.raw_images = std::move(*images);
  parsed.line = row.line;
  if (parsed.topology_epoch_id == 0 || parsed.record.layout_signature == 0 ||
      parsed.record.assignment_signature == 0 ||
      parsed.record.evidence_relative_relation_signature !=
          parsed.record.assignment_signature ||
      ((parsed.record.evidence_compatible_hypothesis_count == 0) !=
       parsed.record.evidence_compatible_hypothesis_hash.empty()) ||
      ((parsed.record.evidence_compatible_hypothesis_count == 0) !=
       parsed.record.evidence_compatible_pair_hash.empty()) ||
      ((parsed.record.evidence_compatible_hypothesis_count == 0) !=
       (parsed.record.evidence_compatible_contact_count == 0)) ||
      ((parsed.record.evidence_compatible_hypothesis_count == 0) !=
       (parsed.record.evidence_supported_relation_count == 0)) ||
      parsed.component_count != parsed.raw_images.size() ||
      parsed.candidate_set_scope != "retained_certified_band") {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::ReferentialIntegrity, reader.path(), row.line,
        "retained candidate identity metadata is inconsistent"));
  }

  auto shell =
      parse_integer<std::size_t>(reader, row, "component_search_shell_radius");
  auto audit_known = parse_bool(reader, row, "search_domain_audit_known");
  auto evidence_complete =
      parse_bool(reader, row, "search_domain_evidence_complete");
  auto touches = parse_optional_bool(
      reader, row, "selected_assignment_touches_shell_boundary");
  auto expansion_attempted =
      parse_bool(reader, row, "domain_expansion_attempted");
  auto expansion_exhausted =
      parse_bool(reader, row, "domain_expansion_exhausted");
  auto bounded = parse_bool(reader, row, "bounded_domain_only");
  auto domain_completeness = parse_domain_completeness(reader, row);
  if (!shell)
    return std::unexpected(shell.error());
  if (!audit_known)
    return std::unexpected(audit_known.error());
  if (!evidence_complete)
    return std::unexpected(evidence_complete.error());
  if (!touches)
    return std::unexpected(touches.error());
  if (!expansion_attempted)
    return std::unexpected(expansion_attempted.error());
  if (!expansion_exhausted)
    return std::unexpected(expansion_exhausted.error());
  if (!bounded)
    return std::unexpected(bounded.error());
  if (!domain_completeness)
    return std::unexpected(domain_completeness.error());
  if (*audit_known != touches->has_value()) {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::ReferentialIntegrity, reader.path(), row.line,
        "selected shell-boundary known/na state is inconsistent"));
  }
  parsed.frame_search_domain.shell_radius = *shell;
  parsed.frame_search_domain.audit_known = *audit_known;
  parsed.frame_search_domain.evidence_complete = *evidence_complete;
  parsed.frame_search_domain.expansion_attempted = *expansion_attempted;
  parsed.frame_search_domain.expansion_exhausted = *expansion_exhausted;
  parsed.frame_search_domain.bounded_domain_only = *bounded;
  parsed.frame_search_domain.completeness = *domain_completeness;
  parsed.record.domain_evidence.boundary_known = touches->has_value();
  parsed.record.domain_evidence.touches_shell_boundary =
      touches->value_or(false);

  auto identified = parse_bool(reader, row, "identified");
  auto exhaustive = parse_bool(reader, row, "uniqueness_search_exhaustive");
  auto objective_known = parse_bool(reader, row, "objective_uniqueness_known");
  auto objective_unique =
      parse_bool(reader, row, "objective_unique_within_search_domain");
  auto evidence_known = parse_bool(reader, row, "evidence_uniqueness_known");
  auto evidence_unique =
      parse_bool(reader, row, "evidence_unique_within_search_domain");
  auto feasible_enumerated =
      parse_bool(reader, row, "feasible_assignment_set_enumerated");
  auto feasible_semantics = parse_feasible_set_semantics(reader, row);
  auto equivalent_complete = parse_bool(reader, row, "equivalent_set_complete");
  auto credible_complete =
      parse_bool(reader, row, "credible_alternative_set_complete");
  auto retained = parse_integer<std::size_t>(
      reader, row, "retained_certified_assignment_count");
  auto equivalent_exported = parse_integer<std::size_t>(
      reader, row, "search_equivalent_assignments_exported");
  auto equivalent_total = parse_integer<std::size_t>(
      reader, row, "equivalent_best_assignments_within_domain");
  auto feasible_total = parse_integer<std::size_t>(
      reader, row, "feasible_assignments_within_domain");
  auto soft_total = parse_integer<std::size_t>(
      reader, row, "soft_score_valid_assignments_within_domain");
  auto domain_exported = parse_integer<std::size_t>(
      reader, row, "search_domain_assignments_exported");
  if (!identified)
    return std::unexpected(identified.error());
  if (!exhaustive)
    return std::unexpected(exhaustive.error());
  if (!objective_known)
    return std::unexpected(objective_known.error());
  if (!objective_unique)
    return std::unexpected(objective_unique.error());
  if (!evidence_known)
    return std::unexpected(evidence_known.error());
  if (!evidence_unique)
    return std::unexpected(evidence_unique.error());
  if (!feasible_enumerated)
    return std::unexpected(feasible_enumerated.error());
  if (!feasible_semantics)
    return std::unexpected(feasible_semantics.error());
  if (!equivalent_complete)
    return std::unexpected(equivalent_complete.error());
  if (!credible_complete)
    return std::unexpected(credible_complete.error());
  if (!retained)
    return std::unexpected(retained.error());
  if (!equivalent_exported)
    return std::unexpected(equivalent_exported.error());
  if (!equivalent_total)
    return std::unexpected(equivalent_total.error());
  if (!feasible_total)
    return std::unexpected(feasible_total.error());
  if (!soft_total)
    return std::unexpected(soft_total.error());
  if (!domain_exported)
    return std::unexpected(domain_exported.error());
  parsed.frame_evidence.identified = *identified;
  parsed.frame_evidence.uniqueness_search_exhaustive = *exhaustive;
  parsed.frame_evidence.objective_uniqueness_known = *objective_known;
  parsed.frame_evidence.objective_unique_within_search_domain =
      *objective_unique;
  parsed.frame_evidence.evidence_uniqueness_known = *evidence_known;
  parsed.frame_evidence.evidence_unique_within_search_domain = *evidence_unique;
  parsed.frame_evidence.feasible_assignment_set_enumerated =
      *feasible_enumerated;
  parsed.frame_evidence.feasible_set_semantics = *feasible_semantics;
  if ((*feasible_enumerated) !=
      (*feasible_semantics ==
       HoloLiftFeasibleSetSemantics::HardConstraintsWithinSearchDomain)) {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::ReferentialIntegrity, reader.path(), row.line,
        "feasible-set enumeration flag and semantics disagree"));
  }
  parsed.frame_evidence.equivalent_set_complete = *equivalent_complete;
  parsed.frame_evidence.credible_alternative_set_complete = *credible_complete;
  parsed.frame_evidence.retained_certified_assignment_count = *retained;
  parsed.frame_evidence.search_equivalent_assignments_exported =
      *equivalent_exported;
  parsed.frame_evidence.equivalent_best_assignments_within_domain =
      *equivalent_total;
  parsed.frame_evidence.feasible_assignments_within_domain = *feasible_total;
  parsed.frame_evidence.soft_score_valid_assignments_within_domain =
      *soft_total;
  parsed.frame_evidence.search_domain_assignments_exported = *domain_exported;

  constexpr std::array<std::string_view, 6> gram_columns{
      "normalized_gram_g00", "normalized_gram_g01", "normalized_gram_g02",
      "normalized_gram_g11", "normalized_gram_g12", "normalized_gram_g22"};
  for (std::size_t idx = 0; idx < gram_columns.size(); ++idx) {
    auto value = parse_double(reader, row, gram_columns[idx]);
    if (!value)
      return std::unexpected(value.error());
    parsed.normalized_gram[idx] = *value;
  }

  auto broken = parse_integer<std::size_t>(reader, row, "broken_edges");
  auto continuity_max = parse_double(reader, row, "continuity_max_d2", true);
  auto continuity_path = parse_double(reader, row, "continuity_path", true);
  auto anchor_max = parse_double(reader, row, "anchor_max_d2", true);
  auto anchor_mst = parse_double(reader, row, "anchor_mst2", true);
  auto min_pair = parse_double(reader, row, "min_pair_d2", true);
  auto shift_norm = parse_double(reader, row, "shift_norm2", true);
  if (!broken)
    return std::unexpected(broken.error());
  if (!continuity_max)
    return std::unexpected(continuity_max.error());
  if (!continuity_path)
    return std::unexpected(continuity_path.error());
  if (!anchor_max)
    return std::unexpected(anchor_max.error());
  if (!anchor_mst)
    return std::unexpected(anchor_mst.error());
  if (!min_pair)
    return std::unexpected(min_pair.error());
  if (!shift_norm)
    return std::unexpected(shift_norm.error());
  parsed.record.vibe_score_summary = {
      *broken,     *continuity_max, *continuity_path, *anchor_max,
      *anchor_mst, *min_pair,       *shift_norm};

  for (const std::string_view gauge :
       {"global_gauge_x", "global_gauge_y", "global_gauge_z"}) {
    auto value = parse_integer<std::int64_t>(reader, row, gauge);
    if (!value)
      return std::unexpected(value.error());
  }
  return parsed;
}

bool same_frame_evidence(const HoloLiftFrameEvidence &lhs,
                         const HoloLiftFrameEvidence &rhs) {
  return lhs.identified == rhs.identified &&
         lhs.uniqueness_search_exhaustive == rhs.uniqueness_search_exhaustive &&
         lhs.objective_uniqueness_known == rhs.objective_uniqueness_known &&
         lhs.objective_unique_within_search_domain ==
             rhs.objective_unique_within_search_domain &&
         lhs.evidence_uniqueness_known == rhs.evidence_uniqueness_known &&
         lhs.evidence_unique_within_search_domain ==
             rhs.evidence_unique_within_search_domain &&
         lhs.feasible_assignment_set_enumerated ==
             rhs.feasible_assignment_set_enumerated &&
         lhs.feasible_set_semantics == rhs.feasible_set_semantics &&
         lhs.equivalent_set_complete == rhs.equivalent_set_complete &&
         lhs.credible_alternative_set_complete ==
             rhs.credible_alternative_set_complete &&
         lhs.retained_certified_assignment_count ==
             rhs.retained_certified_assignment_count &&
         lhs.search_equivalent_assignments_exported ==
             rhs.search_equivalent_assignments_exported &&
         lhs.equivalent_best_assignments_within_domain ==
             rhs.equivalent_best_assignments_within_domain &&
         lhs.feasible_assignments_within_domain ==
             rhs.feasible_assignments_within_domain &&
         lhs.soft_score_valid_assignments_within_domain ==
             rhs.soft_score_valid_assignments_within_domain &&
         lhs.search_domain_assignments_exported ==
             rhs.search_domain_assignments_exported;
}

bool frame_report_evidence_matches(const HoloLiftFrameEvidence &report,
                                   const HoloLiftFrameEvidence &candidate) {
  return report.identified == candidate.identified &&
         report.uniqueness_search_exhaustive ==
             candidate.uniqueness_search_exhaustive &&
         report.objective_uniqueness_known ==
             candidate.objective_uniqueness_known &&
         report.objective_unique_within_search_domain ==
             candidate.objective_unique_within_search_domain &&
         report.evidence_uniqueness_known ==
             candidate.evidence_uniqueness_known &&
         report.evidence_unique_within_search_domain ==
             candidate.evidence_unique_within_search_domain &&
         report.feasible_assignment_set_enumerated ==
             candidate.feasible_assignment_set_enumerated &&
         report.feasible_set_semantics == candidate.feasible_set_semantics &&
         report.feasible_assignments_within_domain ==
             candidate.feasible_assignments_within_domain &&
         report.equivalent_best_assignments_within_domain ==
             candidate.equivalent_best_assignments_within_domain &&
         report.soft_score_valid_assignments_within_domain ==
             candidate.soft_score_valid_assignments_within_domain;
}

bool nearly_equal(double lhs, double rhs) {
  const double scale = std::max({1.0, std::fabs(lhs), std::fabs(rhs)});
  return std::fabs(lhs - rhs) <= 1.0e-10 * scale;
}

bool certificate_graph_matches_hard_graph(
    titan_pbctopo::PbctopoCertificateGraphSource certificate,
    titan_pbctopo::PbctopoHardGraphSource hard_graph) {
  using Certificate = titan_pbctopo::PbctopoCertificateGraphSource;
  using Hard = titan_pbctopo::PbctopoHardGraphSource;
  switch (hard_graph) {
  case Hard::ExplicitTopology:
    return certificate == Certificate::GromacsTopology;
  case Hard::ValidatedMetadata:
    return certificate == Certificate::MetadataChainResidue;
  case Hard::MixedHard:
    return certificate == Certificate::Mixed;
  case Hard::None:
    return certificate == Certificate::None;
  }
  return false;
}

std::expected<void, ImportError>
validate_temporal_path(const HoloLiftVibeImportOptions &options,
                       ContractTracker &contract,
                       HoloLiftObservationStoreBuilder &builder,
                       std::span<const std::size_t> experimental_frames) {
  const auto path = options.package_directory / "pbctopo_temporal_path.csv";
  std::error_code exists_error;
  const bool exists = std::filesystem::exists(path, exists_error);
  if (exists_error) {
    return std::unexpected(
        make_error(HoloLiftVibeImportCode::IoError, path, 0,
                   "cannot inspect temporal path: " + exists_error.message()));
  }
  if (exists && !std::filesystem::is_regular_file(path, exists_error)) {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::IoError, path, 0,
        exists_error ? "cannot inspect temporal path: " + exists_error.message()
                     : "temporal path is not a regular file"));
  }
  if (experimental_frames.empty() && !exists)
    return {};
  if (!experimental_frames.empty() &&
      !options.accept_experimental_temporal_diagnostics) {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::UnsupportedObservation, path, 0,
        "experimental temporal diagnostics are disabled by import options"));
  }
  if (!exists) {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::MissingArtifact, path, 0,
        "experimental frame reports require pbctopo_temporal_path.csv"));
  }

  const auto required = with_common(
      {"temporal_policy_status", "frame", "time_ps", "selected_rank",
       "assignment_signature", "changed_from_greedy", "replay_applied"});
  auto opened = CsvReader::open(path, required);
  if (!opened)
    return std::unexpected(opened.error());
  CsvReader reader = std::move(*opened);
  CsvRow row;
  std::size_t temporal_index = 0;
  while (true) {
    auto next = reader.next(row);
    if (!next)
      return std::unexpected(next.error());
    if (!*next)
      break;
    const auto observed =
        contract.observe(reader, row, "experimental_temporal_path");
    if (!observed)
      return std::unexpected(observed.error());
    if (reader.get(row, "temporal_policy_status") !=
        "experimental_viterbi_diagnostic") {
      return std::unexpected(
          make_error(HoloLiftVibeImportCode::ContractMismatch, path, row.line,
                     "temporal path is not marked experimental diagnostic"));
    }
    if (temporal_index >= experimental_frames.size()) {
      return std::unexpected(
          make_error(HoloLiftVibeImportCode::ReferentialIntegrity, path,
                     row.line, "temporal path contains an extra frame"));
    }
    auto source_frame = parse_integer<std::size_t>(reader, row, "frame");
    auto time = parse_double(reader, row, "time_ps");
    auto rank = parse_integer<std::size_t>(reader, row, "selected_rank");
    auto signature =
        parse_integer<std::uint64_t>(reader, row, "assignment_signature");
    auto changed = parse_bool(reader, row, "changed_from_greedy");
    auto replay = parse_bool(reader, row, "replay_applied");
    if (!source_frame)
      return std::unexpected(source_frame.error());
    if (!time)
      return std::unexpected(time.error());
    if (!rank)
      return std::unexpected(rank.error());
    if (!signature)
      return std::unexpected(signature.error());
    if (!changed)
      return std::unexpected(changed.error());
    if (!replay)
      return std::unexpected(replay.error());

    const auto &frame = builder.frames[experimental_frames[temporal_index]];
    if (*source_frame != frame.frame || !nearly_equal(*time, frame.time_ps) ||
        *rank == 0 || *rank > frame.candidates.count) {
      return std::unexpected(make_error(
          HoloLiftVibeImportCode::ReferentialIntegrity, path, row.line,
          "temporal path frame, time, or rank does not match its frame"));
    }
    const auto &temporal_candidate =
        builder.candidates[frame.candidates.begin + *rank - 1];
    const auto &framewise_candidate =
        builder.candidates[frame.selected_candidate_index];
    if (*signature != temporal_candidate.assignment_signature ||
        *signature != frame.provenance.temporal_assignment_signature ||
        *changed != (*signature != framewise_candidate.assignment_signature) ||
        (*replay &&
         frame.provenance.output_assignment_signature != *signature)) {
      return std::unexpected(make_error(
          HoloLiftVibeImportCode::ReferentialIntegrity, path, row.line,
          "temporal path signature provenance is inconsistent"));
    }
    ++temporal_index;
  }
  if (temporal_index != experimental_frames.size()) {
    return std::unexpected(
        make_error(HoloLiftVibeImportCode::ReferentialIntegrity, path, 0,
                   "temporal path does not cover every experimental frame"));
  }
  return {};
}

std::expected<void, ImportError> import_frames_and_candidates(
    const HoloLiftVibeImportOptions &options, ContractTracker &contract,
    const std::unordered_map<std::uint64_t, EpochLookup> &epochs,
    HoloLiftObservationStoreBuilder &builder, HoloLiftVibeImportStats &stats) {
  const auto frame_path =
      options.package_directory / "pbctopo_frame_report.csv";
  const auto frame_required =
      with_common({"temporal_policy_status",
                   "frame",
                   "time_ps",
                   "trajectory_frame_index",
                    "frame_binding_valid",
                    "spatial_lift_policy_version",
                    "spatial_lift_identity",
                    "spatial_lift_ambiguous_hard_edges",
                    "spatial_lift_cycle_residuals",
                    "spatial_lift_valid",
                   "box_b00",
                   "box_b01",
                   "box_b02",
                   "box_b10",
                   "box_b11",
                   "box_b12",
                   "box_b20",
                   "box_b21",
                   "box_b22",
                    "box_hash",
                    "atom_selection_hash",
                    "canonical_atom_universe_hash",
                    "wrapped_coordinate_hash",
                   "status",
                   "certificate_scope",
                   "certificate_graph_source",
                   "hard_graph_source",
                   "search_evidence_source",
                   "hard_feasible",
                   "hard_evaluated",
                   "hard_certified",
                   "topology_edges_available",
                   "metadata_edges_available",
                   "hard_topology_edges_checked",
                   "hard_metadata_edges_checked",
                   "hard_edge_residuals",
                   "hard_atom_image_residuals",
                   "hard_assignment_point_residuals",
                   "hard_topology_cycle_residuals",
                   "hard_unsupported_winding_rank",
                   "hard_steric_audit_available",
                   "hard_steric_decision_complete",
                   "hard_steric_pair_enumeration_complete",
                   "hard_steric_termination_reason",
                   "hard_steric_scan_complete",
                   "hard_clashes",
                   "hard_metadata_internal_clashes",
                   "hard_required_contacts_lost",
                   "evidence_state",
                   "evidence_consistency_evaluated",
                   "evidence_consistent",
                   "evidence_graph_connected",
                   "soft_observed_contact_pair_hash",
                   "soft_observed_lost_pair_hash",
                   "soft_observed_all_hypotheses_hash",
                   "soft_observed_selected_hypothesis_hash",
                   "soft_observed_hypothesis_hash_algorithm",
                   "soft_observed_hypothesis_hash_domain",
                   "soft_observed_selected_compatible_pair_hash",
                   "soft_observed_hypothesis_count",
                   "soft_observed_component_relation_count",
                   "soft_observed_ambiguous_relation_count",
                   "soft_observed_compatible_hypothesis_count",
                   "soft_observed_selected_compatible_contact_count",
                   "soft_observed_alternative_hypothesis_contact_count",
                   "soft_observed_selected_compatible_contact_loss_count",
                   "soft_observed_no_support_relation_count",
                   "soft_observed_selected_relation_has_support",
                   "soft_observed_hypothesis_construction_attempted",
                   "soft_observed_hypothesis_construction_complete",
                   "soft_observed_hypothesis_mic_ambiguity_failures",
                   "soft_observed_hypothesis_component_mapping_failures",
                   "soft_observed_hypothesis_internal_edges_ignored",
                   "soft_observed_contact_hash_algorithm",
                   "soft_observed_contact_hash_domain",
                   "soft_observed_contact_cutoff_policy",
                   "soft_observed_contact_audit_policy_version",
                   "soft_observed_contact_max_pairs_per_unit_pair",
                   "soft_observed_contact_selection_tie_tolerance_d2_A2",
                   "soft_observed_contact_loss_tolerance_d2_A2",
                   "soft_observed_contacts_checked",
                   "soft_observed_contacts_lost",
                   "soft_observed_contact_cutoff_A",
                   "soft_observed_contact_construction_attempted",
                   "soft_observed_contact_construction_complete",
                   "soft_observed_contact_cell_setup_failures",
                   "soft_observed_contact_atom_index_failures",
                   "soft_observed_contact_mic_query_failures",
                   "identified",
                   "uniqueness_search_exhaustive",
                   "objective_uniqueness_known",
                   "objective_unique_within_search_domain",
                   "evidence_uniqueness_known",
                   "evidence_unique_within_search_domain",
                   "feasible_assignment_set_enumerated",
                   "feasible_assignment_set_semantics",
                   "feasible_assignments_within_domain",
                   "equivalent_best_assignments_within_domain",
                   "committed_candidate_count",
                   "component_soft_score_valid_assignments_within_domain",
                   "topology_epoch_id",
                   "component_layout_signature",
                   "assignment_signature",
                   "assignment_component_count",
                   "temporal_selected",
                   "temporal_changed_from_greedy",
                   "temporal_assignment_signature"});
  auto opened_frames = CsvReader::open(frame_path, frame_required);
  if (!opened_frames)
    return std::unexpected(opened_frames.error());
  CsvReader frame_reader = std::move(*opened_frames);

  const auto candidate_path =
      options.package_directory / "pbctopo_assignment_candidates.csv";
  const auto candidate_required =
      with_common({"topology_epoch_id",
                   "frame",
                   "time_ps",
                   "candidate_rank",
                   "selected_framewise",
                   "candidate_set_scope",
                   "observation_class",
                   "search_equivalence_known",
                   "equivalent_to_search_best",
                   "equivalent_under_output_order",
                   "equivalent_set_complete",
                   "retained_certified_assignment_count",
                   "search_equivalent_assignments_exported",
                   "equivalent_best_assignments_within_domain",
                   "identified",
                   "unique_within_search_domain",
                   "uniqueness_search_exhaustive",
                   "objective_uniqueness_known",
                   "objective_unique_within_search_domain",
                   "evidence_uniqueness_known",
                   "evidence_unique_within_search_domain",
                   "feasible_assignment_set_enumerated",
                   "feasible_assignment_set_semantics",
                   "feasible_assignments_within_domain",
                   "soft_score_valid_assignments_within_domain",
                   "search_domain_assignments_exported",
                   "credible_alternative_set_complete",
                   "component_search_shell_radius",
                   "search_domain_audit_known",
                   "search_domain_evidence_complete",
                   "selected_assignment_touches_shell_boundary",
                   "domain_expansion_attempted",
                   "domain_expansion_exhausted",
                   "bounded_domain_only",
                   "domain_completeness",
                   "hard_feasible",
                   "layout_signature",
                   "assignment_signature",
                   "evidence_relative_relation_signature",
                   "evidence_compatible_hypothesis_hash",
                   "evidence_compatible_pair_hash",
                   "evidence_compatible_hypothesis_count",
                   "evidence_supported_relation_count",
                   "evidence_compatible_contact_count",
                   "evidence_no_support_relation_count",
                   "component_count",
                   "global_gauge_x",
                   "global_gauge_y",
                   "global_gauge_z",
                   "component_assignment",
                   "normalized_gram_g00",
                   "normalized_gram_g01",
                   "normalized_gram_g02",
                   "normalized_gram_g11",
                   "normalized_gram_g12",
                   "normalized_gram_g22",
                   "broken_edges",
                   "continuity_max_d2",
                   "continuity_path",
                   "anchor_max_d2",
                   "anchor_mst2",
                   "min_pair_d2",
                   "shift_norm2"});
  auto opened_candidates = CsvReader::open(candidate_path, candidate_required);
  if (!opened_candidates)
    return std::unexpected(opened_candidates.error());
  CsvReader candidate_reader = std::move(*opened_candidates);

  std::optional<ParsedCandidate> pending_candidate;
  auto read_candidate = [&]() -> std::expected<void, ImportError> {
    CsvRow row;
    auto next = candidate_reader.next(row);
    if (!next)
      return std::unexpected(next.error());
    if (!*next) {
      pending_candidate.reset();
      return {};
    }
    const auto observed =
        contract.observe(candidate_reader, row, "framewise_bounded_domain");
    if (!observed)
      return std::unexpected(observed.error());
    auto parsed = parse_candidate_row(candidate_reader, row);
    if (!parsed)
      return std::unexpected(parsed.error());
    pending_candidate = std::move(*parsed);
    ++stats.candidate_rows;
    return {};
  };
  auto initial_candidate = read_candidate();
  if (!initial_candidate)
    return initial_candidate;

  bool have_previous_source_frame = false;
  std::size_t previous_source_frame = 0;
  double previous_source_time = 0.0;
  std::vector<std::size_t> experimental_frames;
  CsvRow frame_csv_row;
  while (true) {
    auto next = frame_reader.next(frame_csv_row);
    if (!next)
      return std::unexpected(next.error());
    if (!*next)
      break;
    ++stats.source_frame_rows;
    const auto observed = contract.observe(frame_reader, frame_csv_row,
                                           "framewise_bounded_domain");
    if (!observed)
      return std::unexpected(observed.error());
    auto parsed_frame = parse_frame_row(frame_reader, frame_csv_row);
    if (!parsed_frame)
      return std::unexpected(parsed_frame.error());
    if (have_previous_source_frame &&
        (parsed_frame->report.frame <= previous_source_frame ||
         parsed_frame->report.time_ps < previous_source_time)) {
      return std::unexpected(
          make_error(HoloLiftVibeImportCode::ReferentialIntegrity, frame_path,
                     parsed_frame->line,
                     "source frame ids/times are not strictly ordered"));
    }
    have_previous_source_frame = true;
    previous_source_frame = parsed_frame->report.frame;
    previous_source_time = parsed_frame->report.time_ps;

    if (pending_candidate &&
        pending_candidate->frame < parsed_frame->report.frame) {
      return std::unexpected(
          make_error(HoloLiftVibeImportCode::ReferentialIntegrity,
                     candidate_path, pending_candidate->line,
                     "candidate references a missing earlier frame report"));
    }
    std::vector<ParsedCandidate> frame_candidates;
    while (pending_candidate &&
           pending_candidate->frame == parsed_frame->report.frame) {
      frame_candidates.push_back(std::move(*pending_candidate));
      auto read = read_candidate();
      if (!read)
        return read;
    }

    const bool importable = parsed_frame->report.status ==
                                titan_pbctopo::PbctopoFrameStatus::Certified ||
                            parsed_frame->report.status ==
                                titan_pbctopo::PbctopoFrameStatus::Rescued ||
                            parsed_frame->report.status ==
                                titan_pbctopo::PbctopoFrameStatus::
                                    WeakObservation;
    const bool contradicted =
        parsed_frame->report.status ==
        titan_pbctopo::PbctopoFrameStatus::Contradicted;
    if ((importable && parsed_frame->committed_candidate_count != 1) ||
        (contradicted && parsed_frame->committed_candidate_count != 0)) {
      return std::unexpected(make_error(
          HoloLiftVibeImportCode::ReferentialIntegrity, frame_path,
          parsed_frame->line,
          "frame status and committed candidate count are inconsistent"));
    }
    if (!importable && !contradicted) {
      if (!frame_candidates.empty()) {
        return std::unexpected(
            make_error(HoloLiftVibeImportCode::ReferentialIntegrity,
                       candidate_path, frame_candidates.front().line,
                       "uncertified frame has retained certified candidates"));
      }
      if (parsed_frame->report.status ==
          titan_pbctopo::PbctopoFrameStatus::LocalUnwrap) {
        ++stats.skipped_local_unwrap_frames;
      } else {
        ++stats.skipped_fallback_frames;
      }
      continue;
    }
    if (!parsed_frame->report.hard_feasible ||
        (importable && frame_candidates.empty()) ||
        (contradicted && !frame_candidates.empty())) {
      return std::unexpected(make_error(
          HoloLiftVibeImportCode::ReferentialIntegrity, frame_path,
          parsed_frame->line,
          "hard-certified frame evidence state disagrees with candidate band"));
    }
    if (!parsed_frame->frame_binding_valid) {
      return std::unexpected(make_error(
          HoloLiftVibeImportCode::ReferentialIntegrity, frame_path,
          parsed_frame->line,
          "hard-certified frame lacks a trajectory binding"));
    }
    if (!certificate_graph_matches_hard_graph(
            parsed_frame->report.certificate_graph_source,
            parsed_frame->report.hard_graph_source)) {
      return std::unexpected(make_error(
          HoloLiftVibeImportCode::ReferentialIntegrity, frame_path,
          parsed_frame->line,
          "certificate graph source does not match the hard graph source"));
    }
    const auto epoch_it = epochs.find(parsed_frame->report.topology_epoch_id);
    if (epoch_it == epochs.end()) {
      return std::unexpected(make_error(
          HoloLiftVibeImportCode::ReferentialIntegrity, frame_path,
          parsed_frame->line, "frame references a missing topology epoch"));
    }
    const EpochLookup &epoch = epoch_it->second;
    if (parsed_frame->layout_signature != epoch.layout_signature ||
        parsed_frame->hard_graph_source != epoch.hard_graph_source ||
        parsed_frame->canonical_atom_universe_hash !=
            epoch.canonical_atom_universe_hash) {
      return std::unexpected(
          make_error(HoloLiftVibeImportCode::ReferentialIntegrity, frame_path,
                     parsed_frame->line,
                     "frame layout/hard graph does not match topology epoch"));
    }
    const auto &epoch_record = builder.topology_epochs[epoch.dense_index];
    const auto components =
        std::span<const HoloLiftComponentRecord>(builder.components)
            .subspan(epoch_record.components.begin,
                     epoch_record.components.count);
    if (parsed_frame->report.assignment_component_count != components.size()) {
      return std::unexpected(make_error(
          HoloLiftVibeImportCode::ReferentialIntegrity, frame_path,
          parsed_frame->line,
          "frame assignment component count does not match topology epoch"));
    }

    HoloLiftFrameRecord frame;
    frame.frame = parsed_frame->report.frame;
    frame.time_ps = parsed_frame->report.time_ps;
    frame.source_sequence_index = stats.source_frame_rows - 1;
    frame.trajectory_frame_index = parsed_frame->trajectory_frame_index;
    frame.box_matrix = parsed_frame->box_matrix;
    frame.box_hash = parsed_frame->box_hash;
    frame.atom_selection_hash = parsed_frame->atom_selection_hash;
    frame.canonical_atom_universe_hash =
        parsed_frame->canonical_atom_universe_hash;
    frame.wrapped_coordinate_hash = parsed_frame->wrapped_coordinate_hash;
    frame.frame_binding_valid = parsed_frame->frame_binding_valid;
    frame.spatial_lift_policy_version =
        parsed_frame->spatial_lift_policy_version;
    frame.spatial_lift_identity = parsed_frame->spatial_lift_identity;
    frame.spatial_lift_ambiguous_hard_edges =
        parsed_frame->spatial_lift_ambiguous_hard_edges;
    frame.spatial_lift_cycle_residuals =
        parsed_frame->spatial_lift_cycle_residuals;
    frame.spatial_lift_valid = parsed_frame->spatial_lift_valid;
    frame.topology_epoch_index = epoch.dense_index;
    frame.topology_epoch_id = parsed_frame->report.topology_epoch_id;
    frame.layout_signature = epoch.layout_signature;
    frame.candidate_set_scope =
        importable ? HoloLiftCandidateSetScope::RetainedCertifiedBand
                   : HoloLiftCandidateSetScope::Unknown;

    HoloLiftVibeFrameAppendInput transaction;
    transaction.candidates.reserve(frame_candidates.size());
    if (contradicted)
      frame.evidence = parsed_frame->evidence;
    if (contradicted) {
      frame.evidence.retained_certified_assignment_count = 0;
      frame.evidence.search_equivalent_assignments_exported = 0;
      frame.evidence.search_domain_assignments_exported = 0;
      frame.evidence.equivalent_set_complete = false;
      frame.evidence.credible_alternative_set_complete = false;
      frame.normalized_gram =
          titan_pbctopo::pbctopo_temporal_normalized_gram(frame.box_matrix);
    }
    std::size_t selected_count = 0;
    std::uint64_t framewise_signature = 0;
    for (std::size_t idx = 0; idx < frame_candidates.size(); ++idx) {
      auto &source = frame_candidates[idx];
      if (source.record.rank != idx + 1 ||
          source.topology_epoch_id != frame.topology_epoch_id ||
          source.record.layout_signature != frame.layout_signature ||
          source.component_count != components.size() ||
          !nearly_equal(source.time_ps, frame.time_ps) ||
          source.candidate_set_scope != "retained_certified_band") {
        return std::unexpected(
            make_error(HoloLiftVibeImportCode::ReferentialIntegrity,
                       candidate_path, source.line,
                       "candidate rank/frame/layout metadata is inconsistent"));
      }
      if (idx == 0) {
        frame.normalized_gram = source.normalized_gram;
        frame.evidence = source.frame_evidence;
        frame.search_domain = source.frame_search_domain;
      } else if (source.normalized_gram != frame.normalized_gram ||
                  !same_frame_evidence(source.frame_evidence, frame.evidence) ||
                  source.frame_search_domain != frame.search_domain) {
        return std::unexpected(make_error(
            HoloLiftVibeImportCode::ReferentialIntegrity, candidate_path,
            source.line,
            "candidate frame-constant evidence/domain is inconsistent"));
      }
      if (source.record.evidence.selected_framewise) {
        ++selected_count;
        framewise_signature = source.record.assignment_signature;
      }
      transaction.candidates.push_back(
          {std::move(source.record), std::move(source.raw_images)});
    }
    frame.candidates = {0, transaction.candidates.size()};
    if (importable && (selected_count != 1 ||
        frame.evidence.retained_certified_assignment_count !=
            frame.candidates.count ||
        !frame_report_evidence_matches(parsed_frame->evidence,
                                       frame.evidence))) {
      return std::unexpected(make_error(
          HoloLiftVibeImportCode::ReferentialIntegrity, candidate_path,
          frame_candidates.front().line,
          "candidate band selection or frame evidence is inconsistent"));
    }
    const std::uint64_t provenance_framewise_signature =
        contradicted ? parsed_frame->report.assignment_signature
                     : framewise_signature;
    auto provenance = hololift_vibe_frame_provenance(
        parsed_frame->report, provenance_framewise_signature);
    if (!provenance) {
      return std::unexpected(
          make_error(HoloLiftVibeImportCode::ReferentialIntegrity, frame_path,
                     parsed_frame->line, provenance.error()));
    }
    frame.provenance = *provenance;
    if (!builder.frames.empty() &&
        frame.source_sequence_index ==
            builder.frames.back().source_sequence_index + 1 &&
        frame.topology_epoch_id == builder.frames.back().topology_epoch_id) {
      frame.source_relation =
          HoloLiftSourceFrameRelation::ContiguousSourceFrame;
    } else {
      frame.source_relation = HoloLiftSourceFrameRelation::SegmentStart;
      ++stats.observation_segments;
    }
    transaction.frame = std::move(frame);
    const auto appended =
        append_hololift_vibe_frame(builder, std::move(transaction));
    if (!appended) {
      return std::unexpected(make_error(
          HoloLiftVibeImportCode::ReferentialIntegrity, candidate_path,
          frame_candidates.empty() ? parsed_frame->line
                                   : frame_candidates.front().line,
          appended.error()));
    }
    ++stats.imported_frames;
    if (parsed_frame->report.status ==
        titan_pbctopo::PbctopoFrameStatus::WeakObservation) {
      ++stats.imported_weak_observation_frames;
    }
    if (parsed_frame->report.temporal_inference_experimental) {
      experimental_frames.push_back(*appended);
      ++stats.experimental_temporal_frames;
    }
  }
  if (pending_candidate) {
    return std::unexpected(
        make_error(HoloLiftVibeImportCode::ReferentialIntegrity, candidate_path,
                   pending_candidate->line,
                   "candidate references a missing later frame report"));
  }
  if (stats.source_frame_rows == 0) {
    return std::unexpected(
        make_error(HoloLiftVibeImportCode::UnsupportedObservation, frame_path,
                   0, "frame report contains no rows"));
  }
  if (stats.imported_frames == 0) {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::UnsupportedObservation, frame_path, 0,
        "VIBE package contains no hard-certified observation band"));
  }
  return validate_temporal_path(options, contract, builder,
                                experimental_frames);
}

} // namespace

std::string_view
hololift_vibe_import_code_name(HoloLiftVibeImportCode code) noexcept {
  switch (code) {
  case HoloLiftVibeImportCode::InvalidOptions:
    return "invalid_options";
  case HoloLiftVibeImportCode::MissingArtifact:
    return "missing_artifact";
  case HoloLiftVibeImportCode::IoError:
    return "io_error";
  case HoloLiftVibeImportCode::CsvSyntax:
    return "csv_syntax";
  case HoloLiftVibeImportCode::MissingColumn:
    return "missing_column";
  case HoloLiftVibeImportCode::DuplicateColumn:
    return "duplicate_column";
  case HoloLiftVibeImportCode::InvalidValue:
    return "invalid_value";
  case HoloLiftVibeImportCode::ContractMismatch:
    return "contract_mismatch";
  case HoloLiftVibeImportCode::ReferentialIntegrity:
    return "referential_integrity";
  case HoloLiftVibeImportCode::UnsupportedObservation:
    return "unsupported_observation";
  case HoloLiftVibeImportCode::StoreValidationFailed:
    return "store_validation_failed";
  }
  return "invalid_value";
}

std::expected<HoloLiftVibeImportResult, HoloLiftVibeImportError>
import_hololift_vibe_package(const HoloLiftVibeImportOptions &options) {
  if (options.package_directory.empty() || options.package_id.empty() ||
      options.trajectory_id.empty()) {
    return std::unexpected(make_error(
        HoloLiftVibeImportCode::InvalidOptions, options.package_directory, 0,
        "package directory, package id, and trajectory id are required"));
  }
  ContractTracker contract;
  HoloLiftObservationStoreBuilder builder;
  auto epochs = import_layout(options, contract, builder);
  if (!epochs)
    return std::unexpected(epochs.error());
  HoloLiftVibeImportStats stats;
  stats.topology_epochs = epochs->size();
  auto frames =
      import_frames_and_candidates(options, contract, *epochs, builder, stats);
  if (!frames)
    return std::unexpected(frames.error());
  builder.source_coverage = {
      HoloLiftSourceCoverageScope::CompleteSourceManifest,
      stats.source_frame_rows,
      stats.imported_frames,
      stats.skipped_local_unwrap_frames,
      stats.skipped_fallback_frames,
  };
  auto finalized = finalize_hololift_observation_store(std::move(builder));
  if (!finalized) {
    const auto &validation = finalized.error();
    return std::unexpected(
        make_error(HoloLiftVibeImportCode::StoreValidationFailed,
                   options.package_directory, 0,
                   std::string(hololift_validation_code_name(validation.code)) +
                       ": " + validation.message));
  }
  return HoloLiftVibeImportResult{std::move(*finalized), stats};
}

} // namespace titan_hololift
