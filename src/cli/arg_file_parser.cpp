#include "arguments.h"
#include "cli/option_validation.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::string trim_copy(std::string value) {
  const auto first =
      std::find_if_not(value.begin(), value.end(), [](unsigned char character) {
        return std::isspace(character) != 0;
      });
  if (first == value.end())
    return {};
  const auto last = std::find_if_not(value.rbegin(), value.rend(),
                                     [](unsigned char character) {
                                       return std::isspace(character) != 0;
                                     })
                        .base();
  return std::string(first, last);
}

std::string strip_comment(std::string_view line) {
  bool in_single_quote = false;
  bool in_double_quote = false;
  bool escaped = false;
  for (std::size_t index = 0; index < line.size(); ++index) {
    const char character = line[index];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (character == '\\') {
      escaped = true;
      continue;
    }
    if (character == '\'' && !in_double_quote)
      in_single_quote = !in_single_quote;
    else if (character == '"' && !in_single_quote)
      in_double_quote = !in_double_quote;
    else if (character == '#' && !in_single_quote && !in_double_quote)
      return std::string(line.substr(0, index));
  }
  return std::string(line);
}

std::string unquote_value(std::string value) {
  if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                            (value.front() == '\'' && value.back() == '\''))) {
    value = value.substr(1, value.size() - 2);
  }
  return value;
}

std::string line_location(const std::string &filename,
                          std::size_t line_number) {
  return filename + ":" + std::to_string(line_number);
}

} // namespace

int arguments::tokenize_arg_file(const std::string &filename,
                                 std::vector<std::string> &tokens) {
  tokens.clear();
  std::ifstream input(filename);
  if (!input.is_open()) {
    TitanUI::print_error("Argument file '" + filename + "' was not found.");
    return titan_cli::kNoArgumentFile;
  }

  titan_cli::ValidationResult validation;
  std::vector<titan_cli::OptionOccurrence> occurrences;
  bool saw_option = false;
  std::size_t line_number = 0;
  for (std::string raw_line; std::getline(input, raw_line);) {
    ++line_number;
    if (!raw_line.empty() && raw_line.back() == '\r')
      raw_line.pop_back();
    const std::string line = trim_copy(strip_comment(raw_line));
    if (line.empty())
      continue;

    const std::string location = line_location(filename, line_number);
    if (!titan_cli::looks_like_option(line)) {
      validation.error("CLI005",
                       location + ": expected '-option=value'; text outside an "
                                  "option assignment is not allowed.");
      continue;
    }
    const std::size_t equals = line.find('=');
    if (equals == std::string::npos) {
      const std::string key = trim_copy(line);
      const std::size_t whitespace = key.find_first_of(" \t");
      if (whitespace != std::string::npos) {
        const std::string option = key.substr(0, whitespace);
        validation.append(titan_cli::validate_option_name(option, location));
        validation.error("CLI005",
                         location + ": invalid option line; value options use "
                                    "'-option=value' without whitespace; the "
                                    "'=' sign is mandatory.");
        continue;
      }
      if (!titan_cli::looks_like_option(key)) {
        validation.error("CLI005", location +
                                       ": expected '-option=value' or a bare "
                                       "boolean/presence switch.");
        continue;
      }
      validation.append(titan_cli::validate_option_name(key, location));
      if (!titan_cli::allows_implicit_true(key)) {
        validation.error("CLI005",
                         location + ": option '" + key +
                             "' requires '-option=value' in arg.in; the "
                             "'=' sign is mandatory for value options.");
        continue;
      }
      occurrences.push_back({key, {}, false, true});
      tokens.push_back(key);
      saw_option = true;
      continue;
    }

    const std::string raw_key = line.substr(0, equals);
    const std::string key = trim_copy(raw_key);
    std::string value = unquote_value(trim_copy(line.substr(equals + 1)));
    if (key.empty() || raw_key != key ||
        key.find_first_of(" \t") != std::string::npos ||
        !titan_cli::looks_like_option(key)) {
      validation.error("CLI005",
                       location + ": invalid option name; use '-option=value' "
                                  "without whitespace before '='.");
      continue;
    }
    validation.append(titan_cli::validate_option_name(key, location));
    if (value.empty()) {
      validation.error("CLI003", location + ": option '" + key +
                                     "' requires a non-empty value after '='.");
      continue;
    }
    saw_option = true;

    const bool boolean_option = titan_cli::is_boolean_option(key);
    const bool value_option = titan_cli::is_value_option(key);
    titan_cli::OptionOccurrence occurrence{key, value, true,
                                           !titan_cli::is_false_value(value)};
    occurrences.push_back(occurrence);

    // Standalone downstream builds may compile the compatibility translation
    // unit without Makefile-generated registry metadata. Keep arg.in usable in
    // that reduced mode; production builds always take the strict branches
    // below with complete option classification.
    if (!titan_cli::registry_available()) {
      if (titan_cli::is_boolean_value(value)) {
        if (titan_cli::is_true_value(value))
          tokens.push_back(key);
      } else {
        tokens.push_back(key);
        tokens.push_back(value);
      }
      continue;
    }

    if (boolean_option && !titan_cli::is_boolean_value(value)) {
      validation.error("CLI004", location + ": boolean option '" + key +
                                     "' requires true/false (also 1/0, yes/no, "
                                     "on/off).");
      continue;
    }
    if (boolean_option) {
      if (titan_cli::is_true_value(value))
        tokens.push_back(key);
      else
        tokens.push_back(key + "=false");
      continue;
    }
    if (!value_option && !boolean_option) {
      if (!titan_cli::is_boolean_value(value)) {
        validation.error("CLI009", location + ": presence-only switch '" + key +
                                       "' requires '=true' or '=false'.");
        continue;
      }
      if (titan_cli::is_true_value(value))
        tokens.push_back(key);
      continue;
    }

    // Internally the legacy dispatcher still consumes argv-style pairs. The
    // strict source grammar is enforced above before this representation is
    // created, and compound values remain a single token.
    tokens.push_back(key);
    tokens.push_back(value);
  }

  validation.append(titan_cli::validate_occurrence_constraints(occurrences));
  titan_cli::emit_diagnostics(validation);
  if (!validation.ok()) {
    tokens.clear();
    return titan_cli::kInvalidArguments;
  }
  if (tokens.empty() && !saw_option)
    return titan_cli::kNoArgumentFile;
  return titan_cli::kParseSuccess;
}

int arguments::parse_arg_file() {
  std::vector<std::string> tokens;
  const int tokenize_status = tokenize_arg_file(arg_filename, tokens);
  if (tokenize_status != titan_cli::kParseSuccess)
    return tokenize_status;

  file_par = true;
  namespace fs = std::filesystem;

  std::string arg_dir;
  if (arg_filename.find('\\') != std::string::npos ||
      arg_filename.find('/') != std::string::npos) {
    arg_dir = arg_filename.substr(0, arg_filename.find_last_of("\\/"));
  }
  if (!arg_dir.empty()) {
    const fs::path arg_dir_path(arg_dir);
    const auto is_path_flag = [](const std::string &flag) {
      return flag == "-topo" || flag == "-topo2" || flag == "-common_topo" ||
             flag == "-common-topo" || flag == "-f" || flag == "-s" ||
             flag == "-n" || flag == "-i" || flag == "-traj" ||
             flag == "-dham-temps" || flag == "-dham-ener" ||
             flag == "-mbar-temps" || flag == "-mbar-ener" ||
             flag == "-mbar-demux-map" || flag == "-path-a" ||
              flag == "-path-n" || flag == "-pbc-ref" ||
              flag == "-pathway-file" || flag == "-pathway-previous-status" ||
              flag == "-segment-config" || flag == "-interface-config" ||
             flag == "-membrane-leaflet-file" || flag == "-pbe-test-pqr" ||
             flag == "-pbe-binding-receptor-pqr" ||
             flag == "-pbe-binding-ligand-pqr";
    };

    for (std::size_t index = 0; index + 1 < tokens.size(); ++index) {
      if (!is_path_flag(tokens[index]))
        continue;
      if (tokens[index + 1].empty() || tokens[index + 1] == "no")
        continue;
      fs::path candidate(tokens[index + 1]);
      if (candidate.is_relative())
        tokens[index + 1] =
            (arg_dir_path / candidate).lexically_normal().string();
      ++index;
    }
  }

  std::vector<std::string> argv_storage;
  argv_storage.reserve(tokens.size() + 1);
  argv_storage.push_back("titan");
  for (std::string &token : tokens)
    argv_storage.push_back(std::move(token));

  std::vector<char *> argv_ptrs;
  argv_ptrs.reserve(argv_storage.size());
  for (std::string &value : argv_storage)
    argv_ptrs.push_back(value.data());

  const bool previous_context = parsing_arg_file_tokens;
  parsing_arg_file_tokens = true;
  int result = titan_cli::kInvalidArguments;
  try {
    result =
        parse_arguments(static_cast<int>(argv_ptrs.size()), argv_ptrs.data());
  } catch (...) {
    parsing_arg_file_tokens = previous_context;
    throw;
  }
  parsing_arg_file_tokens = previous_context;

  if (!arg_dir.empty()) {
    const fs::path arg_dir_path(arg_dir);
    auto resolve = [&](std::string &path) {
      if (!path.empty() && path != "no") {
        fs::path candidate(path);
        if (candidate.is_relative())
          path = (arg_dir_path / candidate).lexically_normal().string();
      }
    };
    if (flag_topo_1)
      resolve(topo_name);
    if (flag_topo_2)
      resolve(topo_name_2);
    if (flag_1_pdb)
      resolve(pdb_name);
    if (flag_index)
      resolve(index_name);
    if (flag_listfile)
      resolve(list_filename);
    if (flag_xdr)
      resolve(traj_file);
    if (pbc.flag_pbc_ref)
      resolve(pbc.ref_pdb_name);
    if (!membrane.leaflet_file.empty())
      resolve(membrane.leaflet_file);
    resolve(path_amino_acid);
    resolve(path_nonbonded);
    if (dham.flag_dham) {
      resolve(dham.temps_file);
      resolve(dham.ener_list_file);
    }
    if (mbar.flag_mbar) {
      resolve(mbar.temps_file);
      resolve(mbar.ener_list_file);
      resolve(mbar.demux_map_file);
    }
    resolve(pbe.test_pqr);
    resolve(pbe.binding_receptor_pqr);
    resolve(pbe.binding_ligand_pqr);
  }

  return result;
}
