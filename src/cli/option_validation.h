#ifndef TITAN_CLI_OPTION_VALIDATION_H
#define TITAN_CLI_OPTION_VALIDATION_H

#include "titan_ui.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__has_include)
#if __has_include("titan_cli_registry_generated.h")
#include "titan_cli_registry_generated.h"
#endif
#endif

#ifndef TITAN_CLI_REGISTRY_GENERATED
#define TITAN_CLI_REGISTRY_GENERATED 0
namespace titan_cli::generated {
inline constexpr std::array<std::string_view, 0> known_options{};
inline constexpr std::array<std::string_view, 0> value_options{};
inline constexpr std::array<std::string_view, 0> boolean_options{};
} // namespace titan_cli::generated
#endif

namespace titan_cli {

inline constexpr int kParseSuccess = 0;
// The parser converts several hundred option values with std::sto*, whose
// exceptions carry only "stoi" or "stod" and no indication of which option was
// being read. Recording the token currently under inspection lets the
// top-level handler name it, without restructuring the parse loop.
inline std::string &active_option_token() {
  static thread_local std::string token;
  return token;
}

inline constexpr int kNoArgumentFile = 1;
inline constexpr int kHandledAndExit = 2;
inline constexpr int kInvalidArguments = 3;

enum class DiagnosticSeverity { warning, error };

struct Diagnostic {
  DiagnosticSeverity severity{DiagnosticSeverity::error};
  std::string code{};
  std::string message{};
};

struct ValidationResult {
  std::vector<Diagnostic> diagnostics{};

  [[nodiscard]] bool ok() const noexcept {
    return std::none_of(diagnostics.begin(), diagnostics.end(),
                        [](const Diagnostic &diagnostic) {
                          return diagnostic.severity ==
                                 DiagnosticSeverity::error;
                        });
  }

  void error(std::string code, std::string message) {
    diagnostics.push_back(
        {DiagnosticSeverity::error, std::move(code), std::move(message)});
  }

  void warning(std::string code, std::string message) {
    diagnostics.push_back(
        {DiagnosticSeverity::warning, std::move(code), std::move(message)});
  }

  void append(ValidationResult other) {
    diagnostics.insert(diagnostics.end(),
                       std::make_move_iterator(other.diagnostics.begin()),
                       std::make_move_iterator(other.diagnostics.end()));
  }
};

struct OptionOccurrence {
  std::string option{};
  std::string value{};
  bool has_value{false};
  bool enabled{true};
};

struct SubgroupSelection {
  std::vector<int> molecule_1{};
  std::vector<int> molecule_2{};
};

inline std::string lowercase_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  return value;
}

inline bool is_false_value(std::string_view value) {
  const std::string lowered = lowercase_copy(std::string(value));
  return lowered == "0" || lowered == "false" || lowered == "no" ||
         lowered == "off";
}

inline bool is_true_value(std::string_view value) {
  const std::string lowered = lowercase_copy(std::string(value));
  return lowered == "1" || lowered == "true" || lowered == "yes" ||
         lowered == "on";
}

inline bool is_boolean_value(std::string_view value) {
  return is_true_value(value) || is_false_value(value);
}

inline std::string option_key(std::string_view token) {
  const std::size_t equals = token.find('=');
  return std::string(token.substr(0, equals));
}

inline bool looks_like_option(std::string_view token) {
  if (token.size() < 2 || token.front() != '-')
    return false;
  std::size_t offset = 1;
  if (token[offset] == '-') {
    ++offset;
    if (offset >= token.size())
      return false;
  }
  if (std::isalpha(static_cast<unsigned char>(token[offset])) != 0)
    return true;
  // Preserve the historical -1pdb switch without classifying negative
  // numeric values as options.
  constexpr std::string_view one_pdb = "1pdb";
  return token.substr(offset).starts_with(one_pdb) &&
         (token.size() == offset + one_pdb.size() ||
          token[offset + one_pdb.size()] == '=');
}

template <std::size_t Size>
inline bool sorted_contains(const std::array<std::string_view, Size> &values,
                            std::string_view value) {
  return std::binary_search(values.begin(), values.end(), value);
}

inline std::string canonical_option(std::string_view raw_option) {
  std::string option(raw_option);
  if (option.rfind("--", 0) == 0)
    option.erase(option.begin());
  constexpr std::string_view old_prefix = "-pbe-tabi-";
  constexpr std::string_view new_prefix = "-pbe-bie-";
  if (option.rfind(old_prefix, 0) == 0)
    option.replace(0, old_prefix.size(), new_prefix);
  return option;
}

inline bool registry_available() noexcept {
  return TITAN_CLI_REGISTRY_GENERATED != 0;
}

inline bool is_known_option(std::string_view option) {
  if (!registry_available())
    return true;
  if (sorted_contains(generated::known_options, option))
    return true;
  const std::string canonical = canonical_option(option);
  return canonical != option &&
         sorted_contains(generated::known_options, canonical);
}

inline bool is_value_option(std::string_view option) {
  const std::string canonical = canonical_option(option);
  return sorted_contains(generated::value_options, option) ||
         sorted_contains(generated::value_options, canonical);
}

inline bool is_boolean_option(std::string_view option) {
  const std::string canonical = canonical_option(option);
  return sorted_contains(generated::boolean_options, option) ||
         sorted_contains(generated::boolean_options, canonical);
}

inline bool is_presence_only_option(std::string_view option) {
  if (!registry_available())
    return true;
  return is_known_option(option) && !is_value_option(option) &&
         !is_boolean_option(option);
}

inline bool allows_implicit_true(std::string_view option) {
  return is_boolean_option(option) || is_presence_only_option(option);
}

inline bool is_compound_terminal_value_option(std::string_view option) {
  const std::string canonical = canonical_option(option);
  return canonical == "-sg" || canonical == "-pbsa-components";
}

inline std::size_t edit_distance(std::string_view left,
                                 std::string_view right) {
  std::vector<std::size_t> previous(right.size() + 1);
  std::vector<std::size_t> current(right.size() + 1);
  for (std::size_t column = 0; column <= right.size(); ++column)
    previous[column] = column;
  for (std::size_t row = 1; row <= left.size(); ++row) {
    current[0] = row;
    for (std::size_t column = 1; column <= right.size(); ++column) {
      const std::size_t substitution =
          previous[column - 1] + (left[row - 1] == right[column - 1] ? 0 : 1);
      current[column] = std::min(
          {previous[column] + 1, current[column - 1] + 1, substitution});
    }
    previous.swap(current);
  }
  return previous.back();
}

inline std::string trim_ascii_copy(std::string_view value) {
  std::size_t first = 0;
  while (first < value.size() &&
         std::isspace(static_cast<unsigned char>(value[first])) != 0)
    ++first;
  std::size_t last = value.size();
  while (last > first &&
         std::isspace(static_cast<unsigned char>(value[last - 1])) != 0)
    --last;
  return std::string(value.substr(first, last - first));
}

inline bool parse_subgroup_side(std::string_view text, std::vector<int> &groups,
                                std::string &error) {
  groups.clear();
  std::size_t index = 0;
  while (index < text.size()) {
    while (index < text.size() &&
           (std::isspace(static_cast<unsigned char>(text[index])) != 0 ||
            text[index] == ','))
      ++index;
    if (index == text.size())
      break;
    if (std::isdigit(static_cast<unsigned char>(text[index])) == 0) {
      error = "group lists may contain only non-negative integer IDs "
              "separated by spaces or commas";
      return false;
    }
    int value = 0;
    while (index < text.size() &&
           std::isdigit(static_cast<unsigned char>(text[index])) != 0) {
      const int digit = text[index] - '0';
      if (value > (std::numeric_limits<int>::max() - digit) / 10) {
        error = "group ID is outside the supported integer range";
        return false;
      }
      value = value * 10 + digit;
      ++index;
    }
    if (index < text.size() && text[index] != ',' &&
        std::isspace(static_cast<unsigned char>(text[index])) == 0) {
      error = "group lists may contain only non-negative integer IDs "
              "separated by spaces or commas";
      return false;
    }
    groups.push_back(value);
  }
  if (groups.empty()) {
    error = "both molecule group lists must contain at least one group ID";
    return false;
  }
  return true;
}

inline bool parse_subgroup_expression(std::string_view raw_value,
                                      SubgroupSelection &selection,
                                      std::string &error) {
  selection = {};
  error.clear();
  const std::string value = trim_ascii_copy(raw_value);
  const std::size_t separator = value.find_first_of("-&");
  if (separator == std::string::npos) {
    error = "missing molecule separator '-' (legacy '&' is also accepted)";
    return false;
  }
  if (value.find_first_of("-&", separator + 1) != std::string::npos) {
    error = "exactly one molecule separator is allowed";
    return false;
  }
  if (!parse_subgroup_side(std::string_view(value).substr(0, separator),
                           selection.molecule_1, error))
    return false;
  if (!parse_subgroup_side(std::string_view(value).substr(separator + 1),
                           selection.molecule_2, error))
    return false;
  return true;
}

inline bool normalize_pbsa_component_selection(std::string_view raw_value,
                                               std::string &normalized,
                                               std::string &error) {
  normalized.clear();
  error.clear();
  std::string value = lowercase_copy(trim_ascii_copy(raw_value));
  if (value == "auto") {
    normalized = "auto";
    return true;
  }
  for (char &character : value) {
    if (character == ',' || character == ';' || character == '-' ||
        std::isspace(static_cast<unsigned char>(character)) != 0)
      character = ' ';
  }

  std::vector<std::string> components;
  std::istringstream input(value);
  for (std::string component; input >> component;) {
    constexpr std::array<std::string_view, 3> valid = {"complex", "ligand",
                                                       "receptor"};
    if (std::find(valid.begin(), valid.end(), component) == valid.end()) {
      std::string suggestion;
      std::size_t best_distance = std::numeric_limits<std::size_t>::max();
      for (const std::string_view candidate : valid) {
        const std::size_t distance = edit_distance(component, candidate);
        if (distance < best_distance) {
          best_distance = distance;
          suggestion = std::string(candidate);
        }
      }
      error = "unknown PBE/PBSA component '" + component + "'";
      if (best_distance <= 2)
        error += "; did you mean '" + suggestion + "'?";
      return false;
    }
    if (std::find(components.begin(), components.end(), component) !=
        components.end()) {
      error = "duplicate PBE/PBSA component '" + component + "'";
      return false;
    }
    components.push_back(std::move(component));
  }

  std::sort(components.begin(), components.end());
  const std::vector<std::string> required = {"complex", "ligand", "receptor"};
  if (components != required) {
    error = "expected exactly complex, ligand, and receptor";
    return false;
  }
  normalized = "complex,receptor,ligand";
  return true;
}

inline std::string_view without_leading_hyphens(std::string_view option) {
  while (!option.empty() && option.front() == '-')
    option.remove_prefix(1);
  return option;
}

inline std::vector<std::string> suggest_options(std::string_view unknown,
                                                std::size_t limit = 3) {
  struct Candidate {
    std::size_t distance{};
    std::size_t style_penalty{};
    std::string option{};
  };
  std::vector<Candidate> candidates;
  if (!registry_available())
    return {};
  const bool long_style = unknown.rfind("--", 0) == 0;
  const std::string_view needle = without_leading_hyphens(unknown);
  const std::size_t threshold = std::max<std::size_t>(2, needle.size() / 4);
  for (const std::string_view known : generated::known_options) {
    const std::size_t distance =
        edit_distance(needle, without_leading_hyphens(known));
    if (distance > threshold)
      continue;
    candidates.push_back({distance,
                          (known.rfind("--", 0) == 0) == long_style ? 0U : 1U,
                          std::string(known)});
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &left, const Candidate &right) {
              if (left.distance != right.distance)
                return left.distance < right.distance;
              if (left.style_penalty != right.style_penalty)
                return left.style_penalty < right.style_penalty;
              return left.option < right.option;
            });
  std::vector<std::string> suggestions;
  for (const Candidate &candidate : candidates) {
    if (std::find(suggestions.begin(), suggestions.end(), candidate.option) !=
        suggestions.end())
      continue;
    suggestions.push_back(candidate.option);
    if (suggestions.size() == limit)
      break;
  }
  return suggestions;
}

inline std::string format_suggestions(const std::vector<std::string> &values) {
  if (values.empty())
    return {};
  std::ostringstream message;
  message << " Did you mean ";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0)
      message << (index + 1 == values.size() ? " or " : ", ");
    message << '\'' << values[index] << '\'';
  }
  message << '?';
  return message.str();
}

inline ValidationResult validate_option_name(std::string_view option,
                                             std::string_view location) {
  ValidationResult result;
  const std::string canonical = canonical_option(option);
  const auto removed = [&](std::string_view old_option,
                           std::string_view replacement) {
    if (canonical != old_option)
      return false;
    result.error("CLI006", std::string(location) + ": obsolete option '" +
                               std::string(option) + "' was removed; use " +
                               std::string(replacement) + ".");
    return true;
  };
  if (removed("-useForceOptionforEnergy",
              "'-edecomp true'; force maps use '-fmap <mode>'") ||
      removed("-emap", "'-edecomp true'") ||
      removed("-emap-level", "'-edecomp-level <value>'") ||
      removed("-energy-res", "'-edecomp-level residue'") ||
      removed("-new-emap", "'-edecomp true'") ||
      removed("-s2-plateau-window", "'-s2-fit-window <value>'"))
    return result;

  if (canonical == "-contact-map") {
    result.warning("CLIW01",
                   std::string(location) + ": obsolete alias '" +
                       std::string(option) +
                       "' is deprecated; use '-cmap true -cmap-scope interface "
                       "-cmap-mode sidechain-com'.");
  }
  if (canonical == "-allostery-cut-mode") {
    result.warning("CLIW03", std::string(location) + ": deprecated alias '" +
                                 std::string(option) +
                                 "'; use '-allostery-prune-window'.");
  }
  if (option.rfind("-pbe-tabi-", 0) == 0 ||
      option.rfind("--pbe-tabi-", 0) == 0) {
    std::string replacement(option);
    const std::size_t prefix = replacement.rfind("--", 0) == 0 ? 2 : 1;
    replacement.replace(prefix, std::string("pbe-tabi-").size(), "pbe-bie-");
    result.warning("CLIW02", std::string(location) + ": obsolete PBE alias '" +
                                 std::string(option) + "'; use '" +
                                 replacement + "'.");
  }
  if (!is_known_option(option)) {
    result.error("CLI002", std::string(location) + ": unknown option '" +
                               std::string(option) + "'." +
                               format_suggestions(suggest_options(option)));
  }
  return result;
}

inline void add_conflict(ValidationResult &result, std::string_view left,
                         std::string_view right, std::string_view guidance) {
  result.error("CLI007", "Conflicting options '" + std::string(left) +
                             "' and '" + std::string(right) + "'. " +
                             std::string(guidance));
}

inline ValidationResult validate_occurrence_constraints(
    const std::vector<OptionOccurrence> &occurrences) {
  ValidationResult result;
  std::unordered_map<std::string, const OptionOccurrence *> enabled;
  for (const OptionOccurrence &occurrence : occurrences) {
    if (occurrence.enabled)
      enabled[canonical_option(occurrence.option)] = &occurrence;
  }
  const auto has = [&](std::string_view option) {
    return enabled.contains(std::string(option));
  };
  if (has("-cmap") && has("-contact-map"))
    add_conflict(result, "-cmap", "-contact-map",
                 "Use only the modern -cmap option family.");
  if (has("-cas") && has("-nocal"))
    add_conflict(result, "-cas", "-nocal",
                 "CAS requires GBSA energy evaluation.");
  if (has("-awh-ifp") && has("-nocal"))
    add_conflict(result, "-awh-ifp", "-nocal",
                 "AWH-IFP requires two-group energy analysis.");

  constexpr std::array<std::string_view, 11> legacy_pbc = {
      "-pbc",  "-pbci",     "-pbci2",    "-pbci3",   "-pbcn",  "-pbca",
      "-pbcm", "-pbcclust", "-pbc-envi", "-pbc-ref", "-refpbc"};
  if (has("-pbctopo")) {
    for (const std::string_view option : legacy_pbc) {
      if (has(option)) {
        add_conflict(
            result, "-pbctopo", option,
            "Select PBCTOPO or one legacy/reference PBC mode, not both.");
        break;
      }
    }
  }

  const auto cycle = enabled.find("-pbe-binding-cycle");
  if (cycle != enabled.end() && cycle->second->has_value &&
      lowercase_copy(cycle->second->value) == "pqr-components") {
    std::vector<std::string> missing;
    if (!has("-pbe-test-pqr"))
      missing.emplace_back("-pbe-test-pqr (complex)");
    if (!has("-pbe-binding-receptor-pqr"))
      missing.emplace_back("-pbe-binding-receptor-pqr");
    if (!has("-pbe-binding-ligand-pqr"))
      missing.emplace_back("-pbe-binding-ligand-pqr");
    if (!missing.empty()) {
      std::ostringstream message;
      message << "PBE pqr-components requires separate complex, receptor, and "
                 "ligand inputs; missing ";
      for (std::size_t index = 0; index < missing.size(); ++index) {
        if (index != 0)
          message << (index + 1 == missing.size() ? " and " : ", ");
        message << missing[index];
      }
      message << '.';
      result.error("CLI008", message.str());
    }
  }
  return result;
}

inline ValidationResult validate_terminal_tokens(int argc, char **argv) {
  ValidationResult result;
  std::vector<OptionOccurrence> occurrences;
  for (int index = 1; index < argc; ++index) {
    const std::string token(argv[index]);
    if (!looks_like_option(token)) {
      result.error(
          "CLI010",
          "terminal: unexpected positional argument '" + token +
              "'; values must immediately follow an option that accepts a "
              "value. Boolean/presence options may stand alone or use an "
              "explicit true/false value.");
      continue;
    }
    const std::string key = option_key(token);
    result.append(validate_option_name(key, "terminal"));
    const std::size_t inline_equals = token.find('=');
    if (inline_equals != std::string::npos) {
      result.error("CLI001",
                   "terminal: invalid syntax '" + token + "'; use '" +
                       token.substr(0, inline_equals) + " " +
                       token.substr(inline_equals + 1) +
                       "'. The -option=value form is reserved for arg.in.");
    }

    OptionOccurrence occurrence{key, {}, false, true};
    const bool value_option = is_value_option(key);
    const bool boolean_option = is_boolean_option(key);
    if ((value_option || boolean_option) &&
        inline_equals != std::string::npos) {
      occurrence.value = token.substr(inline_equals + 1);
      occurrence.has_value = !occurrence.value.empty();
      occurrence.enabled = !is_false_value(occurrence.value);
      if (boolean_option && occurrence.has_value &&
          !is_boolean_value(occurrence.value)) {
        result.error("CLI004",
                     "terminal: boolean option '" + key +
                         "' requires true/false (also 1/0, yes/no, on/off).");
      }
    } else if (value_option) {
      if (index + 1 >= argc || looks_like_option(argv[index + 1])) {
        result.error("CLI003", "terminal: option '" + key +
                                   "' requires a separate value: '" + key +
                                   " <value>'.");
      } else {
        occurrence.value = argv[index + 1];
        occurrence.has_value = true;
        occurrence.enabled = !is_false_value(occurrence.value);
        ++index;
        if (is_compound_terminal_value_option(key)) {
          while (index + 1 < argc &&
                 !looks_like_option(argv[index + 1])) {
            occurrence.value.push_back(' ');
            occurrence.value += argv[++index];
          }
        }
      }
    } else if (boolean_option && index + 1 < argc &&
               is_boolean_value(argv[index + 1])) {
      occurrence.value = argv[index + 1];
      occurrence.has_value = true;
      occurrence.enabled = !is_false_value(occurrence.value);
      ++index;
    }
    occurrences.push_back(std::move(occurrence));
  }
  result.append(validate_occurrence_constraints(occurrences));
  return result;
}

inline void emit_diagnostics(const ValidationResult &result) {
  for (const Diagnostic &diagnostic : result.diagnostics) {
    const std::string message = diagnostic.code + ": " + diagnostic.message;
    if (diagnostic.severity == DiagnosticSeverity::error)
      TitanUI::print_error(message);
    else
      TitanUI::print_warning(message);
  }
}

} // namespace titan_cli

#endif
