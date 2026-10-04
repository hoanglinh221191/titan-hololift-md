#include "titan_segment_config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace titan_segments {
namespace {

std::string trim_copy(const std::string &value) {
  const auto begin =
      std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
      });
  const auto end =
      std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
      }).base();
  if (begin >= end)
    return {};
  return std::string(begin, end);
}

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  return value;
}

std::vector<std::string> split_words(const std::string &line) {
  std::istringstream input(line);
  std::vector<std::string> words;
  for (std::string word; input >> word;)
    words.push_back(word);
  return words;
}

[[noreturn]] void fail(const std::string &filename, std::size_t line,
                       const std::string &message) {
  std::ostringstream out;
  out << "Segment config '" << filename << "'";
  if (line > 0)
    out << " line " << line;
  out << ": " << message;
  throw std::runtime_error(out.str());
}

std::size_t parse_nonnegative(const std::string &filename, std::size_t line,
                              const std::string &value,
                              const std::string &field) {
  try {
    std::size_t parsed_chars = 0;
    const unsigned long long parsed =
        std::stoull(value, &parsed_chars, 10);
    if (parsed_chars != value.size() ||
        parsed > std::numeric_limits<std::size_t>::max()) {
      fail(filename, line, field + " must be a nonnegative integer");
    }
    return static_cast<std::size_t>(parsed);
  } catch (const std::exception &) {
    fail(filename, line, field + " must be a nonnegative integer");
  }
}

std::uint8_t parse_molecule(const std::string &filename, std::size_t line,
                            const std::string &value) {
  const std::string normalized = lower_copy(value);
  if (normalized == "mol1" || normalized == "1")
    return 1;
  if (normalized == "mol2" || normalized == "2")
    return 2;
  fail(filename, line, "molecule must be mol1 or mol2");
}

CoordinateKind parse_coordinate(const std::string &filename, std::size_t line,
                                const std::string &value) {
  const std::string normalized = lower_copy(value);
  if (normalized == "ordinal")
    return CoordinateKind::Ordinal;
  if (normalized == "resnr" || normalized == "residue-number")
    return CoordinateKind::ResidueNumber;
  if (normalized == "internal" || normalized == "internal-resnr")
    return CoordinateKind::Internal;
  fail(filename, line,
       "segment coordinate must be ordinal, resnr, or internal");
}

} // namespace

const Entity *Config::find_entity(const std::string &name) const {
  const auto found =
      std::find_if(entities.begin(), entities.end(), [&](const Entity &entity) {
        return entity.name == name;
      });
  return found == entities.end() ? nullptr : &*found;
}

const Entity *Config::entity_for_role(const std::string &role) const {
  const auto role_it = roles.find(role);
  return role_it == roles.end() ? nullptr : find_entity(role_it->second);
}

std::vector<const Segment *>
Config::segments_for_entity(const std::string &entity) const {
  std::vector<const Segment *> result;
  for (const Segment &segment : segments) {
    if (segment.entity == entity)
      result.push_back(&segment);
  }
  return result;
}

Config load_config(const std::string &filename) {
  std::ifstream input(filename);
  if (!input.is_open())
    fail(filename, 0, "could not open file");

  Config config;
  bool version_seen = false;
  std::unordered_set<std::string> entity_names;
  std::unordered_set<std::string> segment_names;

  std::string raw_line;
  std::size_t line_number = 0;
  while (std::getline(input, raw_line)) {
    ++line_number;
    const auto comment = raw_line.find('#');
    if (comment != std::string::npos)
      raw_line.erase(comment);
    const std::string line = trim_copy(raw_line);
    if (line.empty())
      continue;

    const std::vector<std::string> words = split_words(line);
    const std::string keyword = lower_copy(words.front());
    if (keyword == "version") {
      if (words.size() != 2)
        fail(filename, line_number, "version syntax is: version 1");
      if (version_seen)
        fail(filename, line_number, "version may be declared only once");
      const std::size_t version =
          parse_nonnegative(filename, line_number, words[1], "version");
      if (version != 1)
        fail(filename, line_number, "only version 1 is supported");
      config.version = 1;
      version_seen = true;
      continue;
    }

    if (keyword == "entity") {
      if (words.size() != 5) {
        fail(filename, line_number,
             "entity syntax is: entity NAME mol1|mol2 chain|chainnum VALUE");
      }
      if (!entity_names.insert(words[1]).second)
        fail(filename, line_number, "duplicate entity name '" + words[1] + "'");

      Entity entity;
      entity.name = words[1];
      entity.molecule = parse_molecule(filename, line_number, words[2]);
      const std::string selector = lower_copy(words[3]);
      if (selector == "chain") {
        entity.selector_kind = EntitySelectorKind::Chain;
        entity.chain = words[4];
      } else if (selector == "chainnum") {
        entity.selector_kind = EntitySelectorKind::ChainNumber;
        entity.chain_number =
            parse_nonnegative(filename, line_number, words[4], "chainnum");
      } else {
        fail(filename, line_number,
             "entity selector must be chain or chainnum");
      }
      config.entities.push_back(std::move(entity));
      continue;
    }

    if (keyword == "role") {
      if (words.size() != 3)
        fail(filename, line_number, "role syntax is: role ROLE ENTITY");
      if (!config.roles.emplace(words[1], words[2]).second)
        fail(filename, line_number, "duplicate role '" + words[1] + "'");
      continue;
    }

    if (keyword == "segment") {
      if (words.size() != 6) {
        fail(filename, line_number,
             "segment syntax is: segment ENTITY NAME "
             "ordinal|resnr|internal BEGIN END");
      }
      const std::string unique_name = words[1] + '\n' + words[2];
      if (!segment_names.insert(unique_name).second) {
        fail(filename, line_number,
             "duplicate segment '" + words[2] + "' for entity '" + words[1] +
                 "'");
      }
      Segment segment;
      segment.entity = words[1];
      segment.name = words[2];
      segment.coordinate =
          parse_coordinate(filename, line_number, words[3]);
      segment.begin =
          parse_nonnegative(filename, line_number, words[4], "segment begin");
      segment.end =
          parse_nonnegative(filename, line_number, words[5], "segment end");
      if (segment.begin == 0 || segment.end == 0 ||
          segment.end < segment.begin) {
        fail(filename, line_number,
             "segment range must satisfy 1 <= BEGIN <= END");
      }
      config.segments.push_back(std::move(segment));
      continue;
    }

    fail(filename, line_number, "unknown keyword '" + words.front() + "'");
  }

  if (!version_seen)
    fail(filename, 0, "missing 'version 1'");
  if (config.entities.empty())
    fail(filename, 0, "at least one entity is required");

  for (const auto &[role, entity_name] : config.roles) {
    if (config.find_entity(entity_name) == nullptr) {
      fail(filename, 0,
           "role '" + role + "' references unknown entity '" + entity_name +
               "'");
    }
  }
  for (const Segment &segment : config.segments) {
    if (config.find_entity(segment.entity) == nullptr) {
      fail(filename, 0,
           "segment '" + segment.name + "' references unknown entity '" +
               segment.entity + "'");
    }
  }
  return config;
}

bool entity_matches(const Entity &entity,
                    const ResidueDescriptor &residue) noexcept {
  if (entity.molecule != residue.molecule)
    return false;
  if (entity.selector_kind == EntitySelectorKind::ChainNumber)
    return entity.chain_number == residue.chain_number;
  return entity.chain == "*" || entity.chain == residue.chain;
}

bool segment_matches(const Segment &segment,
                     const ResidueDescriptor &residue) noexcept {
  std::size_t coordinate = 0;
  switch (segment.coordinate) {
  case CoordinateKind::Ordinal:
    coordinate = residue.ordinal;
    break;
  case CoordinateKind::ResidueNumber:
    coordinate = residue.residue_number;
    break;
  case CoordinateKind::Internal:
    coordinate = residue.internal_residue_number;
    break;
  }
  return coordinate >= segment.begin && coordinate <= segment.end;
}

const char *coordinate_name(CoordinateKind coordinate) noexcept {
  switch (coordinate) {
  case CoordinateKind::Ordinal:
    return "ordinal";
  case CoordinateKind::ResidueNumber:
    return "resnr";
  case CoordinateKind::Internal:
    return "internal";
  }
  return "unknown";
}

} // namespace titan_segments
