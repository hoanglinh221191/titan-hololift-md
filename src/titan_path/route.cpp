#include "titan_path/route.h"
#include "titan_sha256.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace titan_path {
namespace {

constexpr double kGeometryTolerance = 1.0e-10;

std::string file_sha256(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot open route file: " + path.string());
  titan_hash::Sha256Builder hash;
  std::array<char, 8192> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0) {
      hash.update(std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(buffer.data()),
          static_cast<std::size_t>(count)));
    }
  }
  if (!input.eof())
    throw std::runtime_error("failed while hashing route file: " +
                             path.string());
  constexpr char digits[] = "0123456789abcdef";
  std::string output;
  output.reserve(64);
  for (const std::byte value : hash.finish256()) {
    const unsigned int byte = std::to_integer<unsigned int>(value);
    output.push_back(digits[byte >> 4U]);
    output.push_back(digits[byte & 0x0fU]);
  }
  return output;
}

std::string trim(std::string_view input) {
  std::size_t begin = 0;
  while (begin < input.size() &&
         std::isspace(static_cast<unsigned char>(input[begin])) != 0)
    ++begin;
  std::size_t end = input.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(input[end - 1])) != 0)
    --end;
  return std::string(input.substr(begin, end - begin));
}

std::vector<std::string> split_words(const std::string &value) {
  std::istringstream stream(value);
  std::vector<std::string> words;
  for (std::string word; stream >> word;)
    words.push_back(std::move(word));
  return words;
}

bool finite_vec(const Vec3 &value) {
  return std::isfinite(value[0]) && std::isfinite(value[1]) &&
         std::isfinite(value[2]);
}

Vec3 subtract(const Vec3 &lhs, const Vec3 &rhs) {
  return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
}

Vec3 cross(const Vec3 &lhs, const Vec3 &rhs) {
  return {lhs[1] * rhs[2] - lhs[2] * rhs[1],
          lhs[2] * rhs[0] - lhs[0] * rhs[2],
          lhs[0] * rhs[1] - lhs[1] * rhs[0]};
}

double norm_sq(const Vec3 &value) {
  return value[0] * value[0] + value[1] * value[1] +
         value[2] * value[2];
}

double distance(const Vec3 &lhs, const Vec3 &rhs) {
  return std::sqrt(norm_sq(subtract(lhs, rhs)));
}

std::size_t parse_size(const std::string &text, std::size_t line,
                       const char *field) {
  std::size_t consumed = 0;
  unsigned long long value = 0;
  try {
    value = std::stoull(text, &consumed);
  } catch (const std::exception &) {
    throw std::runtime_error("line " + std::to_string(line) + ": invalid " +
                             field + " value '" + text + "'");
  }
  if (consumed != text.size() ||
      value > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("line " + std::to_string(line) + ": invalid " +
                             field + " value '" + text + "'");
  }
  return static_cast<std::size_t>(value);
}

double parse_real(const std::string &text, std::size_t line,
                  const char *field) {
  std::size_t consumed = 0;
  double value = 0.0;
  try {
    value = std::stod(text, &consumed);
  } catch (const std::exception &) {
    throw std::runtime_error("line " + std::to_string(line) + ": invalid " +
                             field + " value '" + text + "'");
  }
  if (consumed != text.size() || !std::isfinite(value)) {
    throw std::runtime_error("line " + std::to_string(line) + ": invalid " +
                             field + " value '" + text + "'");
  }
  return value;
}

bool is_sha256(const std::string &value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](unsigned char ch) {
           return std::isxdigit(ch) != 0;
         });
}

bool anchors_non_collinear(const std::vector<RouteAnchor> &anchors) {
  if (anchors.size() < 3)
    return false;
  const Vec3 origin = anchors.front().reference_position_angstrom;
  for (std::size_t i = 1; i + 1 < anchors.size(); ++i) {
    const Vec3 first = subtract(anchors[i].reference_position_angstrom, origin);
    if (norm_sq(first) <= kGeometryTolerance)
      continue;
    for (std::size_t j = i + 1; j < anchors.size(); ++j) {
      const Vec3 second =
          subtract(anchors[j].reference_position_angstrom, origin);
      if (norm_sq(cross(first, second)) > kGeometryTolerance)
        return true;
    }
  }
  return false;
}

void validate(RouteDefinition &route) {
  if (route.schema_version != 1)
    throw std::runtime_error("unsupported TITAN_PATH schema version");
  if (route.route_id.empty())
    throw std::runtime_error("route_id is required");
  if (route.source.empty())
    throw std::runtime_error("source is required");
  if (!is_sha256(route.reference_structure_sha256))
    throw std::runtime_error(
        "reference_structure_sha256 must contain 64 hexadecimal characters");
  if (route.waypoints.size() < 2)
    throw std::runtime_error("at least two waypoints are required");
  if (route.start_index != 0 ||
      route.terminal_index + 1 != route.waypoints.size())
    throw std::runtime_error(
        "schema v1 requires explicit start_index=0 and terminal_index=N-1");

  double total_length = 0.0;
  for (std::size_t i = 0; i < route.waypoints.size(); ++i) {
    const auto &waypoint = route.waypoints[i];
    if (waypoint.index != i)
      throw std::runtime_error("waypoint indices must be contiguous from zero");
    if (!finite_vec(waypoint.position_angstrom) ||
        !std::isfinite(waypoint.radius_angstrom) ||
        waypoint.radius_angstrom <= 0.0)
      throw std::runtime_error(
          "waypoint coordinates must be finite and radii must be positive");
    if (i != 0) {
      const double segment = distance(route.waypoints[i - 1].position_angstrom,
                                      waypoint.position_angstrom);
      if (segment <= kGeometryTolerance)
        throw std::runtime_error("consecutive waypoints must be distinct");
      total_length += segment;
    }
  }
  route.total_length_angstrom = total_length;

  std::set<std::string> anchor_keys;
  for (const auto &anchor : route.anchors) {
    if (anchor.original_atom_id == 0 || anchor.atom_name.empty() ||
        !finite_vec(anchor.reference_position_angstrom))
      throw std::runtime_error(
          "anchors require a positive original atom id, atom name, and finite coordinates");
    std::ostringstream key;
    key << anchor.original_atom_id << ':' << anchor.chain << ':'
        << anchor.residue_position << ':' << anchor.atom_name;
    if (!anchor_keys.insert(key.str()).second)
      throw std::runtime_error("duplicate route anchor identity");
  }
  if (route.registration == RegistrationMode::Rigid) {
    if (route.anchors.size() < 3)
      throw std::runtime_error(
          "rigid registration requires at least three anchors");
    if (!anchors_non_collinear(route.anchors))
      throw std::runtime_error(
          "rigid registration anchors must contain a non-collinear triplet");
  }
}

} // namespace

const char *registration_mode_name(RegistrationMode mode) {
  return mode == RegistrationMode::Rigid ? "rigid" : "static";
}

RouteLoadResult load_route_file(const std::filesystem::path &path) {
  RouteLoadResult result;
  std::ifstream input(path);
  if (!input) {
    result.error = "cannot open route file: " + path.string();
    return result;
  }

  try {
    result.source_file_sha256 = file_sha256(path);
    RouteDefinition route;
    bool header_seen = false;
    bool records_started = false;
    double unit_scale = 1.0;
    bool units_seen = false;
    std::set<std::string> singleton_keys;
    std::string raw_line;
    for (std::size_t line_number = 1; std::getline(input, raw_line);
         ++line_number) {
      if (const std::size_t comment = raw_line.find('#');
          comment != std::string::npos)
        raw_line.erase(comment);
      const std::string line = trim(raw_line);
      if (line.empty())
        continue;
      if (!header_seen) {
        const auto words = split_words(line);
        if (words.size() != 2 || words[0] != "TITAN_PATH")
          throw std::runtime_error(
              "line " + std::to_string(line_number) +
              ": expected 'TITAN_PATH 1' header");
        route.schema_version = static_cast<int>(
            parse_size(words[1], line_number, "schema version"));
        header_seen = true;
        continue;
      }

      const std::size_t equals = line.find('=');
      if (equals == std::string::npos)
        throw std::runtime_error("line " + std::to_string(line_number) +
                                 ": expected key = value");
      const std::string key = trim(std::string_view(line).substr(0, equals));
      const std::string value =
          trim(std::string_view(line).substr(equals + 1));
      if (key.empty() || value.empty())
        throw std::runtime_error("line " + std::to_string(line_number) +
                                 ": empty key or value");

      if (key != "anchor" && key != "waypoint") {
        if (!singleton_keys.insert(key).second)
          throw std::runtime_error("line " + std::to_string(line_number) +
                                   ": duplicate key '" + key + "'");
      }

      if (key == "route_id") {
        route.route_id = value;
      } else if (key == "branch_id") {
        route.branch_id = value;
      } else if (key == "source") {
        route.source = value;
      } else if (key == "reference_structure_sha256") {
        route.reference_structure_sha256 = value;
      } else if (key == "registration") {
        if (value == "rigid")
          route.registration = RegistrationMode::Rigid;
        else if (value == "static")
          route.registration = RegistrationMode::Static;
        else
          throw std::runtime_error("line " + std::to_string(line_number) +
                                   ": registration must be rigid or static");
      } else if (key == "units") {
        if (records_started)
          throw std::runtime_error(
              "line " + std::to_string(line_number) +
              ": units must be declared before anchors and waypoints");
        if (value == "angstrom")
          unit_scale = 1.0;
        else if (value == "nm")
          unit_scale = 10.0;
        else
          throw std::runtime_error("line " + std::to_string(line_number) +
                                   ": units must be angstrom or nm");
        units_seen = true;
      } else if (key == "start_index") {
        route.start_index = parse_size(value, line_number, "start_index");
      } else if (key == "terminal_index") {
        route.terminal_index =
            parse_size(value, line_number, "terminal_index");
      } else if (key == "anchor") {
        if (!units_seen)
          throw std::runtime_error("line " + std::to_string(line_number) +
                                   ": declare units before anchor records");
        records_started = true;
        const auto words = split_words(value);
        if (words.size() != 7)
          throw std::runtime_error(
              "line " + std::to_string(line_number) +
              ": anchor requires id chain respos atom x y z");
        RouteAnchor anchor;
        anchor.original_atom_id =
            parse_size(words[0], line_number, "anchor atom id");
        anchor.chain = words[1] == "-" ? "" : words[1];
        anchor.residue_position =
            parse_size(words[2], line_number, "anchor residue position");
        anchor.atom_name = words[3];
        for (std::size_t axis = 0; axis < 3; ++axis)
          anchor.reference_position_angstrom[axis] =
              unit_scale * parse_real(words[4 + axis], line_number,
                                      "anchor coordinate");
        route.anchors.push_back(std::move(anchor));
      } else if (key == "waypoint") {
        if (!units_seen)
          throw std::runtime_error("line " + std::to_string(line_number) +
                                   ": declare units before waypoint records");
        records_started = true;
        const auto words = split_words(value);
        if (words.size() != 5)
          throw std::runtime_error(
              "line " + std::to_string(line_number) +
              ": waypoint requires index x y z radius");
        RouteWaypoint waypoint;
        waypoint.index =
            parse_size(words[0], line_number, "waypoint index");
        for (std::size_t axis = 0; axis < 3; ++axis)
          waypoint.position_angstrom[axis] =
              unit_scale * parse_real(words[1 + axis], line_number,
                                      "waypoint coordinate");
        waypoint.radius_angstrom =
            unit_scale * parse_real(words[4], line_number, "waypoint radius");
        route.waypoints.push_back(std::move(waypoint));
      } else {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                 ": unknown key '" + key + "'");
      }
    }
    if (!header_seen)
      throw std::runtime_error("missing TITAN_PATH header");
    if (!units_seen)
      throw std::runtime_error("units is required");
    validate(route);
    result.route = std::move(route);
    result.ok = true;
  } catch (const std::exception &error) {
    result.error = error.what();
  }
  return result;
}

std::string serialize_route_canonical(const RouteDefinition &route) {
  std::ostringstream output;
  output << std::setprecision(17);
  output << "TITAN_PATH " << route.schema_version << '\n';
  output << "route_id = " << route.route_id << '\n';
  output << "branch_id = " << route.branch_id << '\n';
  output << "source = " << route.source << '\n';
  output << "reference_structure_sha256 = "
         << route.reference_structure_sha256 << '\n';
  output << "registration = " << registration_mode_name(route.registration)
         << '\n';
  output << "units = angstrom\n";
  output << "start_index = " << route.start_index << '\n';
  output << "terminal_index = " << route.terminal_index << '\n';
  for (const auto &anchor : route.anchors) {
    output << "anchor = " << anchor.original_atom_id << ' '
           << (anchor.chain.empty() ? "-" : anchor.chain) << ' '
           << anchor.residue_position << ' ' << anchor.atom_name << ' '
           << anchor.reference_position_angstrom[0] << ' '
           << anchor.reference_position_angstrom[1] << ' '
           << anchor.reference_position_angstrom[2] << '\n';
  }
  for (const auto &waypoint : route.waypoints) {
    output << "waypoint = " << waypoint.index << ' '
           << waypoint.position_angstrom[0] << ' '
           << waypoint.position_angstrom[1] << ' '
           << waypoint.position_angstrom[2] << ' '
           << waypoint.radius_angstrom << '\n';
  }
  return output.str();
}

} // namespace titan_path
