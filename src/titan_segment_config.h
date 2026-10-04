#ifndef TITAN_SEGMENT_CONFIG_H
#define TITAN_SEGMENT_CONFIG_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace titan_segments {

enum class EntitySelectorKind : std::uint8_t { Chain, ChainNumber };
enum class CoordinateKind : std::uint8_t { Ordinal, ResidueNumber, Internal };

struct Entity {
  std::string name;
  std::uint8_t molecule{1};
  EntitySelectorKind selector_kind{EntitySelectorKind::Chain};
  std::string chain{"*"};
  std::size_t chain_number{0};
};

struct Segment {
  std::string entity;
  std::string name;
  CoordinateKind coordinate{CoordinateKind::Ordinal};
  std::size_t begin{0};
  std::size_t end{0};
};

struct Config {
  int version{1};
  std::vector<Entity> entities;
  std::vector<Segment> segments;
  std::map<std::string, std::string> roles;

  [[nodiscard]] const Entity *find_entity(const std::string &name) const;
  [[nodiscard]] const Entity *entity_for_role(const std::string &role) const;
  [[nodiscard]] std::vector<const Segment *>
  segments_for_entity(const std::string &entity) const;
};

struct ResidueDescriptor {
  std::uint8_t molecule{1};
  std::size_t chain_number{0};
  std::string chain;
  std::size_t residue_number{0};
  std::size_t internal_residue_number{0};
  std::size_t ordinal{0};
  std::string residue_name;
};

Config load_config(const std::string &filename);

[[nodiscard]] bool entity_matches(const Entity &entity,
                                  const ResidueDescriptor &residue) noexcept;
[[nodiscard]] bool segment_matches(const Segment &segment,
                                   const ResidueDescriptor &residue) noexcept;
[[nodiscard]] const char *coordinate_name(CoordinateKind coordinate) noexcept;

} // namespace titan_segments

#endif // TITAN_SEGMENT_CONFIG_H
