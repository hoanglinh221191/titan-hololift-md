#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace titan_nb {
struct Coefficients { double c6{}, c12{}; };
struct Atom { std::uint32_t type{}; double charge{}; std::string name; };
struct ListedPair { Coefficients lj; double charge_product{}; };
struct PairTerms {
  Coefficients normal, listed;
  double normal_charge{}, listed_charge{};
  bool has_listed{};
};

// Immutable, identity-bound A-state LJ/Coulomb topology. Coordinates are not
// used to generate exclusions. All distances used by consumers are in nm.
struct Topology {
  std::vector<Atom> atoms;
  std::vector<Coefficients> nonbond_matrix;
  std::uint32_t type_count{};
  std::unordered_set<std::uint64_t> exclusions;
  std::unordered_map<std::uint64_t, ListedPair> listed_pairs;
  std::vector<std::filesystem::path> input_files;
  int combination_rule{};
  double fudge_qq{1}, fudge_lj{1};
  PairTerms pair(std::size_t a, std::size_t b) const;
};

std::uint64_t pair_key(std::size_t a, std::size_t b);
// Includes and basic #define/#ifdef/#ifndef are handled in source order.
// Unsupported preprocessor/interaction semantics throw with a location.
std::shared_ptr<const Topology> read_topology(const std::filesystem::path &path);
} // namespace titan_nb
