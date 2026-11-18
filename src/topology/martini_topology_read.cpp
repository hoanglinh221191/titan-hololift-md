#include "topology/martini_topology.h"

#include "titan_ui.h"
#include "topology/include_utils.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

bool resolve_topology_include_path(std::string_view current_file,
                                   std::string_view include_target,
                                   std::string &resolved_path) {
  namespace fs = std::filesystem;
  for (const auto &candidate :
       TitanTopo::build_include_candidates(current_file, include_target)) {
    std::error_code ec;
    if (fs::exists(candidate, ec)) {
      resolved_path = candidate.string();
      return true;
    }
  }
  return false;
}

} // namespace

namespace titan_martini {

void MartiniTopology::finalize_pair_table() {
  type_to_id.clear();
  auto register_type = [&](const std::string &name) {
    if (!name.empty() && !type_to_id.contains(name))
      type_to_id.emplace(name, static_cast<int>(type_to_id.size()));
  };

  for (const auto &[name, _] : atomtype_params)
    register_type(name);
  for (const auto &rule : nonbond_rules) {
    register_type(rule.type_i);
    register_type(rule.type_j);
  }

  const std::size_t ntypes = type_to_id.size();
  pair_sigma_table.assign(ntypes * ntypes, -1.0);
  pair_epsilon_table.assign(ntypes * ntypes, -1.0);

  auto set_pair = [&](int ti, int tj, const LJParam &lj) {
    const std::size_t ij = static_cast<std::size_t>(ti) * ntypes +
                           static_cast<std::size_t>(tj);
    const std::size_t ji = static_cast<std::size_t>(tj) * ntypes +
                           static_cast<std::size_t>(ti);
    pair_sigma_table[ij] = lj.sigma;
    pair_sigma_table[ji] = lj.sigma;
    pair_epsilon_table[ij] = lj.epsilon;
    pair_epsilon_table[ji] = lj.epsilon;
  };

  for (const auto &[name, lj] : atomtype_params) {
    const int tid = type_to_id.at(name);
    set_pair(tid, tid, lj);
  }

  explicit_pair_rules = 0;
  for (const auto &rule : nonbond_rules) {
    const auto it_i = type_to_id.find(rule.type_i);
    const auto it_j = type_to_id.find(rule.type_j);
    if (it_i == type_to_id.end() || it_j == type_to_id.end())
      continue;
    set_pair(it_i->second, it_j->second, rule.lj);
    ++explicit_pair_rules;
  }

  fallback_pair_rules = 0;
  for (std::size_t i = 0; i < ntypes; ++i) {
    const double sigma_i = pair_sigma_table[i * ntypes + i];
    const double eps_i = pair_epsilon_table[i * ntypes + i];
    for (std::size_t j = i + 1; j < ntypes; ++j) {
      const std::size_t ij = i * ntypes + j;
      if (pair_sigma_table[ij] >= 0.0 && pair_epsilon_table[ij] >= 0.0)
        continue;
      const double sigma_j = pair_sigma_table[j * ntypes + j];
      const double eps_j = pair_epsilon_table[j * ntypes + j];
      if (sigma_i >= 0.0 && sigma_j >= 0.0 && eps_i >= 0.0 && eps_j >= 0.0) {
        const double sigma_ij = 0.5 * (sigma_i + sigma_j);
        const double epsilon_ij = std::sqrt(eps_i * eps_j);
        pair_sigma_table[ij] = pair_sigma_table[j * ntypes + i] = sigma_ij;
        pair_epsilon_table[ij] =
            pair_epsilon_table[j * ntypes + i] = epsilon_ij;
        ++fallback_pair_rules;
      }
    }
  }
}

std::string MartiniTopology::get_directive(const std::string &line) {
  size_t p1 = line.find('[');
  size_t p2 = line.find(']');
  if (p1 != std::string::npos && p2 != std::string::npos && p2 > p1) {
    std::string dir = line.substr(p1 + 1, p2 - p1 - 1);
    size_t s = dir.find_first_not_of(" \t");
    size_t e = dir.find_last_not_of(" \t");
    if (s != std::string::npos)
      return dir.substr(s, e - s + 1);
  }
  return "";
}

void MartiniTopology::parse_files(const std::string &top_file,
                                  std::fstream &log_file) {
  std::vector<std::string> lines;
  std::vector<std::string> include_stack;

  std::function<void(const std::string &)> read_file =
      [&](const std::string &filename) {
        const std::string normalized =
            std::filesystem::path(filename).lexically_normal().string();
        if (std::find(include_stack.begin(), include_stack.end(), normalized) !=
            include_stack.end()) {
          TitanUI::log_message(
              log_file,
              "[WARNING] MARTINI Parser: Skipping cyclic include: " +
                  normalized);
          return;
        }

        include_stack.emplace_back(normalized);
        std::fstream f(normalized, std::ios::in);
        if (!f.is_open()) {
          TitanUI::print_info(
              "\033[1;33m[WARNING]\033[0m MARTINI Parser: Cannot open "
              "included file: " +
              filename + " — check if file exists.");
          TitanUI::log_message(
              log_file,
              "[WARNING] Cannot open included file: " + normalized);
          include_stack.pop_back();
          return;
        }
        files_read++;
        std::string line;
        while (std::getline(f, line)) {
          size_t comment_pos = line.find(';');
          if (comment_pos != std::string::npos)
            line = line.substr(0, comment_pos); // Xóa comment

          size_t start = line.find_first_not_of(" \t\r\n");
          if (start == std::string::npos)
            continue; // Bỏ qua dòng rỗng
          size_t end = line.find_last_not_of(" \t\r\n");
          line = line.substr(start, end - start + 1);

          // Phân giải #include
          if (const auto include_target =
                  TitanTopo::extract_include_target(line)) {
            std::string resolved_include;
            if (resolve_topology_include_path(normalized, *include_target,
                                              resolved_include)) {
              read_file(resolved_include);
            } else {
              TitanUI::print_info(
                  "\033[1;33m[WARNING]\033[0m MARTINI Parser: Cannot open "
                  "included file: " +
                  *include_target + " â€” check if file exists.");
              TitanUI::log_message(
                  log_file,
                  "[WARNING] Cannot open included file: " + *include_target +
                      " (from " + normalized + ")");
            }
            continue;
          }
          lines.emplace_back(line);
        }
        include_stack.pop_back();
      };

  read_file(top_file);

  std::string current_dir = "";
  std::string current_mol = "";

  for (const auto &line : lines) {
    std::string dir = get_directive(line);
    if (!dir.empty()) {
      current_dir = dir;
      if (dir == "moleculetype")
        current_mol = "";
      continue;
    }

    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token)
      tokens.emplace_back(token);
    if (tokens.empty())
      continue;

    if (current_dir == "nonbond_params") {
      if (tokens.size() >= 5) {
        try {
          double sigma = std::stod(tokens[3]);
          double epsilon = std::stod(tokens[4]);
          nonbond_rules.push_back({tokens[0], tokens[1], {sigma, epsilon}});
        } catch (...) {
        }
      }
    } else if (current_dir == "atomtypes") {
      if (tokens.size() >= 6) {
        try {
          double sigma = std::stod(tokens[tokens.size() - 2]);
          double epsilon = std::stod(tokens[tokens.size() - 1]);
          atomtype_params[tokens[0]] = {sigma, epsilon};
        } catch (...) {
        }
      }
    } else if (current_dir == "moleculetype") {
      if (current_mol.empty())
        current_mol = tokens[0];
    } else if (current_dir == "atoms") {
      if (!current_mol.empty() && tokens.size() >= 7) {
        try {
          std::string type = tokens[1];
          int resnr = std::stoi(tokens[2]);
          std::string resname = tokens[3];
          std::string atomname = tokens[4];
          double charge = std::stod(tokens[6]);
          molecule_types[current_mol].atoms.emplace_back(type, resnr, resname,
                                                         atomname, charge);
        } catch (...) {
        }
      }
    } else if (current_dir == "molecules") {
      if (tokens.size() >= 2) {
        try {
          system_molecules.emplace_back(tokens[0], std::stoi(tokens[1]));
        } catch (...) {
        }
      }
    }
  }

  finalize_pair_table();

  for (const auto &comp : system_molecules) {
    if (molecule_types.find(comp.first) == molecule_types.end()) {
      TitanUI::log_message(
          log_file,
          "[ERROR FATAL] Molecule name '" + comp.first +
              "' in [ molecules ] DOES NOT MATCH any [ moleculetype ]!");
    }
  }
}

void MartiniTopology::build_flat_system() {
  for (const auto &comp : system_molecules) {
    std::string m_name = comp.first;
    int count = comp.second;
    if (molecule_types.count(m_name)) {
      for (int c = 0; c < count; ++c) {
        for (const auto &atom : molecule_types[m_name].atoms) {
          flat_system.emplace_back(atom);
        }
      }
    }
  }
}

int MartiniTopology::get_molecule_count(
    const std::string &target_mol) const {
  for (const auto &comp : system_molecules) {
    if (comp.first == target_mol) {
      return comp.second;
    }
  }
  return 1;
}

} // namespace titan_martini
