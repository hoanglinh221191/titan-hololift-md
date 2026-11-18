#include "ptopo_gromacs_topology.h"

#include "../topology/include_utils.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <string_view>
#include <unordered_map>

namespace titan_ptopo {
namespace {

namespace fs = std::filesystem;

std::string_view trim(std::string_view s) {
  const std::size_t first = s.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos)
    return {};
  const std::size_t last = s.find_last_not_of(" \t\r\n");
  return s.substr(first, last - first + 1);
}

std::vector<std::string_view> fields(std::string_view s) {
  std::vector<std::string_view> out;
  std::size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
      ++i;
    const std::size_t begin = i;
    while (i < s.size() && !std::isspace(static_cast<unsigned char>(s[i])))
      ++i;
    if (i > begin)
      out.push_back(s.substr(begin, i - begin));
  }
  return out;
}

template <class T> bool parse_int(std::string_view s, T &out) {
  const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
  return ec == std::errc{} && ptr == s.data() + s.size();
}

bool parse_double(std::string_view s, double &out) {
  // std::strtod: floating from_chars is not available in every TITAN
  // toolchain's standard library.
  const std::string text(s);
  char *end = nullptr;
  out = std::strtod(text.c_str(), &end);
  return end == text.c_str() + text.size() && !text.empty();
}

std::string lower(std::string_view s) {
  std::string out(s);
  for (char &c : out)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

struct Conditional {
  bool parent_active = true;
  bool taken = false; // the #ifdef/#ifndef condition held
  bool in_else = false;
};

class Reader {
public:
  explicit Reader(const TopologyReadOptions &options) : options_(options) {
    for (const std::string &raw : options.defines) {
      std::string_view d = trim(raw);
      if (d.starts_with("-D"))
        d.remove_prefix(2);
      const std::size_t eq = d.find('=');
      const std::string name(trim(d.substr(0, eq)));
      if (!name.empty())
        defines_[name] =
            eq == std::string_view::npos ? std::string() : std::string(d.substr(eq + 1));
    }
  }

  Result<GromacsTopology> run(const fs::path &root) {
    if (auto ok = read_file(root, 0); !ok)
      return std::unexpected(ok.error());
    if (!conditionals_.empty())
      return make_error(ErrorCode::InvalidInput,
                        "unterminated #ifdef/#ifndef at end of topology");
    if (out_.molecules.empty())
      return make_error(ErrorCode::InvalidInput,
                        "topology has no [ molecules ] entries");
    for (const TopologyMoleculeType &type : out_.molecule_types)
      for (const TopologyBond &bond : type.bonds)
        if (bond.a >= type.atoms.size() || bond.b >= type.atoms.size() ||
            bond.a == bond.b)
          return make_error(ErrorCode::InvalidInput,
                            "molecule type '" + type.name +
                                "' has a bond/constraint with an invalid atom "
                                "index");
    for (const TopologyMoleculeBlock &block : out_.molecules)
      if (!type_index_.contains(block.name))
        return make_error(ErrorCode::InvalidInput,
                          "[ molecules ] names undefined molecule type '" +
                              block.name + "'");
    return std::move(out_);
  }

private:
  bool active() const {
    if (conditionals_.empty())
      return true;
    const Conditional &c = conditionals_.back();
    return c.parent_active && (c.in_else ? !c.taken : c.taken);
  }

  std::unexpected<Error> fail(const fs::path &file, std::size_t line,
                              const std::string &what) const {
    return make_error(ErrorCode::InvalidInput,
                      file.string() + ":" + std::to_string(line) + ": " + what);
  }

  Result<void> read_file(const fs::path &file, std::size_t depth) {
    if (depth > options_.max_include_depth)
      return make_error(ErrorCode::ResourceLimit,
                        "topology include depth exceeds declared maximum at " +
                            file.string());
    const fs::path normal = file.lexically_normal();
    if (std::find(stack_.begin(), stack_.end(), normal) != stack_.end())
      return make_error(ErrorCode::InvalidInput,
                        "recursive #include of " + normal.string());
    std::ifstream in(file);
    if (!in.is_open())
      return make_error(ErrorCode::InvalidInput,
                        "cannot open topology file " + file.string());
    stack_.push_back(normal);
    out_.files.push_back(normal.string());
    const std::size_t conditional_depth = conditionals_.size();
    std::string raw, logical;
    std::size_t line_no = 0, first_line = 0;
    while (std::getline(in, raw)) {
      ++line_no;
      if (!raw.empty() && raw.back() == '\r')
        raw.pop_back();
      // Continuation: a trailing backslash joins the next line.
      if (logical.empty())
        first_line = line_no;
      if (!raw.empty() && raw.back() == '\\') {
        raw.pop_back();
        logical += raw;
        logical += ' ';
        continue;
      }
      logical += raw;
      std::string line = std::move(logical);
      logical.clear();
      if (auto ok = handle_line(file, first_line, line, depth); !ok)
        return ok;
    }
    if (!logical.empty())
      if (auto ok = handle_line(file, first_line, logical, depth); !ok)
        return ok;
    if (conditionals_.size() != conditional_depth)
      return fail(file, line_no, "#ifdef/#ifndef not closed in this file");
    stack_.pop_back();
    return {};
  }

  Result<void> handle_line(const fs::path &file, std::size_t line_no,
                           const std::string &raw, std::size_t depth) {
    std::string_view text = raw;
    if (const std::size_t semicolon = text.find(';');
        semicolon != std::string_view::npos)
      text = text.substr(0, semicolon);
    text = trim(text);
    if (text.empty())
      return {};
    if (text.front() == '#')
      return directive(file, line_no, text, depth);
    if (!active())
      return {};
    if (text.front() == '[') {
      const std::size_t close = text.find(']');
      if (close == std::string_view::npos)
        return fail(file, line_no, "unterminated section header");
      section_ = lower(trim(text.substr(1, close - 1)));
      if (section_ == "intermolecular_interactions") {
        intermolecular_ = true;
        current_.reset();
        out_.notes.push_back(
            file.string() + ":" + std::to_string(line_no) +
            ": [ intermolecular_interactions ] ignored; its bonds are not "
            "hard edges");
      } else if (section_ == "moleculetype" || section_ == "system" ||
                 section_ == "molecules") {
        intermolecular_ = false;
        if (section_ != "moleculetype")
          current_.reset();
      }
      return {};
    }
    return data(file, line_no, fields(text));
  }

  Result<void> directive(const fs::path &file, std::size_t line_no,
                         std::string_view text, std::size_t depth) {
    text.remove_prefix(1);
    text = trim(text);
    const std::size_t space = text.find_first_of(" \t");
    const std::string_view name = text.substr(0, space);
    const std::string_view arg =
        space == std::string_view::npos ? std::string_view{}
                                        : trim(text.substr(space));
    if (name == "ifdef" || name == "ifndef") {
      const std::string key(fields(arg).empty() ? std::string_view{}
                                                : fields(arg).front());
      if (key.empty())
        return fail(file, line_no, "#" + std::string(name) + " without a name");
      const bool defined = defines_.contains(key);
      conditionals_.push_back(
          {active(), name == "ifdef" ? defined : !defined, false});
      return {};
    }
    if (name == "else") {
      if (conditionals_.empty() || conditionals_.back().in_else)
        return fail(file, line_no, "#else without matching #ifdef/#ifndef");
      conditionals_.back().in_else = true;
      return {};
    }
    if (name == "endif") {
      if (conditionals_.empty())
        return fail(file, line_no, "#endif without matching #ifdef/#ifndef");
      conditionals_.pop_back();
      return {};
    }
    if (name == "if" || name == "elif")
      return fail(file, line_no,
                  "#" + std::string(name) +
                      " is not supported by the GROMACS preprocessor subset");
    if (!active())
      return {};
    if (name == "define") {
      const auto parts = fields(arg);
      if (parts.empty())
        return fail(file, line_no, "#define without a name");
      const std::string key(parts.front());
      const std::string_view rest = trim(arg.substr(parts.front().size()));
      defines_[key] = std::string(rest);
      return {};
    }
    if (name == "undef") {
      const auto parts = fields(arg);
      if (!parts.empty())
        defines_.erase(std::string(parts.front()));
      return {};
    }
    if (name == "include") {
      const auto target = TitanTopo::extract_include_target(
          std::string("#include ") + std::string(arg));
      if (!target)
        return fail(file, line_no, "malformed #include");
      auto candidates =
          TitanTopo::build_include_candidates(file.string(), *target);
      for (const fs::path &dir : options_.include_directories)
        candidates.push_back((dir / *target).lexically_normal());
      for (const fs::path &candidate : candidates) {
        std::error_code ec;
        if (fs::is_regular_file(candidate, ec))
          return read_file(candidate, depth + 1);
      }
      return fail(file, line_no, "#include target not found: " + *target);
    }
    return fail(file, line_no,
                "unsupported preprocessor directive #" + std::string(name));
  }

  Result<void> data(const fs::path &file, std::size_t line_no,
                    const std::vector<std::string_view> &f) {
    if (section_ == "moleculetype") {
      if (f.empty())
        return {};
      const std::string name(f[0]);
      if (type_index_.contains(name))
        return fail(file, line_no, "duplicate molecule type '" + name + "'");
      TopologyMoleculeType type;
      type.name = name;
      type.source = file.string();
      if (f.size() > 1 && !parse_int(f[1], type.nrexcl))
        return fail(file, line_no, "invalid nrexcl");
      type_index_[name] = out_.molecule_types.size();
      current_ = out_.molecule_types.size();
      out_.molecule_types.push_back(std::move(type));
      return {};
    }
    if (section_ == "molecules") {
      std::uint64_t count = 0;
      if (f.size() < 2 || !parse_int(f[1], count))
        return fail(file, line_no, "invalid [ molecules ] line");
      out_.molecules.push_back({std::string(f[0]), count});
      return {};
    }
    const bool atoms = section_ == "atoms";
    const bool bonds = section_ == "bonds";
    const bool constraints = section_ == "constraints";
    const bool settles = section_ == "settles";
    const bool vsites = section_.starts_with("virtual_sites");
    if (!atoms && !bonds && !constraints && !settles && !vsites)
      return {};
    if (intermolecular_)
      return {};
    if (!current_)
      return fail(file, line_no,
                  "[ " + section_ + " ] outside a [ moleculetype ]");
    TopologyMoleculeType &type = out_.molecule_types[*current_];
    if (atoms) {
      // nr type resnr residue atom cgnr [charge [mass]]
      if (f.size() < 5)
        return fail(file, line_no, "short [ atoms ] line");
      std::uint64_t nr = 0;
      TopologyAtom atom;
      if (!parse_int(f[0], nr) || !parse_int(f[2], atom.residue_number))
        return fail(file, line_no, "invalid [ atoms ] numbers");
      if (nr != type.atoms.size() + 1)
        return fail(file, line_no,
                    "atoms of '" + type.name + "' are not numbered "
                    "consecutively from 1");
      atom.type = std::string(f[1]);
      atom.residue_name = std::string(f[3]);
      atom.atom_name = std::string(f[4]);
      if (f.size() > 6 && !parse_double(f[6], atom.charge))
        return fail(file, line_no, "invalid charge");
      if (f.size() > 7 && !parse_double(f[7], atom.mass))
        return fail(file, line_no, "invalid mass");
      type.atoms.push_back(std::move(atom));
      return {};
    }
    if (settles) {
      // OW funct doh dhh: hydrogens follow the oxygen.
      std::uint32_t ow = 0;
      if (f.empty() || !parse_int(f[0], ow) || ow == 0)
        return fail(file, line_no, "invalid [ settles ] line");
      type.bonds.push_back({ow - 1, ow, 0, TopologyLinkKind::Settle});
      type.bonds.push_back({ow - 1, ow + 1, 0, TopologyLinkKind::Settle});
      return {};
    }
    if (vsites)
      return virtual_site(file, line_no, f, type);
    std::uint32_t ai = 0, aj = 0;
    int funct = 0;
    if (f.size() < 3 || !parse_int(f[0], ai) || !parse_int(f[1], aj) ||
        !parse_int(f[2], funct) || ai == 0 || aj == 0)
      return fail(file, line_no, "invalid [ " + section_ + " ] line");
    type.bonds.push_back({ai - 1, aj - 1, funct,
                          constraints ? TopologyLinkKind::Constraint
                                      : TopologyLinkKind::Bond});
    return {};
  }

  // virtual_sites1: site ai funct; virtual_sites2: site ai aj funct ...;
  // virtual_sites3: site ai aj ak funct ...; virtual_sites4: site ai aj ak al
  // funct ...; virtual_sitesn: site funct a1 a2 ... (funct 3, COW: atom
  // weight pairs).
  Result<void> virtual_site(const fs::path &file, std::size_t line_no,
                            const std::vector<std::string_view> &f,
                            TopologyMoleculeType &type) {
    std::uint32_t site = 0;
    if (f.empty() || !parse_int(f[0], site) || site == 0)
      return fail(file, line_no, "invalid [ " + section_ + " ] site");
    std::vector<std::string_view> from;
    int funct = 0;
    if (section_ == "virtual_sitesn") {
      if (f.size() < 3 || !parse_int(f[1], funct))
        return fail(file, line_no, "invalid [ virtual_sitesn ] line");
      for (std::size_t k = 2; k < f.size(); k += funct == 3 ? 2 : 1)
        from.push_back(f[k]);
    } else {
      std::size_t count = 0;
      if (section_ == "virtual_sites1")
        count = 1;
      else if (section_ == "virtual_sites2")
        count = 2;
      else if (section_ == "virtual_sites3")
        count = 3;
      else if (section_ == "virtual_sites4")
        count = 4;
      else
        return fail(file, line_no, "unknown section [ " + section_ + " ]");
      if (f.size() < count + 2 || !parse_int(f[count + 1], funct))
        return fail(file, line_no, "short [ " + section_ + " ] line");
      from.assign(f.begin() + 1, f.begin() + 1 + static_cast<std::ptrdiff_t>(count));
    }
    for (const std::string_view atom : from) {
      std::uint32_t a = 0;
      if (!parse_int(atom, a) || a == 0)
        return fail(file, line_no,
                    "invalid constructing atom in [ " + section_ + " ]");
      type.bonds.push_back({site - 1, a - 1, funct, TopologyLinkKind::VirtualSite});
    }
    return {};
  }

  const TopologyReadOptions &options_;
  GromacsTopology out_;
  std::unordered_map<std::string, std::string> defines_;
  std::unordered_map<std::string, std::size_t> type_index_;
  std::vector<fs::path> stack_;
  std::vector<Conditional> conditionals_;
  std::string section_;
  std::optional<std::size_t> current_;
  bool intermolecular_ = false;
};

} // namespace

Result<GromacsTopology> read_gromacs_topology(const fs::path &topology,
                                              const TopologyReadOptions &options) {
  Reader reader(options);
  return reader.run(topology);
}

} // namespace titan_ptopo
