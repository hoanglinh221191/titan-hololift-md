#ifndef TITAN_PBE_TABI_SES_PATCH_PROVENANCE_H
#define TITAN_PBE_TABI_SES_PATCH_PROVENANCE_H

#include "tabi_analytic_ses.h"
#include "tabi_ses_patch_fingerprint.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace TitanPBE::Tabi::analytic_ses {

static_assert(static_cast<std::uint32_t>(PatchKind::contact) == 0U);
static_assert(static_cast<std::uint32_t>(PatchKind::toroidal) == 1U);
static_assert(static_cast<std::uint32_t>(PatchKind::reentrant) == 2U);

// Stable compact identity used only for analytic-patch provenance.  The high
// two bits store PatchKind and the low thirty bits store the kind-local index.
// Kind value three is reserved for the invalid sentinel, so every conversion
// from size_t must be checked.
class PatchRefId {
public:
  static constexpr unsigned index_bits = 30U;
  static constexpr std::uint32_t index_mask =
      (std::uint32_t{1} << index_bits) - 1U;
  static constexpr std::uint32_t invalid_value =
      std::numeric_limits<std::uint32_t>::max();

  constexpr PatchRefId() noexcept = default;

  [[nodiscard]] static bool try_make(PatchKind kind, std::size_t index,
                                     PatchRefId &result) noexcept {
    const auto kind_value = static_cast<std::uint32_t>(kind);
    if (kind_value > static_cast<std::uint32_t>(PatchKind::reentrant) ||
        index > index_mask) {
      result = {};
      return false;
    }
    result.value_ =
        (kind_value << index_bits) | static_cast<std::uint32_t>(index);
    return true;
  }

  [[nodiscard]] static bool try_make(LocalPatchRef ref,
                                     PatchRefId &result) noexcept {
    return try_make(ref.kind, ref.patch_index, result);
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return value_ != invalid_value &&
           (value_ >> index_bits) <=
               static_cast<std::uint32_t>(PatchKind::reentrant);
  }

  [[nodiscard]] constexpr PatchKind kind() const noexcept {
    return valid() ? static_cast<PatchKind>(value_ >> index_bits)
                   : PatchKind::contact;
  }

  [[nodiscard]] constexpr std::size_t index() const noexcept {
    return static_cast<std::size_t>(value_ & index_mask);
  }

  [[nodiscard]] constexpr std::uint32_t packed() const noexcept {
    return value_;
  }

  [[nodiscard]] friend constexpr bool operator==(PatchRefId,
                                                 PatchRefId) noexcept = default;

  [[nodiscard]] friend constexpr bool operator<(PatchRefId left,
                                                PatchRefId right) noexcept {
    return left.value_ < right.value_;
  }

private:
  std::uint32_t value_{invalid_value};
};

static_assert(sizeof(PatchRefId) == sizeof(std::uint32_t));
static_assert(std::is_trivially_copyable_v<PatchRefId>);

enum class PatchRelation : std::uint8_t {
  same_surface,
  curve_adjacent,
  point_only,
  separated,
  uncertified
};

// Shared certificate for the bounded analytic seams.  It is built by the
// exact logic used by the closest-point index, so provenance and projection
// cannot silently drift to different seam tolerances.
struct ReentrantEdgeSeamCertificate {
  PatchRefId reentrant;
  PatchRefId torus;
  std::uint8_t edge{0};
  PatchRelation relation{PatchRelation::uncertified};
};

struct AnalyticPatchSeamCertificate {
  // Side zero is the beta seam on atom_i and side one is the beta seam on
  // atom_j.  Keeping them separate prevents one invalid side from granting
  // contact adjacency on the other.
  std::vector<std::array<std::uint8_t, 2>> torus_contact_curve_certified;
  // Exactly three records per reentrant patch, in patch-major/edge-major
  // order, when structure_certified is true.
  std::vector<ReentrantEdgeSeamCertificate> reentrant_edges;
  std::size_t certified_torus_contact_seams{0};
  std::size_t certified_torus_reentrant_seams{0};
  std::size_t uncertified_seams{0};
  bool structure_certified{false};
  bool all_seams_certified{false};
  std::string status{"not_built"};
  const PatchComplex *source_complex{nullptr};

  // Exact vector payload capacity in bytes. Fixed object fields and the
  // implementation-dependent small-string/status allocation are excluded.
  [[nodiscard]] std::size_t retained_payload_bytes() const noexcept {
    return torus_contact_curve_certified.capacity() *
               sizeof(std::array<std::uint8_t, 2>) +
           reentrant_edges.capacity() * sizeof(ReentrantEdgeSeamCertificate);
  }
};

enum class SeamCertificateStorage : std::uint8_t {
  summary_only,
  retain_relations
};

[[nodiscard]] AnalyticPatchSeamCertificate
build_analytic_patch_seam_certificate(
    const PatchComplex &complex,
    SeamCertificateStorage storage = SeamCertificateStorage::retain_relations,
    std::span<const std::size_t> atom_offsets = {},
    std::span<const LocalPatchRef> atom_patches = {});

// One canonical ref is retained for every unique crossing.  Only crossings
// carrying multiple patch identities allocate a sparse seam record and a
// complete sorted ref list.  This represents every ref while avoiding a full
// per-crossing CSR offset array.
struct SparsePatchSeamGroup {
  std::uint32_t crossing_index{0};
  std::uint32_t refs_begin{0};
  std::uint32_t refs_end{0};

  [[nodiscard]] friend constexpr bool
  operator==(SparsePatchSeamGroup, SparsePatchSeamGroup) noexcept = default;
};

// A single-patch coordinate normally carries one analytic root.  A sparse
// record is retained when two-or-more roots of that same patch merge at the
// coordinate tolerance. This is non-certifying evidence: without crossing
// orientation or an implicit derivative, the roots may still delimit a real
// narrow entry/exit interval and remain unclassified in Phase 2A.
struct SparseRootMultiplicity {
  std::uint32_t crossing_index{0};
  std::uint32_t multiplicity{0};

  [[nodiscard]] friend constexpr bool
  operator==(SparseRootMultiplicity, SparseRootMultiplicity) noexcept = default;
};

struct AxisPatchProvenance {
  std::vector<PatchRefId> canonical_refs;
  std::vector<SparsePatchSeamGroup> seam_groups;
  std::vector<PatchRefId> seam_refs;
  // Parallel to seam_refs.  This preserves each patch's root multiplicity
  // without widening the 16-byte patch-major event.
  std::vector<std::uint32_t> seam_ref_multiplicities;
  std::vector<SparseRootMultiplicity> single_ref_multiplicities;

  [[nodiscard]] std::span<const PatchRefId>
  refs(std::size_t crossing_index) const noexcept {
    if (crossing_index >= canonical_refs.size())
      return {};
    const auto found = std::lower_bound(
        seam_groups.begin(), seam_groups.end(), crossing_index,
        [](const SparsePatchSeamGroup &group, std::size_t index) {
          return static_cast<std::size_t>(group.crossing_index) < index;
        });
    if (found == seam_groups.end() || found->crossing_index != crossing_index)
      return {canonical_refs.data() + crossing_index, 1U};
    if (found->refs_begin >= found->refs_end ||
        found->refs_end > seam_refs.size())
      return {};
    return {seam_refs.data() + found->refs_begin,
            static_cast<std::size_t>(found->refs_end - found->refs_begin)};
  }

  [[nodiscard]] std::uint32_t
  ref_multiplicity(std::size_t crossing_index,
                   std::size_t ref_ordinal) const noexcept {
    if (crossing_index >= canonical_refs.size())
      return 0U;
    const auto seam = std::lower_bound(
        seam_groups.begin(), seam_groups.end(), crossing_index,
        [](const SparsePatchSeamGroup &group, std::size_t index) {
          return static_cast<std::size_t>(group.crossing_index) < index;
        });
    if (seam != seam_groups.end() && seam->crossing_index == crossing_index) {
      const std::size_t count =
          static_cast<std::size_t>(seam->refs_end - seam->refs_begin);
      if (ref_ordinal >= count ||
          seam->refs_begin + ref_ordinal >= seam_ref_multiplicities.size())
        return 0U;
      return seam_ref_multiplicities[seam->refs_begin + ref_ordinal];
    }
    if (ref_ordinal != 0U)
      return 0U;
    const auto repeated = std::lower_bound(
        single_ref_multiplicities.begin(), single_ref_multiplicities.end(),
        crossing_index,
        [](const SparseRootMultiplicity &group, std::size_t index) {
          return static_cast<std::size_t>(group.crossing_index) < index;
        });
    return repeated != single_ref_multiplicities.end() &&
                   repeated->crossing_index == crossing_index
               ? repeated->multiplicity
               : 1U;
  }

  [[nodiscard]] bool root_event_count(std::size_t crossing_index,
                                      std::size_t &count) const noexcept {
    const auto patch_refs = refs(crossing_index);
    count = 0U;
    if (patch_refs.empty())
      return false;
    for (std::size_t ordinal = 0; ordinal < patch_refs.size(); ++ordinal) {
      const std::uint32_t multiplicity =
          ref_multiplicity(crossing_index, ordinal);
      if (multiplicity == 0U ||
          count > std::numeric_limits<std::size_t>::max() - multiplicity) {
        count = 0U;
        return false;
      }
      count += multiplicity;
    }
    return true;
  }

  // refs_ must already be sorted and unique.  multiplicities_ is either empty
  // (all ones) or parallel to refs_ and strictly positive.
  [[nodiscard]] bool
  append(std::span<const PatchRefId> refs_,
         std::span<const std::uint32_t> multiplicities_ = {}) {
    return append_with_limits(refs_, multiplicities_,
                              std::numeric_limits<std::uint32_t>::max(),
                              std::numeric_limits<std::uint32_t>::max(),
                              std::numeric_limits<std::uint32_t>::max(),
                              std::numeric_limits<std::uint32_t>::max());
  }

#ifdef TITAN_TABI_SES_TEST_HOOKS
  [[nodiscard]] bool
  append_with_limits_for_test(std::span<const PatchRefId> refs_,
                              std::span<const std::uint32_t> multiplicities_,
                              std::size_t maximum_crossing_index,
                              std::size_t maximum_seam_refs,
                              std::size_t maximum_sparse_groups) {
    return append_with_limits(refs_, multiplicities_, maximum_crossing_index,
                              maximum_seam_refs, maximum_sparse_groups,
                              maximum_sparse_groups);
  }
#endif

private:
  [[nodiscard]] bool append_with_limits(
      std::span<const PatchRefId> refs_,
      std::span<const std::uint32_t> multiplicities_,
      std::size_t maximum_crossing_index, std::size_t maximum_seam_refs,
      std::size_t maximum_seam_groups, std::size_t maximum_single_ref_groups) {
    if (refs_.empty())
      return false;
    if (!multiplicities_.empty() && multiplicities_.size() != refs_.size())
      return false;
    for (std::size_t index = 0; index < refs_.size(); ++index) {
      if (!refs_[index].valid() ||
          (index != 0U && !(refs_[index - 1U] < refs_[index])) ||
          (!multiplicities_.empty() && multiplicities_[index] == 0U))
        return false;
    }
    const std::size_t crossing_index = canonical_refs.size();
    const std::uint32_t single_multiplicity =
        multiplicities_.empty() ? 1U : multiplicities_.front();
    const bool needs_single_multiplicity =
        refs_.size() == 1U && single_multiplicity > 1U;
    if (seam_ref_multiplicities.size() != seam_refs.size() ||
        seam_refs.size() > maximum_seam_refs ||
        seam_groups.size() > maximum_seam_groups ||
        single_ref_multiplicities.size() > maximum_single_ref_groups)
      return false;
    if (maximum_crossing_index > std::numeric_limits<std::uint32_t>::max() ||
        maximum_seam_refs > std::numeric_limits<std::uint32_t>::max() ||
        maximum_seam_groups > std::numeric_limits<std::uint32_t>::max() ||
        maximum_single_ref_groups > std::numeric_limits<std::uint32_t>::max() ||
        crossing_index > maximum_crossing_index ||
        (refs_.size() > 1U &&
         (refs_.size() > maximum_seam_refs ||
          seam_refs.size() > maximum_seam_refs - refs_.size() ||
          seam_groups.size() >= maximum_seam_groups)) ||
        (needs_single_multiplicity &&
         single_ref_multiplicities.size() >= maximum_single_ref_groups))
      return false;
    const std::size_t old_canonical_size = canonical_refs.size();
    const std::size_t old_group_size = seam_groups.size();
    const std::size_t old_seam_ref_size = seam_refs.size();
    const std::size_t old_seam_multiplicity_size =
        seam_ref_multiplicities.size();
    const std::size_t old_single_multiplicity_size =
        single_ref_multiplicities.size();
    try {
      canonical_refs.push_back(refs_.front());
      if (refs_.size() > 1U) {
        const auto begin = static_cast<std::uint32_t>(seam_refs.size());
        seam_refs.insert(seam_refs.end(), refs_.begin(), refs_.end());
        if (multiplicities_.empty()) {
          seam_ref_multiplicities.insert(seam_ref_multiplicities.end(),
                                         refs_.size(), 1U);
        } else {
          seam_ref_multiplicities.insert(seam_ref_multiplicities.end(),
                                         multiplicities_.begin(),
                                         multiplicities_.end());
        }
        seam_groups.push_back({static_cast<std::uint32_t>(crossing_index),
                               begin,
                               static_cast<std::uint32_t>(seam_refs.size())});
      } else if (needs_single_multiplicity) {
        single_ref_multiplicities.push_back(
            {static_cast<std::uint32_t>(crossing_index), single_multiplicity});
      }
    } catch (...) {
      canonical_refs.resize(old_canonical_size);
      seam_groups.resize(old_group_size);
      seam_refs.resize(old_seam_ref_size);
      seam_ref_multiplicities.resize(old_seam_multiplicity_size);
      single_ref_multiplicities.resize(old_single_multiplicity_size);
      throw;
    }
    return true;
  }

public:
  [[nodiscard]] bool validate(std::size_t crossing_count) const noexcept {
    if (canonical_refs.size() != crossing_count)
      return false;
    for (const PatchRefId ref : canonical_refs) {
      if (!ref.valid())
        return false;
    }
    if (seam_ref_multiplicities.size() != seam_refs.size())
      return false;
    std::size_t expected_refs_begin = 0U;
    std::size_t previous_crossing = 0U;
    bool first_group = true;
    for (const SparsePatchSeamGroup &group : seam_groups) {
      if (group.crossing_index >= crossing_count ||
          (!first_group && group.crossing_index <= previous_crossing) ||
          group.refs_begin != expected_refs_begin ||
          group.refs_end <= group.refs_begin + 1U ||
          group.refs_end > seam_refs.size())
        return false;
      const auto group_refs = std::span<const PatchRefId>(
          seam_refs.data() + group.refs_begin,
          static_cast<std::size_t>(group.refs_end - group.refs_begin));
      if (group_refs.front() != canonical_refs[group.crossing_index])
        return false;
      for (std::size_t index = 0; index < group_refs.size(); ++index) {
        if (!group_refs[index].valid() ||
            seam_ref_multiplicities[group.refs_begin + index] == 0U ||
            (index != 0U && !(group_refs[index - 1U] < group_refs[index])))
          return false;
      }
      expected_refs_begin = group.refs_end;
      previous_crossing = group.crossing_index;
      first_group = false;
    }
    if (expected_refs_begin != seam_refs.size())
      return false;
    std::size_t previous_multiplicity_crossing = 0U;
    bool first_multiplicity = true;
    for (const SparseRootMultiplicity &group : single_ref_multiplicities) {
      if (group.crossing_index >= crossing_count || group.multiplicity <= 1U ||
          (!first_multiplicity &&
           group.crossing_index <= previous_multiplicity_crossing))
        return false;
      const auto seam = std::lower_bound(
          seam_groups.begin(), seam_groups.end(), group.crossing_index,
          [](const SparsePatchSeamGroup &candidate, std::uint32_t crossing) {
            return candidate.crossing_index < crossing;
          });
      if (seam != seam_groups.end() &&
          seam->crossing_index == group.crossing_index)
        return false;
      previous_multiplicity_crossing = group.crossing_index;
      first_multiplicity = false;
    }
    return true;
  }

  [[nodiscard]] bool retained_payload_bytes(std::size_t &bytes) const noexcept {
    bytes = 0U;
    const auto add_vector = [&](std::size_t capacity, std::size_t width) {
      if (capacity > std::numeric_limits<std::size_t>::max() / width)
        return false;
      const std::size_t contribution = capacity * width;
      if (bytes > std::numeric_limits<std::size_t>::max() - contribution)
        return false;
      bytes += contribution;
      return true;
    };
    return add_vector(canonical_refs.capacity(), sizeof(PatchRefId)) &&
           add_vector(seam_groups.capacity(), sizeof(SparsePatchSeamGroup)) &&
           add_vector(seam_refs.capacity(), sizeof(PatchRefId)) &&
           add_vector(seam_ref_multiplicities.capacity(),
                      sizeof(std::uint32_t)) &&
           add_vector(single_ref_multiplicities.capacity(),
                      sizeof(SparseRootMultiplicity));
  }

  [[nodiscard]] std::size_t retained_payload_bytes() const noexcept {
    std::size_t bytes = 0U;
    return retained_payload_bytes(bytes)
               ? bytes
               : std::numeric_limits<std::size_t>::max();
  }
};

struct PatchRayProvenance {
  std::array<AxisPatchProvenance, 3> axes;
  PatchComplexFingerprint complex_fingerprint;
  // Exact audit-on identity of the lattice that generated the retained CSR.
  // Scalars are stored as IEEE-754 bit patterns, so stale signed zero or a
  // one-ULP spacing/lower change cannot silently alter row decoding.
  struct LatticeIdentity {
    std::array<std::uint64_t, 4> scalar_bits{};
    std::array<std::size_t, 3> counts{};
    bool valid{false};

    [[nodiscard]] friend constexpr bool
    operator==(LatticeIdentity, LatticeIdentity) noexcept = default;
  } lattice_identity;
  std::size_t unique_crossings{0};
  std::size_t seam_groups{0};
  std::size_t refs_total{0};
  std::size_t max_refs_per_crossing{0};
  std::size_t roots_total{0};
  std::size_t repeated_root_groups{0};
  // Multiplicity alone does not prove a tangency: two merged roots of one
  // primitive can still bound a very narrow entry/exit interval.  Until
  // crossing orientation/derivative evidence is retained, these remain
  // explicitly unclassified.
  std::size_t same_primitive_near_pair_unclassified{0};
  std::size_t retained_bytes{0};
  std::size_t peak_temporary_bytes{0};
  double coordinate_merge_tolerance{0.0};
  double near_pair_tolerance{0.0};
  double build_ms{0.0};
  bool enabled{false};
  bool certified{false};
  std::string status{"disabled"};
};

class AnalyticPatchAdjacencyOracle {
public:
  // This is a topology-level potential relation. In particular,
  // curve_adjacent does not authorize a mesh edge at an arbitrary location;
  // Phase 2B must additionally certify that the edge lies on that seam.
  [[nodiscard]] PatchRelation relation(PatchRefId first,
                                       PatchRefId second) const noexcept;
  [[nodiscard]] bool certified() const noexcept { return certified_; }
  [[nodiscard]] std::string_view status() const noexcept { return status_; }
  [[nodiscard]] double build_ms() const noexcept { return build_ms_; }
  // Exact vector payload capacity; fixed fields/status storage are excluded.
  [[nodiscard]] std::size_t retained_payload_bytes() const noexcept;
  [[nodiscard]] const AnalyticPatchSeamCertificate &
  seam_certificate() const noexcept {
    return seams_;
  }

private:
  struct Bounds {
    Vec3 lower;
    Vec3 upper;
  };
  struct ContactData {
    std::uint32_t atom{0};
    Vec3 center;
    double radius{0.0};
    Bounds bounds;
  };
  struct TorusAtoms {
    std::uint32_t atom_i{0};
    std::uint32_t atom_j{0};
    Bounds bounds;
  };
  struct ReentrantData {
    std::array<std::uint32_t, 3> atoms{};
    Vec3 probe_center;
    std::array<Vec3, 3> contact_directions{};
    Bounds bounds;
  };
  struct TripleJunction {
    PatchRefId contact;
    PatchRefId torus_first;
    PatchRefId torus_second;
    PatchRefId reentrant;
    Vec3 point;
    std::uint8_t certified{0};
  };

  std::vector<ContactData> contacts_;
  std::vector<TorusAtoms> torus_atoms_;
  std::vector<ReentrantData> reentrant_;
  // Patch-major, then reentrant vertex/atom-major.
  std::vector<TripleJunction> junctions_;
  // Indices of certified junctions, sorted by the unordered torus pair.
  std::vector<std::uint32_t> torus_junction_order_;
  AnalyticPatchSeamCertificate seams_;
  double probe_radius_{0.0};
  double build_ms_{0.0};
  bool certified_{false};
  std::string status_{"not_built"};

  friend AnalyticPatchAdjacencyOracle build_analytic_patch_adjacency_oracle(
      const PatchComplex &, const struct LocalPatchProjectionIndex &);
};

struct LocalPatchProjectionIndex;

// The prepared closest-point index is identity-bound to the same PatchComplex.
// Cross-patch relations fail closed if that binding or its complete bounded
// domain certificate is absent. A valid identical ref remains same_surface
// even when an unrelated seam elsewhere in the complex is uncertified.
[[nodiscard]] AnalyticPatchAdjacencyOracle
build_analytic_patch_adjacency_oracle(const PatchComplex &complex,
                                      const LocalPatchProjectionIndex &index);

struct PatchRayProvenanceAudit {
  std::array<std::size_t, 5> relation_counts{};
  std::size_t seam_groups_audited{0};
  std::size_t curve_adjacent_groups{0};
  // A point-only topology relation never authorizes a mesh edge; Phase 2B
  // would still need the crossing location to equal that certified junction.
  std::size_t point_only_groups{0};
  std::size_t incompatible_crossing_groups{0};
  std::size_t uncertified_crossing_groups{0};
  std::size_t same_primitive_merged_groups{0};
  std::size_t same_primitive_adjacent_near_pairs{0};
  // Summary of the preceding two non-certifying same-primitive cases.
  std::size_t same_primitive_near_pair_unclassified{0};
  std::size_t adjacent_near_pairs{0};
  std::array<std::size_t, 5> near_pair_relation_counts{};
  std::size_t incompatible_near_pairs{0};
  std::size_t uncertified_near_pairs{0};
  std::size_t incompatible_active_cells{0};
  std::size_t uncertified_active_cells{0};
  std::size_t incompatible_mesh_edges{0};
  std::size_t uncertified_mesh_edges{0};
  std::size_t incompatible_mesh_faces{0};
  std::size_t uncertified_mesh_faces{0};
  std::size_t oracle_retained_bytes{0};
  double oracle_build_ms{0.0};
  double audit_ms{0.0};
  // True only when every retained pair received a topology-level relation.
  // This does not prove that coordinate-merged refs lie on their shared seam.
  bool relation_audit_complete{false};
  // Phase 2A deliberately retains neither each ref's raw coordinate spread nor
  // crossing derivative/orientation, so it makes no location certificate.
  bool crossing_location_certified{false};
  bool active_cell_mapping_certified{false};
  bool mesh_mapping_certified{false};
  std::string status{"not_run"};
  std::string active_cell_mapping_status{"active_cell_mapping_uncertified"};
  std::string mesh_mapping_status{"mesh_mapping_uncertified"};
};

struct PatchRayGrid;

// Audit-only: coordinates, parity and connectivity are never mutated.
[[nodiscard]] PatchRayProvenanceAudit
audit_patch_ray_provenance(const PatchRayGrid &grid,
                           const AnalyticPatchAdjacencyOracle &oracle);

} // namespace TitanPBE::Tabi::analytic_ses

#endif // TITAN_PBE_TABI_SES_PATCH_PROVENANCE_H
