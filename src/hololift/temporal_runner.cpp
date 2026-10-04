// HoloLift temporal selection on a VIBE package: the body of TITAN's -pbctopo-temporal=hololift and of
// tools/hololift_temporal_value_runner.cpp (moved here unchanged apart from reading HoloLiftRunOptions
// instead of the environment), with the coordinate writer of the selected path.
#include "temporal_runner.h"

#include "frame_binding.h"
#include "identity.h"
#include "phase1.h"
#include "validation.h"
#include "vibe_importer.h"
#include "../box_type.h"
#include "../pbctopo/lattice_math.h"
#include "../gmxtraj/include/xdrfile.h"
#include "../gmxtraj/include/xdrfile_xtc.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <sstream>
#include <thread>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace titan_hololift;

namespace {

constexpr double kNmToAngstrom = 10.0;
constexpr double kTimeTolerancePs = 1.0e-4;

struct TrajectoryFrame {
  std::size_t trajectory_frame_index = 0;
  std::vector<HoloLiftSourceAtomKey> atom_order;
  std::vector<std::array<double, 3>> wrapped_coordinates;
  std::array<double, 9> box_matrix{};
};

struct TransportBandAuditRow {
  std::size_t frame_index = 0;
  double time_ps = 0.0;
  std::size_t source_frame_index = 0;
  std::ptrdiff_t source_offset = 0;
  bool source_band_candidate = false;
  std::size_t source_candidate_rank = 0;
  std::size_t candidate_rank = 0;
  bool source_selected_framewise = false;
  bool selected_framewise = false;
  bool synthetic_rank_inversion_frame = false;
  bool synthetic_domain_miss_frame = false;
  bool synthetic_domain_fence_frame = false;
  HoloLiftHash128 assignment_identity;
  std::vector<HoloLiftLatticeImage> images;
};

[[nodiscard]] bool same_source_segment(const HoloLiftObservationStore &store,
                                       std::size_t lhs,
                                       std::size_t rhs) noexcept {
  if (lhs >= store.frames().size() || rhs >= store.frames().size())
    return false;
  if (lhs > rhs)
    std::swap(lhs, rhs);
  if (store.frames()[lhs].topology_epoch_index !=
      store.frames()[rhs].topology_epoch_index)
    return false;
  for (std::size_t index = lhs + 1; index <= rhs; ++index) {
    if (store.frames()[index].source_relation !=
        HoloLiftSourceFrameRelation::ContiguousSourceFrame)
      return false;
  }
  return true;
}

[[nodiscard]] std::expected<
    std::pair<HoloLiftObservationStore, std::vector<TransportBandAuditRow>>,
    std::string>
build_temporal_transport_band(const HoloLiftObservationStore &source,
                              std::size_t radius,
                              bool synthetic_rank_inversion_v1,
                              bool synthetic_domain_miss_v1,
                              bool synthetic_domain_miss_v2) {
  if (radius == 0)
    return std::unexpected("temporal transport radius must be positive");
  const unsigned synthetic_mode_count =
      static_cast<unsigned>(synthetic_rank_inversion_v1) +
      static_cast<unsigned>(synthetic_domain_miss_v1) +
      static_cast<unsigned>(synthetic_domain_miss_v2);
  if (synthetic_mode_count > 1U) {
    return std::unexpected(
        "synthetic control modes are mutually exclusive");
  }

  HoloLiftObservationStoreBuilder builder;
  builder.api_version = source.api_version();
  builder.source_contract = source.source_contract();
  builder.source_contract.producer_version +=
      synthetic_domain_miss_v2
          ? "+hololift-temporal-transport-band-v9-domain-miss2"
          : synthetic_domain_miss_v1
                ? "+hololift-temporal-transport-band-v8-domain-miss1"
          : synthetic_rank_inversion_v1
                ? "+hololift-temporal-transport-band-v7-challenge1"
                : "+hololift-temporal-transport-band-v3";
  builder.source_contract.package_id +=
      "+transport-r" + std::to_string(radius);
  if (synthetic_rank_inversion_v1)
    builder.source_contract.package_id += "+synthetic-rank-inversion-v1";
  if (synthetic_domain_miss_v1)
    builder.source_contract.package_id += "+synthetic-domain-miss-v1";
  if (synthetic_domain_miss_v2)
    builder.source_contract.package_id += "+synthetic-domain-miss-v2";
  builder.source_coverage = source.source_coverage();
  builder.topology_epochs.assign(source.topology_epochs().begin(),
                                 source.topology_epochs().end());
  builder.components.assign(source.components().begin(),
                            source.components().end());
  builder.exact_atom_memberships.assign(source.exact_atom_memberships().begin(),
                                        source.exact_atom_memberships().end());
  builder.exact_atom_masses.assign(source.exact_atom_masses().begin(),
                                   source.exact_atom_masses().end());
  builder.hard_edges.assign(source.hard_edges().begin(),
                            source.hard_edges().end());
  builder.frames.reserve(source.frames().size());
  builder.candidates.reserve(source.candidates().size() *
                             (2 * radius + 1));
  builder.component_images.reserve(source.component_images().size() *
                                   (2 * radius + 1));

  std::vector<TransportBandAuditRow> audit_rows;
  audit_rows.reserve(source.frames().size() * (2 * radius + 1));

  struct Proposal {
    std::size_t source_frame_index = 0;
    std::ptrdiff_t source_offset = 0;
    std::size_t source_candidate_index = HOLOLIFT_NO_INDEX;
    bool source_band_candidate = false;
    std::vector<HoloLiftLatticeImage> images;
  };

  for (std::size_t frame_index = 0; frame_index < source.frames().size();
       ++frame_index) {
    const auto &source_frame = source.frames()[frame_index];
    HoloLiftFrameRecord frame = source_frame;
    frame.candidates = {builder.candidates.size(), 0};

    if (source_frame.candidates.count == 0 ||
        source_frame.selected_candidate_index == HOLOLIFT_NO_INDEX) {
      frame.selected_candidate_index = HOLOLIFT_NO_INDEX;
      builder.frames.push_back(std::move(frame));
      continue;
    }
    if (source_frame.topology_epoch_index >= source.topology_epochs().size())
      return std::unexpected("transport frame references an invalid epoch");
    const auto &epoch =
        source.topology_epochs()[source_frame.topology_epoch_index];
    if (!epoch.components.valid_for(source.components().size()))
      return std::unexpected("transport frame epoch component range is invalid");
    const auto components = source.components().subspan(
        epoch.components.begin, epoch.components.count);

    std::vector<Proposal> proposals;
    proposals.reserve(2 * radius + 1);
    const auto append_candidate = [&](std::size_t neighbor,
                                      std::ptrdiff_t offset,
                                      std::size_t candidate_index,
                                      bool source_band_candidate)
        -> std::expected<void, std::string> {
      if (!same_source_segment(source, frame_index, neighbor))
        return {};
      if (candidate_index == HOLOLIFT_NO_INDEX ||
          candidate_index >= source.candidates().size())
        return {};
      const auto &candidate = source.candidates()[candidate_index];
      if (!candidate.component_images.valid_for(source.component_images().size()) ||
          candidate.component_images.count != components.size()) {
        return std::unexpected(
            "transport source candidate component-image range is invalid");
      }
      std::vector<HoloLiftLatticeImage> images(
          source.component_images().begin() +
              static_cast<std::ptrdiff_t>(candidate.component_images.begin),
          source.component_images().begin() +
              static_cast<std::ptrdiff_t>(candidate.component_images.begin +
                                          candidate.component_images.count));
      const bool duplicate = std::any_of(
          proposals.begin(), proposals.end(), [&](const Proposal &existing) {
            return existing.images == images;
      });
      if (!duplicate)
        proposals.push_back({neighbor, offset, candidate_index,
                             source_band_candidate, std::move(images)});
      return {};
    };

    const std::size_t schedule_position = frame_index % 40;
    const bool synthetic_rank_inversion_frame =
        synthetic_rank_inversion_v1 && schedule_position >= 10 &&
        schedule_position < 15;
    const bool synthetic_domain_miss_frame =
        (synthetic_domain_miss_v1 || synthetic_domain_miss_v2) &&
        schedule_position >= 10 &&
        schedule_position < 15;
    const bool synthetic_domain_fence_frame =
        synthetic_domain_miss_v2 &&
        (schedule_position == 9 || schedule_position == 15);
    const bool incomplete_domain_frame =
        synthetic_domain_miss_frame || synthetic_domain_fence_frame;
    const bool synthetic_selected_frame =
        synthetic_rank_inversion_frame || synthetic_domain_miss_frame;
    std::size_t derived_selected_candidate_index =
        source_frame.selected_candidate_index;
    if (synthetic_selected_frame) {
      if (components.size() < 2)
        return std::unexpected(
            "synthetic selection control requires at least two components");
      const auto &source_selected =
          source.candidates()[source_frame.selected_candidate_index];
      if (!source_selected.component_images.valid_for(
              source.component_images().size()) ||
          source_selected.component_images.count != components.size()) {
        return std::unexpected(
            "synthetic selection control source selection is invalid");
      }
      std::vector<HoloLiftLatticeImage> target(
          source.component_images().begin() +
              static_cast<std::ptrdiff_t>(
                  source_selected.component_images.begin),
          source.component_images().begin() +
              static_cast<std::ptrdiff_t>(
                  source_selected.component_images.begin +
                  source_selected.component_images.count));
      if (target[1].z == std::numeric_limits<std::int64_t>::max())
        return std::unexpected(
            "synthetic selection control target image overflows");
      ++target[1].z;
      derived_selected_candidate_index = HOLOLIFT_NO_INDEX;
      for (std::size_t local_index = 0;
           local_index < source_frame.candidates.count; ++local_index) {
        const std::size_t candidate_index =
            source_frame.candidates.begin + local_index;
        const auto &candidate = source.candidates()[candidate_index];
        if (!candidate.component_images.valid_for(
                source.component_images().size()) ||
            candidate.component_images.count != components.size()) {
          return std::unexpected(
              "synthetic selection control candidate range is invalid");
        }
        const auto images = source.component_images().subspan(
            candidate.component_images.begin,
            candidate.component_images.count);
        if (std::equal(images.begin(), images.end(), target.begin())) {
          derived_selected_candidate_index = candidate_index;
          break;
        }
      }
      if (derived_selected_candidate_index == HOLOLIFT_NO_INDEX)
        return std::unexpected(
            "synthetic selection control target is absent from source band");
    }

    auto selected = append_candidate(frame_index, 0,
                                     derived_selected_candidate_index, true);
    if (!selected)
      return std::unexpected(selected.error());
    for (std::size_t local_index = 0;
         local_index < source_frame.candidates.count; ++local_index) {
      const std::size_t candidate_index =
          source_frame.candidates.begin + local_index;
      if (candidate_index == derived_selected_candidate_index)
        continue;
      if (synthetic_domain_miss_v2 && incomplete_domain_frame)
        continue;
      if (synthetic_domain_miss_frame &&
          candidate_index == source_frame.selected_candidate_index)
        continue;
      auto source_candidate =
          append_candidate(frame_index, 0, candidate_index, true);
      if (!source_candidate)
        return std::unexpected(source_candidate.error());
    }
    for (std::size_t distance = 1;
         !incomplete_domain_frame && distance <= radius; ++distance) {
      if (distance <= frame_index) {
        const std::size_t previous_frame = frame_index - distance;
        auto previous = append_candidate(
            previous_frame, -static_cast<std::ptrdiff_t>(distance),
            source.frames()[previous_frame].selected_candidate_index, false);
        if (!previous)
          return std::unexpected(previous.error());
      }
      if (distance <= source.frames().size() - frame_index - 1) {
        const std::size_t next_frame = frame_index + distance;
        auto next = append_candidate(
            next_frame, static_cast<std::ptrdiff_t>(distance),
            source.frames()[next_frame].selected_candidate_index, false);
        if (!next)
          return std::unexpected(next.error());
      }
    }
    if (proposals.empty())
      return std::unexpected("transport band contains no candidate");

    const auto &selected_source =
        source.candidates()[source_frame.selected_candidate_index];
    for (std::size_t local_index = 0; local_index < proposals.size();
         ++local_index) {
      const auto &proposal = proposals[local_index];
      HoloLiftCandidateRecord candidate =
          proposal.source_band_candidate
              ? source.candidates()[proposal.source_candidate_index]
              : selected_source;
      candidate.rank = local_index + 1;
      candidate.component_images = {builder.component_images.size(),
                                    proposal.images.size()};
      candidate.evidence.selected_framewise = local_index == 0;
      candidate.evidence.hard_feasible = true;
      candidate.evidence.search_equivalence_known =
          !incomplete_domain_frame;
      candidate.evidence.equivalent_to_search_best =
          !incomplete_domain_frame && local_index == 0;
      candidate.evidence.equivalent_under_output_order =
          !incomplete_domain_frame && local_index == 0;
      const bool source_touches_boundary =
          proposal.source_band_candidate &&
          candidate.domain_evidence.boundary_known &&
          candidate.domain_evidence.touches_shell_boundary;
      candidate.domain_evidence.boundary_known = true;
      const auto absolute_offset = proposal.source_offset < 0
                                       ? -proposal.source_offset
                                       : proposal.source_offset;
      candidate.domain_evidence.touches_shell_boundary =
          source_touches_boundary ||
          (proposal.source_offset != 0 &&
           static_cast<std::size_t>(absolute_offset) == radius);

      const auto signature = hololift_assignment_signature_canonical(
          frame.layout_signature, components, proposal.images);
      if (!signature)
        return std::unexpected(signature.error());
      const auto identity = hololift_assignment_identity128_canonical(
          frame.layout_identity, components, proposal.images);
      if (!identity)
        return std::unexpected(identity.error());
      candidate.assignment_signature = *signature;
      candidate.assignment_identity = *identity;
      candidate.evidence_relative_relation_signature = *signature;
      if (synthetic_selected_frame && local_index == 0) {
        candidate.evidence_compatible_hypothesis_hash =
            selected_source.evidence_compatible_hypothesis_hash;
        candidate.evidence_compatible_pair_hash =
            selected_source.evidence_compatible_pair_hash;
        candidate.evidence_compatible_hypothesis_count =
            selected_source.evidence_compatible_hypothesis_count;
        candidate.evidence_supported_relation_count =
            selected_source.evidence_supported_relation_count;
        candidate.evidence_compatible_contact_count =
            selected_source.evidence_compatible_contact_count;
        candidate.evidence_no_support_relation_count =
            selected_source.evidence_no_support_relation_count;
      }
      if (!proposal.source_band_candidate) {
        candidate.evidence_compatible_hypothesis_hash = {};
        candidate.evidence_compatible_pair_hash = {};
        candidate.evidence_compatible_hypothesis_count = 0;
        candidate.evidence_supported_relation_count = 0;
        candidate.evidence_compatible_contact_count = 0;
        candidate.evidence_no_support_relation_count =
            frame.provenance.soft_observed_component_relation_count;
      }
      const bool strong_bounded_singleton =
          !incomplete_domain_frame && local_index == 0 &&
          proposals.size() == 1 &&
          frame.evidence.identified &&
          (frame.provenance.status == HoloLiftFrameStatus::Certified ||
           frame.provenance.status == HoloLiftFrameStatus::Rescued) &&
          (frame.provenance.certificate_scope ==
               HoloLiftCertificateScope::ImageIntercomponentSteric ||
           frame.provenance.certificate_scope ==
               HoloLiftCertificateScope::ImageIntercomponentStericInterface) &&
          frame.provenance.certificate_graph_source !=
              HoloLiftCertificateGraphSource::None &&
          frame.provenance.hard_graph_source != HoloLiftHardGraphSource::None &&
          frame.provenance.temporal_policy ==
              HoloLiftTemporalPolicyStatus::FramewiseProduction &&
          !candidate.domain_evidence.touches_shell_boundary &&
          frame.spatial_lift_ambiguous_hard_edges == 0 &&
          frame.spatial_lift_cycle_residuals == 0;
      candidate.observation_class =
          !frame.evidence.identified
              ? HoloLiftObservationClass::WeakObservation
              : strong_bounded_singleton
                    ? HoloLiftObservationClass::StrongBoundedAnchor
                    : HoloLiftObservationClass::AmbiguousAnchor;

      builder.component_images.insert(builder.component_images.end(),
                                      proposal.images.begin(),
                                      proposal.images.end());
      builder.candidates.push_back(candidate);
      audit_rows.push_back(
          {frame_index, frame.time_ps, proposal.source_frame_index,
           proposal.source_offset, proposal.source_band_candidate,
           source.candidates()[proposal.source_candidate_index].rank,
           candidate.rank,
           proposal.source_candidate_index ==
               source_frame.selected_candidate_index,
           candidate.evidence.selected_framewise,
           synthetic_rank_inversion_frame,
           synthetic_domain_miss_frame,
           synthetic_domain_fence_frame,
           candidate.assignment_identity, proposal.images});
    }

    frame.candidates.count = proposals.size();
    frame.selected_candidate_index = frame.candidates.begin;
    frame.search_domain.shell_radius =
        std::max(radius, source_frame.search_domain.shell_radius);
    frame.search_domain.audit_known = true;
    frame.search_domain.evidence_complete = !incomplete_domain_frame;
    frame.search_domain.expansion_attempted = true;
    frame.search_domain.expansion_exhausted = !incomplete_domain_frame;
    frame.search_domain.bounded_domain_only = true;
    frame.search_domain.completeness =
        incomplete_domain_frame
            ? HoloLiftDomainCompleteness::Unknown
            : HoloLiftDomainCompleteness::BoundedExhaustive;
    frame.evidence.uniqueness_search_exhaustive =
        !incomplete_domain_frame;
    frame.evidence.objective_uniqueness_known =
        !incomplete_domain_frame;
    frame.evidence.objective_unique_within_search_domain =
        !incomplete_domain_frame && frame.evidence.identified;
    frame.evidence.evidence_uniqueness_known =
        !incomplete_domain_frame;
    frame.evidence.evidence_unique_within_search_domain =
        !incomplete_domain_frame && frame.evidence.identified &&
        proposals.size() == 1;
    frame.evidence.feasible_assignment_set_enumerated =
        !incomplete_domain_frame;
    frame.evidence.feasible_set_semantics =
        incomplete_domain_frame
            ? HoloLiftFeasibleSetSemantics::NotEnumerated
            : HoloLiftFeasibleSetSemantics::HardConstraintsWithinSearchDomain;
    frame.evidence.equivalent_set_complete =
        !incomplete_domain_frame;
    frame.evidence.credible_alternative_set_complete =
        !incomplete_domain_frame;
    frame.evidence.retained_certified_assignment_count = proposals.size();
    frame.evidence.search_equivalent_assignments_exported =
        incomplete_domain_frame ? 0 : 1;
    frame.evidence.equivalent_best_assignments_within_domain =
        incomplete_domain_frame ? 0 : 1;
    frame.evidence.feasible_assignments_within_domain =
        incomplete_domain_frame ? 0 : proposals.size();
    frame.evidence.soft_score_valid_assignments_within_domain =
        incomplete_domain_frame ? 0 : proposals.size();
    frame.evidence.search_domain_assignments_exported =
        incomplete_domain_frame ? 0 : proposals.size();
    const auto &derived_selected =
        builder.candidates[frame.selected_candidate_index];
    frame.provenance.framewise_assignment_signature =
        derived_selected.assignment_signature;
    frame.provenance.framewise_assignment_identity =
        derived_selected.assignment_identity;
    frame.provenance.output_assignment_signature =
        derived_selected.assignment_signature;
    frame.provenance.output_assignment_identity =
        derived_selected.assignment_identity;
    builder.frames.push_back(std::move(frame));
  }

  auto finalized = finalize_hololift_observation_store(std::move(builder));
  if (!finalized) {
    return std::unexpected(std::string(hololift_validation_code_name(
                               finalized.error().code)) +
                           ": " + finalized.error().message);
  }
  return std::pair<HoloLiftObservationStore,
                   std::vector<TransportBandAuditRow>>{
      std::move(*finalized), std::move(audit_rows)};
}

[[nodiscard]] std::string import_error_text(
    const HoloLiftVibeImportError &error) {
  std::ostringstream out;
  out << hololift_vibe_import_code_name(error.code) << ": "
      << error.message;
  if (!error.artifact.empty())
    out << " [" << error.artifact.string() << ']';
  if (error.line != 0)
    out << " line " << error.line;
  return out.str();
}

[[nodiscard]] double wrap_unit(double value) noexcept {
  value -= std::floor(value);
  if (value >= 1.0)
    value -= 1.0;
  if (value < 0.0)
    value += 1.0;
  return value;
}

[[nodiscard]] std::array<double, 3>
wrap_cartesian(const std::array<double, 3> &coordinate,
               const std::array<double, 9> &box) {
  const double ax = box[0];
  const double bx = box[3];
  const double by = box[4];
  const double cx = box[6];
  const double cy = box[7];
  const double cz = box[8];
  if (ax == 0.0 || by == 0.0 || cz == 0.0)
    return coordinate;

  double sz = coordinate[2] / cz;
  double sy = (coordinate[1] - cy * sz) / by;
  double sx = (coordinate[0] - bx * sy - cx * sz) / ax;
  sx = wrap_unit(sx);
  sy = wrap_unit(sy);
  sz = wrap_unit(sz);
  return {sx * box[0] + sy * box[3] + sz * box[6],
          sx * box[1] + sy * box[4] + sz * box[7],
          sx * box[2] + sy * box[5] + sz * box[8]};
}

[[nodiscard]] std::array<double, 9>
rebuild_titan_box_matrix(const matrix box_nm) {
  constexpr double kPi = 3.14159265358979;
  constexpr double kOrthorhombicToleranceNm = 1.0e-7;
  const bool orthorhombic =
      std::fabs(static_cast<double>(box_nm[0][1])) <=
          kOrthorhombicToleranceNm &&
      std::fabs(static_cast<double>(box_nm[0][2])) <=
          kOrthorhombicToleranceNm &&
      std::fabs(static_cast<double>(box_nm[1][0])) <=
          kOrthorhombicToleranceNm &&
      std::fabs(static_cast<double>(box_nm[1][2])) <=
          kOrthorhombicToleranceNm &&
      std::fabs(static_cast<double>(box_nm[2][0])) <=
          kOrthorhombicToleranceNm &&
      std::fabs(static_cast<double>(box_nm[2][1])) <=
          kOrthorhombicToleranceNm;
  if (orthorhombic) {
    return {std::fabs(static_cast<double>(box_nm[0][0])) * kNmToAngstrom,
            0.0,
            0.0,
            0.0,
            std::fabs(static_cast<double>(box_nm[1][1])) * kNmToAngstrom,
            0.0,
            0.0,
            0.0,
            std::fabs(static_cast<double>(box_nm[2][2])) * kNmToAngstrom};
  }

  const std::array<double, 3> v1{
      static_cast<double>(box_nm[0][0]) * kNmToAngstrom,
      static_cast<double>(box_nm[0][1]) * kNmToAngstrom,
      static_cast<double>(box_nm[0][2]) * kNmToAngstrom};
  const std::array<double, 3> v2{
      static_cast<double>(box_nm[1][0]) * kNmToAngstrom,
      static_cast<double>(box_nm[1][1]) * kNmToAngstrom,
      static_cast<double>(box_nm[1][2]) * kNmToAngstrom};
  const std::array<double, 3> v3{
      static_cast<double>(box_nm[2][0]) * kNmToAngstrom,
      static_cast<double>(box_nm[2][1]) * kNmToAngstrom,
      static_cast<double>(box_nm[2][2]) * kNmToAngstrom};
  const auto norm = [](const std::array<double, 3> &v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  };
  const auto dot = [](const std::array<double, 3> &a,
                      const std::array<double, 3> &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  };
  const double a = norm(v1);
  const double b = norm(v2);
  const double c = norm(v3);
  const double cos_alpha = std::clamp(dot(v2, v3) / (b * c), -1.0, 1.0);
  const double cos_beta = std::clamp(dot(v1, v3) / (a * c), -1.0, 1.0);
  const double cos_gamma = std::clamp(dot(v1, v2) / (a * b), -1.0, 1.0);

  // Derive the lattice vectors with TITAN's own box class rather than a
  // second formula for the same thing.  The two are equal on paper and not
  // in doubles: writing v3z as sqrt(c^2 - v3x^2 - v3y^2) instead of
  // c*sqrt(gram)/sin(gamma) moved it by 7.1e-15 A on the AB42 dodecahedron.
  // PBCTOPO wraps and hashes coordinates with the box this class builds, and
  // validate_hololift_frame_binding compares that hash bit for bit, so any
  // last-bit difference here fails every frame.  It never showed on the
  // orthorhombic systems because both paths short-circuit to the diagonal.
  constexpr double kRadToDeg = 180.0 / 3.14159265358979;
  box frame_box;
  frame_box.set_a(a);
  frame_box.set_b(b);
  frame_box.set_c(c);
  frame_box.set_alpha(std::acos(cos_alpha) * kRadToDeg);
  frame_box.set_beta(std::acos(cos_beta) * kRadToDeg);
  frame_box.set_gamma(std::acos(cos_gamma) * kRadToDeg);
  double w1[3]{}, w2[3]{}, w3[3]{};
  frame_box.get_v1(w1);
  frame_box.get_v2(w2);
  frame_box.get_v3(w3);
  return {w1[0], w1[1], w1[2], w2[0], w2[1], w2[2], w3[0], w3[1], w3[2]};
}

[[nodiscard]] double max_box_difference(const std::array<double, 9> &lhs,
                                        const std::array<double, 9> &rhs) {
  double maximum = 0.0;
  for (std::size_t idx = 0; idx < lhs.size(); ++idx)
    maximum = std::max(maximum, std::fabs(lhs[idx] - rhs[idx]));
  return maximum;
}

int skip_xtc_coordinates(XDRFILE *handle, int expected_atoms,
                         std::int64_t file_size) {
  float box_values[DIM * DIM]{};
  if (xdrfile_read_float(box_values, DIM * DIM, handle) != DIM * DIM)
    return exdrFLOAT;
  int coordinate_count = 0;
  if (xdrfile_read_int(&coordinate_count, 1, handle) != 1)
    return exdrINT;
  if (coordinate_count < 0 || coordinate_count != expected_atoms)
    return exdr3DX;
  if (coordinate_count <= 9) {
    float raw_coordinates[9 * DIM]{};
    const int value_count = coordinate_count * DIM;
    return xdrfile_read_float(raw_coordinates, value_count, handle) ==
                   value_count
               ? exdrOK
               : exdrFLOAT;
  }

  float precision = 0.0f;
  int minimum[DIM]{};
  int maximum[DIM]{};
  int small_index = 0;
  if (xdrfile_read_float(&precision, 1, handle) != 1)
    return exdrFLOAT;
  if (xdrfile_read_int(minimum, DIM, handle) != DIM ||
      xdrfile_read_int(maximum, DIM, handle) != DIM ||
      xdrfile_read_int(&small_index, 1, handle) != 1)
    return exdrINT;

  std::int64_t opaque_bytes = 0;
  if (xdrfile_get_xtc_magic(handle) == XTC_NEW_MAGIC) {
    if (xdrfile_read_int64(&opaque_bytes, 1, handle) != 1)
      return exdrINT;
  } else {
    int legacy_bytes = 0;
    if (xdrfile_read_int(&legacy_bytes, 1, handle) != 1)
      return exdrINT;
    opaque_bytes = legacy_bytes;
  }
  if (opaque_bytes < 0 ||
      opaque_bytes > std::numeric_limits<std::int64_t>::max() - 3)
    return exdr3DX;
  const std::int64_t padded_bytes = (opaque_bytes + 3) & ~std::int64_t(3);
  const std::int64_t payload_start = xdr_tell(handle);
  if (payload_start < 0 || payload_start > file_size ||
      padded_bytes > file_size - payload_start)
    return exdr3DX;
  return xdr_seek(handle, padded_bytes, SEEK_CUR);
}

[[nodiscard]] std::vector<HoloLiftSourceAtomKey>
atom_order_for_epoch(const HoloLiftObservationStore &store,
                     std::uint32_t epoch_index) {
  if (epoch_index >= store.topology_epochs().size())
    return {};
  const auto &epoch = store.topology_epochs()[epoch_index];
  if (!epoch.components.valid_for(store.components().size()))
    return {};
  std::vector<HoloLiftSourceAtomKey> result;
  result.reserve(epoch.atom_count);
  const auto components =
      store.components().subspan(epoch.components.begin, epoch.components.count);
  for (const auto &component : components) {
    if (!component.exact_atom_membership.valid_for(
            store.exact_atom_memberships().size()))
      return {};
    const auto membership = store.exact_atom_memberships().subspan(
        component.exact_atom_membership.begin,
        component.exact_atom_membership.count);
    result.insert(result.end(), membership.begin(), membership.end());
  }
  std::sort(result.begin(), result.end(), [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.owner, lhs.source_atom_id) <
           std::tie(rhs.owner, rhs.source_atom_id);
  });
  if (std::adjacent_find(result.begin(), result.end()) != result.end())
    return {};
  return result;
}

// Runs task(index) for every index in [0, count) on up to `workers` threads, the calling thread included.
template <typename Task> void run_indexed_tasks(std::size_t count, std::size_t workers, Task &&task) {
  std::atomic<std::size_t> next{0};
  const auto drain = [&] {
    for (std::size_t index = next++; index < count; index = next++)
      task(index);
  };
  std::vector<std::thread> pool;
  const std::size_t threads = std::min(std::max<std::size_t>(workers, 1), count);
  for (std::size_t worker = 1; worker < threads; ++worker) {
    try {
      pool.emplace_back(drain);
    } catch (...) {
      break;
    }
  }
  drain();
  for (auto &thread : pool)
    thread.join();
}

// The XTC frames that the observation store binds, read a block at a time as TITAN's other modules read a trajectory.
// A frame carries the selected atoms as the XTC stores them (Angstrom, not yet wrapped into the cell);
// bind_trajectory_frame wraps them and checks the VIBE binding, so that this per-frame work can run in parallel.
class BoundTrajectoryReader {
public:
  [[nodiscard]] static std::expected<std::unique_ptr<BoundTrajectoryReader>, std::string>
  open(const HoloLiftObservationStore &store, const fs::path &xtc_path) {
    if (store.frames().empty())
      return std::unexpected("HoloLift observation store has no frames");
    for (std::size_t idx = 1; idx < store.frames().size(); ++idx) {
      if (store.frames()[idx].time_ps <= store.frames()[idx - 1].time_ps)
        return std::unexpected("observation frame times are not strictly increasing");
    }
    std::string xtc_string = xtc_path.string();
    int atom_count = 0;
    if (read_xtc_natoms(xtc_string.data(), &atom_count) != exdrOK || atom_count <= 0)
      return std::unexpected("failed to read XTC atom count");
    std::error_code size_error;
    const auto unsigned_size = fs::file_size(xtc_path, size_error);
    if (size_error ||
        unsigned_size > static_cast<std::uintmax_t>(std::numeric_limits<std::int64_t>::max()))
      return std::unexpected("failed to determine XTC file size");
    XDRFILE *handle = xdrfile_open(xtc_string.c_str(), "r");
    if (handle == nullptr)
      return std::unexpected("failed to open XTC trajectory");
    return std::unique_ptr<BoundTrajectoryReader>(
        new BoundTrajectoryReader(store, handle, atom_count, static_cast<std::int64_t>(unsigned_size)));
  }

  ~BoundTrajectoryReader() {
    if (handle_ != nullptr)
      xdrfile_close(handle_);
  }
  BoundTrajectoryReader(const BoundTrajectoryReader &) = delete;
  BoundTrajectoryReader &operator=(const BoundTrajectoryReader &) = delete;

  [[nodiscard]] bool done() const noexcept { return next_ >= store_.frames().size(); }

  // The next `count` observation frames (fewer at the end), skipping the XTC frames that are not observations.
  [[nodiscard]] std::expected<std::vector<TrajectoryFrame>, std::string> read(std::size_t count) {
    std::vector<TrajectoryFrame> frames;
    frames.reserve(std::min(count, store_.frames().size() - next_));
    while (frames.size() < count && next_ < store_.frames().size()) {
      int frame_atoms = 0;
      int step = 0;
      float time_ps = 0.0f;
      if (xtc_header(handle_, &frame_atoms, &step, &time_ps, 1) != exdrOK)
        return std::unexpected("XTC ended before every observation frame was read");
      if (frame_atoms != atom_count_)
        return std::unexpected("XTC atom count changed between frames");

      const auto &observation = store_.frames()[next_];
      const double frame_time = static_cast<double>(time_ps);
      if (frame_time < observation.time_ps - kTimeTolerancePs) {
        if (skip_xtc_coordinates(handle_, atom_count_, file_size_) != exdrOK)
          return std::unexpected("failed to skip an unselected XTC frame");
        continue;
      }
      if (std::fabs(frame_time - observation.time_ps) > kTimeTolerancePs) {
        return std::unexpected("XTC does not contain the requested observation time " +
                               std::to_string(observation.time_ps));
      }

      matrix raw_box{};
      if (xdrfile_read_float(raw_box[0], DIM * DIM, handle_) != DIM * DIM)
        return std::unexpected("failed to read selected XTC box");
      int decompressed_atoms = atom_count_;
      float precision = 0.0f;
      if (xdrfile_decompress_coord_float(full_coordinates_[0], &decompressed_atoms, &precision, handle_) !=
              atom_count_ ||
          decompressed_atoms != atom_count_) {
        return std::unexpected("failed to decompress selected XTC coordinates");
      }
      if (max_box_difference(rebuild_titan_box_matrix(raw_box), observation.box_matrix) > 2.0e-8)
        return std::unexpected("selected XTC box disagrees with VIBE observation");

      const auto order = epoch_atom_order(observation);
      if (!order)
        return std::unexpected(order.error());
      TrajectoryFrame frame;
      frame.trajectory_frame_index = observation.trajectory_frame_index;
      frame.box_matrix = observation.box_matrix;
      frame.atom_order = **order;
      frame.wrapped_coordinates.reserve(frame.atom_order.size());
      for (const auto &atom : frame.atom_order) {
        if (atom.source_atom_id >= static_cast<std::uint64_t>(atom_count_))
          return std::unexpected("source atom id is outside the XTC atom range");
        const auto &raw = full_coordinates_[static_cast<std::size_t>(atom.source_atom_id)];
        frame.wrapped_coordinates.push_back({static_cast<double>(raw[0]) * kNmToAngstrom,
                                             static_cast<double>(raw[1]) * kNmToAngstrom,
                                             static_cast<double>(raw[2]) * kNmToAngstrom});
      }
      frames.push_back(std::move(frame));
      ++next_;
    }
    return frames;
  }

private:
  BoundTrajectoryReader(const HoloLiftObservationStore &store, XDRFILE *handle, int atom_count,
                        std::int64_t file_size)
      : store_(store), handle_(handle), atom_count_(atom_count), file_size_(file_size),
        full_coordinates_(std::make_unique<rvec[]>(static_cast<std::size_t>(atom_count))),
        epoch_orders_(store.topology_epochs().size()) {}

  struct EpochAtomOrder {
    bool built = false;
    std::vector<HoloLiftSourceAtomKey> order;
    HoloLiftHash128 hash;
  };

  // The canonical atom order of the frame's topology epoch, built once per epoch, checked against the selection
  // VIBE recorded for the frame.
  [[nodiscard]] std::expected<const std::vector<HoloLiftSourceAtomKey> *, std::string>
  epoch_atom_order(const HoloLiftFrameRecord &observation) {
    if (observation.topology_epoch_index >= epoch_orders_.size())
      return std::unexpected("failed to reconstruct observation atom order");
    auto &cached = epoch_orders_[observation.topology_epoch_index];
    if (!cached.built) {
      cached.order = atom_order_for_epoch(store_, observation.topology_epoch_index);
      cached.hash = hash_hololift_atom_selection(cached.order);
      cached.built = true;
    }
    if (cached.order.size() != store_.topology_epochs()[observation.topology_epoch_index].atom_count)
      return std::unexpected("failed to reconstruct observation atom order");
    if (cached.hash != observation.atom_selection_hash)
      return std::unexpected("canonical atom order does not match VIBE selection order");
    return &cached.order;
  }

  const HoloLiftObservationStore &store_;
  XDRFILE *handle_ = nullptr;
  int atom_count_ = 0;
  std::int64_t file_size_ = 0;
  std::unique_ptr<rvec[]> full_coordinates_;
  std::vector<EpochAtomOrder> epoch_orders_;
  std::size_t next_ = 0;
};

// Wraps a frame's coordinates into its cell and checks them against the hash VIBE recorded for the frame.
[[nodiscard]] std::expected<void, std::string> bind_trajectory_frame(TrajectoryFrame &frame,
                                                                     const HoloLiftFrameRecord &observation) {
  for (auto &coordinate : frame.wrapped_coordinates)
    coordinate = wrap_cartesian(coordinate, frame.box_matrix);
  if (hash_hololift_wrapped_coordinates(frame.wrapped_coordinates) == observation.wrapped_coordinate_hash)
    return {};
  if (std::getenv("HOLOLIFT_BINDING_DEBUG") != nullptr) {
    const auto computed = hash_hololift_wrapped_coordinates(frame.wrapped_coordinates);
    std::ostringstream debug;
    debug << "BINDING_DEBUG frame=" << observation.trajectory_frame_index
          << " atoms=" << frame.wrapped_coordinates.size() << " computed=" << std::hex << computed.hi
          << computed.lo << " expected=" << observation.wrapped_coordinate_hash.hi
          << observation.wrapped_coordinate_hash.lo << std::dec << '\n';
    for (std::size_t probe = 0; probe < 3 && probe < frame.wrapped_coordinates.size(); ++probe) {
      debug << "  order[" << probe << "] source_atom_id=" << frame.atom_order[probe].source_atom_id
            << " owner=" << frame.atom_order[probe].owner << std::setprecision(17)
            << " xyz=" << frame.wrapped_coordinates[probe][0] << ' ' << frame.wrapped_coordinates[probe][1] << ' '
            << frame.wrapped_coordinates[probe][2] << '\n';
    }
    std::cerr << debug.str();
  }
  return std::unexpected("wrapped XTC coordinates do not match VIBE binding");
}

// Frames per block: TITAN's XDR streaming budget (64 MB, at most 64 frames per chunk) at about 256 bytes per selected
// atom (coordinates, atom order and audit scratch), but never fewer frames than workers.
[[nodiscard]] std::size_t trajectory_block_frames(const HoloLiftRunOptions &options,
                                                  const HoloLiftObservationStore &store, std::size_t workers) {
  if (options.trajectory_block_frames != 0)
    return options.trajectory_block_frames;
  std::size_t atoms = 1;
  for (const auto &epoch : store.topology_epochs())
    atoms = std::max<std::size_t>(atoms, epoch.atom_count);
  constexpr std::size_t kBudgetBytes = 64ULL << 20;
  constexpr std::size_t kBytesPerAtom = 256;
  constexpr std::size_t kMaximumFrames = 64;
  return std::max(workers, std::clamp<std::size_t>(kBudgetBytes / (atoms * kBytesPerAtom), 1, kMaximumFrames));
}

// Feeds the bound trajectory to process(first_frame, frames) block by block, decoding the next block on a second
// thread meanwhile, as TITAN's chunk streamer does.  Returns 0, 3 after a trajectory error, or the nonzero code that
// process returned.
template <typename Process>
int stream_bound_trajectory(const HoloLiftObservationStore &store, const fs::path &xtc_path, std::size_t block,
                            Process &&process) {
  auto opened = BoundTrajectoryReader::open(store, xtc_path);
  if (!opened) {
    std::cerr << "TRAJECTORY_ERROR " << opened.error() << '\n';
    return 3;
  }
  BoundTrajectoryReader &reader = **opened;
  const auto read_block = [&reader, block] { return reader.read(block); };
  auto pending = std::async(std::launch::async | std::launch::deferred, read_block);
  std::size_t first = 0;
  while (pending.valid()) {
    auto frames = pending.get();
    if (!frames) {
      std::cerr << "TRAJECTORY_ERROR " << frames.error() << '\n';
      return 3;
    }
    if (!reader.done())
      pending = std::async(std::launch::async | std::launch::deferred, read_block);
    if (const int status = process(first, *frames); status != 0)
      return status;
    first += frames->size();
  }
  return 0;
}

[[nodiscard]] std::string digest_hex(std::span<const std::byte, 32> digest) {
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (const std::byte value : digest)
    out << std::setw(2) << std::to_integer<unsigned int>(value);
  return out.str();
}

[[nodiscard]] std::string candidate_identity(
    const HoloLiftObservationStore &store, std::size_t candidate_index) {
  if (candidate_index == HOLOLIFT_NO_INDEX ||
      candidate_index >= store.candidates().size())
    return "na";
  return format_hololift_hash128(
      store.candidates()[candidate_index].assignment_identity);
}

[[nodiscard]] std::string index_text(std::size_t value) {
  return value == HOLOLIFT_NO_INDEX ? "na" : std::to_string(value);
}

[[nodiscard]] std::string candidate_rank_text(
    const HoloLiftObservationStore &store, std::size_t candidate_index) {
  return candidate_index == HOLOLIFT_NO_INDEX ||
                 candidate_index >= store.candidates().size()
             ? "na"
             : std::to_string(store.candidates()[candidate_index].rank);
}

[[nodiscard]] std::string candidate_images_text(
    const HoloLiftObservationStore &store, std::size_t candidate_index) {
  if (candidate_index == HOLOLIFT_NO_INDEX ||
      candidate_index >= store.candidates().size())
    return "na";
  const auto &candidate = store.candidates()[candidate_index];
  if (!candidate.component_images.valid_for(store.component_images().size()))
    return "invalid";
  const auto images = store.component_images().subspan(
      candidate.component_images.begin, candidate.component_images.count);
  std::ostringstream out;
  for (std::size_t index = 0; index < images.size(); ++index) {
    if (index != 0)
      out << '|';
    out << images[index].x << ':' << images[index].y << ':'
        << images[index].z;
  }
  return out.str();
}

[[nodiscard]] std::string spatial_representative_images_text(
    const HoloLiftPhase1PreparedFrame &prepared) {
  std::ostringstream out;
  const auto representatives =
      prepared.component_fractional_representatives();
  for (std::size_t component = 0; component < representatives.size();
       ++component) {
    if (component != 0)
      out << '|';
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const double value = representatives[component][axis];
      const double image = std::floor(value);
      if (!std::isfinite(value) ||
          image < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
          image > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
        return "invalid";
      }
      if (axis != 0)
        out << ':';
      out << static_cast<std::int64_t>(image);
    }
  }
  return out.str();
}

void write_transport_band_audit(
    const fs::path &path, std::string_view system, std::size_t radius,
    std::span<const TransportBandAuditRow> rows) {
  std::ofstream out(path, std::ios::trunc);
  out.imbue(std::locale::classic());
  out << "system,transport_radius,frame_index,source_frame_index,"
         "time_ps,source_offset,source_band_candidate,source_candidate_rank,"
         "candidate_rank,source_selected_framewise,selected_framewise,"
         "synthetic_rank_inversion_frame,"
         "synthetic_domain_miss_frame,"
         "synthetic_domain_fence_frame,"
         "assignment_identity,component_assignment\n";
  for (const auto &row : rows) {
    out << system << ',' << radius << ',' << row.frame_index << ','
        << row.source_frame_index << ',' << std::setprecision(17)
        << row.time_ps << ',' << row.source_offset << ','
        << (row.source_band_candidate ? 1 : 0) << ','
        << row.source_candidate_rank << ',' << row.candidate_rank << ','
        << (row.source_selected_framewise ? 1 : 0) << ','
        << (row.selected_framewise ? 1 : 0) << ','
        << (row.synthetic_rank_inversion_frame ? 1 : 0) << ','
        << (row.synthetic_domain_miss_frame ? 1 : 0) << ','
        << (row.synthetic_domain_fence_frame ? 1 : 0) << ','
        << format_hololift_hash128(row.assignment_identity) << ',';
    for (std::size_t index = 0; index < row.images.size(); ++index) {
      if (index != 0)
        out << '|';
      out << row.images[index].x << ':' << row.images[index].y << ':'
          << row.images[index].z;
    }
    out << '\n';
  }
}

// ---------------------------------------------------------------------------
// Band policies (H10, September 2026).
//
// The domain the solver optimizes over is declared, not assumed.  A frame
// whose band holds no evidence-compatible candidate (K_eff = 0) is one at
// which the provider reports no evidence, so its ordering there is a
// tie-break rather than a measurement, and what to admit as the domain is a
// modelling choice.  Three policies are offered; a run records which it used.
//
//   strict   the supplied list is the domain at every frame, including those
//            frames.  This is the behaviour of every run made before this
//            option existed, and remains the default.
//   carry    at an evidence-absent frame, the images of the preceding
//            (already augmented) frame's band that the frame lacks are
//            admitted, so a held image may pass through the frame, and
//            Phase 1 charges no rank cost at such frames.  This is the
//            evidence-absent analogue of an observation gap.
//   restore  at every frame, the assignments of the radius-2 product that
//            the provider did not list are admitted (two-component band
//            only), priced by the same rank cost.  This is the domain a
//            provider that exported its rejections would supply.
//
// Admitted assignments are flagged (HoloLiftCandidateRecord::carried), carry
// no evidence, and lower the frame's evidence-completeness flag.  Everything
// the provider wrote is copied unchanged, the frame's selected candidate is
// the provider's, the optimum stays exact on the declared graph, and no
// certificate condition is relaxed.

enum class EvidenceAbsentPolicy { Strict, Carry, Restore };

struct EvidencePolicyStats {
  std::size_t evidence_absent_frames = 0;
  std::size_t augmented_frames = 0;
  std::size_t added_candidates = 0;
};

struct EvidencePolicyReport {
  std::string_view name = "strict";
  EvidencePolicyStats stats;
  std::size_t carried_selections = 0;
};

[[nodiscard]] constexpr std::string_view
evidence_absent_policy_name(EvidenceAbsentPolicy policy) noexcept {
  switch (policy) {
  case EvidenceAbsentPolicy::Carry:
    return "carry";
  case EvidenceAbsentPolicy::Restore:
    return "restore";
  case EvidenceAbsentPolicy::Strict:
  default:
    return "strict";
  }
}

[[nodiscard]] bool frame_evidence_absent(const HoloLiftObservationStore &store,
                                         const HoloLiftFrameRecord &frame) {
  if (frame.candidates.count == 0 ||
      !frame.candidates.valid_for(store.candidates().size()))
    return false;
  for (std::size_t local = 0; local < frame.candidates.count; ++local) {
    const auto &candidate = store.candidates()[frame.candidates.begin + local];
    if (candidate.evidence.hard_feasible &&
        candidate.observation_class !=
            HoloLiftObservationClass::NotHardFeasible &&
        candidate.evidence_compatible_hypothesis_count > 0)
      return false;
  }
  return true;
}

// The selected candidate carries no evidence-compatible contact although
// another candidate of its frame does: a frame the path leaves unsupported.
[[nodiscard]] bool selection_evidence_unsupported(
    const HoloLiftObservationStore &store, const HoloLiftFrameRecord &frame,
    std::size_t selected) {
  if (frame.candidates.count == 0 || selected >= store.candidates().size() ||
      frame_evidence_absent(store, frame))
    return false;
  const auto &candidate = store.candidates()[selected];
  return !(candidate.evidence.hard_feasible &&
           candidate.observation_class !=
               HoloLiftObservationClass::NotHardFeasible &&
           candidate.evidence_compatible_hypothesis_count > 0);
}

using RejectedAssignments = std::unordered_map<
    std::size_t, std::unordered_set<std::uint64_t>>;

[[nodiscard]] std::vector<std::string> ledger_csv_fields(const std::string &line) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (c == '"') {
      if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
        field += '"'; ++i;
      } else quoted = !quoted;
    } else if (c == ',' && !quoted) {
      fields.push_back(field); field.clear();
    } else if (c != '\r') field += c;
  }
  if (quoted) throw std::runtime_error("unterminated rejected-ledger CSV field");
  fields.push_back(field);
  return fields;
}

[[nodiscard]] RejectedAssignments read_rejected_ledger(
    const fs::path &package, const HoloLiftObservationStore &store) {
  RejectedAssignments result;
  const auto path = package / "pbctopo_rejected_candidates.csv";
  if (!fs::exists(path)) return result;
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read rejected-assignment ledger");
  std::string line;
  if (!std::getline(in, line)) throw std::runtime_error("empty rejected ledger");
  const auto headers = ledger_csv_fields(line);
  std::unordered_map<std::string, std::size_t> columns;
  for (std::size_t i = 0; i < headers.size(); ++i)
    if (!columns.emplace(headers[i], i).second)
      throw std::runtime_error("duplicate rejected-ledger column");
  std::unordered_map<std::size_t, std::size_t> source_frame_to_dense;
  for (std::size_t i = 0; i < store.frames().size(); ++i)
    source_frame_to_dense.emplace(store.frames()[i].frame, i);
  while (std::getline(in, line)) {
    if (line.empty() || line == "\r") continue;
    const auto fields = ledger_csv_fields(line);
    if (fields.size() != headers.size())
      throw std::runtime_error("rejected-ledger column count differs");
    const auto get = [&](const char *name) -> const std::string & {
      const auto c = columns.find(name);
      if (c == columns.end()) throw std::runtime_error(std::string("rejected-ledger missing ") + name);
      return fields[c->second];
    };
    if (get("schema_version") != "1" ||
        get("record_scope") != "provider_rejected_audit_ledger" ||
        get("provider_status") != "provider_rejected" ||
        get("hard_feasible") != "0" || get("selected_framewise") != "0")
      throw std::runtime_error("unsupported rejected-ledger contract");
    const auto source_frame = static_cast<std::size_t>(std::stoull(get("frame")));
    const auto found = source_frame_to_dense.find(source_frame);
    if (found == source_frame_to_dense.end()) continue; // omitted spatial gaps
    const auto &frame = store.frames()[found->second];
    if (std::stoull(get("layout_signature")) != frame.layout_signature ||
        std::stoull(get("topology_epoch_id")) != frame.topology_epoch_id ||
        std::abs(std::stod(get("time_ps")) - frame.time_ps) > kTimeTolerancePs)
      throw std::runtime_error("rejected-ledger frame binding mismatch");
    if (get("hard_certificate_evaluated") == "1")
      result[found->second].insert(std::stoull(get("assignment_signature")));
    else if (get("hard_certificate_evaluated") != "0")
      throw std::runtime_error("invalid hard-certificate evaluation flag");
  }
  return result;
}

[[nodiscard]] std::expected<
    std::pair<HoloLiftObservationStore, EvidencePolicyStats>, std::string>
build_evidence_absent_policy_store(const HoloLiftObservationStore &source,
                                    EvidenceAbsentPolicy policy,
                                    const RejectedAssignments &rejected) {
  if (policy == EvidenceAbsentPolicy::Strict)
    return std::unexpected("no policy store is needed for the strict policy");
  HoloLiftObservationStoreBuilder builder;
  builder.api_version = source.api_version();
  builder.source_contract = source.source_contract();
  builder.source_contract.producer_version +=
      std::string("+hololift-evidence-absent-") +
      std::string(evidence_absent_policy_name(policy));
  builder.source_contract.package_id +=
      std::string("+evidence-absent-") +
      std::string(evidence_absent_policy_name(policy));
  builder.source_coverage = source.source_coverage();
  builder.topology_epochs.assign(source.topology_epochs().begin(),
                                 source.topology_epochs().end());
  builder.components.assign(source.components().begin(),
                            source.components().end());
  builder.exact_atom_memberships.assign(source.exact_atom_memberships().begin(),
                                        source.exact_atom_memberships().end());
  builder.exact_atom_masses.assign(source.exact_atom_masses().begin(),
                                   source.exact_atom_masses().end());
  builder.hard_edges.assign(source.hard_edges().begin(),
                            source.hard_edges().end());
  builder.frames.reserve(source.frames().size());
  builder.candidates.reserve(source.candidates().size() + source.frames().size());
  builder.component_images.reserve(source.component_images().size() +
                                   2 * source.frames().size());

  EvidencePolicyStats stats;
  using Images = std::vector<HoloLiftLatticeImage>;
  std::vector<Images> previous_images;   // augmented band of the frame before
  std::vector<HoloLiftHash128> previous_identities;

  for (std::size_t frame_index = 0; frame_index < source.frames().size();
       ++frame_index) {
    const auto &source_frame = source.frames()[frame_index];
    HoloLiftFrameRecord frame = source_frame;
    frame.candidates = {builder.candidates.size(), 0};
    if (source_frame.candidates.count == 0 ||
        source_frame.selected_candidate_index == HOLOLIFT_NO_INDEX) {
      frame.selected_candidate_index = HOLOLIFT_NO_INDEX;
      builder.frames.push_back(std::move(frame));
      previous_images.clear();
      previous_identities.clear();
      continue;
    }
    if (source_frame.topology_epoch_index >= source.topology_epochs().size())
      return std::unexpected("policy frame references an invalid epoch");
    const auto &epoch =
        source.topology_epochs()[source_frame.topology_epoch_index];
    if (!epoch.components.valid_for(source.components().size()))
      return std::unexpected("policy frame epoch component range is invalid");
    const auto components = source.components().subspan(
        epoch.components.begin, epoch.components.count);

    // the provider's band, copied verbatim
    std::vector<Images> images_here;
    images_here.reserve(source_frame.candidates.count);
    for (std::size_t local = 0; local < source_frame.candidates.count; ++local) {
      const std::size_t candidate_index = source_frame.candidates.begin + local;
      HoloLiftCandidateRecord candidate = source.candidates()[candidate_index];
      if (!candidate.component_images.valid_for(
              source.component_images().size()) ||
          candidate.component_images.count != components.size()) {
        return std::unexpected(
            "policy source candidate component-image range is invalid");
      }
      Images images(source.component_images().begin() +
                        static_cast<std::ptrdiff_t>(candidate.component_images.begin),
                    source.component_images().begin() +
                        static_cast<std::ptrdiff_t>(candidate.component_images.begin +
                                                    candidate.component_images.count));
      candidate.component_images = {builder.component_images.size(), images.size()};
      builder.component_images.insert(builder.component_images.end(),
                                      images.begin(), images.end());
      builder.candidates.push_back(candidate);
      images_here.push_back(std::move(images));
    }
    frame.selected_candidate_index =
        frame.candidates.begin +
        (source_frame.selected_candidate_index - source_frame.candidates.begin);
    const bool absent = frame_evidence_absent(source, source_frame);
    if (absent)
      ++stats.evidence_absent_frames;

    // what the policy adds
    std::vector<Images> add;
    const auto propose = [&](const Images &images) {
      const auto same = [&](const Images &other) { return other == images; };
      if (std::any_of(images_here.begin(), images_here.end(), same) ||
          std::any_of(add.begin(), add.end(), same))
        return;
      add.push_back(images);
    };
    if (policy == EvidenceAbsentPolicy::Carry && absent && frame_index > 0 &&
        same_source_segment(source, frame_index - 1, frame_index)) {
      for (const auto &images : previous_images)
        propose(images);
    }
    if (policy == EvidenceAbsentPolicy::Restore) {
      if (components.size() != 2)
        return std::unexpected(
            "the restore policy is defined for a two-component band only");
      const auto &selected_images =
          images_here[frame.selected_candidate_index - frame.candidates.begin];
      for (std::int64_t x = -2; x <= 2; ++x)
        for (std::int64_t y = -2; y <= 2; ++y)
          for (std::int64_t z = -2; z <= 2; ++z)
            propose({selected_images[0], HoloLiftLatticeImage{x, y, z}});
    }

    const HoloLiftCandidateRecord template_candidate =
        builder.candidates[frame.selected_candidate_index];
    for (const auto &images : add) {
      HoloLiftCandidateRecord candidate = template_candidate;
      // appended after the provider's rows, in the order proposed
      candidate.rank = builder.candidates.size() - frame.candidates.begin + 1;
      candidate.component_images = {builder.component_images.size(), images.size()};
      candidate.evidence.selected_framewise = false;
      // Admission to a diagnostic graph is not a provider hard certificate.
      candidate.evidence.hard_feasible = false;
      candidate.provider_evaluation = HoloLiftProviderEvaluation::NotEvaluated;
      candidate.policy_admission =
          policy == EvidenceAbsentPolicy::Carry
              ? HoloLiftPolicyAdmission::CarryPreviousFrame
              : HoloLiftPolicyAdmission::RestoreDeclaredDomain;
      candidate.policy_source_frame_index = HOLOLIFT_NO_INDEX;
      candidate.policy_source_assignment_identity = {};
      candidate.vibe_score_summary = {};
      candidate.evidence.search_equivalence_known = false;
      candidate.evidence.equivalent_to_search_best = false;
      candidate.evidence.equivalent_under_output_order = false;
      // The store's validator requires every candidate of a frame to agree
      // with the frame's search-domain audit on whether a boundary is known,
      // so the provider's value is kept; the added candidate is identified by
      // its own flag, not by pretending its provenance is unknown.
      candidate.domain_evidence.boundary_known =
          frame.search_domain.audit_known;
      candidate.domain_evidence.touches_shell_boundary = false;
      if (frame.search_domain.audit_known) {
        const auto radius = static_cast<std::int64_t>(frame.search_domain.shell_radius);
        for (const auto &image : images)
          candidate.domain_evidence.touches_shell_boundary |=
              image.x <= -radius || image.x >= radius ||
              image.y <= -radius || image.y >= radius ||
              image.z <= -radius || image.z >= radius;
      }
      const auto signature = hololift_assignment_signature_canonical(
          frame.layout_signature, components, images);
      if (!signature)
        return std::unexpected(signature.error());
      const auto identity = hololift_assignment_identity128_canonical(
          frame.layout_identity, components, images);
      if (!identity)
        return std::unexpected(identity.error());
      candidate.assignment_signature = *signature;
      candidate.assignment_identity = *identity;
      if (const auto found = rejected.find(frame_index);
          found != rejected.end() && found->second.contains(*signature))
        candidate.provider_evaluation = HoloLiftProviderEvaluation::Recorded;
      if (policy == EvidenceAbsentPolicy::Carry) {
        const auto found = std::find(previous_images.begin(), previous_images.end(), images);
        if (found == previous_images.end())
          return std::unexpected("carried assignment lacks preceding-band provenance");
        candidate.policy_source_frame_index = frame_index - 1;
        candidate.policy_source_assignment_identity =
            previous_identities[static_cast<std::size_t>(found - previous_images.begin())];
      }
      candidate.evidence_relative_relation_signature = *signature;
      candidate.evidence_compatible_hypothesis_hash = {};
      candidate.evidence_compatible_pair_hash = {};
      candidate.evidence_compatible_hypothesis_count = 0;
      candidate.evidence_supported_relation_count = 0;
      candidate.evidence_compatible_contact_count = 0;
      candidate.evidence_no_support_relation_count =
          frame.provenance.soft_observed_component_relation_count;
      // The class the validator recomputes for a hard-feasible candidate that
      // is not the framewise selection.
      candidate.observation_class = HoloLiftObservationClass::NotHardFeasible;
      candidate.carried = true;
      builder.component_images.insert(builder.component_images.end(),
                                      images.begin(), images.end());
      builder.candidates.push_back(candidate);
      ++stats.added_candidates;
    }
    frame.candidates.count = builder.candidates.size() - frame.candidates.begin;
    frame.evidence.retained_certified_assignment_count = frame.candidates.count;
    if (!add.empty()) {
      ++stats.augmented_frames;
      // The band is no longer the provider's enumeration of its search
      // domain, so the frame stops claiming an exhaustive bounded domain and
      // a complete alternative set.  A candidate that was a strong bounded
      // anchor on that claim is demoted with it.
      frame.search_domain.evidence_complete = false;
      frame.search_domain.completeness = HoloLiftDomainCompleteness::Unknown;
      frame.evidence.credible_alternative_set_complete = false;
      for (std::size_t local = 0; local < frame.candidates.count; ++local) {
        auto &existing = builder.candidates[frame.candidates.begin + local];
        if (existing.observation_class ==
            HoloLiftObservationClass::StrongBoundedAnchor)
          existing.observation_class =
              HoloLiftObservationClass::AmbiguousAnchor;
      }
    }
    if (policy == EvidenceAbsentPolicy::Carry) {
      previous_images = std::move(images_here);
      previous_images.insert(previous_images.end(), add.begin(), add.end());
      previous_identities.clear();
      for (std::size_t local = 0; local < frame.candidates.count; ++local)
        previous_identities.push_back(
            builder.candidates[frame.candidates.begin + local].assignment_identity);
    }
    builder.frames.push_back(std::move(frame));
  }

  auto finalized = finalize_hololift_observation_store(std::move(builder));
  if (!finalized) {
    return std::unexpected(std::string(hololift_validation_code_name(
                               finalized.error().code)) +
                           ": " + finalized.error().message);
  }
  return std::pair<HoloLiftObservationStore, EvidencePolicyStats>{
      std::move(*finalized), stats};
}

void write_frame_results(const fs::path &path, std::string_view system,
                         const HoloLiftObservationStore &store,
                         std::span<const HoloLiftPhase1PreparedFrame> prepared,
                         const HoloLiftPhase1Result &result) {
  std::ofstream out(path, std::ios::trunc);
  out.imbue(std::locale::classic());
  out << std::setprecision(17);
  out << "system,frame_index,time_ps,trajectory_frame_index,"
         "retained_candidate_count,framewise_candidate_index,"
         "temporal_candidate_index,framewise_candidate_rank,"
         "temporal_candidate_rank,framewise_assignment_identity,"
         "temporal_assignment_identity,framewise_component_images,"
         "temporal_component_images,spatial_representative_images,"
         "changed_from_framewise,frame_use,"
         "preceding_transition_gap,segment_index,emission_cost,"
         "transition_cost,cumulative_segment_cost,component_displacement_rms_A,"
         "component_displacement_max_A,component_nearest_image_violations,"
         "component_injectivity_uncertified,global_gauge_increment_ambiguous,"
         "atom_temporal_nearest_image_violations,"
         "atom_temporal_injectivity_uncertified,"
         "space_time_curvature_residuals,evidence_absent_frame,"
         "selected_carried,selected_provider_hard_feasible,"
         "selected_provider_unsupported,preceding_transition_provider_unsupported,"
         "policy_admission,provider_evaluation,selected_evidence_unsupported\n";
  for (std::size_t idx = 0; idx < store.frames().size(); ++idx) {
    const auto &source = store.frames()[idx];
    const auto &selected = result.frames[idx];
    const bool selected_carried =
        selected.candidate_index < store.candidates().size() &&
        store.candidates()[selected.candidate_index].carried;
    out << system << ',' << idx << ',' << source.time_ps << ','
        << source.trajectory_frame_index << ',' << source.candidates.count << ','
        << index_text(source.selected_candidate_index) << ','
        << index_text(selected.candidate_index) << ','
        << candidate_rank_text(store, source.selected_candidate_index) << ','
        << candidate_rank_text(store, selected.candidate_index) << ','
        << candidate_identity(store, source.selected_candidate_index) << ','
        << candidate_identity(store, selected.candidate_index) << ','
        << candidate_images_text(store, source.selected_candidate_index) << ','
        << candidate_images_text(store, selected.candidate_index) << ','
        << (idx < prepared.size()
                ? spatial_representative_images_text(prepared[idx])
                : "invalid")
        << ','
        << (selected.changed_from_framewise ? 1 : 0) << ','
        << hololift_phase1_frame_use_name(selected.use) << ','
        << hololift_phase1_transition_gap_name(
               selected.preceding_transition_gap)
        << ',' << index_text(selected.segment_index) << ','
        << selected.emission_cost << ',' << selected.transition_cost << ','
        << selected.cumulative_segment_cost << ','
        << selected.component_displacement_rms_A << ','
        << selected.component_displacement_max_A << ','
        << selected.component_nearest_image_violations << ','
        << selected.component_injectivity_uncertified << ','
        << (selected.global_gauge_increment_ambiguous ? 1 : 0) << ','
        << selected.atom_temporal_nearest_image_violations << ','
        << selected.atom_temporal_injectivity_uncertified << ','
        << selected.independent_space_time_curvature_residuals << ','
        << (frame_evidence_absent(store, source) ? 1 : 0) << ','
        << (selected_carried ? 1 : 0) << ','
        << (selected.candidate_index < store.candidates().size() &&
            store.candidates()[selected.candidate_index].evidence.hard_feasible ? 1 : 0) << ','
        << (selected.selected_candidate_provider_unsupported ? 1 : 0) << ','
        << (selected.preceding_transition_provider_unsupported ? 1 : 0) << ',';
    if (selected.candidate_index < store.candidates().size()) {
      const auto &c = store.candidates()[selected.candidate_index];
      out << static_cast<unsigned>(c.policy_admission) << ','
          << static_cast<unsigned>(c.provider_evaluation);
    } else out << "na,na";
    out << ','
        << (selection_evidence_unsupported(store, source, selected.candidate_index) ? 1 : 0)
        << '\n';
  }
}

void write_policy_admissions(const fs::path &path,
                             const HoloLiftObservationStore &store) {
  std::ofstream out(path);
  out.imbue(std::locale::classic());
  out << std::setprecision(17)
      << "frame,time_ps,candidate_rank,assignment_identity,component_images,"
         "policy_admission,provider_evaluation,provider_hard_feasible,"
         "source_frame_index,source_assignment_identity,touches_shell_boundary\n";
  for (std::size_t f = 0; f < store.frames().size(); ++f) {
    const auto &frame = store.frames()[f];
    for (std::size_t local = 0; local < frame.candidates.count; ++local) {
      const auto i = frame.candidates.begin + local;
      const auto &c = store.candidates()[i];
      if (!c.carried) continue;
      out << f << ',' << frame.time_ps << ',' << c.rank << ','
          << format_hololift_hash128(c.assignment_identity) << ','
          << candidate_images_text(store, i) << ','
          << static_cast<unsigned>(c.policy_admission) << ','
          << static_cast<unsigned>(c.provider_evaluation) << ','
          << (c.evidence.hard_feasible ? 1 : 0) << ','
          << index_text(c.policy_source_frame_index) << ','
          << format_hololift_hash128(c.policy_source_assignment_identity) << ','
          << (c.domain_evidence.touches_shell_boundary ? 1 : 0) << '\n';
    }
  }
}

void write_segment_results(const fs::path &path, std::string_view system,
                           const HoloLiftPhase1Result &result) {
  std::ofstream out(path, std::ios::trunc);
  out.imbue(std::locale::classic());
  out << std::setprecision(17);
  out << "system,segment_index,frame_begin,frame_count,topology_epoch_index,"
         "objective,second_best_objective,absolute_path_gap,relative_path_gap,"
         "optimal_path_count_capped,temporal_transition_count,"
         "ambiguous_gauge_transition_count,exact_on_retained_graph,"
         "unique_on_retained_graph,retained_candidate_bands_complete,"
         "all_atom_reconstruction_audited,temporal_lift_locally_certified,"
         "time_reversal_consistent,peak_candidate_images,"
         "peak_candidate_image_bytes_estimate\n";
  for (std::size_t idx = 0; idx < result.segments.size(); ++idx) {
    const auto &segment = result.segments[idx];
    out << system << ',' << idx << ',' << segment.frame_begin << ','
        << segment.frame_count << ',' << segment.topology_epoch_index << ','
        << segment.objective << ',' << segment.second_best_objective << ','
        << segment.absolute_path_gap << ',' << segment.relative_path_gap << ','
        << segment.optimal_path_count_capped << ','
        << segment.temporal_transition_count << ','
        << segment.ambiguous_gauge_transition_count << ','
        << (segment.exact_on_retained_graph ? 1 : 0) << ','
        << (segment.unique_on_configured_retained_state_graph ? 1 : 0) << ','
        << (segment.retained_candidate_bands_complete ? 1 : 0) << ','
        << (segment.all_atom_reconstruction_audited ? 1 : 0) << ','
        << (segment.temporal_lift_locally_certified ? 1 : 0) << ','
        << (segment.selected_transition_time_reversal_consistent ? 1 : 0)
        << ',' << segment.peak_live_pretransformed_candidate_component_images
        << ','
        << segment
               .peak_live_pretransformed_candidate_component_image_bytes_estimate
        << '\n';
  }
}

void write_system_summary(const fs::path &path, std::string_view system,
                          const HoloLiftObservationStore &store,
                          const HoloLiftVibeImportStats &stats,
                          const HoloLiftPhase1Result &result,
                          double elapsed_ms, std::string_view digest,
                          const HoloLiftPhase1Config &config,
                          const EvidencePolicyReport &policy) {
  std::size_t multiple_candidate_frames = 0;
  std::size_t retained_candidates = 0;
  for (const auto &frame : store.frames()) {
    retained_candidates += frame.candidates.count;
    if (frame.candidates.count > 1)
      ++multiple_candidate_frames;
  }
  const double multiple_fraction = store.frames().empty()
                                       ? 0.0
                                       : static_cast<double>(
                                             multiple_candidate_frames) /
                                             static_cast<double>(
                                                 store.frames().size());
  std::ofstream out(path, std::ios::trunc);
  out.imbue(std::locale::classic());
  out << std::setprecision(17);
  out << "system,source_frame_count,imported_observation_frames,"
         "retained_candidate_rows,frames_with_multiple_candidates,"
         "multiple_candidate_fraction,temporal_observation_frames,"
         "observation_gap_frames,changed_from_framewise_frames,segment_count,"
         "objective,all_segment_objectives_unique,"
          "all_observation_bands_complete,exact_on_retained_graph,"
          "all_atom_reconstruction_audited,temporal_lift_locally_certified,"
          "complete_domain_trajectory_claim_eligible,"
          "source_trajectory_coverage_complete,elapsed_ms,"
         "peak_candidate_image_bytes_estimate,audit_digest_sha256,"
         "ordinal_rank_weight,physical_transition_weight,ordinal_rank_form,"
         "evidence_absent_policy,evidence_absent_frames,"
         "policy_augmented_frames,policy_added_candidates,"
          "carried_selections,unique_evidence_mismatch_weight,"
          "allow_policy_admitted_candidates,evidence_absent_zero_rank,dp_workers,"
          "supported_evidence_mismatch_weight,evidence_unsupported_selections\n";
  std::size_t evidence_unsupported = 0;
  for (std::size_t idx = 0;
       idx < store.frames().size() && idx < result.frames.size(); ++idx) {
    if (selection_evidence_unsupported(store, store.frames()[idx],
                                       result.frames[idx].candidate_index))
      ++evidence_unsupported;
  }
  out << system << ',' << store.source_coverage().source_frame_count << ','
      << stats.imported_frames << ',' << retained_candidates << ','
      << multiple_candidate_frames << ',' << multiple_fraction << ','
      << result.temporal_observation_frames << ','
      << result.observation_gap_frames << ','
      << result.changed_from_framewise_frames << ',' << result.segments.size()
      << ',' << result.objective << ','
      << (result.all_segment_objectives_unique ? 1 : 0) << ','
      << (result.all_observation_bands_complete ? 1 : 0) << ','
      << (result.exact_on_retained_graph ? 1 : 0) << ','
       << (result.certificate.all_atom_reconstruction_audited ? 1 : 0) << ','
       << (result.certificate.temporal_lift_locally_certified ? 1 : 0) << ','
       << (result.certificate.complete_domain_trajectory_claim_eligible ? 1 : 0)
       << ','
       << (result.certificate.source_trajectory_coverage_complete ? 1 : 0)
      << ',' << elapsed_ms << ','
      << result
             .peak_live_pretransformed_candidate_component_image_bytes_estimate
      << ',' << digest << ',' << config.ordinal_rank_weight << ','
      << config.physical_transition_weight << ','
      << (config.ordinal_rank_saturated ? "saturated" : "linear") << ','
      << policy.name << ',' << policy.stats.evidence_absent_frames << ','
      << policy.stats.augmented_frames << ','
      << policy.stats.added_candidates << ','
       << policy.carried_selections << ',' << config.unique_evidence_mismatch_weight
       << ',' << (config.allow_policy_admitted_candidates ? 1 : 0)
       << ',' << (config.evidence_absent_zero_rank ? 1 : 0)
       << ',' << config.dp_worker_count
       << ',' << config.supported_evidence_mismatch_weight
       << ',' << evidence_unsupported << '\n';
}

// ---------------------------------------------------------------- coordinates of the selected path

// The anchor of the toroidal translation: the largest component of the epoch, ties broken by the smallest component
// identity, i.e. HoloLift's canonical gauge root.
[[nodiscard]] std::size_t anchor_component(const HoloLiftObservationStore &store,
                                           const HoloLiftTopologyEpochRecord &epoch) {
  std::size_t best = epoch.components.begin;
  for (std::size_t c = epoch.components.begin; c < epoch.components.begin + epoch.components.count; ++c) {
    const auto &candidate = store.components()[c];
    const auto &current = store.components()[best];
    if (candidate.exact_atom_membership.count > current.exact_atom_membership.count ||
        (candidate.exact_atom_membership.count == current.exact_atom_membership.count &&
         std::tie(candidate.component_identity.hi, candidate.component_identity.lo) <
             std::tie(current.component_identity.hi, current.component_identity.lo)))
      best = c;
  }
  return best;
}

struct ReferenceAtom {
  std::string record; // columns 1-30 of the reference PDB line (record name to residue sequence and insertion code)
  std::string tail;   // columns 55 onwards (occupancy, B-factor, element, charge)
};

// ATOM/HETATM records of the reference structure in file order; a source atom id is the zero-based position in it.
[[nodiscard]] std::vector<ReferenceAtom> read_reference_atoms(const fs::path &path) {
  std::vector<ReferenceAtom> atoms;
  if (path.empty())
    return atoms;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("ATOM", 0) != 0 && line.rfind("HETATM", 0) != 0)
      continue;
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    line.resize(std::max<std::size_t>(line.size(), 80), ' ');
    atoms.push_back({line.substr(0, 30), line.substr(54, 26)});
  }
  return atoms;
}

class PdbWriter {
public:
  PdbWriter(const fs::path &path, std::span<const HoloLiftSourceAtomKey> order,
            const std::vector<ReferenceAtom> &reference)
      : out_(path) {
    out_.imbue(std::locale::classic());
    out_ << "REMARK   HoloLift-MD selected path; toroidal or lattice coordinates as in hololift_coordinates.csv\n";
    records_.reserve(order.size());
    for (std::size_t a = 0; a < order.size(); ++a) {
      const std::uint64_t id = order[a].source_atom_id;
      if (id < reference.size()) {
        auto record = reference[id];
        // serial numbers follow the output order, capped at the 5-column field
        const std::string serial = std::to_string((a + 1) % 100000);
        record.record.replace(6, 5, std::string(5 - serial.size(), ' ') + serial);
        records_.push_back(std::move(record));
      } else {
        std::ostringstream generic;
        generic << "ATOM  " << std::setw(5) << (a + 1) % 100000 << "  X   UNK X" << std::setw(4) << (id % 10000)
                << "    ";
        records_.push_back({generic.str().substr(0, 30), "  1.00  0.00           X  "});
      }
    }
  }
  void write_model(std::size_t model, std::span<const std::array<double, 3>> x, const std::array<double, 9> &box) {
    // CRYST1 from the cell edge lengths and angles of this frame
    const std::array<double, 3> a{box[0], box[1], box[2]}, b{box[3], box[4], box[5]}, c{box[6], box[7], box[8]};
    const auto dot = [](const std::array<double, 3> &u, const std::array<double, 3> &v) {
      return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    };
    const double la = std::sqrt(dot(a, a)), lb = std::sqrt(dot(b, b)), lc = std::sqrt(dot(c, c));
    constexpr double kDeg = 57.29577951308232;
    char cryst[96];
    std::snprintf(cryst, sizeof cryst, "CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1\n", la, lb, lc,
                  std::acos(std::clamp(dot(b, c) / (lb * lc), -1.0, 1.0)) * kDeg,
                  std::acos(std::clamp(dot(a, c) / (la * lc), -1.0, 1.0)) * kDeg,
                  std::acos(std::clamp(dot(a, b) / (la * lb), -1.0, 1.0)) * kDeg);
    out_ << cryst << "MODEL " << std::setw(8) << model << '\n';
    char xyz[32];
    for (std::size_t i = 0; i < x.size(); ++i) {
      std::snprintf(xyz, sizeof xyz, "%8.3f%8.3f%8.3f", x[i][0], x[i][1], x[i][2]);
      out_ << records_[i].record << xyz << records_[i].tail << '\n';
    }
    out_ << "ENDMDL\n";
  }
  bool finish() {
    out_ << "END\n";
    out_.flush();
    return static_cast<bool>(out_);
  }

private:
  std::ofstream out_;
  std::vector<ReferenceAtom> records_;
};

// Writes the selected path as coordinates (lattice: every component on its selected image plus the cumulative lattice
// gauge; toroidal: the same relative images, with the whole assembly translated so that the anchor component's
// centre follows the displacement-preserving toroidal scheme of Bullerjahn et al., J. Chem. Theory Comput. 2023,
// c(t) = c(t-1) + the nearest-image displacement between its wrapped centres at t-1 and t in the cell of t).  Under
// constant pressure, lattice coordinates of a component that has crossed n cells carry n dB box-rescaling terms in
// their absolute motion; the toroidal translation removes them without changing any relative image.  A frame after an
// observation gap restarts the anchor at its lattice position; gap frames are not written.  Blocks of frames arrive
// in order with the lattice coordinates the all-atom audit reconstructed; the anchor path, the translation and the
// files follow in frame order.  The files are written under .partial names and take their final names in commit(),
// once the audit has sealed the result, so a failed run leaves no coordinates behind.
//   hololift_coordinates.xtc   selected atoms in the bound observation order, nm
//   hololift_coordinates.pdb   the first written frame, or every frame as MODEL records (pdb == All)
//   hololift_coordinates.csv   per frame: use, anchor centre on the lattice and on the toroidal path, translation
class SelectedCoordinateWriter {
public:
  [[nodiscard]] static std::expected<std::unique_ptr<SelectedCoordinateWriter>, std::string>
  open(const HoloLiftRunOptions &options, const HoloLiftObservationStore &store) {
    std::unique_ptr<SelectedCoordinateWriter> writer(new SelectedCoordinateWriter(options, store));
    writer->xtc_ = xdrfile_open(writer->partial_path(kXtcName).string().c_str(), "w");
    if (writer->xtc_ == nullptr)
      return std::unexpected("failed to open hololift_coordinates.xtc");
    writer->table_.open(writer->partial_path(kCsvName));
    if (!writer->table_)
      return std::unexpected("failed to open hololift_coordinates.csv");
    writer->table_.imbue(std::locale::classic());
    writer->table_ << std::setprecision(10)
                   << "frame_index,time_ps,frame_use,coordinates,anchor_component_id,anchor_atoms,anchor_lattice_x_A,"
                      "anchor_lattice_y_A,anchor_lattice_z_A,anchor_written_x_A,anchor_written_y_A,anchor_written_z_A,"
                      "translation_x_A,translation_y_A,translation_z_A\n";
    writer->reference_ = read_reference_atoms(options.reference_pdb);
    return writer;
  }

  ~SelectedCoordinateWriter() {
    close_files();
    if (!committed_) {
      std::error_code ignored;
      for (const char *name : {kXtcName, kCsvName, kPdbName})
        fs::remove(partial_path(name), ignored);
    }
  }
  SelectedCoordinateWriter(const SelectedCoordinateWriter &) = delete;
  SelectedCoordinateWriter &operator=(const SelectedCoordinateWriter &) = delete;

  // Frames [first, first + frames.size()), with the lattice coordinates of each (empty for a frame that is not a
  // temporal observation).
  [[nodiscard]] std::expected<void, std::string>
  write_block(std::size_t first, std::span<const TrajectoryFrame> frames,
              std::span<const std::vector<std::array<double, 3>>> lattice, const HoloLiftPhase1Result &result,
              std::span<const HoloLiftPhase1PreparedFrame> prepared) {
    if (lattice.size() != frames.size() || first + frames.size() > store_.frames().size() ||
        result.frames.size() != store_.frames().size() || prepared.size() != store_.frames().size())
      return std::unexpected("coordinate output inputs are inconsistent");
    for (std::size_t offset = 0; offset < frames.size(); ++offset) {
      const std::size_t idx = first + offset;
      const auto &frame = store_.frames()[idx];
      const auto &selection = result.frames[idx];
      if (selection.use != HoloLiftPhase1FrameUse::TemporalObservation) {
        table_ << idx << ',' << frame.time_ps << ',' << hololift_phase1_frame_use_name(selection.use)
               << ",not_written,,,,,,,,,,,\n";
        have_previous_ = false;
        continue;
      }
      const auto &x = lattice[offset];
      const auto &observed = frames[offset];
      if (x.size() != observed.atom_order.size())
        return std::unexpected("frame " + std::to_string(idx) + ": reconstructed atom count differs");
      if (!have_anchor_ || frame.topology_epoch_index != anchor_epoch_ || anchor_order_ != observed.atom_order) {
        const auto &component =
            store_.components()[anchor_component(store_, store_.topology_epochs()[frame.topology_epoch_index])];
        std::unordered_map<HoloLiftSourceAtomKey, std::size_t, KeyHash> position;
        position.reserve(observed.atom_order.size());
        for (std::size_t a = 0; a < observed.atom_order.size(); ++a)
          position.emplace(observed.atom_order[a], a);
        const bool masses = store_.exact_atom_masses().size() == store_.exact_atom_memberships().size();
        anchor_atoms_.clear();
        anchor_weights_.clear();
        for (std::size_t m = component.exact_atom_membership.begin;
             m < component.exact_atom_membership.begin + component.exact_atom_membership.count; ++m) {
          const auto found = position.find(store_.exact_atom_memberships()[m]);
          if (found == position.end())
            return std::unexpected("anchor atom missing from the bound atom order");
          anchor_atoms_.push_back(found->second);
          anchor_weights_.push_back(masses && store_.exact_atom_masses()[m] > 0.0 ? store_.exact_atom_masses()[m]
                                                                                  : 1.0);
        }
        anchor_epoch_ = frame.topology_epoch_index;
        anchor_order_ = observed.atom_order;
        anchor_id_ = component.component_id;
        have_anchor_ = true;
        have_previous_ = false;
      }
      std::array<double, 3> centre{};
      double total = 0.0;
      for (std::size_t a = 0; a < anchor_atoms_.size(); ++a) {
        for (std::size_t c = 0; c < 3; ++c)
          centre[c] += anchor_weights_[a] * x[anchor_atoms_[a]][c];
        total += anchor_weights_[a];
      }
      for (auto &value : centre)
        value /= total;
      const titan_pbctopo::PbctopoLatticeMetric metric(prepared[idx].box_matrix());
      if (!metric.valid())
        return std::unexpected("frame " + std::to_string(idx) + ": invalid cell");
      // the anchor centre wrapped into the cell of this frame
      const auto scaled = metric.cartesian_to_scaled(centre[0], centre[1], centre[2]);
      const std::array<double, 3> wrapped = metric.scaled_to_cartesian(
          scaled[0] - std::floor(scaled[0]), scaled[1] - std::floor(scaled[1]), scaled[2] - std::floor(scaled[2]));
      if (!toroidal_ || !have_previous_) {
        written_centre_ = centre;
      } else {
        const auto step = metric.nearest_image_cartesian(wrapped[0] - previous_wrapped_[0],
                                                         wrapped[1] - previous_wrapped_[1],
                                                         wrapped[2] - previous_wrapped_[2]);
        written_centre_[0] += step.x;
        written_centre_[1] += step.y;
        written_centre_[2] += step.z;
      }
      previous_wrapped_ = wrapped;
      have_previous_ = true;
      const std::array<double, 3> translation{written_centre_[0] - centre[0], written_centre_[1] - centre[1],
                                              written_centre_[2] - centre[2]};
      buffer_.resize(x.size());
      std::vector<std::array<double, 3>> shifted;
      const bool need_pdb = options_.pdb == HoloLiftPdbOutput::All ||
                            (options_.pdb == HoloLiftPdbOutput::First && written_ == 0);
      if (need_pdb)
        shifted.resize(x.size());
      for (std::size_t a = 0; a < x.size(); ++a) {
        for (std::size_t c = 0; c < 3; ++c) {
          const double value = x[a][c] + translation[c];
          buffer_[a][c] = static_cast<float>(value / kNmToAngstrom);
          if (need_pdb)
            shifted[a][c] = value;
        }
      }
      matrix box_nm{};
      const auto &box = prepared[idx].box_matrix();
      for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
          box_nm[r][c] = static_cast<float>(box[r * 3 + c] / kNmToAngstrom);
      if (write_xtc(xtc_, static_cast<int>(buffer_.size()), static_cast<int>(observed.trajectory_frame_index),
                    static_cast<float>(frame.time_ps), box_nm, reinterpret_cast<rvec *>(buffer_.data()),
                    1000.0F) != exdrOK)
        return std::unexpected("failed to write XTC frame " + std::to_string(idx));
      if (need_pdb) {
        if (!pdb_)
          pdb_.emplace(partial_path(kPdbName), observed.atom_order, reference_);
        std::array<double, 9> box_copy{};
        std::copy(box.begin(), box.end(), box_copy.begin());
        pdb_->write_model(written_ + 1, shifted, box_copy);
      }
      table_ << idx << ',' << frame.time_ps << ',' << hololift_phase1_frame_use_name(selection.use) << ','
             << (toroidal_ ? "toroidal" : "lattice") << ',' << anchor_id_ << ',' << anchor_atoms_.size() << ','
             << centre[0] << ',' << centre[1] << ',' << centre[2] << ',' << written_centre_[0] << ','
             << written_centre_[1] << ',' << written_centre_[2] << ',' << translation[0] << ',' << translation[1]
             << ',' << translation[2] << '\n';
      ++written_;
    }
    return {};
  }

  // Closes the files and gives them their final names; returns the number of frames written.
  [[nodiscard]] std::expected<std::size_t, std::string> commit() {
    const bool pdb_written = pdb_.has_value();
    if (pdb_ && !pdb_->finish())
      return std::unexpected("failed to write hololift_coordinates.pdb");
    table_.flush();
    if (!table_)
      return std::unexpected("failed to write hololift_coordinates.csv");
    close_files();
    std::vector<const char *> names{kXtcName, kCsvName};
    if (pdb_written)
      names.push_back(kPdbName);
    for (const char *name : names) {
      const fs::path final_path = options_.output_directory / name;
      std::error_code error;
      fs::remove(final_path, error);
      fs::rename(partial_path(name), final_path, error);
      if (error)
        return std::unexpected(std::string("failed to move ") + name + " into place: " + error.message());
    }
    committed_ = true;
    return written_;
  }

private:
  static constexpr const char *kXtcName = "hololift_coordinates.xtc";
  static constexpr const char *kCsvName = "hololift_coordinates.csv";
  static constexpr const char *kPdbName = "hololift_coordinates.pdb";

  struct KeyHash {
    std::size_t operator()(const HoloLiftSourceAtomKey &k) const noexcept {
      return std::hash<std::uint64_t>{}(k.source_atom_id) ^ (std::hash<int>{}(k.owner) << 1);
    }
  };

  SelectedCoordinateWriter(const HoloLiftRunOptions &options, const HoloLiftObservationStore &store)
      : options_(options), store_(store), toroidal_(options.coordinates == HoloLiftCoordinateOutput::Toroidal) {}

  [[nodiscard]] fs::path partial_path(const char *name) const {
    return options_.output_directory / (std::string(name) + ".partial");
  }

  void close_files() {
    if (xtc_ != nullptr) {
      xdrfile_close(xtc_);
      xtc_ = nullptr;
    }
    if (table_.is_open())
      table_.close();
    pdb_.reset();
  }

  const HoloLiftRunOptions &options_;
  const HoloLiftObservationStore &store_;
  const bool toroidal_;
  XDRFILE *xtc_ = nullptr;
  std::ofstream table_;
  std::vector<ReferenceAtom> reference_;
  std::optional<PdbWriter> pdb_;
  bool committed_ = false;

  bool have_anchor_ = false;
  std::uint32_t anchor_epoch_ = HOLOLIFT_NO_DENSE_INDEX;
  std::vector<HoloLiftSourceAtomKey> anchor_order_;
  std::uint64_t anchor_id_ = 0;
  std::vector<std::size_t> anchor_atoms_;
  std::vector<double> anchor_weights_;
  bool have_previous_ = false;
  std::array<double, 3> previous_wrapped_{}, written_centre_{};
  std::size_t written_ = 0;
  std::vector<std::array<float, 3>> buffer_;
};

int run(const HoloLiftRunOptions &options) {
  const std::string &system = options.system_label;
  const fs::path &package_directory = options.package_directory;
  const fs::path &xtc_path = options.trajectory_path;
  const fs::path &output_directory = options.output_directory;
  const std::size_t transport_radius = options.transport_radius;
  const bool synthetic_rank_inversion_v1 = options.synthetic_rank_inversion_v1;
  const bool synthetic_domain_miss_v1 = options.synthetic_domain_miss_v1;
  const bool synthetic_domain_miss_v2 = options.synthetic_domain_miss_v2;
  // wall time of each stage on stderr, as TIMING lines
  const bool timing = options.report_stage_times;
  auto stage_clock = std::chrono::steady_clock::now();
  const auto stage = [&](const char *name) {
    if (!timing)
      return;
    const auto now = std::chrono::steady_clock::now();
    std::cerr << "TIMING " << name << ' ' << std::chrono::duration<double>(now - stage_clock).count() << " s\n";
    stage_clock = now;
  };
  HoloLiftVibeImportOptions import_options;
  import_options.package_directory = package_directory;
  import_options.package_id = "temporal-value-" + std::string(system);
  import_options.trajectory_id = xtc_path.filename().string();
  import_options.accept_experimental_temporal_diagnostics = false;
  auto imported = import_hololift_vibe_package(import_options);
  if (!imported) {
    std::cerr << "IMPORT_ERROR " << import_error_text(imported.error()) << '\n';
    return 2;
  }
  stage("import");
  std::optional<HoloLiftObservationStore> transported_store;
  std::vector<TransportBandAuditRow> transport_rows;
  if (transport_radius != 0) {
    auto transported =
        build_temporal_transport_band(imported->store, transport_radius,
                                      synthetic_rank_inversion_v1,
                                      synthetic_domain_miss_v1,
                                      synthetic_domain_miss_v2);
    if (!transported) {
      std::cerr << "TRANSPORT_BAND_ERROR " << transported.error() << '\n';
      return 12;
    }
    transported_store.emplace(std::move(transported->first));
    transport_rows = std::move(transported->second);
  }
  // Evidence-absent admission: strict (default, the recorded behaviour; "hard" is accepted as the former name of
  // strict), carry or restore.  The policy store is built from the store Phase 1 would otherwise see, so it
  // composes with the transport band.
  EvidenceAbsentPolicy policy = EvidenceAbsentPolicy::Strict;
  {
    const std::string_view name(options.evidence_absent);
    if (name == "carry") {
      policy = EvidenceAbsentPolicy::Carry;
    } else if (name == "restore") {
      policy = EvidenceAbsentPolicy::Restore;
    } else if (!name.empty() && name != "strict" && name != "hard") {
      std::cerr << "CONFIG_ERROR the evidence-absent policy must be strict, "
                   "carry or restore\n";
      return 2;
    }
  }
  std::optional<HoloLiftObservationStore> policy_store;
  EvidencePolicyStats policy_stats;
  if (policy != EvidenceAbsentPolicy::Strict) {
    const auto rejected = read_rejected_ledger(package_directory, imported->store);
    auto built = build_evidence_absent_policy_store(
        transport_radius == 0 ? imported->store : *transported_store, policy,
        rejected);
    if (!built) {
      std::cerr << "EVIDENCE_POLICY_ERROR " << built.error() << '\n';
      return 13;
    }
    policy_store.emplace(std::move(built->first));
    policy_stats = built->second;
  }
  auto &store = policy_store ? *policy_store
                : transport_radius == 0 ? imported->store
                                        : *transported_store;
  stage("policy_and_transport");
  HoloLiftPhase1Config config;
  // The weights are otherwise the struct defaults and are written into the system summary, so a run says what it
  // ran with.  Unset options leave the defaults untouched, so the default path is bit-identical to the runs made
  // before the options existed.
  if (options.rank_weight)
    config.ordinal_rank_weight = *options.rank_weight;
  if (options.transition_weight)
    config.physical_transition_weight = *options.transition_weight;
  // rank cost "saturated" switches E_t to w_r [rank != 1]; "linear" keeps Eq. 1.  Recorded in the system summary as
  // ordinal_rank_form.
  if (options.rank_cost == "saturated") {
    config.ordinal_rank_saturated = true;
  } else if (!options.rank_cost.empty() && options.rank_cost != "linear") {
    std::cerr << "CONFIG_ERROR the rank cost must be linear or saturated\n";
    return 2;
  }
  if (policy == EvidenceAbsentPolicy::Carry)
    config.evidence_absent_zero_rank = true;
  config.allow_policy_admitted_candidates = policy != EvidenceAbsentPolicy::Strict;
  if (options.unique_evidence_weight) {
    if (!std::isfinite(*options.unique_evidence_weight) || *options.unique_evidence_weight < 0.0) {
      std::cerr << "CONFIG_ERROR the unique-evidence weight must be finite and nonnegative\n";
      return 2;
    }
    config.unique_evidence_mismatch_weight = *options.unique_evidence_weight;
  }
  if (options.supported_evidence_weight) {
    if (!std::isfinite(*options.supported_evidence_weight) || *options.supported_evidence_weight < 0.0) {
      std::cerr << "CONFIG_ERROR the supported-evidence weight must be finite and nonnegative\n";
      return 2;
    }
    config.supported_evidence_mismatch_weight = *options.supported_evidence_weight;
  }
  if (options.dp_workers) {
    if (*options.dp_workers == 0 || *options.dp_workers > 64) {
      std::cerr << "CONFIG_ERROR the DP worker count must be in [1,64]\n";
      return 2;
    }
    config.dp_worker_count = *options.dp_workers;
  }

  auto context = make_hololift_phase1_preparation_context(store);
  if (!context) {
    std::cerr << "PREPARATION_ERROR " << context.error() << '\n';
    return 4;
  }
  // The trajectory is read twice, a block at a time, and never held whole: once to bind and prepare every frame
  // (only the component representatives are kept), and once after the solve to audit the all-atom reconstruction of
  // the selected path and write it out.  Within a block, frames are bound, prepared and audited on `workers` threads
  // while the next block is decoded; results are taken in frame order, so they do not depend on either number.
  const std::size_t workers = std::clamp<std::size_t>(options.trajectory_workers, 1, 64);
  const std::size_t block = trajectory_block_frames(options, store, workers);
  std::vector<HoloLiftPhase1PreparedFrame> prepared;
  prepared.reserve(store.frames().size());
  int status = stream_bound_trajectory(
      store, xtc_path, block, [&](std::size_t first, std::vector<TrajectoryFrame> &frames) {
        std::vector<std::optional<HoloLiftPhase1PreparedFrame>> tokens(frames.size());
        std::vector<std::pair<int, std::string>> errors(frames.size(), {0, std::string()});
        run_indexed_tasks(frames.size(), workers, [&](std::size_t offset) {
          auto &frame = frames[offset];
          const std::size_t idx = first + offset;
          try {
            if (auto bound = bind_trajectory_frame(frame, store.frames()[idx]); !bound) {
              errors[offset] = {3, bound.error()};
              return;
            }
            auto token = prepare_hololift_phase1_frame(*context, store, idx, frame.trajectory_frame_index,
                                                       frame.box_matrix, frame.atom_order, frame.wrapped_coordinates);
            if (!token) {
              errors[offset] = {5, token.error()};
              return;
            }
            tokens[offset].emplace(std::move(*token));
          } catch (const std::exception &error) {
            errors[offset] = {5, error.what()};
          }
        });
        for (std::size_t offset = 0; offset < frames.size(); ++offset) {
          if (errors[offset].first == 3) {
            std::cerr << "TRAJECTORY_ERROR " << errors[offset].second << '\n';
            return 3;
          }
          if (errors[offset].first != 0) {
            std::cerr << "PREPARATION_ERROR frame=" << first + offset << ' ' << errors[offset].second << '\n';
            return 5;
          }
          prepared.push_back(std::move(*tokens[offset]));
        }
        return 0;
      });
  if (status != 0)
    return status;
  stage("read_bind_prepare");

  const auto started = std::chrono::steady_clock::now();
  auto result = solve_hololift_phase1_temporal_path(store, prepared, config);
  if (!result) {
    std::cerr << "SOLVE_ERROR " << result.error() << '\n';
    return 6;
  }
  stage("solve");

  std::error_code directory_error;
  fs::create_directories(output_directory, directory_error);
  if (directory_error) {
    std::cerr << "OUTPUT_ERROR " << directory_error.message() << '\n';
    return 10;
  }
  auto audit = begin_hololift_phase1_audit(*context, store, prepared, std::move(*result));
  if (!audit) {
    std::cerr << "AUDIT_ERROR " << audit.error() << '\n';
    return 7;
  }
  std::unique_ptr<SelectedCoordinateWriter> writer;
  if (options.coordinates != HoloLiftCoordinateOutput::None) {
    auto opened = SelectedCoordinateWriter::open(options, store);
    if (!opened) {
      std::cerr << "COORDINATE_ERROR " << opened.error() << '\n';
      return 14;
    }
    writer = std::move(*opened);
  }
  status = stream_bound_trajectory(
      store, xtc_path, block, [&](std::size_t first, std::vector<TrajectoryFrame> &frames) {
        std::vector<std::optional<std::string>> errors(frames.size());
        run_indexed_tasks(frames.size(), workers, [&](std::size_t offset) {
          try {
            if (auto bound = bind_trajectory_frame(frames[offset], store.frames()[first + offset]); !bound)
              errors[offset] = bound.error();
          } catch (const std::exception &error) {
            errors[offset] = error.what();
          }
        });
        for (const auto &error : errors) {
          if (error) {
            std::cerr << "TRAJECTORY_ERROR " << *error << '\n';
            return 3;
          }
        }
        std::vector<HoloLiftPhase1TrajectoryFrameView> views;
        views.reserve(frames.size());
        for (const auto &frame : frames)
          views.push_back({frame.trajectory_frame_index, frame.atom_order, frame.wrapped_coordinates});
        auto lattice = audit->audit_block(first, views, workers);
        if (!lattice) {
          std::cerr << "AUDIT_ERROR " << lattice.error() << '\n';
          return 7;
        }
        if (writer) {
          const auto *selected = audit->result_if_valid();
          if (selected == nullptr) {
            std::cerr << "AUDIT_ERROR audit stream lost its result\n";
            return 7;
          }
          if (auto written = writer->write_block(first, frames, *lattice, *selected, prepared); !written) {
            std::cerr << "COORDINATE_ERROR " << written.error() << '\n';
            return 14;
          }
        }
        return 0;
      });
  if (status != 0)
    return status;
  auto audited = audit->finish();
  if (!audited) {
    std::cerr << "AUDIT_ERROR " << audited.error() << '\n';
    return 7;
  }
  stage("audit_and_coordinates");
  auto validation =
      validate_hololift_phase1_audited_result(store, prepared, *audited);
  if (!validation) {
    std::cerr << "VALIDATION_ERROR " << validation.error() << '\n';
    return 8;
  }
  const auto finished = std::chrono::steady_clock::now();
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(finished - started).count();
  const auto *sealed_result = audited->result_if_valid();
  if (sealed_result == nullptr) {
    std::cerr << "AUDIT_ERROR sealed result is empty\n";
    return 9;
  }
  const std::string digest =
      digest_hex(audited->trajectory_audit_digest_sha256());

  if (transport_radius != 0) {
    write_transport_band_audit(output_directory /
                                   "hololift_transport_band_candidates.csv",
                               system, transport_radius, transport_rows);
  }
  write_frame_results(output_directory / "hololift_frame_results.csv", system,
                      store, prepared, *sealed_result);
  write_policy_admissions(output_directory / "hololift_policy_admissions.csv", store);
  write_segment_results(output_directory / "hololift_segment_results.csv",
                        system, *sealed_result);
  EvidencePolicyReport policy_report{evidence_absent_policy_name(policy),
                                     policy_stats, 0};
  for (const auto &selection : sealed_result->frames) {
    if (selection.candidate_index < store.candidates().size() &&
        store.candidates()[selection.candidate_index].carried)
      ++policy_report.carried_selections;
  }
  write_system_summary(output_directory / "hololift_system_summary.csv",
                       system, store, imported->stats, *sealed_result,
                       elapsed_ms, digest, config, policy_report);
  std::string coordinate_note;
  if (writer) {
    auto written = writer->commit();
    if (!written) {
      std::cerr << "COORDINATE_ERROR " << written.error() << '\n';
      return 14;
    }
    coordinate_note = std::string(" coordinates=") +
                      (options.coordinates == HoloLiftCoordinateOutput::Toroidal ? "toroidal" : "lattice") +
                      " written=" + std::to_string(*written);
  }
  std::size_t evidence_unsupported = 0;
  for (std::size_t idx = 0;
       idx < store.frames().size() && idx < sealed_result->frames.size(); ++idx) {
    if (selection_evidence_unsupported(store, store.frames()[idx],
                                       sealed_result->frames[idx].candidate_index))
      ++evidence_unsupported;
  }
  std::cout << "PASS system=" << system << " frames=" << store.frames().size()
            << " policy=" << policy_report.name
            << " carried=" << policy_report.carried_selections
            << " evidence_unsupported=" << evidence_unsupported
            << " candidates=" << store.candidates().size()
            << " transport_radius=" << transport_radius
            << " synthetic_rank_inversion_v1="
            << (synthetic_rank_inversion_v1 ? 1 : 0)
            << " synthetic_domain_miss_v1="
            << (synthetic_domain_miss_v1 ? 1 : 0)
            << " synthetic_domain_miss_v2="
            << (synthetic_domain_miss_v2 ? 1 : 0)
            << " changed=" << sealed_result->changed_from_framewise_frames
            << " gaps=" << sealed_result->observation_gap_frames
            << " elapsed_ms=" << std::setprecision(10) << elapsed_ms
            << " audit=" << digest << coordinate_note << '\n';
  return 0;
}

} // namespace

namespace titan_hololift {

int run_hololift_temporal_value(const HoloLiftRunOptions &options) { return run(options); }

std::optional<HoloLiftCoordinateOutput> parse_hololift_coordinate_output(const std::string &name) {
  if (name == "none")
    return HoloLiftCoordinateOutput::None;
  if (name == "lattice")
    return HoloLiftCoordinateOutput::Lattice;
  if (name == "toroidal")
    return HoloLiftCoordinateOutput::Toroidal;
  return std::nullopt;
}

std::optional<HoloLiftPdbOutput> parse_hololift_pdb_output(const std::string &name) {
  if (name == "none")
    return HoloLiftPdbOutput::None;
  if (name == "first")
    return HoloLiftPdbOutput::First;
  if (name == "all")
    return HoloLiftPdbOutput::All;
  return std::nullopt;
}

} // namespace titan_hololift
