#include "serialization.h"

#include "../titan_sha256.h"
#include "durable_artifact.h"
#include "validation.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <span>
#include <streambuf>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace titan_hololift {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{
    'H', 'L', 'I', 'F', 'T', '2', '\r', '\n'};
constexpr std::uint64_t kPayloadHashDomain = 0x484c5041594c4433ULL;
using PayloadDigest = std::array<std::byte, 32>;

class PayloadChecksum {
public:
  PayloadChecksum() noexcept { hash_.word(kPayloadHashDomain); }

  void update(std::uint8_t value) noexcept {
    hash_.byte(value);
    ++size_;
  }

  void update(std::span<const std::byte> bytes) noexcept {
    hash_.update(bytes);
    size_ += static_cast<std::uint64_t>(bytes.size());
  }

  [[nodiscard]] std::uint64_t size() const noexcept { return size_; }

  [[nodiscard]] PayloadDigest finish() const noexcept {
    return hash_.finish256();
  }

private:
  titan_hash::Sha256Builder hash_;
  std::uint64_t size_ = 0;
};

class NullOutputBuffer final : public std::streambuf {
protected:
  int_type overflow(int_type character) override {
    return traits_type::not_eof(character);
  }

  std::streamsize xsputn(const char *, std::streamsize count) override {
    return count;
  }
};

std::string format_payload_digest(const PayloadDigest &digest) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.resize(digest.size() * 2);
  for (std::size_t index = 0; index < digest.size(); ++index) {
    const auto value = std::to_integer<unsigned int>(digest[index]);
    result[2 * index] = digits[value >> 4];
    result[2 * index + 1] = digits[value & 0x0fU];
  }
  return result;
}

std::expected<PayloadDigest, std::string>
payload_checksum_stream(std::istream &in, std::size_t byte_count) {
  PayloadChecksum checksum;
  std::array<std::byte, 64 * 1024> buffer{};
  std::size_t remaining = byte_count;
  while (remaining != 0) {
    const std::size_t chunk = std::min(remaining, buffer.size());
    in.read(reinterpret_cast<char *>(buffer.data()),
            static_cast<std::streamsize>(chunk));
    if (!in || static_cast<std::size_t>(in.gcount()) != chunk) {
      return std::unexpected(
          "failed to stream HoloLift binary payload checksum");
    }
    checksum.update(std::span<const std::byte>(buffer).first(chunk));
    remaining -= chunk;
  }
  return checksum.finish();
}

class ByteWriter {
public:
  explicit ByteWriter(std::ostream &out,
                      PayloadChecksum *checksum = nullptr) noexcept
      : out_(out), checksum_(checksum) {}

  void u8(std::uint8_t value) {
    if (!ok_)
      return;
    out_.put(static_cast<char>(value));
    if (!out_) {
      ok_ = false;
      return;
    }
    if (checksum_ != nullptr)
      checksum_->update(value);
    if (bytes_written_ == std::numeric_limits<std::uint64_t>::max()) {
      ok_ = false;
      return;
    }
    ++bytes_written_;
  }

  void u32(std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8)
      u8(static_cast<std::uint8_t>(value >> shift));
  }

  void u64(std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8)
      u8(static_cast<std::uint8_t>(value >> shift));
  }

  void i64(std::int64_t value) { u64(std::bit_cast<std::uint64_t>(value)); }
  void real(double value) { u64(std::bit_cast<std::uint64_t>(value)); }
  void boolean(bool value) { u8(value ? 1U : 0U); }
  void size(std::size_t value) { u64(static_cast<std::uint64_t>(value)); }
  void hash(HoloLiftHash128 value) {
    u64(value.lo);
    u64(value.hi);
  }
  void digest(const PayloadDigest &value) {
    for (const std::byte byte : value)
      u8(std::to_integer<std::uint8_t>(byte));
  }
  void range(HoloLiftRange value) {
    size(value.begin);
    size(value.count);
  }
  void text(std::string_view value) {
    size(value.size());
    for (const char character : value) {
      u8(static_cast<std::uint8_t>(
          static_cast<unsigned char>(character)));
    }
  }
  [[nodiscard]] bool ok() const noexcept { return ok_ && out_.good(); }
  [[nodiscard]] std::uint64_t bytes_written() const noexcept {
    return bytes_written_;
  }

private:
  std::ostream &out_;
  PayloadChecksum *checksum_ = nullptr;
  std::uint64_t bytes_written_ = 0;
  bool ok_ = true;
};

class ByteReader {
public:
  explicit ByteReader(std::span<const std::byte> bytes) : bytes_(bytes) {}
  ByteReader(std::istream &stream, std::size_t byte_count)
      : stream_(&stream), stream_unbuffered_(byte_count) {}

  std::uint8_t u8() {
    if (!ok())
      return 0;
    if (stream_ != nullptr) {
      if (stream_buffer_offset_ == stream_buffer_size_ &&
          !refill_stream_buffer()) {
        return 0;
      }
      return std::to_integer<std::uint8_t>(
          stream_buffer_[stream_buffer_offset_++]);
    }
    if (offset_ >= bytes_.size()) {
      fail("truncated HoloLift binary artifact");
      return 0;
    }
    return std::to_integer<std::uint8_t>(bytes_[offset_++]);
  }
  std::uint32_t u32() {
    std::uint32_t value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8)
      value |= static_cast<std::uint32_t>(u8()) << shift;
    return value;
  }
  std::uint64_t u64() {
    std::uint64_t value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8)
      value |= static_cast<std::uint64_t>(u8()) << shift;
    return value;
  }
  std::int64_t i64() { return std::bit_cast<std::int64_t>(u64()); }
  double real() { return std::bit_cast<double>(u64()); }
  bool boolean() {
    const std::uint8_t value = u8();
    if (value > 1)
      fail("invalid binary boolean");
    return value != 0;
  }
  std::size_t size() {
    const std::uint64_t value = u64();
    if (value > std::numeric_limits<std::size_t>::max()) {
      fail("binary size exceeds this build's size_t range");
      return 0;
    }
    return static_cast<std::size_t>(value);
  }
  int integer() {
    const std::int64_t value = i64();
    if (value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
      fail("binary integer exceeds int range");
      return 0;
    }
    return static_cast<int>(value);
  }
  HoloLiftHash128 hash() { return {u64(), u64()}; }
  PayloadDigest digest() {
    PayloadDigest value{};
    for (std::byte &byte : value)
      byte = static_cast<std::byte>(u8());
    return value;
  }
  HoloLiftRange range() { return {size(), size()}; }
  std::string text(std::size_t max_count) {
    const std::size_t count = size();
    if (count > max_count) {
      fail("binary string exceeds the configured size limit");
      return {};
    }
    if (!require(count))
      return {};
    std::string value;
    value.resize(count);
    for (std::size_t idx = 0; idx < count; ++idx)
      value[idx] = static_cast<char>(u8());
    if (!ok())
      return {};
    return value;
  }
  template <typename Enum> Enum enumeration(std::uint8_t max_value) {
    static_assert(std::is_enum_v<Enum>);
    const std::uint8_t value = u8();
    if (value > max_value)
      fail("binary enum value is outside its schema domain");
    return static_cast<Enum>(value);
  }
  [[nodiscard]] std::size_t remaining() const noexcept {
    if (stream_ != nullptr) {
      return stream_unbuffered_ +
             (stream_buffer_size_ - stream_buffer_offset_);
    }
    return offset_ <= bytes_.size() ? bytes_.size() - offset_ : 0;
  }
  [[nodiscard]] bool ok() const noexcept { return error_.empty(); }
  [[nodiscard]] const std::string &error() const noexcept { return error_; }
  void fail(std::string message) {
    if (error_.empty())
      error_ = std::move(message);
  }

private:
  bool refill_stream_buffer() {
    if (stream_ == nullptr || stream_unbuffered_ == 0) {
      fail("truncated HoloLift binary artifact");
      return false;
    }
    const std::size_t count =
        std::min(stream_unbuffered_, stream_buffer_.size());
    stream_->read(reinterpret_cast<char *>(stream_buffer_.data()),
                  static_cast<std::streamsize>(count));
    if (!*stream_ || static_cast<std::size_t>(stream_->gcount()) != count) {
      fail("truncated HoloLift binary artifact");
      return false;
    }
    stream_unbuffered_ -= count;
    stream_buffer_offset_ = 0;
    stream_buffer_size_ = count;
    return true;
  }

  bool require(std::size_t count) {
    if (!ok())
      return false;
    if (count > remaining()) {
      fail("truncated HoloLift binary artifact");
      return false;
    }
    return true;
  }

  std::span<const std::byte> bytes_;
  std::size_t offset_ = 0;
  std::istream *stream_ = nullptr;
  std::array<std::byte, 64 * 1024> stream_buffer_{};
  std::size_t stream_buffer_offset_ = 0;
  std::size_t stream_buffer_size_ = 0;
  std::size_t stream_unbuffered_ = 0;
  std::string error_;
};

void write_source(ByteWriter &out, const HoloLiftSourceContract &source) {
  out.u32(source.schema_version);
  out.text(source.producer_version);
  out.text(source.package_id);
  out.text(source.trajectory_id);
  out.text(source.coordinate_unit);
  out.text(source.time_unit);
  out.text(source.endianness);
  out.text(source.floating_format);
  out.text(source.identity_hash_algorithm);
  out.text(source.binding_hash_algorithm);
  out.text(source.payload_checksum_algorithm);
  out.text(source.binding_mode);
  out.text(source.negative_zero_policy);
  out.text(source.nonfinite_policy);
  out.text(source.observation_scope);
}

HoloLiftSourceContract read_source(ByteReader &in,
                                   std::size_t max_string_bytes) {
  HoloLiftSourceContract source;
  source.schema_version = in.u32();
  source.producer_version = in.text(max_string_bytes);
  source.package_id = in.text(max_string_bytes);
  source.trajectory_id = in.text(max_string_bytes);
  source.coordinate_unit = in.text(max_string_bytes);
  source.time_unit = in.text(max_string_bytes);
  source.endianness = in.text(max_string_bytes);
  source.floating_format = in.text(max_string_bytes);
  source.identity_hash_algorithm = in.text(max_string_bytes);
  source.binding_hash_algorithm = in.text(max_string_bytes);
  source.payload_checksum_algorithm = in.text(max_string_bytes);
  source.binding_mode = in.text(max_string_bytes);
  source.negative_zero_policy = in.text(max_string_bytes);
  source.nonfinite_policy = in.text(max_string_bytes);
  source.observation_scope = in.text(max_string_bytes);
  return source;
}

void write_source_coverage(ByteWriter &out,
                           const HoloLiftSourceCoverage &coverage) {
  out.u8(static_cast<std::uint8_t>(coverage.scope));
  out.size(coverage.source_frame_count);
  out.size(coverage.imported_observation_frame_count);
  out.size(coverage.skipped_local_unwrap_frame_count);
  out.size(coverage.skipped_fallback_frame_count);
}

HoloLiftSourceCoverage read_source_coverage(ByteReader &in) {
  HoloLiftSourceCoverage coverage;
  coverage.scope = in.enumeration<HoloLiftSourceCoverageScope>(1);
  coverage.source_frame_count = in.size();
  coverage.imported_observation_frame_count = in.size();
  coverage.skipped_local_unwrap_frame_count = in.size();
  coverage.skipped_fallback_frame_count = in.size();
  return coverage;
}

void write_provenance(ByteWriter &out,
                      const HoloLiftFrameProvenance &value) {
  out.u8(static_cast<std::uint8_t>(value.status));
  out.u8(static_cast<std::uint8_t>(value.certificate_scope));
  out.u8(static_cast<std::uint8_t>(value.certificate_graph_source));
  out.u8(static_cast<std::uint8_t>(value.hard_graph_source));
  out.u8(static_cast<std::uint8_t>(value.search_evidence_source));
  out.u8(static_cast<std::uint8_t>(value.temporal_policy));
  out.u8(static_cast<std::uint8_t>(value.evidence_state));
  out.boolean(value.evidence_consistency_evaluated);
  out.boolean(value.evidence_consistent);
  out.boolean(value.evidence_graph_connected);
  out.hash(value.soft_observed_contact_pair_hash);
  out.hash(value.soft_observed_lost_pair_hash);
  out.hash(value.soft_observed_all_hypotheses_hash);
  out.hash(value.soft_observed_selected_hypothesis_hash);
  out.hash(value.soft_observed_selected_compatible_pair_hash);
  out.size(value.soft_observed_hypothesis_count);
  out.size(value.soft_observed_component_relation_count);
  out.size(value.soft_observed_ambiguous_relation_count);
  out.size(value.soft_observed_compatible_hypothesis_count);
  out.size(value.soft_observed_selected_compatible_contact_count);
  out.size(value.soft_observed_alternative_hypothesis_contact_count);
  out.size(value.soft_observed_selected_compatible_contact_loss_count);
  out.size(value.soft_observed_no_support_relation_count);
  out.boolean(value.soft_observed_selected_relation_has_support);
  out.boolean(value.soft_observed_hypothesis_construction_attempted);
  out.boolean(value.soft_observed_hypothesis_construction_complete);
  out.size(value.soft_observed_hypothesis_mic_ambiguity_failures);
  out.size(value.soft_observed_hypothesis_component_mapping_failures);
  out.size(value.soft_observed_hypothesis_internal_edges_ignored);
  out.size(value.soft_observed_contacts_checked);
  out.size(value.soft_observed_contacts_lost);
  out.real(value.soft_observed_contact_cutoff_A);
  out.boolean(value.soft_observed_contact_construction_attempted);
  out.boolean(value.soft_observed_contact_construction_complete);
  out.size(value.soft_observed_contact_cell_setup_failures);
  out.size(value.soft_observed_contact_atom_index_failures);
  out.size(value.soft_observed_contact_mic_query_failures);
  out.u64(value.framewise_assignment_signature);
  out.u64(value.output_assignment_signature);
  out.u64(value.temporal_assignment_signature);
  out.boolean(value.temporal_selected);
  out.boolean(value.temporal_changed_from_framewise);
  out.boolean(value.source_frame_report_present);
  out.hash(value.framewise_assignment_identity);
  out.hash(value.output_assignment_identity);
  out.hash(value.temporal_assignment_identity);
}

HoloLiftFrameProvenance read_provenance(ByteReader &in) {
  HoloLiftFrameProvenance value;
  value.status = in.enumeration<HoloLiftFrameStatus>(5);
  value.certificate_scope = in.enumeration<HoloLiftCertificateScope>(3);
  value.certificate_graph_source =
      in.enumeration<HoloLiftCertificateGraphSource>(4);
  value.hard_graph_source = in.enumeration<HoloLiftHardGraphSource>(3);
  value.search_evidence_source =
      in.enumeration<HoloLiftSearchEvidenceSource>(4);
  value.temporal_policy = in.enumeration<HoloLiftTemporalPolicyStatus>(1);
  value.evidence_state = in.enumeration<HoloLiftEvidenceState>(5);
  value.evidence_consistency_evaluated = in.boolean();
  value.evidence_consistent = in.boolean();
  value.evidence_graph_connected = in.boolean();
  value.soft_observed_contact_pair_hash = in.hash();
  value.soft_observed_lost_pair_hash = in.hash();
  value.soft_observed_all_hypotheses_hash = in.hash();
  value.soft_observed_selected_hypothesis_hash = in.hash();
  value.soft_observed_selected_compatible_pair_hash = in.hash();
  value.soft_observed_hypothesis_count = in.size();
  value.soft_observed_component_relation_count = in.size();
  value.soft_observed_ambiguous_relation_count = in.size();
  value.soft_observed_compatible_hypothesis_count = in.size();
  value.soft_observed_selected_compatible_contact_count = in.size();
  value.soft_observed_alternative_hypothesis_contact_count = in.size();
  value.soft_observed_selected_compatible_contact_loss_count = in.size();
  value.soft_observed_no_support_relation_count = in.size();
  value.soft_observed_selected_relation_has_support = in.boolean();
  value.soft_observed_hypothesis_construction_attempted = in.boolean();
  value.soft_observed_hypothesis_construction_complete = in.boolean();
  value.soft_observed_hypothesis_mic_ambiguity_failures = in.size();
  value.soft_observed_hypothesis_component_mapping_failures = in.size();
  value.soft_observed_hypothesis_internal_edges_ignored = in.size();
  value.soft_observed_contacts_checked = in.size();
  value.soft_observed_contacts_lost = in.size();
  value.soft_observed_contact_cutoff_A = in.real();
  value.soft_observed_contact_construction_attempted = in.boolean();
  value.soft_observed_contact_construction_complete = in.boolean();
  value.soft_observed_contact_cell_setup_failures = in.size();
  value.soft_observed_contact_atom_index_failures = in.size();
  value.soft_observed_contact_mic_query_failures = in.size();
  value.framewise_assignment_signature = in.u64();
  value.output_assignment_signature = in.u64();
  value.temporal_assignment_signature = in.u64();
  value.temporal_selected = in.boolean();
  value.temporal_changed_from_framewise = in.boolean();
  value.source_frame_report_present = in.boolean();
  value.framewise_assignment_identity = in.hash();
  value.output_assignment_identity = in.hash();
  value.temporal_assignment_identity = in.hash();
  return value;
}

void write_frame_evidence(ByteWriter &out, const HoloLiftFrameEvidence &value) {
  out.boolean(value.identified);
  out.boolean(value.uniqueness_search_exhaustive);
  out.boolean(value.objective_uniqueness_known);
  out.boolean(value.objective_unique_within_search_domain);
  out.boolean(value.evidence_uniqueness_known);
  out.boolean(value.evidence_unique_within_search_domain);
  out.boolean(value.feasible_assignment_set_enumerated);
  out.u8(static_cast<std::uint8_t>(value.feasible_set_semantics));
  out.boolean(value.equivalent_set_complete);
  out.boolean(value.credible_alternative_set_complete);
  out.size(value.retained_certified_assignment_count);
  out.size(value.search_equivalent_assignments_exported);
  out.size(value.equivalent_best_assignments_within_domain);
  out.size(value.feasible_assignments_within_domain);
  out.size(value.soft_score_valid_assignments_within_domain);
  out.size(value.search_domain_assignments_exported);
}

HoloLiftFrameEvidence read_frame_evidence(ByteReader &in) {
  HoloLiftFrameEvidence value;
  value.identified = in.boolean();
  value.uniqueness_search_exhaustive = in.boolean();
  value.objective_uniqueness_known = in.boolean();
  value.objective_unique_within_search_domain = in.boolean();
  value.evidence_uniqueness_known = in.boolean();
  value.evidence_unique_within_search_domain = in.boolean();
  value.feasible_assignment_set_enumerated = in.boolean();
  value.feasible_set_semantics =
      in.enumeration<HoloLiftFeasibleSetSemantics>(1);
  value.equivalent_set_complete = in.boolean();
  value.credible_alternative_set_complete = in.boolean();
  value.retained_certified_assignment_count = in.size();
  value.search_equivalent_assignments_exported = in.size();
  value.equivalent_best_assignments_within_domain = in.size();
  value.feasible_assignments_within_domain = in.size();
  value.soft_score_valid_assignments_within_domain = in.size();
  value.search_domain_assignments_exported = in.size();
  return value;
}

void write_payload(ByteWriter &out, const HoloLiftObservationStore &store) {
  out.u32(store.api_version());
  write_source(out, store.source_contract());
  write_source_coverage(out, store.source_coverage());

  out.size(store.topology_epochs().size());
  for (const auto &value : store.topology_epochs()) {
    out.u64(value.topology_epoch_id);
    out.u64(value.layout_signature);
    out.range(value.components);
    out.range(value.hard_edges);
    out.u8(static_cast<std::uint8_t>(value.hard_graph_source));
    out.hash(value.layout_identity);
    out.hash(value.topology_epoch_identity);
    out.hash(value.canonical_atom_universe_hash);
    out.size(value.atom_count);
    out.size(value.hard_graph_cycle_rank);
  }
  out.size(store.components().size());
  for (const auto &value : store.components()) {
    out.u64(value.component_id);
    out.i64(value.owner);
    out.boolean(value.mixed_owner);
    out.i64(value.root_atom.owner);
    out.u64(value.root_atom.source_atom_id);
    out.range(value.exact_atom_membership);
    out.hash(value.component_identity);
  }
  out.size(store.exact_atom_memberships().size());
  for (const auto &value : store.exact_atom_memberships()) {
    out.i64(value.owner);
    out.u64(value.source_atom_id);
  }
  out.size(store.exact_atom_masses().size());
  for (const double value : store.exact_atom_masses())
    out.real(value);
  out.size(store.hard_edges().size());
  for (const auto &value : store.hard_edges()) {
    out.u32(value.atom_a);
    out.u32(value.atom_b);
    out.u8(static_cast<std::uint8_t>(value.kind));
  }
  out.size(store.frames().size());
  for (const auto &value : store.frames()) {
    out.size(value.frame);
    out.real(value.time_ps);
    out.size(value.source_sequence_index);
    out.size(value.trajectory_frame_index);
    out.u8(static_cast<std::uint8_t>(value.source_relation));
    out.u32(value.topology_epoch_index);
    out.u64(value.topology_epoch_id);
    out.u64(value.layout_signature);
    for (double element : value.normalized_gram)
      out.real(element);
    for (double element : value.box_matrix)
      out.real(element);
    out.hash(value.box_hash);
    out.hash(value.atom_selection_hash);
    out.hash(value.canonical_atom_universe_hash);
    out.hash(value.wrapped_coordinate_hash);
    out.boolean(value.frame_binding_valid);
    out.u32(value.spatial_lift_policy_version);
    out.hash(value.spatial_lift_identity);
    out.size(value.spatial_lift_ambiguous_hard_edges);
    out.size(value.spatial_lift_cycle_residuals);
    out.boolean(value.spatial_lift_valid);
    out.range(value.candidates);
    out.size(value.selected_candidate_index);
    out.u8(static_cast<std::uint8_t>(value.candidate_set_scope));
    write_provenance(out, value.provenance);
    write_frame_evidence(out, value.evidence);
    out.size(value.search_domain.shell_radius);
    out.boolean(value.search_domain.audit_known);
    out.boolean(value.search_domain.evidence_complete);
    out.boolean(value.search_domain.expansion_attempted);
    out.boolean(value.search_domain.expansion_exhausted);
    out.boolean(value.search_domain.bounded_domain_only);
    out.u8(static_cast<std::uint8_t>(value.search_domain.completeness));
    out.hash(value.topology_epoch_identity);
    out.hash(value.layout_identity);
  }
  out.size(store.candidates().size());
  for (const auto &value : store.candidates()) {
    out.size(value.rank);
    out.u64(value.layout_signature);
    out.u64(value.assignment_signature);
    out.u64(value.evidence_relative_relation_signature);
    out.hash(value.evidence_compatible_hypothesis_hash);
    out.hash(value.evidence_compatible_pair_hash);
    out.size(value.evidence_compatible_hypothesis_count);
    out.size(value.evidence_supported_relation_count);
    out.size(value.evidence_compatible_contact_count);
    out.size(value.evidence_no_support_relation_count);
    out.range(value.component_images);
    out.u8(static_cast<std::uint8_t>(value.observation_class));
    out.boolean(value.evidence.selected_framewise);
    out.boolean(value.evidence.hard_feasible);
    out.boolean(value.evidence.search_equivalence_known);
    out.boolean(value.evidence.equivalent_to_search_best);
    out.boolean(value.evidence.equivalent_under_output_order);
    out.boolean(value.domain_evidence.boundary_known);
    out.boolean(value.domain_evidence.touches_shell_boundary);
    out.size(value.vibe_score_summary.broken_edges);
    out.real(value.vibe_score_summary.continuity_max_d2);
    out.real(value.vibe_score_summary.continuity_path);
    out.real(value.vibe_score_summary.anchor_max_d2);
    out.real(value.vibe_score_summary.anchor_mst2);
    out.real(value.vibe_score_summary.min_pair_d2);
    out.real(value.vibe_score_summary.shift_norm2);
    out.hash(value.layout_identity);
    out.hash(value.assignment_identity);
    out.boolean(value.carried);
    out.u8(static_cast<std::uint8_t>(value.policy_admission));
    out.u8(static_cast<std::uint8_t>(value.provider_evaluation));
    out.size(value.policy_source_frame_index);
    out.hash(value.policy_source_assignment_identity);
  }
  out.size(store.component_images().size());
  for (const auto &value : store.component_images()) {
    out.i64(value.x);
    out.i64(value.y);
    out.i64(value.z);
  }
}

std::size_t read_count(ByteReader &in, std::size_t configured_max,
                       std::size_t minimum_record_bytes,
                       std::string_view table_name) {
  const std::size_t count = in.size();
  if (count > configured_max) {
    in.fail("binary " + std::string(table_name) +
            " count exceeds the configured limit");
    return 0;
  } else if (minimum_record_bytes != 0 &&
             count > in.remaining() / minimum_record_bytes) {
    in.fail("binary " + std::string(table_name) +
            " count exceeds the remaining payload");
    return 0;
  }
  return count;
}

std::expected<HoloLiftObservationStore, std::string>
read_payload(ByteReader &in, const HoloLiftBinaryReadLimits &limits,
             std::uint32_t format_version) {
  HoloLiftObservationStoreBuilder builder;
  builder.api_version = in.u32();
  // Version 9 predates policy admission. Its default candidates remain
  // provider records, never implicitly admitted by the reader.
  if (format_version == 9 && builder.api_version == 9)
    builder.api_version = HOLOLIFT_PHASE0_API_VERSION;
  builder.source_contract = read_source(in, limits.max_string_bytes);
  builder.source_coverage = read_source_coverage(in);

  const std::size_t epoch_count =
      read_count(in, limits.max_topology_epochs, 97, "topology epoch");
  builder.topology_epochs.reserve(epoch_count);
  for (std::size_t idx = 0; idx < epoch_count; ++idx) {
    HoloLiftTopologyEpochRecord value;
    value.topology_epoch_id = in.u64();
    value.layout_signature = in.u64();
    value.components = in.range();
    value.hard_edges = in.range();
    value.hard_graph_source = in.enumeration<HoloLiftHardGraphSource>(3);
    value.layout_identity = in.hash();
    value.topology_epoch_identity = in.hash();
    value.canonical_atom_universe_hash = in.hash();
    value.atom_count = in.size();
    value.hard_graph_cycle_rank = in.size();
    builder.topology_epochs.push_back(value);
  }
  const std::size_t component_count =
      read_count(in, limits.max_components, 65, "component");
  builder.components.reserve(component_count);
  for (std::size_t idx = 0; idx < component_count; ++idx) {
    HoloLiftComponentRecord value;
    value.component_id = in.u64();
    value.owner = in.integer();
    value.mixed_owner = in.boolean();
    value.root_atom.owner = in.integer();
    value.root_atom.source_atom_id = in.u64();
    value.exact_atom_membership = in.range();
    value.component_identity = in.hash();
    builder.components.push_back(value);
  }
  const std::size_t membership_count = read_count(
      in, limits.max_atom_memberships, 16, "atom membership");
  builder.exact_atom_memberships.reserve(membership_count);
  for (std::size_t idx = 0; idx < membership_count; ++idx) {
    builder.exact_atom_memberships.push_back({in.integer(), in.u64()});
  }
  if (format_version >= 6) {
    const std::size_t mass_count = read_count(
        in, limits.max_atom_memberships, sizeof(double), "atom mass");
    builder.exact_atom_masses.reserve(mass_count);
    for (std::size_t idx = 0; idx < mass_count; ++idx)
      builder.exact_atom_masses.push_back(in.real());
  }
  const std::size_t edge_count =
      read_count(in, limits.max_hard_edges, 9, "hard edge");
  builder.hard_edges.reserve(edge_count);
  for (std::size_t idx = 0; idx < edge_count; ++idx) {
    HoloLiftHardEdgeRecord value;
    value.atom_a = in.u32();
    value.atom_b = in.u32();
    value.kind = in.enumeration<HoloLiftHardEdgeKind>(1);
    builder.hard_edges.push_back(value);
  }
  const std::size_t frame_count =
      read_count(in, limits.max_frames, 216, "frame");
  builder.frames.reserve(frame_count);
  for (std::size_t idx = 0; idx < frame_count; ++idx) {
    HoloLiftFrameRecord value;
    value.frame = in.size();
    value.time_ps = in.real();
    value.source_sequence_index = in.size();
    value.trajectory_frame_index = in.size();
    value.source_relation = in.enumeration<HoloLiftSourceFrameRelation>(1);
    value.topology_epoch_index = in.u32();
    value.topology_epoch_id = in.u64();
    value.layout_signature = in.u64();
    for (double &element : value.normalized_gram)
      element = in.real();
    for (double &element : value.box_matrix)
      element = in.real();
    value.box_hash = in.hash();
    value.atom_selection_hash = in.hash();
    value.canonical_atom_universe_hash = in.hash();
    value.wrapped_coordinate_hash = in.hash();
    value.frame_binding_valid = in.boolean();
    value.spatial_lift_policy_version = in.u32();
    value.spatial_lift_identity = in.hash();
    value.spatial_lift_ambiguous_hard_edges = in.size();
    value.spatial_lift_cycle_residuals = in.size();
    value.spatial_lift_valid = in.boolean();
    value.candidates = in.range();
    value.selected_candidate_index = in.size();
    value.candidate_set_scope = in.enumeration<HoloLiftCandidateSetScope>(1);
    value.provenance = read_provenance(in);
    value.evidence = read_frame_evidence(in);
    value.search_domain.shell_radius = in.size();
    value.search_domain.audit_known = in.boolean();
    value.search_domain.evidence_complete = in.boolean();
    value.search_domain.expansion_attempted = in.boolean();
    value.search_domain.expansion_exhausted = in.boolean();
    value.search_domain.bounded_domain_only = in.boolean();
    value.search_domain.completeness =
        in.enumeration<HoloLiftDomainCompleteness>(2);
    value.topology_epoch_identity = in.hash();
    value.layout_identity = in.hash();
    builder.frames.push_back(value);
  }
  const std::size_t candidate_count =
      read_count(in, limits.max_candidates, 100, "candidate");
  builder.candidates.reserve(candidate_count);
  for (std::size_t idx = 0; idx < candidate_count; ++idx) {
    HoloLiftCandidateRecord value;
    value.rank = in.size();
    value.layout_signature = in.u64();
    value.assignment_signature = in.u64();
    if (format_version >= 8) {
      value.evidence_relative_relation_signature = in.u64();
      value.evidence_compatible_hypothesis_hash = in.hash();
      value.evidence_compatible_pair_hash = in.hash();
      value.evidence_compatible_hypothesis_count = in.size();
      value.evidence_supported_relation_count = in.size();
      value.evidence_compatible_contact_count = in.size();
      value.evidence_no_support_relation_count = in.size();
    }
    value.component_images = in.range();
    value.observation_class = in.enumeration<HoloLiftObservationClass>(4);
    value.evidence.selected_framewise = in.boolean();
    value.evidence.hard_feasible = in.boolean();
    value.evidence.search_equivalence_known = in.boolean();
    value.evidence.equivalent_to_search_best = in.boolean();
    value.evidence.equivalent_under_output_order = in.boolean();
    value.domain_evidence.boundary_known = in.boolean();
    value.domain_evidence.touches_shell_boundary = in.boolean();
    value.vibe_score_summary.broken_edges = in.size();
    value.vibe_score_summary.continuity_max_d2 = in.real();
    value.vibe_score_summary.continuity_path = in.real();
    value.vibe_score_summary.anchor_max_d2 = in.real();
    value.vibe_score_summary.anchor_mst2 = in.real();
    value.vibe_score_summary.min_pair_d2 = in.real();
    value.vibe_score_summary.shift_norm2 = in.real();
    value.layout_identity = in.hash();
    value.assignment_identity = in.hash();
    if (format_version >= 10) {
      value.carried = in.boolean();
      value.policy_admission = in.enumeration<HoloLiftPolicyAdmission>(2);
      value.provider_evaluation = in.enumeration<HoloLiftProviderEvaluation>(1);
      value.policy_source_frame_index = in.size();
      value.policy_source_assignment_identity = in.hash();
    }
    builder.candidates.push_back(value);
  }
  const std::size_t image_count = read_count(
      in, limits.max_component_images, 24, "component image");
  builder.component_images.reserve(image_count);
  for (std::size_t idx = 0; idx < image_count; ++idx)
    builder.component_images.push_back({in.i64(), in.i64(), in.i64()});

  if (!in.ok())
    return std::unexpected(in.error());
  if (in.remaining() != 0)
    return std::unexpected("HoloLift binary payload has trailing bytes");
  auto finalized = finalize_hololift_observation_store(std::move(builder));
  if (!finalized) {
    return std::unexpected(
        std::string(hololift_validation_code_name(finalized.error().code)) +
        ": " + finalized.error().message);
  }
  return std::move(*finalized);
}

} // namespace

std::expected<std::string, std::string>
hololift_observation_store_payload_sha256(
    const HoloLiftObservationStore &store) {
  NullOutputBuffer buffer;
  std::ostream sink(&buffer);
  PayloadChecksum checksum;
  ByteWriter payload(sink, &checksum);
  write_payload(payload, store);
  if (!payload.ok() || payload.bytes_written() != checksum.size()) {
    return std::unexpected(
        "failed to hash canonical HoloLift observation payload");
  }
  return format_payload_digest(checksum.finish());
}

HoloLiftArtifactWriteResult write_hololift_observation_store_binary(
    const HoloLiftObservationStore &store,
    const std::filesystem::path &path) {
  static std::atomic<std::uint64_t> temporary_counter{0};
  std::filesystem::path temporary = path;
  temporary += ".tmp." +
               std::to_string(static_cast<std::uint64_t>(
                   std::chrono::steady_clock::now().time_since_epoch().count())) +
               "." + std::to_string(temporary_counter.fetch_add(1));
  const auto cleanup_temporary = [&] {
    std::error_code ignored;
    if (std::filesystem::exists(temporary, ignored) && !ignored)
      std::filesystem::remove(temporary, ignored);
  };
  const auto fail_not_committed = [&](std::string message)
      -> HoloLiftArtifactWriteResult {
    cleanup_temporary();
    return detail::artifact_write_failure(
        HoloLiftArtifactCommitState::NotCommitted, std::move(message));
  };

  std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
  if (!out)
    return fail_not_committed(
        "cannot open temporary HoloLift binary artifact for writing");

  ByteWriter header(out);
  for (const std::uint8_t byte : kMagic)
    header.u8(byte);
  header.u32(HOLOLIFT_BINARY_FORMAT_VERSION);
  header.u64(0);
  header.digest(PayloadDigest{});
  if (!header.ok()) {
    out.close();
    return fail_not_committed("failed to write HoloLift binary header");
  }

  PayloadChecksum checksum;
  ByteWriter payload(out, &checksum);
  write_payload(payload, store);
  if (!payload.ok() || payload.bytes_written() != checksum.size()) {
    out.close();
    return fail_not_committed("failed to stream HoloLift binary payload");
  }
  const std::uint64_t payload_size = payload.bytes_written();
  const PayloadDigest payload_hash = checksum.finish();

  out.flush();
  out.seekp(static_cast<std::streamoff>(kMagic.size() + sizeof(std::uint32_t)),
            std::ios::beg);
  ByteWriter header_patch(out);
  header_patch.u64(payload_size);
  header_patch.digest(payload_hash);
  out.flush();
  if (!header_patch.ok()) {
    out.close();
    return fail_not_committed("failed to finalize HoloLift binary header");
  }
  out.close();
  if (!out)
    return fail_not_committed("failed to close HoloLift binary artifact");

  auto replaced = detail::durably_replace_file(
      temporary, path, "HoloLift binary artifact");
  if (!replaced) {
    auto issue = std::move(replaced.error());
    if (!issue.committed())
      cleanup_temporary();
    return std::unexpected(std::move(issue));
  }
  return *replaced;
}

std::expected<HoloLiftObservationStore, std::string>
read_hololift_observation_store_binary(const std::filesystem::path &path) {
  return read_hololift_observation_store_binary(path, {});
}

std::expected<HoloLiftObservationStore, std::string>
read_hololift_observation_store_binary(
    const std::filesystem::path &path,
    const HoloLiftBinaryReadLimits &limits) {
  constexpr std::size_t header_size = 8 + 4 + 8 + 32;
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return std::unexpected("cannot open HoloLift binary artifact");
  in.seekg(0, std::ios::end);
  const std::streamoff length = in.tellg();
  if (length < static_cast<std::streamoff>(header_size) ||
      static_cast<std::uint64_t>(length) > limits.max_artifact_bytes ||
      static_cast<std::uint64_t>(length) >
          std::numeric_limits<std::size_t>::max()) {
    return std::unexpected("invalid HoloLift binary artifact size");
  }
  in.seekg(0, std::ios::beg);
  std::array<std::byte, header_size> header_bytes{};
  in.read(reinterpret_cast<char *>(header_bytes.data()),
          static_cast<std::streamsize>(header_bytes.size()));
  if (!in)
    return std::unexpected("failed to read HoloLift binary header");

  ByteReader header(header_bytes);
  for (const std::uint8_t expected : kMagic) {
    if (header.u8() != expected)
      return std::unexpected("invalid HoloLift binary magic");
  }
  const std::uint32_t format_version = header.u32();
  if (format_version != 9 && format_version != HOLOLIFT_BINARY_FORMAT_VERSION) {
    return std::unexpected("unsupported HoloLift binary format version");
  }
  const std::uint64_t payload_size_u64 = header.u64();
  const PayloadDigest expected_checksum = header.digest();
  if (!header.ok())
    return std::unexpected(header.error());
  const std::uint64_t file_payload_size =
      static_cast<std::uint64_t>(length) - header_size;
  if (payload_size_u64 != file_payload_size ||
      payload_size_u64 > limits.max_artifact_bytes - header_size ||
      payload_size_u64 > std::numeric_limits<std::size_t>::max() ||
      payload_size_u64 >
          static_cast<std::uint64_t>(
              std::numeric_limits<std::streamsize>::max())) {
    return std::unexpected("HoloLift binary payload size mismatch");
  }
  const std::size_t payload_size =
      static_cast<std::size_t>(payload_size_u64);
  const auto actual_checksum = payload_checksum_stream(in, payload_size);
  if (!actual_checksum)
    return std::unexpected(actual_checksum.error());
  if (*actual_checksum != expected_checksum)
    return std::unexpected("HoloLift binary payload checksum mismatch");
  in.clear();
  in.seekg(static_cast<std::streamoff>(header_size), std::ios::beg);
  if (!in)
    return std::unexpected("failed to rewind HoloLift binary payload");
  ByteReader payload_reader(in, payload_size);
  return read_payload(payload_reader, limits, format_version);
}

} // namespace titan_hololift
