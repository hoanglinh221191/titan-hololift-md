#include "tabi_ses_patch_mapping.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>

namespace TitanPBE::Tabi::analytic_ses {
namespace {

[[nodiscard]] bool finite_vec(const Vec3 &value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] bool checked_add(std::size_t &value,
                               std::size_t increment) noexcept {
  if (value > std::numeric_limits<std::size_t>::max() - increment)
    return false;
  value += increment;
  return true;
}

[[nodiscard]] bool checked_multiply(std::size_t left, std::size_t right,
                                    std::size_t &result) noexcept {
  if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left)
    return false;
  result = left * right;
  return true;
}

[[nodiscard]] bool patch_ref_in_range(const PatchComplex &complex,
                                      PatchRefId ref) noexcept {
  if (!ref.valid())
    return false;
  switch (ref.kind()) {
  case PatchKind::contact:
    return ref.index() < complex.contacts.size();
  case PatchKind::toroidal:
    return ref.index() < complex.tori.size();
  case PatchKind::reentrant:
    return ref.index() < complex.reentrant.size();
  }
  return false;
}

[[nodiscard]] bool add_scaled_coordinate(double lower, double spacing,
                                         std::size_t index,
                                         double &value) noexcept {
  value = lower + spacing * static_cast<double>(index);
  return std::isfinite(value);
}

[[nodiscard]] bool
crossing_inside_lattice_axis(const PatchMappingGridSpec &spec, RayAxis axis,
                             double coordinate) noexcept {
  if (!std::isfinite(coordinate))
    return false;
  double lower = 0.0;
  std::size_t count = 0U;
  switch (axis) {
  case RayAxis::x:
    lower = spec.lower.x;
    count = spec.nx;
    break;
  case RayAxis::y:
    lower = spec.lower.y;
    count = spec.ny;
    break;
  case RayAxis::z:
    lower = spec.lower.z;
    count = spec.nz;
    break;
  }
  if (count == 0U)
    return false;
  double upper = 0.0;
  if (!add_scaled_coordinate(lower, spec.grid_spacing, count - 1U, upper))
    return false;
  const double scale =
      std::max({1.0, std::abs(lower), std::abs(upper), std::abs(coordinate),
                std::abs(upper - lower), std::abs(spec.grid_spacing)});
  const double margin = 2048.0 * std::numeric_limits<double>::epsilon() * scale;
  return coordinate >= lower - margin && coordinate <= upper + margin;
}

[[nodiscard]] bool validate_axis_contract(const AxisRayCrossings &axis,
                                          const AxisPatchProvenance &provenance,
                                          RayAxis expected_axis,
                                          std::size_t expected_u,
                                          std::size_t expected_v) noexcept {
  const std::size_t row_count = axis.row_count();
  if (axis.axis != expected_axis || axis.u_count != expected_u ||
      axis.v_count != expected_v ||
      row_count == std::numeric_limits<std::size_t>::max() ||
      row_count > PatchCrossingAddress::row_mask ||
      axis.offsets.size() != row_count + 1U || axis.offsets.empty() ||
      axis.offsets.front() != 0U ||
      axis.offsets.back() != axis.coordinates.size() ||
      axis.coordinates.size() > std::numeric_limits<std::uint32_t>::max() ||
      !provenance.validate(axis.coordinates.size()))
    return false;
  for (std::size_t row = 0U; row < row_count; ++row) {
    if (axis.offsets[row] > axis.offsets[row + 1U] ||
        axis.offsets[row + 1U] > axis.coordinates.size())
      return false;
    for (std::size_t crossing = axis.offsets[row];
         crossing < axis.offsets[row + 1U]; ++crossing) {
      if (!std::isfinite(axis.coordinates[crossing]) ||
          (crossing > axis.offsets[row] &&
           axis.coordinates[crossing] <= axis.coordinates[crossing - 1U]))
        return false;
    }
  }
  return true;
}

[[nodiscard]] double bounded_patch_tolerance(double root_tolerance,
                                             const Vec3 &point,
                                             double spacing) noexcept {
  if (!std::isfinite(root_tolerance) || root_tolerance < 0.0 ||
      !finite_vec(point) || !std::isfinite(spacing) || !(spacing > 0.0))
    return std::numeric_limits<double>::quiet_NaN();
  const double scale = std::max({1.0, std::abs(point.x), std::abs(point.y),
                                 std::abs(point.z), std::abs(spacing)});
  return std::max(root_tolerance,
                  4096.0 * std::numeric_limits<double>::epsilon() * scale);
}

[[nodiscard]] double point_distance(const Vec3 &left,
                                    const Vec3 &right) noexcept {
  const double dx = left.x - right.x;
  const double dy = left.y - right.y;
  const double dz = left.z - right.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void clear_payload(PreparedPatchCrossingWitnesses &result) noexcept {
  std::vector<CertifiedPatchCrossingWitness>().swap(result.witnesses);
  result.axis_offsets = {};
  result.complex_fingerprint = {};
  result.lattice_identity = {};
  result.refs_audited = 0U;
  result.root_events_audited = 0U;
  result.identity_certified_crossings = 0U;
  result.root_reproduced_crossings = 0U;
  result.bounded_patch_location_certified_crossings = 0U;
  result.transversality_certified_crossings = 0U;
  result.multi_ref_crossings = 0U;
  result.multi_ref_relation_uncertified_crossings = 0U;
  result.primitive_simplicity_uncertified_crossings = 0U;
  result.repeated_root_ambiguities = 0U;
  result.intersection_uncertified = 0U;
  result.roots_not_reproduced = 0U;
  result.root_multiplicity_mismatches = 0U;
  result.projector_uncertified = 0U;
  result.patch_residual_failures = 0U;
  result.nonfinite_bounded_residuals = 0U;
  result.retained_bytes = 0U;
  result.peak_temporary_bytes = 0U;
  result.identity_certificate_complete = false;
  result.root_reproduction_complete = false;
  result.location_certificate_complete = false;
  result.transversality_certificate_complete = false;
}

} // namespace

double patch_crossing_root_match_tolerance(double coordinate_merge_tolerance,
                                           const DecodedPatchRayRow &row,
                                           double coordinate,
                                           double grid_spacing) noexcept {
  if (!std::isfinite(coordinate_merge_tolerance) ||
      coordinate_merge_tolerance < 0.0 || !std::isfinite(row.u) ||
      !std::isfinite(row.v) || !std::isfinite(coordinate) ||
      !std::isfinite(grid_spacing) || !(grid_spacing > 0.0))
    return std::numeric_limits<double>::quiet_NaN();
  const double scale = std::max({1.0, std::abs(row.u), std::abs(row.v),
                                 std::abs(coordinate), std::abs(grid_spacing)});
  return std::max(coordinate_merge_tolerance,
                  2048.0 * std::numeric_limits<double>::epsilon() * scale);
}

bool patch_crossing_residual_within_tolerance(double residual,
                                              double tolerance) noexcept {
  return std::isfinite(residual) && residual >= 0.0 &&
         std::isfinite(tolerance) && tolerance >= 0.0 && residual <= tolerance;
}

CrossingMultiplicityClassification
classify_patch_crossing_multiplicity(std::uint32_t stored_multiplicity,
                                     std::size_t observed_matches) noexcept {
  CrossingMultiplicityClassification result;
  if (stored_multiplicity == 0U ||
      observed_matches != static_cast<std::size_t>(stored_multiplicity)) {
    result.root_status =
        observed_matches == 0U
            ? CrossingRootReproductionStatus::root_not_reproduced
            : CrossingRootReproductionStatus::root_multiplicity_mismatch;
    return result;
  }
  result.root_status = CrossingRootReproductionStatus::reproduced;
  result.transversality_if_location_certified =
      stored_multiplicity == 1U
          ? CrossingTransversalityStatus::
                primitive_filtered_transversal_unoriented
          : CrossingTransversalityStatus::repeated_root_ambiguous;
  return result;
}

CrossingTransversalityStatus classify_patch_crossing_group_transversality(
    std::size_t ref_count, std::uint32_t maximum_ref_multiplicity,
    bool bounded_location_certified,
    bool primitive_simplicity_certified) noexcept {
  if (!bounded_location_certified || ref_count == 0U ||
      maximum_ref_multiplicity == 0U)
    return CrossingTransversalityStatus::location_uncertified;
  if (maximum_ref_multiplicity > 1U)
    return CrossingTransversalityStatus::repeated_root_ambiguous;
  if (ref_count > 1U)
    return CrossingTransversalityStatus::multi_ref_relation_uncertified;
  if (!primitive_simplicity_certified)
    return CrossingTransversalityStatus::primitive_simplicity_uncertified;
  return CrossingTransversalityStatus::
      primitive_filtered_transversal_unoriented;
}

bool PatchCrossingAddress::try_make(RayAxis axis, std::size_t row,
                                    std::size_t crossing,
                                    PatchCrossingAddress &result) noexcept {
  const auto axis_value = static_cast<std::uint64_t>(axis);
  if (axis_value > static_cast<std::uint64_t>(RayAxis::z) || row > row_mask ||
      crossing > std::numeric_limits<std::uint32_t>::max()) {
    result = {};
    return false;
  }
  result.route_ = (axis_value << axis_shift) | static_cast<std::uint64_t>(row);
  result.crossing_ = static_cast<std::uint32_t>(crossing);
  return true;
}

bool validate_patch_mapping_grid_spec(const PatchMappingGridSpec &spec,
                                      std::array<std::size_t, 3> *row_counts,
                                      std::size_t *vertex_count) noexcept {
  if (!finite_vec(spec.lower) || !std::isfinite(spec.grid_spacing) ||
      !(spec.grid_spacing > 0.0) || spec.nx == 0U || spec.ny == 0U ||
      spec.nz == 0U)
    return false;
  std::array<std::size_t, 3> rows{};
  std::size_t xy = 0U;
  std::size_t vertices = 0U;
  if (!checked_multiply(spec.ny, spec.nz, rows[0]) ||
      !checked_multiply(spec.nx, spec.nz, rows[1]) ||
      !checked_multiply(spec.nx, spec.ny, rows[2]) ||
      !checked_multiply(spec.nx, spec.ny, xy) ||
      !checked_multiply(xy, spec.nz, vertices) ||
      rows[0] > PatchCrossingAddress::row_mask ||
      rows[1] > PatchCrossingAddress::row_mask ||
      rows[2] > PatchCrossingAddress::row_mask)
    return false;
  const auto last_coordinate_finite = [&](double lower, std::size_t count) {
    double value = 0.0;
    return add_scaled_coordinate(lower, spec.grid_spacing, count - 1U, value);
  };
  if (!last_coordinate_finite(spec.lower.x, spec.nx) ||
      !last_coordinate_finite(spec.lower.y, spec.ny) ||
      !last_coordinate_finite(spec.lower.z, spec.nz))
    return false;
  if (row_counts != nullptr)
    *row_counts = rows;
  if (vertex_count != nullptr)
    *vertex_count = vertices;
  return true;
}

bool decode_patch_ray_row(const PatchMappingGridSpec &spec, RayAxis axis,
                          std::size_t row,
                          DecodedPatchRayRow &decoded) noexcept {
  decoded = {};
  std::array<std::size_t, 3> row_counts{};
  if (!validate_patch_mapping_grid_spec(spec, &row_counts, nullptr))
    return false;
  const std::size_t axis_index = static_cast<std::size_t>(axis);
  if (axis_index >= row_counts.size() || row >= row_counts[axis_index])
    return false;
  std::size_t nu = 0U;
  double lower_u = 0.0;
  double lower_v = 0.0;
  switch (axis) {
  case RayAxis::x:
    nu = spec.ny;
    lower_u = spec.lower.y;
    lower_v = spec.lower.z;
    break;
  case RayAxis::y:
    nu = spec.nx;
    lower_u = spec.lower.x;
    lower_v = spec.lower.z;
    break;
  case RayAxis::z:
    nu = spec.nx;
    lower_u = spec.lower.x;
    lower_v = spec.lower.y;
    break;
  }
  if (nu == 0U)
    return false;
  decoded.axis = axis;
  decoded.row = row;
  decoded.u_index = row % nu;
  decoded.v_index = row / nu;
  return add_scaled_coordinate(lower_u, spec.grid_spacing, decoded.u_index,
                               decoded.u) &&
         add_scaled_coordinate(lower_v, spec.grid_spacing, decoded.v_index,
                               decoded.v);
}

bool reconstruct_patch_crossing_point(const PatchMappingGridSpec &spec,
                                      const PatchCrossingAddress &address,
                                      double coordinate, Vec3 &point) noexcept {
  point = {};
  if (!address.valid() || !std::isfinite(coordinate))
    return false;
  DecodedPatchRayRow row;
  if (!decode_patch_ray_row(spec, address.axis(), address.row(), row))
    return false;
  switch (address.axis()) {
  case RayAxis::x:
    point = {coordinate, row.u, row.v};
    break;
  case RayAxis::y:
    point = {row.u, coordinate, row.v};
    break;
  case RayAxis::z:
    point = {row.u, row.v, coordinate};
    break;
  }
  return finite_vec(point);
}

PreparedPatchCrossingWitnesses build_certified_patch_crossing_witnesses(
    const PatchComplex &complex, const LocalPatchProjectionIndex &index,
    const PatchRayGrid &grid, const PatchMappingGridSpec &spec) {
  const auto begin = std::chrono::steady_clock::now();
  PreparedPatchCrossingWitnesses result;
  const auto stamp_time = [&]() noexcept {
    result.build_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - begin)
                          .count();
  };
  try {
    std::array<std::size_t, 3> row_counts{};
    std::size_t vertex_count = 0U;
    if (!validate_patch_mapping_grid_spec(spec, &row_counts, &vertex_count)) {
      result.status = "invalid_grid_spec";
      stamp_time();
      return result;
    }
    (void)vertex_count;
    if (grid.status != "pass" || grid.provenance.status != "pass") {
      result.status = "invalid_grid_status";
      stamp_time();
      return result;
    }
    if (!grid.provenance.enabled || !grid.provenance.certified ||
        !std::isfinite(grid.provenance.coordinate_merge_tolerance) ||
        grid.provenance.coordinate_merge_tolerance < 0.0) {
      result.status = "uncertified_provenance";
      stamp_time();
      return result;
    }
    const PatchRayProvenance::LatticeIdentity expected_lattice =
        make_patch_ray_lattice_identity(spec.lower, spec.grid_spacing, spec.nx,
                                        spec.ny, spec.nz);
    if (!expected_lattice.valid || !grid.provenance.lattice_identity.valid ||
        expected_lattice != grid.provenance.lattice_identity) {
      result.status = "stale_grid_lattice";
      stamp_time();
      return result;
    }
    result.lattice_identity = expected_lattice;
    const PatchComplexFingerprint current_fingerprint =
        fingerprint_patch_complex(complex);
    if (!current_fingerprint.valid ||
        !grid.provenance.complex_fingerprint.valid ||
        current_fingerprint != grid.provenance.complex_fingerprint) {
      result.status = "stale_grid_complex";
      stamp_time();
      return result;
    }
    result.complex_fingerprint = current_fingerprint;
    if (index.source_complex != &complex ||
        !index.source_complex_fingerprint.valid ||
        index.source_complex_fingerprint != current_fingerprint ||
        !index.bounded_domains_certified || !index.bvh_invariants_certified) {
      result.status = "invalid_projection_index";
      stamp_time();
      return result;
    }
    const std::array<std::size_t, 3> expected_u = {spec.ny, spec.nx, spec.nx};
    const std::array<std::size_t, 3> expected_v = {spec.nz, spec.nz, spec.ny};
    std::size_t total_crossings = 0U;
    std::size_t maximum_refs_per_crossing = 0U;
    for (std::size_t axis = 0U; axis < 3U; ++axis) {
      if (!validate_axis_contract(grid.axes[axis], grid.provenance.axes[axis],
                                  static_cast<RayAxis>(axis), expected_u[axis],
                                  expected_v[axis]) ||
          grid.axes[axis].row_count() != row_counts[axis] ||
          !checked_add(total_crossings, grid.axes[axis].coordinates.size())) {
        result.status = "invalid_grid_contract";
        stamp_time();
        return result;
      }
      for (const double coordinate : grid.axes[axis].coordinates) {
        if (!crossing_inside_lattice_axis(spec, static_cast<RayAxis>(axis),
                                          coordinate)) {
          result.status = "crossing_out_of_axis_bounds";
          stamp_time();
          return result;
        }
      }
    }
    if (total_crossings == 0U ||
        total_crossings != grid.provenance.unique_crossings) {
      result.status =
          total_crossings == 0U ? "no_crossings" : "invalid_grid_contract";
      stamp_time();
      return result;
    }
    result.witnesses.reserve(total_crossings);
    for (std::size_t axis_index = 0U; axis_index < 3U; ++axis_index) {
      result.axis_offsets[axis_index] = result.witnesses.size();
      const AxisRayCrossings &axis = grid.axes[axis_index];
      const AxisPatchProvenance &provenance = grid.provenance.axes[axis_index];
      for (std::size_t row_index = 0U; row_index < axis.row_count();
           ++row_index) {
        DecodedPatchRayRow decoded;
        if (!decode_patch_ray_row(spec, axis.axis, row_index, decoded)) {
          clear_payload(result);
          result.status = "row_decode_failed";
          stamp_time();
          return result;
        }
        for (std::size_t crossing = axis.offsets[row_index];
             crossing < axis.offsets[row_index + 1U]; ++crossing) {
          CertifiedPatchCrossingWitness witness;
          if (!PatchCrossingAddress::try_make(axis.axis, row_index, crossing,
                                              witness.address) ||
              !reconstruct_patch_crossing_point(spec, witness.address,
                                                axis.coordinates[crossing],
                                                witness.point)) {
            clear_payload(result);
            result.status = "address_or_point_invalid";
            stamp_time();
            return result;
          }
          witness.coordinate = axis.coordinates[crossing];
          witness.root_match_tolerance = patch_crossing_root_match_tolerance(
              grid.provenance.coordinate_merge_tolerance, decoded,
              witness.coordinate, spec.grid_spacing);
          witness.bounded_patch_tolerance = bounded_patch_tolerance(
              witness.root_match_tolerance, witness.point, spec.grid_spacing);
          if (!std::isfinite(witness.root_match_tolerance) ||
              !std::isfinite(witness.bounded_patch_tolerance)) {
            clear_payload(result);
            result.status = "invalid_tolerance";
            stamp_time();
            return result;
          }
          const std::span<const PatchRefId> refs = provenance.refs(crossing);
          if (refs.empty() ||
              refs.size() > std::numeric_limits<std::uint32_t>::max()) {
            clear_payload(result);
            result.status = "invalid_patch_refs";
            stamp_time();
            return result;
          }
          witness.ref_count = static_cast<std::uint32_t>(refs.size());
          maximum_refs_per_crossing =
              std::max(maximum_refs_per_crossing, refs.size());
          if (!checked_add(result.refs_audited, refs.size())) {
            clear_payload(result);
            result.status = "counter_overflow";
            stamp_time();
            return result;
          }
          bool repeated = false;
          for (std::size_t ordinal = 0U; ordinal < refs.size(); ++ordinal) {
            if (!patch_ref_in_range(complex, refs[ordinal])) {
              clear_payload(result);
              result.status = "invalid_patch_refs";
              stamp_time();
              return result;
            }
            const std::uint32_t multiplicity =
                provenance.ref_multiplicity(crossing, ordinal);
            if (multiplicity == 0U) {
              clear_payload(result);
              result.status = "invalid_multiplicity";
              stamp_time();
              return result;
            }
            if (witness.total_root_multiplicity >
                std::numeric_limits<std::uint32_t>::max() - multiplicity) {
              clear_payload(result);
              result.status = "multiplicity_overflow";
              stamp_time();
              return result;
            }
            witness.total_root_multiplicity += multiplicity;
            witness.maximum_ref_multiplicity =
                std::max(witness.maximum_ref_multiplicity, multiplicity);
            repeated = repeated || multiplicity > 1U;
          }
          if (!checked_add(result.root_events_audited,
                           witness.total_root_multiplicity)) {
            clear_payload(result);
            result.status = "counter_overflow";
            stamp_time();
            return result;
          }
          witness.identity_status =
              refs.size() == 1U
                  ? CrossingIdentityCertificateStatus::valid_single_ref
                  : CrossingIdentityCertificateStatus::valid_multi_ref;
          if (!checked_add(result.identity_certified_crossings, 1U) ||
              (refs.size() > 1U &&
               !checked_add(result.multi_ref_crossings, 1U)) ||
              (repeated &&
               !checked_add(result.repeated_root_ambiguities, 1U))) {
            clear_payload(result);
            result.status = "counter_overflow";
            stamp_time();
            return result;
          }

          bool root_reproduced = true;
          bool all_matched_roots_primitive_simple = true;
          for (std::size_t ordinal = 0U; ordinal < refs.size(); ++ordinal) {
            const PatchRefId ref = refs[ordinal];
            const std::uint32_t stored_multiplicity =
                provenance.ref_multiplicity(crossing, ordinal);
            const PatchAxisRoots roots = intersect_patch_axis(
                complex, index.reentrant_probes, ref.kind(), ref.index(),
                axis.axis, decoded.u, decoded.v);
            if (!roots.certified) {
              witness.root_status =
                  CrossingRootReproductionStatus::intersection_uncertified;
              if (!checked_add(result.intersection_uncertified, 1U)) {
                clear_payload(result);
                result.status = "counter_overflow";
                stamp_time();
                return result;
              }
              root_reproduced = false;
              break;
            }
            std::size_t matches = 0U;
            double residual = 0.0;
            for (std::size_t root = 0U; root < roots.count; ++root) {
              const double candidate_residual =
                  std::abs(roots.coordinates[root] - witness.coordinate);
              if (!std::isfinite(candidate_residual)) {
                witness.root_status =
                    CrossingRootReproductionStatus::nonfinite_residual;
                root_reproduced = false;
                break;
              }
              if (patch_crossing_residual_within_tolerance(
                      candidate_residual, witness.root_match_tolerance)) {
                ++matches;
                residual = std::max(residual, candidate_residual);
                all_matched_roots_primitive_simple =
                    all_matched_roots_primitive_simple &&
                    roots.primitive_simplicity_certified[root] != 0U;
              }
            }
            if (!root_reproduced)
              break;
            const CrossingMultiplicityClassification classification =
                classify_patch_crossing_multiplicity(stored_multiplicity,
                                                     matches);
            if (classification.root_status ==
                CrossingRootReproductionStatus::root_not_reproduced) {
              witness.root_status = classification.root_status;
              if (!checked_add(result.roots_not_reproduced, 1U)) {
                clear_payload(result);
                result.status = "counter_overflow";
                stamp_time();
                return result;
              }
              root_reproduced = false;
              break;
            }
            if (classification.root_status !=
                CrossingRootReproductionStatus::reproduced) {
              witness.root_status = classification.root_status;
              if (!checked_add(result.root_multiplicity_mismatches, 1U)) {
                clear_payload(result);
                result.status = "counter_overflow";
                stamp_time();
                return result;
              }
              root_reproduced = false;
              break;
            }
            witness.maximum_root_residual =
                std::max(witness.maximum_root_residual, residual);
          }
          if (root_reproduced) {
            witness.root_status = CrossingRootReproductionStatus::reproduced;
            witness.primitive_simplicity_certified =
                all_matched_roots_primitive_simple;
            if (!checked_add(result.root_reproduced_crossings, 1U)) {
              clear_payload(result);
              result.status = "counter_overflow";
              stamp_time();
              return result;
            }
          }

          bool bounded_location = root_reproduced;
          if (bounded_location) {
            const double search_distance = std::max(
                witness.bounded_patch_tolerance, 0.25 * spec.grid_spacing);
            for (const PatchRefId ref : refs) {
              const LocalProjectionHit hit = project_patch_bounded_closest(
                  complex, index, ref.kind(), ref.index(), witness.point,
                  search_distance);
              if (!hit.found || !hit.bounded_closest_certified ||
                  !hit.normal_certified || hit.kind != ref.kind() ||
                  hit.patch_index != ref.index() ||
                  !std::isfinite(hit.distance) || !finite_vec(hit.point)) {
                witness.location_status =
                    CrossingBoundedLocationStatus::projector_uncertified;
                if (!checked_add(result.projector_uncertified, 1U)) {
                  clear_payload(result);
                  result.status = "counter_overflow";
                  stamp_time();
                  return result;
                }
                bounded_location = false;
                break;
              }
              const double point_residual =
                  point_distance(hit.point, witness.point);
              if (!std::isfinite(point_residual)) {
                witness.location_status =
                    CrossingBoundedLocationStatus::nonfinite_residual;
                if (!checked_add(result.nonfinite_bounded_residuals, 1U)) {
                  clear_payload(result);
                  result.status = "counter_overflow";
                  stamp_time();
                  return result;
                }
                bounded_location = false;
                break;
              }
              const double residual = std::max(hit.distance, point_residual);
              witness.maximum_patch_distance =
                  std::max(witness.maximum_patch_distance, residual);
              if (!patch_crossing_residual_within_tolerance(
                      residual, witness.bounded_patch_tolerance)) {
                witness.location_status =
                    CrossingBoundedLocationStatus::patch_residual_failed;
                if (!checked_add(result.patch_residual_failures, 1U)) {
                  clear_payload(result);
                  result.status = "counter_overflow";
                  stamp_time();
                  return result;
                }
                bounded_location = false;
                break;
              }
            }
          }
          if (!root_reproduced) {
            witness.location_status =
                CrossingBoundedLocationStatus::root_unreproduced;
          } else if (bounded_location) {
            witness.location_status = CrossingBoundedLocationStatus::certified;
            if (!checked_add(result.bounded_patch_location_certified_crossings,
                             1U)) {
              clear_payload(result);
              result.status = "counter_overflow";
              stamp_time();
              return result;
            }
          }
          witness.transversality_status =
              classify_patch_crossing_group_transversality(
                  refs.size(), witness.maximum_ref_multiplicity,
                  bounded_location, witness.primitive_simplicity_certified);
          if (witness.transversality_status ==
              CrossingTransversalityStatus::
                  primitive_filtered_transversal_unoriented) {
            if (!checked_add(result.transversality_certified_crossings, 1U)) {
              clear_payload(result);
              result.status = "counter_overflow";
              stamp_time();
              return result;
            }
          } else if (witness.transversality_status ==
                     CrossingTransversalityStatus::
                         multi_ref_relation_uncertified) {
            if (!checked_add(result.multi_ref_relation_uncertified_crossings,
                             1U)) {
              clear_payload(result);
              result.status = "counter_overflow";
              stamp_time();
              return result;
            }
          } else if (witness.transversality_status ==
                     CrossingTransversalityStatus::
                         primitive_simplicity_uncertified) {
            if (!checked_add(result.primitive_simplicity_uncertified_crossings,
                             1U)) {
              clear_payload(result);
              result.status = "counter_overflow";
              stamp_time();
              return result;
            }
          }
          result.witnesses.push_back(witness);
        }
      }
    }
    result.axis_offsets[3] = result.witnesses.size();
    if (result.witnesses.size() != total_crossings) {
      clear_payload(result);
      result.status = "crossing_count_mismatch";
      stamp_time();
      return result;
    }
    if (result.refs_audited != grid.provenance.refs_total ||
        result.root_events_audited != grid.provenance.roots_total ||
        result.multi_ref_crossings != grid.provenance.seam_groups ||
        result.repeated_root_ambiguities !=
            grid.provenance.repeated_root_groups ||
        result.repeated_root_ambiguities !=
            grid.provenance.same_primitive_near_pair_unclassified ||
        maximum_refs_per_crossing != grid.provenance.max_refs_per_crossing) {
      clear_payload(result);
      result.status = "invalid_provenance_summary";
      stamp_time();
      return result;
    }
    if (result.witnesses.capacity() >
        std::numeric_limits<std::size_t>::max() /
            sizeof(CertifiedPatchCrossingWitness)) {
      clear_payload(result);
      result.status = "byte_overflow";
      stamp_time();
      return result;
    }
    result.retained_bytes =
        result.witnesses.capacity() * sizeof(CertifiedPatchCrossingWitness);
    result.peak_temporary_bytes = result.retained_bytes;
    result.identity_certificate_complete =
        result.identity_certified_crossings == result.witnesses.size();
    result.root_reproduction_complete =
        result.root_reproduced_crossings == result.witnesses.size();
    result.location_certificate_complete =
        result.bounded_patch_location_certified_crossings ==
        result.witnesses.size();
    result.transversality_certificate_complete =
        result.transversality_certified_crossings == result.witnesses.size();
    if (!result.identity_certificate_complete) {
      result.status = "identity_uncertified";
    } else if (!result.root_reproduction_complete) {
      result.status = "root_reproduction_uncertified";
    } else if (!result.location_certificate_complete) {
      result.status = "location_uncertified";
    } else if (!result.transversality_certificate_complete) {
      result.status = result.repeated_root_ambiguities != 0U
                          ? "repeated_root_transversality_ambiguous"
                      : result.multi_ref_relation_uncertified_crossings != 0U
                          ? "multi_ref_relation_uncertified"
                      : result.primitive_simplicity_uncertified_crossings != 0U
                          ? "primitive_simplicity_uncertified"
                          : "transversality_uncertified";
    } else {
      result.status = "pass";
    }
    stamp_time();
    return result;
  } catch (const std::bad_alloc &) {
    clear_payload(result);
    result.status = "alloc_fail";
    stamp_time();
    return result;
  } catch (const std::length_error &) {
    clear_payload(result);
    result.status = "length_err";
    stamp_time();
    return result;
  }
}

PatchGridTopologyEvidenceClassification
classify_patch_grid_topology_evidence(PatchGridTopologyState state,
                                      PatchGridTopologySource source,
                                      bool cavity_fill) noexcept {
  PatchGridTopologyEvidenceClassification result;
  switch (source) {
  case PatchGridTopologySource::unknown:
    return result;
  case PatchGridTopologySource::inside_atom_union:
    if (!cavity_fill && state == PatchGridTopologyState::inside) {
      result.status =
          PatchGridTopologyEvidenceStatus::certified_inside_atom_union;
      result.analytic_evidence = true;
      result.exact_vertex_certified = true;
    }
    return result;
  case PatchGridTopologySource::outside_probe_expanded_union:
    if (!cavity_fill && state == PatchGridTopologyState::outside) {
      result.status = PatchGridTopologyEvidenceStatus::
          certified_outside_probe_expanded_union;
      result.analytic_evidence = true;
      result.exact_vertex_certified = true;
    }
    return result;
  case PatchGridTopologySource::direct_unanimous_parity:
    if (!cavity_fill && state != PatchGridTopologyState::unresolved) {
      // The enum records the production branch only. Slice B has not audited
      // all three prefixes and the transversality of every counted crossing.
      result.status = PatchGridTopologyEvidenceStatus::
          direct_unanimous_parity_not_yet_audited;
    }
    return result;
  case PatchGridTopologySource::analytic_local_repair:
    if (!cavity_fill && state != PatchGridTopologyState::unresolved) {
      result.status =
          PatchGridTopologyEvidenceStatus::analytic_perturbed_point_parity;
      result.analytic_evidence = true;
    }
    return result;
  case PatchGridTopologySource::morphology_fallback:
    if (!cavity_fill && state != PatchGridTopologyState::unresolved)
      result.status =
          PatchGridTopologyEvidenceStatus::morphology_fallback_nonanalytic;
    return result;
  case PatchGridTopologySource::cavity_fill:
    if (cavity_fill && state == PatchGridTopologyState::inside)
      result.status = PatchGridTopologyEvidenceStatus::cavity_fill_nonanalytic;
    return result;
  }
  return result;
}

bool try_patch_grid_vertex_linear_index(const PatchMappingGridSpec &spec,
                                        std::size_t ix, std::size_t iy,
                                        std::size_t iz,
                                        std::size_t &linear_index) noexcept {
  linear_index = 0U;
  if (!validate_patch_mapping_grid_spec(spec, nullptr, nullptr) ||
      ix >= spec.nx || iy >= spec.ny || iz >= spec.nz)
    return false;
  std::size_t yz = 0U;
  std::size_t base = 0U;
  if (!checked_multiply(spec.ny, iz, yz) || !checked_add(yz, iy) ||
      !checked_multiply(spec.nx, yz, base) ||
      base > std::numeric_limits<std::size_t>::max() - ix)
    return false;
  linear_index = base + ix;
  return true;
}

bool PatchGridCellAddress::try_make(const PatchMappingGridSpec &spec,
                                    std::size_t ix, std::size_t iy,
                                    std::size_t iz,
                                    PatchGridCellAddress &result) noexcept {
  result = {};
  if (!validate_patch_mapping_grid_spec(spec, nullptr, nullptr) ||
      spec.nx < 2U || spec.ny < 2U || spec.nz < 2U || ix >= spec.nx - 1U ||
      iy >= spec.ny - 1U || iz >= spec.nz - 1U)
    return false;
  const std::size_t cell_nx = spec.nx - 1U;
  const std::size_t cell_ny = spec.ny - 1U;
  std::size_t yz = 0U;
  std::size_t base = 0U;
  if (!checked_multiply(cell_ny, iz, yz) || !checked_add(yz, iy) ||
      !checked_multiply(cell_nx, yz, base) ||
      base > std::numeric_limits<std::size_t>::max() - ix)
    return false;
  const std::size_t linear = base + ix;
  if constexpr (sizeof(std::size_t) >= sizeof(std::uint64_t)) {
    if (linear >= PatchGridCellAddress::invalid_value)
      return false;
  }
  result.linear_ = static_cast<std::uint64_t>(linear);
  return true;
}

std::size_t enumerate_patch_grid_tetra_sign_changing_edges(
    std::uint8_t tetra_inside_mask,
    std::array<PatchGridLocalTetraEdge, 6> &edges) noexcept {
  edges = {};
  if ((tetra_inside_mask & 0xf0U) != 0U)
    return std::numeric_limits<std::size_t>::max();
  static constexpr std::array<PatchGridLocalTetraEdge, 6> pairs = {
      PatchGridLocalTetraEdge{0U, 1U}, PatchGridLocalTetraEdge{0U, 2U},
      PatchGridLocalTetraEdge{0U, 3U}, PatchGridLocalTetraEdge{1U, 2U},
      PatchGridLocalTetraEdge{1U, 3U}, PatchGridLocalTetraEdge{2U, 3U}};
  std::size_t count = 0U;
  for (const PatchGridLocalTetraEdge pair : pairs) {
    const bool first_inside =
        (tetra_inside_mask & (std::uint8_t{1} << pair.first)) != 0U;
    const bool second_inside =
        (tetra_inside_mask & (std::uint8_t{1} << pair.second)) != 0U;
    if (first_inside != second_inside)
      edges[count++] = pair;
  }
  return count;
}

bool PatchGridCanonicalInterpolationVertexKey::try_make(
    std::size_t first_vertex_linear_index,
    std::size_t second_vertex_linear_index,
    PatchGridCanonicalInterpolationVertexKey &result) noexcept {
  result = {};
  constexpr std::size_t invalid = std::numeric_limits<std::size_t>::max();
  if (first_vertex_linear_index == second_vertex_linear_index ||
      first_vertex_linear_index == invalid ||
      second_vertex_linear_index == invalid)
    return false;
  if (second_vertex_linear_index < first_vertex_linear_index)
    std::swap(first_vertex_linear_index, second_vertex_linear_index);
  result.first_ = first_vertex_linear_index;
  result.second_ = second_vertex_linear_index;
  return true;
}

std::size_t enumerate_patch_grid_pre_orientation_triangles(
    std::uint8_t tetra_inside_mask,
    std::array<PatchGridPreOrientationTriangleEdges, 2> &triangles) noexcept {
  triangles = {};
  if ((tetra_inside_mask & 0xf0U) != 0U)
    return std::numeric_limits<std::size_t>::max();

  std::array<std::uint8_t, 4> inside{};
  std::array<std::uint8_t, 4> outside{};
  std::size_t inside_count = 0U;
  std::size_t outside_count = 0U;
  for (std::uint8_t local = 0U; local < 4U; ++local) {
    if ((tetra_inside_mask & (std::uint8_t{1} << local)) != 0U)
      inside[inside_count++] = local;
    else
      outside[outside_count++] = local;
  }
  const auto edge = [](std::uint8_t first, std::uint8_t second) {
    if (second < first)
      std::swap(first, second);
    return PatchGridLocalTetraEdge{first, second};
  };

  if (inside_count == 1U) {
    triangles[0].edges = {edge(inside[0], outside[0]),
                          edge(inside[0], outside[1]),
                          edge(inside[0], outside[2])};
    return 1U;
  }
  if (inside_count == 3U) {
    triangles[0].edges = {edge(outside[0], inside[0]),
                          edge(outside[0], inside[2]),
                          edge(outside[0], inside[1])};
    return 1U;
  }
  if (inside_count == 2U) {
    triangles[0].edges = {edge(inside[0], outside[0]),
                          edge(inside[0], outside[1]),
                          edge(inside[1], outside[1])};
    triangles[1].edges = {edge(inside[0], outside[0]),
                          edge(inside[1], outside[1]),
                          edge(inside[1], outside[0])};
    return 2U;
  }
  return 0U;
}

namespace {

struct ProvisionalMagnitudeCandidate {
  PatchGridMagnitudeSourceKind kind{
      PatchGridMagnitudeSourceKind::grid_spacing_fallback};
  RayAxis axis{RayAxis::x};
  PatchCrossingAddress address;
  std::size_t crossing{0U};
  double distance{0.0};
  bool raw_exact_owner{false};
  bool raw_tolerance_tied{false};
  bool final_scalar_exact_owner{false};
};

[[nodiscard]] bool topology_metadata_valid(
    const PatchGridVertexMagnitudeRequest &request) noexcept {
  const auto state_value = static_cast<std::uint8_t>(request.topology_state);
  const auto source_value = static_cast<std::uint8_t>(request.topology_source);
  if (state_value > static_cast<std::uint8_t>(PatchGridTopologyState::inside) ||
      source_value >
          static_cast<std::uint8_t>(PatchGridTopologySource::cavity_fill) ||
      request.cavity_fill !=
          (request.topology_source == PatchGridTopologySource::cavity_fill))
    return false;
  if (request.topology_source == PatchGridTopologySource::unknown)
    return request.topology_state == PatchGridTopologyState::unresolved;
  if (request.topology_state == PatchGridTopologyState::unresolved)
    return false;
  if (request.topology_source == PatchGridTopologySource::inside_atom_union ||
      request.topology_source == PatchGridTopologySource::cavity_fill)
    return request.topology_state == PatchGridTopologyState::inside;
  if (request.topology_source ==
      PatchGridTopologySource::outside_probe_expanded_union)
    return request.topology_state == PatchGridTopologyState::outside;
  return true;
}

[[nodiscard]] bool
candidate_less(const ProvisionalMagnitudeCandidate &left,
               const ProvisionalMagnitudeCandidate &right) noexcept {
  if (left.kind != right.kind)
    return left.kind < right.kind;
  if (left.axis != right.axis)
    return left.axis < right.axis;
  return left.address < right.address;
}

[[nodiscard]] bool
candidate_same(const ProvisionalMagnitudeCandidate &left,
               const ProvisionalMagnitudeCandidate &right) noexcept {
  return left.kind == right.kind && left.axis == right.axis &&
         left.address == right.address;
}

[[nodiscard]] bool
vertex_axis_row(const PatchMappingGridSpec &spec,
                const PatchGridVertexMagnitudeRequest &vertex, RayAxis axis,
                std::size_t &row, double &coordinate) noexcept {
  row = 0U;
  coordinate = 0.0;
  std::size_t product = 0U;
  switch (axis) {
  case RayAxis::x:
    if (!checked_multiply(spec.ny, vertex.iz, product) ||
        !checked_add(product, vertex.iy) ||
        !add_scaled_coordinate(spec.lower.x, spec.grid_spacing, vertex.ix,
                               coordinate))
      return false;
    row = product;
    return true;
  case RayAxis::y:
    if (!checked_multiply(spec.nx, vertex.iz, product) ||
        !checked_add(product, vertex.ix) ||
        !add_scaled_coordinate(spec.lower.y, spec.grid_spacing, vertex.iy,
                               coordinate))
      return false;
    row = product;
    return true;
  case RayAxis::z:
    if (!checked_multiply(spec.nx, vertex.iy, product) ||
        !checked_add(product, vertex.ix) ||
        !add_scaled_coordinate(spec.lower.z, spec.grid_spacing, vertex.iz,
                               coordinate))
      return false;
    row = product;
    return true;
  }
  return false;
}

void update_production_nearest(const AxisRayCrossings &axis, std::size_t row,
                               double coordinate, double &distance) noexcept {
  if (row + 1U >= axis.offsets.size())
    return;
  const auto begin =
      axis.coordinates.begin() + static_cast<std::ptrdiff_t>(axis.offsets[row]);
  const auto end = axis.coordinates.begin() +
                   static_cast<std::ptrdiff_t>(axis.offsets[row + 1U]);
  const auto next = std::lower_bound(begin, end, coordinate);
  if (next != end)
    distance = std::min(distance, std::abs(*next - coordinate));
  if (next != begin)
    distance = std::min(distance, std::abs(*std::prev(next) - coordinate));
}

[[nodiscard]] double
vertex_double_tie_tolerance(const PatchMappingGridSpec &spec,
                            const PatchGridVertexMagnitudeRequest &request,
                            double raw) noexcept {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  if (!add_scaled_coordinate(spec.lower.x, spec.grid_spacing, request.ix, x) ||
      !add_scaled_coordinate(spec.lower.y, spec.grid_spacing, request.iy, y) ||
      !add_scaled_coordinate(spec.lower.z, spec.grid_spacing, request.iz, z))
    return std::numeric_limits<double>::quiet_NaN();
  const double scale = std::max({1.0, std::abs(x), std::abs(y), std::abs(z),
                                 std::abs(spec.grid_spacing), std::abs(raw)});
  return 2048.0 * std::numeric_limits<double>::epsilon() * scale;
}

[[nodiscard]] double
sparse_float_tie_tolerance(const PatchGridVertexMagnitudeRequest &request,
                           double raw, double spacing) noexcept {
  if (!request.sparse_magnitude_present)
    return 0.0;
  const double sparse = static_cast<double>(request.sparse_magnitude);
  const double scale =
      std::max({std::abs(sparse), std::abs(raw), 1.0e-9 * std::abs(spacing),
                static_cast<double>(std::numeric_limits<float>::denorm_min())});
  return 4.0 * static_cast<double>(std::numeric_limits<float>::epsilon()) *
         scale;
}

[[nodiscard]] bool within_source_tie(double value, double raw,
                                     double tolerance) noexcept {
  return std::isfinite(value) && value >= 0.0 && std::isfinite(raw) &&
         raw >= 0.0 && std::isfinite(tolerance) && tolerance >= 0.0 &&
         std::abs(value - raw) <= tolerance;
}

[[nodiscard]] const CertifiedPatchCrossingWitness *
find_crossing_witness(const PreparedPatchCrossingWitnesses &prepared,
                      PatchCrossingAddress address) noexcept {
  const auto found = std::lower_bound(
      prepared.witnesses.begin(), prepared.witnesses.end(), address,
      [](const CertifiedPatchCrossingWitness &witness,
         PatchCrossingAddress key) { return witness.address < key; });
  if (found == prepared.witnesses.end() || found->address != address)
    return nullptr;
  return &*found;
}

[[nodiscard]] PatchGridMagnitudeEvidenceStatus classify_crossing_evidence(
    const CertifiedPatchCrossingWitness *witness) noexcept {
  if (witness == nullptr)
    return PatchGridMagnitudeEvidenceStatus::missing_crossing_witness;
  if (!witness->identity_certified())
    return PatchGridMagnitudeEvidenceStatus::identity_uncertified;
  if (!witness->root_reproduced())
    return PatchGridMagnitudeEvidenceStatus::root_uncertified;
  if (!witness->location_certified())
    return PatchGridMagnitudeEvidenceStatus::bounded_location_uncertified;
  switch (witness->transversality_status) {
  case CrossingTransversalityStatus::repeated_root_ambiguous:
    return PatchGridMagnitudeEvidenceStatus::repeated_root_ambiguous;
  case CrossingTransversalityStatus::multi_ref_relation_uncertified:
    return PatchGridMagnitudeEvidenceStatus::multi_ref_relation_uncertified;
  case CrossingTransversalityStatus::primitive_simplicity_uncertified:
    return PatchGridMagnitudeEvidenceStatus::primitive_simplicity_uncertified;
  case CrossingTransversalityStatus::location_uncertified:
    return PatchGridMagnitudeEvidenceStatus::transversality_uncertified;
  case CrossingTransversalityStatus::primitive_filtered_transversal_unoriented:
    return PatchGridMagnitudeEvidenceStatus::certified_analytic_crossing;
  }
  return PatchGridMagnitudeEvidenceStatus::transversality_uncertified;
}

[[nodiscard]] bool checked_vector_payload(std::size_t capacity,
                                          std::size_t element_size,
                                          std::size_t &bytes) noexcept {
  std::size_t contribution = 0U;
  return checked_multiply(capacity, element_size, contribution) &&
         checked_add(bytes, contribution);
}

[[nodiscard]] bool
magnitude_payload_bytes(const PreparedPatchGridVertexMagnitudeWitnesses &result,
                        std::size_t &bytes) noexcept {
  bytes = 0U;
  return checked_vector_payload(result.vertices.capacity(),
                                sizeof(PatchGridVertexMagnitudeWitness),
                                bytes) &&
         checked_vector_payload(result.candidate_offsets.capacity(),
                                sizeof(std::size_t), bytes) &&
         checked_vector_payload(result.candidates.capacity(),
                                sizeof(PatchGridMagnitudeSourceCandidate),
                                bytes) &&
         checked_vector_payload(result.candidate_refs.capacity(),
                                sizeof(PatchRefId), bytes);
}

void clear_magnitude_payload(
    PreparedPatchGridVertexMagnitudeWitnesses &result) noexcept {
  std::vector<PatchGridVertexMagnitudeWitness>().swap(result.vertices);
  std::vector<std::size_t>().swap(result.candidate_offsets);
  std::vector<PatchGridMagnitudeSourceCandidate>().swap(result.candidates);
  std::vector<PatchRefId>().swap(result.candidate_refs);
  result.complex_fingerprint = {};
  result.lattice_identity = {};
  result.analytic_candidates = 0U;
  result.certified_analytic_candidates = 0U;
  result.uncertified_analytic_candidates = 0U;
  result.sparse_candidates = 0U;
  result.fallback_candidates = 0U;
  result.clamp_floor_candidates = 0U;
  result.raw_exact_owner_candidates = 0U;
  result.tolerance_tied_candidates = 0U;
  result.final_scalar_exact_owner_candidates = 0U;
  result.raw_source_certified_vertices = 0U;
  result.final_scalar_source_certified_vertices = 0U;
  result.floor_clamped_vertices = 0U;
  result.sparse_suppressed_yz_vertices = 0U;
  result.topology_analytic_evidence_vertices = 0U;
  result.topology_exact_vertex_certified_vertices = 0U;
  result.topology_nonanalytic_vertices = 0U;
  result.retained_bytes = 0U;
  result.peak_temporary_bytes = 0U;
  result.raw_source_certificate_complete = false;
  result.final_scalar_source_certificate_complete = false;
  result.topology_source_classification_complete = false;
  result.topology_exact_vertex_certificate_complete = false;
  result.surface_location_certified = false;
}

[[nodiscard]] bool validate_magnitude_bindings(
    const PatchComplex &complex, const LocalPatchProjectionIndex &index,
    const PatchRayGrid &grid, const PatchMappingGridSpec &spec,
    const PreparedPatchCrossingWitnesses &prepared,
    PatchComplexFingerprint &fingerprint,
    PatchRayProvenance::LatticeIdentity &lattice) noexcept {
  std::array<std::size_t, 3> row_counts{};
  if (!validate_patch_mapping_grid_spec(spec, &row_counts, nullptr) ||
      grid.status != "pass" || grid.provenance.status != "pass" ||
      !grid.provenance.enabled || !grid.provenance.certified)
    return false;
  lattice = make_patch_ray_lattice_identity(spec.lower, spec.grid_spacing,
                                            spec.nx, spec.ny, spec.nz);
  if (!lattice.valid || !grid.provenance.lattice_identity.valid ||
      grid.provenance.lattice_identity != lattice ||
      prepared.lattice_identity != lattice)
    return false;
  fingerprint = fingerprint_patch_complex(complex);
  if (!fingerprint.valid || !grid.provenance.complex_fingerprint.valid ||
      grid.provenance.complex_fingerprint != fingerprint ||
      prepared.complex_fingerprint != fingerprint ||
      index.source_complex != &complex ||
      !index.source_complex_fingerprint.valid ||
      index.source_complex_fingerprint != fingerprint ||
      !index.bounded_domains_certified || !index.bvh_invariants_certified)
    return false;
  const std::array<std::size_t, 3> expected_u = {spec.ny, spec.nx, spec.nx};
  const std::array<std::size_t, 3> expected_v = {spec.nz, spec.nz, spec.ny};
  std::size_t total_crossings = 0U;
  for (std::size_t axis = 0U; axis < 3U; ++axis) {
    if (!validate_axis_contract(grid.axes[axis], grid.provenance.axes[axis],
                                static_cast<RayAxis>(axis), expected_u[axis],
                                expected_v[axis]) ||
        grid.axes[axis].row_count() != row_counts[axis] ||
        !checked_add(total_crossings, grid.axes[axis].coordinates.size()))
      return false;
  }
  if (total_crossings != grid.provenance.unique_crossings ||
      prepared.axis_offsets[0] != 0U ||
      prepared.axis_offsets[0] > prepared.axis_offsets[1] ||
      prepared.axis_offsets[1] > prepared.axis_offsets[2] ||
      prepared.axis_offsets[2] > prepared.axis_offsets[3] ||
      prepared.axis_offsets[3] != prepared.witnesses.size())
    return false;
  PatchCrossingAddress previous;
  bool have_previous = false;
  for (std::size_t axis_index = 0U; axis_index < 3U; ++axis_index) {
    const AxisRayCrossings &axis = grid.axes[axis_index];
    for (std::size_t position = prepared.axis_offsets[axis_index];
         position < prepared.axis_offsets[axis_index + 1U]; ++position) {
      const CertifiedPatchCrossingWitness &witness =
          prepared.witnesses[position];
      const auto identity_value =
          static_cast<std::uint8_t>(witness.identity_status);
      const auto root_value = static_cast<std::uint8_t>(witness.root_status);
      const auto location_value =
          static_cast<std::uint8_t>(witness.location_status);
      const auto transversality_value =
          static_cast<std::uint8_t>(witness.transversality_status);
      if (!witness.address.valid() ||
          witness.address.axis() != static_cast<RayAxis>(axis_index) ||
          witness.address.row() >= axis.row_count() ||
          witness.address.crossing() >= axis.coordinates.size() ||
          witness.address.crossing() < axis.offsets[witness.address.row()] ||
          witness.address.crossing() >=
              axis.offsets[witness.address.row() + 1U] ||
          witness.coordinate != axis.coordinates[witness.address.crossing()] ||
          !finite_vec(witness.point) || !std::isfinite(witness.coordinate) ||
          !std::isfinite(witness.root_match_tolerance) ||
          witness.root_match_tolerance < 0.0 ||
          !std::isfinite(witness.bounded_patch_tolerance) ||
          witness.bounded_patch_tolerance < 0.0 ||
          !std::isfinite(witness.maximum_root_residual) ||
          witness.maximum_root_residual < 0.0 ||
          !std::isfinite(witness.maximum_patch_distance) ||
          witness.maximum_patch_distance < 0.0 ||
          identity_value > static_cast<std::uint8_t>(
                               CrossingIdentityCertificateStatus::invalid) ||
          root_value >
              static_cast<std::uint8_t>(
                  CrossingRootReproductionStatus::nonfinite_residual) ||
          location_value >
              static_cast<std::uint8_t>(
                  CrossingBoundedLocationStatus::nonfinite_residual) ||
          transversality_value >
              static_cast<std::uint8_t>(
                  CrossingTransversalityStatus::location_uncertified) ||
          (have_previous && !(previous < witness.address)))
        return false;
      Vec3 reconstructed;
      if (!reconstruct_patch_crossing_point(
              spec, witness.address, witness.coordinate, reconstructed) ||
          reconstructed.x != witness.point.x ||
          reconstructed.y != witness.point.y ||
          reconstructed.z != witness.point.z)
        return false;
      const AxisPatchProvenance &axis_provenance =
          grid.provenance.axes[axis_index];
      const std::span<const PatchRefId> refs =
          axis_provenance.refs(witness.address.crossing());
      if (refs.empty() || refs.size() != witness.ref_count)
        return false;
      std::size_t total_multiplicity = 0U;
      std::uint32_t maximum_multiplicity = 0U;
      for (std::size_t ordinal = 0U; ordinal < refs.size(); ++ordinal) {
        const std::uint32_t multiplicity = axis_provenance.ref_multiplicity(
            witness.address.crossing(), ordinal);
        if (multiplicity == 0U ||
            !checked_add(total_multiplicity,
                         static_cast<std::size_t>(multiplicity)))
          return false;
        maximum_multiplicity = std::max(maximum_multiplicity, multiplicity);
      }
      if (total_multiplicity != witness.total_root_multiplicity ||
          maximum_multiplicity != witness.maximum_ref_multiplicity)
        return false;
      previous = witness.address;
      have_previous = true;
    }
  }
  return true;
}

} // namespace

PreparedPatchGridVertexMagnitudeWitnesses
build_patch_grid_vertex_magnitude_witnesses(
    const PatchComplex &complex, const LocalPatchProjectionIndex &index,
    const PatchRayGrid &grid, const PatchMappingGridSpec &spec,
    const PreparedPatchCrossingWitnesses &crossing_witnesses,
    std::span<const PatchGridVertexMagnitudeRequest> requests) {
  const auto begin = std::chrono::steady_clock::now();
  PreparedPatchGridVertexMagnitudeWitnesses result;
  const auto stamp_time = [&]() noexcept {
    result.build_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - begin)
                          .count();
  };
  try {
    if (requests.empty()) {
      result.status = "no_vertices";
      stamp_time();
      return result;
    }
    if (!validate_magnitude_bindings(
            complex, index, grid, spec, crossing_witnesses,
            result.complex_fingerprint, result.lattice_identity)) {
      result.status = "stale_or_invalid_binding";
      stamp_time();
      return result;
    }

    std::vector<std::size_t> request_order(requests.size());
    std::vector<std::size_t> linear_indices(requests.size());
    for (std::size_t request = 0U; request < requests.size(); ++request) {
      const PatchGridVertexMagnitudeRequest &vertex = requests[request];
      request_order[request] = request;
      if (!try_patch_grid_vertex_linear_index(
              spec, vertex.ix, vertex.iy, vertex.iz, linear_indices[request]) ||
          !topology_metadata_valid(vertex) ||
          (vertex.sparse_magnitude_present &&
           (!std::isfinite(vertex.sparse_magnitude) ||
            vertex.sparse_magnitude < 0.0F)) ||
          (vertex.cavity_fill && vertex.sparse_magnitude_present)) {
        clear_magnitude_payload(result);
        result.status = "invalid_vertex_request";
        stamp_time();
        return result;
      }
    }
    std::sort(request_order.begin(), request_order.end(),
              [&](std::size_t left, std::size_t right) {
                return linear_indices[left] < linear_indices[right];
              });
    for (std::size_t position = 1U; position < request_order.size();
         ++position) {
      if (linear_indices[request_order[position - 1U]] ==
          linear_indices[request_order[position]]) {
        clear_magnitude_payload(result);
        result.status = "duplicate_vertex";
        stamp_time();
        return result;
      }
    }

    result.vertices.reserve(requests.size());
    result.candidate_offsets.reserve(requests.size() + 1U);
    result.candidate_offsets.push_back(0U);
    std::vector<ProvisionalMagnitudeCandidate> provisional;
    provisional.reserve(8U);
    bool all_raw_sources_certified = true;
    bool all_final_sources_certified = true;
    bool all_topology_classified = true;
    bool all_topology_exact = true;

    for (const std::size_t request_index : request_order) {
      const PatchGridVertexMagnitudeRequest &request = requests[request_index];
      PatchGridVertexMagnitudeWitness vertex;
      vertex.ix = request.ix;
      vertex.iy = request.iy;
      vertex.iz = request.iz;
      vertex.linear_index = linear_indices[request_index];
      vertex.topology_state = request.topology_state;
      vertex.topology_source = request.topology_source;
      vertex.sparse_magnitude =
          request.sparse_magnitude_present ? request.sparse_magnitude : 0.0F;
      vertex.sparse_magnitude_present = request.sparse_magnitude_present;
      vertex.sparse_suppressed_yz = request.sparse_magnitude_present;
      vertex.cavity_fill = request.cavity_fill;

      std::array<std::size_t, 3> rows{};
      std::array<double, 3> coordinates{};
      for (std::size_t axis = 0U; axis < 3U; ++axis) {
        if (!vertex_axis_row(spec, request, static_cast<RayAxis>(axis),
                             rows[axis], coordinates[axis])) {
          clear_magnitude_payload(result);
          result.status = "vertex_row_overflow";
          stamp_time();
          return result;
        }
      }

      // This block intentionally mirrors patch_ray_field_value: the stored
      // float is converted directly to double; X is unconditional; Y/Z are
      // suppressed only when the sparse value exists; std::min order is kept.
      double raw = request.sparse_magnitude_present
                       ? static_cast<double>(request.sparse_magnitude)
                       : spec.grid_spacing;
      update_production_nearest(grid.axes[0], rows[0], coordinates[0], raw);
      if (!request.sparse_magnitude_present) {
        update_production_nearest(grid.axes[1], rows[1], coordinates[1], raw);
        update_production_nearest(grid.axes[2], rows[2], coordinates[2], raw);
      }
      const double floor = 1.0e-9 * spec.grid_spacing;
      const double clamped = std::max(raw, floor);
      if (!std::isfinite(raw) || raw < 0.0 || !std::isfinite(clamped)) {
        clear_magnitude_payload(result);
        result.status = "invalid_scalar";
        stamp_time();
        return result;
      }
      vertex.raw_magnitude = raw;
      vertex.clamped_magnitude = clamped;
      vertex.floor_clamp_applied = raw < floor;
      vertex.double_arithmetic_tie_tolerance =
          vertex_double_tie_tolerance(spec, request, raw);
      vertex.sparse_float_tie_tolerance =
          sparse_float_tie_tolerance(request, raw, spec.grid_spacing);
      vertex.source_tie_tolerance = vertex.double_arithmetic_tie_tolerance +
                                    vertex.sparse_float_tie_tolerance;
      if (!std::isfinite(vertex.source_tie_tolerance)) {
        clear_magnitude_payload(result);
        result.status = "invalid_tie_tolerance";
        stamp_time();
        return result;
      }

      provisional.clear();
      const auto retain_nonanalytic = [&](PatchGridMagnitudeSourceKind kind,
                                          double distance) {
        if (!within_source_tie(distance, raw, vertex.source_tie_tolerance))
          return;
        ProvisionalMagnitudeCandidate candidate;
        candidate.kind = kind;
        candidate.distance = distance;
        candidate.raw_exact_owner = distance == raw;
        candidate.raw_tolerance_tied = !candidate.raw_exact_owner;
        candidate.final_scalar_exact_owner =
            candidate.raw_exact_owner && !vertex.floor_clamp_applied;
        provisional.push_back(candidate);
      };
      if (request.sparse_magnitude_present) {
        retain_nonanalytic(PatchGridMagnitudeSourceKind::sparse_sample,
                           static_cast<double>(request.sparse_magnitude));
      } else {
        retain_nonanalytic(PatchGridMagnitudeSourceKind::grid_spacing_fallback,
                           spec.grid_spacing);
      }

      const std::size_t allowed_axes =
          request.sparse_magnitude_present ? 1U : 3U;
      for (std::size_t axis_index = 0U; axis_index < allowed_axes;
           ++axis_index) {
        const AxisRayCrossings &axis = grid.axes[axis_index];
        const std::size_t row = rows[axis_index];
        const std::size_t begin_crossing = axis.offsets[row];
        const std::size_t end_crossing = axis.offsets[row + 1U];
        const auto begin_iterator = axis.coordinates.begin() +
                                    static_cast<std::ptrdiff_t>(begin_crossing);
        const auto end_iterator = axis.coordinates.begin() +
                                  static_cast<std::ptrdiff_t>(end_crossing);
        const auto next = std::lower_bound(begin_iterator, end_iterator,
                                           coordinates[axis_index]);
        const auto retain_crossing = [&](auto iterator) {
          const std::size_t crossing =
              static_cast<std::size_t>(iterator - axis.coordinates.begin());
          const double distance = std::abs(*iterator - coordinates[axis_index]);
          if (!within_source_tie(distance, raw, vertex.source_tie_tolerance))
            return;
          ProvisionalMagnitudeCandidate candidate;
          candidate.kind = PatchGridMagnitudeSourceKind::analytic_crossing;
          candidate.axis = static_cast<RayAxis>(axis_index);
          candidate.crossing = crossing;
          candidate.distance = distance;
          candidate.raw_exact_owner = distance == raw;
          candidate.raw_tolerance_tied = !candidate.raw_exact_owner;
          candidate.final_scalar_exact_owner =
              candidate.raw_exact_owner && !vertex.floor_clamp_applied;
          if (!PatchCrossingAddress::try_make(candidate.axis, row, crossing,
                                              candidate.address))
            throw std::length_error("crossing_address");
          provisional.push_back(candidate);
        };
        if (next != end_iterator)
          retain_crossing(next);
        if (next != begin_iterator)
          retain_crossing(std::prev(next));
      }
      if (vertex.floor_clamp_applied) {
        ProvisionalMagnitudeCandidate candidate;
        candidate.kind = PatchGridMagnitudeSourceKind::clamp_floor;
        candidate.distance = clamped;
        candidate.final_scalar_exact_owner = true;
        provisional.push_back(candidate);
      }
      std::sort(provisional.begin(), provisional.end(), candidate_less);
      provisional.erase(
          std::unique(provisional.begin(), provisional.end(), candidate_same),
          provisional.end());
      if (provisional.empty()) {
        clear_magnitude_payload(result);
        result.status = "no_magnitude_source";
        stamp_time();
        return result;
      }

      bool vertex_has_analytic = false;
      bool vertex_raw_certified = true;
      for (const ProvisionalMagnitudeCandidate &source : provisional) {
        PatchGridMagnitudeSourceCandidate candidate;
        candidate.kind = source.kind;
        candidate.axis = source.axis;
        candidate.crossing = source.address;
        candidate.distance = source.distance;
        candidate.raw_exact_owner = source.raw_exact_owner;
        candidate.raw_tolerance_tied = source.raw_tolerance_tied;
        candidate.final_scalar_exact_owner = source.final_scalar_exact_owner;
        if (source.kind == PatchGridMagnitudeSourceKind::analytic_crossing) {
          vertex_has_analytic = true;
          if (!checked_add(result.analytic_candidates, 1U))
            throw std::length_error("counter");
          const AxisPatchProvenance &axis_provenance =
              grid.provenance.axes[static_cast<std::size_t>(source.axis)];
          const std::span<const PatchRefId> refs =
              axis_provenance.refs(source.crossing);
          if (refs.empty() ||
              refs.size() > std::numeric_limits<std::uint32_t>::max() ||
              result.candidate_refs.size() >
                  std::numeric_limits<std::uint32_t>::max() ||
              result.candidate_refs.size() >
                  std::numeric_limits<std::uint32_t>::max() - refs.size())
            throw std::length_error("candidate_refs");
          candidate.refs_begin =
              static_cast<std::uint32_t>(result.candidate_refs.size());
          for (const PatchRefId ref : refs) {
            if (!patch_ref_in_range(complex, ref))
              throw std::length_error("candidate_ref");
            result.candidate_refs.push_back(ref);
          }
          candidate.refs_end =
              static_cast<std::uint32_t>(result.candidate_refs.size());
          candidate.evidence_status = classify_crossing_evidence(
              find_crossing_witness(crossing_witnesses, source.address));
          if (candidate.source_evidence_certified()) {
            if (!checked_add(result.certified_analytic_candidates, 1U))
              throw std::length_error("counter");
          } else {
            vertex_raw_certified = false;
            if (!checked_add(result.uncertified_analytic_candidates, 1U))
              throw std::length_error("counter");
          }
        } else if (source.kind == PatchGridMagnitudeSourceKind::sparse_sample) {
          candidate.evidence_status =
              PatchGridMagnitudeEvidenceStatus::sparse_sample_nonanalytic;
          vertex_raw_certified = false;
          if (!checked_add(result.sparse_candidates, 1U))
            throw std::length_error("counter");
        } else if (source.kind ==
                   PatchGridMagnitudeSourceKind::grid_spacing_fallback) {
          candidate.evidence_status = PatchGridMagnitudeEvidenceStatus::
              grid_spacing_fallback_nonanalytic;
          vertex_raw_certified = false;
          if (!checked_add(result.fallback_candidates, 1U))
            throw std::length_error("counter");
        } else {
          candidate.evidence_status =
              PatchGridMagnitudeEvidenceStatus::clamp_floor_nonanalytic;
          if (!checked_add(result.clamp_floor_candidates, 1U))
            throw std::length_error("counter");
        }
        if (candidate.raw_exact_owner &&
            !checked_add(result.raw_exact_owner_candidates, 1U))
          throw std::length_error("counter");
        if (candidate.raw_tolerance_tied &&
            !checked_add(result.tolerance_tied_candidates, 1U))
          throw std::length_error("counter");
        if (candidate.final_scalar_exact_owner &&
            !checked_add(result.final_scalar_exact_owner_candidates, 1U))
          throw std::length_error("counter");
        result.candidates.push_back(candidate);
      }
      vertex.raw_source_certificate_complete =
          vertex_has_analytic && vertex_raw_certified;
      vertex.final_scalar_source_certificate_complete =
          vertex.raw_source_certificate_complete && !vertex.floor_clamp_applied;
      if (vertex.raw_source_certificate_complete) {
        if (!checked_add(result.raw_source_certified_vertices, 1U))
          throw std::length_error("counter");
      } else {
        all_raw_sources_certified = false;
      }
      if (vertex.final_scalar_source_certificate_complete) {
        if (!checked_add(result.final_scalar_source_certified_vertices, 1U))
          throw std::length_error("counter");
      } else {
        all_final_sources_certified = false;
      }
      if (vertex.floor_clamp_applied &&
          !checked_add(result.floor_clamped_vertices, 1U))
        throw std::length_error("counter");
      if (vertex.sparse_suppressed_yz &&
          !checked_add(result.sparse_suppressed_yz_vertices, 1U))
        throw std::length_error("counter");

      const PatchGridTopologyEvidenceClassification topology =
          classify_patch_grid_topology_evidence(request.topology_state,
                                                request.topology_source,
                                                request.cavity_fill);
      vertex.topology_evidence_status = topology.status;
      vertex.topology_analytic_evidence = topology.analytic_evidence;
      vertex.topology_exact_vertex_certified = topology.exact_vertex_certified;
      if (topology.status == PatchGridTopologyEvidenceStatus::unknown) {
        all_topology_classified = false;
        all_topology_exact = false;
      } else if (topology.analytic_evidence) {
        if (!checked_add(result.topology_analytic_evidence_vertices, 1U))
          throw std::length_error("counter");
        if (topology.exact_vertex_certified) {
          if (!checked_add(result.topology_exact_vertex_certified_vertices, 1U))
            throw std::length_error("counter");
        } else {
          all_topology_exact = false;
        }
      } else {
        all_topology_exact = false;
        if (!checked_add(result.topology_nonanalytic_vertices, 1U))
          throw std::length_error("counter");
      }
      result.vertices.push_back(vertex);
      result.candidate_offsets.push_back(result.candidates.size());

      std::size_t retained = 0U;
      std::size_t scratch = 0U;
      if (!magnitude_payload_bytes(result, retained) ||
          !checked_vector_payload(request_order.capacity(), sizeof(std::size_t),
                                  scratch) ||
          !checked_vector_payload(linear_indices.capacity(),
                                  sizeof(std::size_t), scratch) ||
          !checked_vector_payload(provisional.capacity(),
                                  sizeof(ProvisionalMagnitudeCandidate),
                                  scratch) ||
          !checked_add(retained, scratch))
        throw std::length_error("byte_count");
      result.peak_temporary_bytes =
          std::max(result.peak_temporary_bytes, retained);
    }

    if (!magnitude_payload_bytes(result, result.retained_bytes))
      throw std::length_error("byte_count");
    result.raw_source_certificate_complete =
        all_raw_sources_certified &&
        result.raw_source_certified_vertices == result.vertices.size() &&
        result.certified_analytic_candidates != 0U;
    result.final_scalar_source_certificate_complete =
        all_final_sources_certified &&
        result.final_scalar_source_certified_vertices ==
            result.vertices.size() &&
        result.raw_source_certificate_complete;
    result.topology_source_classification_complete = all_topology_classified;
    result.topology_exact_vertex_certificate_complete =
        all_topology_exact && result.topology_exact_vertex_certified_vertices ==
                                  result.vertices.size();
    result.surface_location_certified = false;
    if (result.certified_analytic_candidates == 0U) {
      result.status = "no_usable_analytic_evidence";
    } else if (!result.raw_source_certificate_complete) {
      result.status = "raw_source_uncertified";
    } else if (!result.final_scalar_source_certificate_complete) {
      result.status = "final_scalar_source_uncertified";
    } else if (!result.topology_source_classification_complete) {
      result.status = "topology_source_unclassified";
    } else if (!result.topology_exact_vertex_certificate_complete) {
      result.status = "topology_exact_vertex_uncertified";
    } else {
      result.status = "pass";
    }
    stamp_time();
    return result;
  } catch (const std::bad_alloc &) {
    clear_magnitude_payload(result);
    result.status = "alloc_fail";
    stamp_time();
    return result;
  } catch (const std::length_error &) {
    clear_magnitude_payload(result);
    result.status = "length_err";
    stamp_time();
    return result;
  }
}

PreparedPatchGridVertexMagnitudeWitnesses
select_patch_grid_vertex_magnitude_sources(
    const PatchComplex &complex, const LocalPatchProjectionIndex &index,
    const PatchRayGrid &grid, const PatchMappingGridSpec &spec,
    const PreparedPatchCrossingWitnesses &crossing_witnesses,
    const PatchGridVertexMagnitudeRequest &request) {
  return build_patch_grid_vertex_magnitude_witnesses(
      complex, index, grid, spec, crossing_witnesses,
      std::span<const PatchGridVertexMagnitudeRequest>(&request, 1U));
}

namespace {

template <class Value>
[[nodiscard]] bool bit_equal(Value left, Value right) noexcept {
  static_assert(std::is_floating_point_v<Value>);
  if constexpr (sizeof(Value) == sizeof(std::uint32_t)) {
    return std::bit_cast<std::uint32_t>(left) ==
           std::bit_cast<std::uint32_t>(right);
  } else {
    return std::bit_cast<std::uint64_t>(left) ==
           std::bit_cast<std::uint64_t>(right);
  }
}

[[nodiscard]] bool
same_vertex_semantics(const PatchGridVertexMagnitudeWitness &left,
                      const PatchGridVertexMagnitudeWitness &right) noexcept {
  return left.linear_index == right.linear_index && left.ix == right.ix &&
         left.iy == right.iy && left.iz == right.iz &&
         bit_equal(left.raw_magnitude, right.raw_magnitude) &&
         bit_equal(left.clamped_magnitude, right.clamped_magnitude) &&
         bit_equal(left.double_arithmetic_tie_tolerance,
                   right.double_arithmetic_tie_tolerance) &&
         bit_equal(left.sparse_float_tie_tolerance,
                   right.sparse_float_tie_tolerance) &&
         bit_equal(left.source_tie_tolerance, right.source_tie_tolerance) &&
         left.topology_state == right.topology_state &&
         left.topology_source == right.topology_source &&
         bit_equal(left.sparse_magnitude, right.sparse_magnitude) &&
         left.sparse_magnitude_present == right.sparse_magnitude_present &&
         left.sparse_suppressed_yz == right.sparse_suppressed_yz &&
         left.cavity_fill == right.cavity_fill &&
         left.floor_clamp_applied == right.floor_clamp_applied &&
         left.topology_evidence_status == right.topology_evidence_status &&
         left.topology_analytic_evidence == right.topology_analytic_evidence &&
         left.topology_exact_vertex_certified ==
             right.topology_exact_vertex_certified &&
         left.raw_source_certificate_complete ==
             right.raw_source_certificate_complete &&
         left.final_scalar_source_certificate_complete ==
             right.final_scalar_source_certificate_complete &&
         left.surface_location_certified == right.surface_location_certified;
}

[[nodiscard]] bool same_candidate_semantics(
    const PatchGridMagnitudeSourceCandidate &left,
    const PatchGridMagnitudeSourceCandidate &right) noexcept {
  return left.kind == right.kind && left.axis == right.axis &&
         left.crossing == right.crossing &&
         bit_equal(left.distance, right.distance) &&
         left.refs_begin == right.refs_begin &&
         left.refs_end == right.refs_end &&
         left.evidence_status == right.evidence_status &&
         left.raw_exact_owner == right.raw_exact_owner &&
         left.raw_tolerance_tied == right.raw_tolerance_tied &&
         left.final_scalar_exact_owner == right.final_scalar_exact_owner;
}

[[nodiscard]] bool same_magnitude_semantics(
    const PreparedPatchGridVertexMagnitudeWitnesses &left,
    const PreparedPatchGridVertexMagnitudeWitnesses &right) noexcept {
  if (left.complex_fingerprint != right.complex_fingerprint ||
      left.lattice_identity != right.lattice_identity ||
      left.vertices.size() != right.vertices.size() ||
      left.candidates.size() != right.candidates.size() ||
      left.candidate_offsets != right.candidate_offsets ||
      left.candidate_refs != right.candidate_refs ||
      left.analytic_candidates != right.analytic_candidates ||
      left.certified_analytic_candidates !=
          right.certified_analytic_candidates ||
      left.uncertified_analytic_candidates !=
          right.uncertified_analytic_candidates ||
      left.sparse_candidates != right.sparse_candidates ||
      left.fallback_candidates != right.fallback_candidates ||
      left.clamp_floor_candidates != right.clamp_floor_candidates ||
      left.raw_exact_owner_candidates != right.raw_exact_owner_candidates ||
      left.tolerance_tied_candidates != right.tolerance_tied_candidates ||
      left.final_scalar_exact_owner_candidates !=
          right.final_scalar_exact_owner_candidates ||
      left.raw_source_certified_vertices !=
          right.raw_source_certified_vertices ||
      left.final_scalar_source_certified_vertices !=
          right.final_scalar_source_certified_vertices ||
      left.floor_clamped_vertices != right.floor_clamped_vertices ||
      left.sparse_suppressed_yz_vertices !=
          right.sparse_suppressed_yz_vertices ||
      left.topology_analytic_evidence_vertices !=
          right.topology_analytic_evidence_vertices ||
      left.topology_exact_vertex_certified_vertices !=
          right.topology_exact_vertex_certified_vertices ||
      left.topology_nonanalytic_vertices !=
          right.topology_nonanalytic_vertices ||
      left.raw_source_certificate_complete !=
          right.raw_source_certificate_complete ||
      left.final_scalar_source_certificate_complete !=
          right.final_scalar_source_certificate_complete ||
      left.topology_source_classification_complete !=
          right.topology_source_classification_complete ||
      left.topology_exact_vertex_certificate_complete !=
          right.topology_exact_vertex_certificate_complete ||
      left.surface_location_certified != right.surface_location_certified ||
      left.status != right.status)
    return false;
  for (std::size_t index = 0U; index < left.vertices.size(); ++index) {
    if (!same_vertex_semantics(left.vertices[index], right.vertices[index]))
      return false;
  }
  for (std::size_t index = 0U; index < left.candidates.size(); ++index) {
    if (!same_candidate_semantics(left.candidates[index],
                                  right.candidates[index]))
      return false;
  }
  return true;
}

void clear_cell_lineage_payload(PreparedPatchGridCellLineage &result) noexcept {
  std::vector<PatchGridCanonicalCellLineage>().swap(result.cells);
  std::vector<std::size_t>().swap(result.cell_tetra_offsets);
  std::vector<PatchGridCanonicalTetraLineage>().swap(result.tetrahedra);
  std::vector<std::size_t>().swap(result.tetra_edge_offsets);
  std::vector<PatchGridSignChangingEdgeLineage>().swap(result.edges);
  std::vector<std::size_t>().swap(result.edge_endpoint_patch_offsets);
  std::vector<PatchRefId>().swap(result.endpoint_patch_refs);
  std::vector<std::size_t>().swap(result.edge_relation_offsets);
  std::vector<PatchGridEdgePatchRelationEvidence>().swap(result.relations);
  std::vector<std::size_t>().swap(result.edge_location_ref_offsets);
  std::vector<PatchRefId>().swap(result.location_patch_refs);
  result.complex_fingerprint = {};
  result.lattice_identity = {};
  result.relation_counts = {};
  result.axis_edges = 0U;
  result.diagonal_edges = 0U;
  result.scalar_lineage_certified_edges = 0U;
  result.surface_location_certified_edges = 0U;
  result.topology_sign_certified_edges = 0U;
  result.oracle_accounted_setup_bytes = 0U;
  result.oracle_build_ms = 0.0;
  result.relation_oracle_binding =
      PatchGridRelationOracleBindingStatus::unavailable;
  result.retained_bytes = 0U;
  result.accounted_peak_temporary_bytes = 0U;
  result.scalar_lineage_complete = false;
  result.patch_candidate_collection_complete = false;
  result.patch_relation_classification_complete = false;
  result.surface_location_certificate_complete = false;
  result.topology_sign_certificate_complete = false;
}

struct CanonicalCellInput {
  PatchGridCellAddress address;
  PatchGridActiveCellRequest request;
};

[[nodiscard]] bool
canonical_cell_less(const CanonicalCellInput &left,
                    const CanonicalCellInput &right) noexcept {
  return left.address < right.address;
}

[[nodiscard]] const PatchGridVertexMagnitudeWitness *
find_vertex_witness(const PreparedPatchGridVertexMagnitudeWitnesses &prepared,
                    std::size_t linear_index) noexcept {
  const auto found = std::lower_bound(
      prepared.vertices.begin(), prepared.vertices.end(), linear_index,
      [](const PatchGridVertexMagnitudeWitness &vertex, std::size_t key) {
        return vertex.linear_index < key;
      });
  if (found == prepared.vertices.end() || found->linear_index != linear_index)
    return nullptr;
  return &*found;
}

[[nodiscard]] std::size_t vertex_witness_position(
    const PreparedPatchGridVertexMagnitudeWitnesses &prepared,
    const PatchGridVertexMagnitudeWitness *vertex) noexcept {
  return static_cast<std::size_t>(vertex - prepared.vertices.data());
}

[[nodiscard]] std::span<const PatchGridMagnitudeSourceCandidate>
vertex_magnitude_candidates(
    const PreparedPatchGridVertexMagnitudeWitnesses &prepared,
    std::size_t vertex_position) noexcept {
  if (vertex_position + 1U >= prepared.candidate_offsets.size())
    return {};
  const std::size_t begin = prepared.candidate_offsets[vertex_position];
  const std::size_t end = prepared.candidate_offsets[vertex_position + 1U];
  if (begin > end || end > prepared.candidates.size())
    return {};
  return {prepared.candidates.data() + begin, end - begin};
}

[[nodiscard]] PatchGridEdgeScalarLineageStatus candidate_scalar_status(
    const PatchGridMagnitudeSourceCandidate &candidate) noexcept {
  if (candidate.raw_tolerance_tied)
    return PatchGridEdgeScalarLineageStatus::tolerance_tied_source;
  switch (candidate.kind) {
  case PatchGridMagnitudeSourceKind::grid_spacing_fallback:
    return PatchGridEdgeScalarLineageStatus::grid_spacing_fallback;
  case PatchGridMagnitudeSourceKind::sparse_sample:
    return PatchGridEdgeScalarLineageStatus::sparse_sample;
  case PatchGridMagnitudeSourceKind::clamp_floor:
    return PatchGridEdgeScalarLineageStatus::clamp_floor;
  case PatchGridMagnitudeSourceKind::analytic_crossing:
    break;
  }
  switch (candidate.evidence_status) {
  case PatchGridMagnitudeEvidenceStatus::certified_analytic_crossing:
    return PatchGridEdgeScalarLineageStatus::certified;
  case PatchGridMagnitudeEvidenceStatus::grid_spacing_fallback_nonanalytic:
    return PatchGridEdgeScalarLineageStatus::grid_spacing_fallback;
  case PatchGridMagnitudeEvidenceStatus::sparse_sample_nonanalytic:
    return PatchGridEdgeScalarLineageStatus::sparse_sample;
  case PatchGridMagnitudeEvidenceStatus::clamp_floor_nonanalytic:
    return PatchGridEdgeScalarLineageStatus::clamp_floor;
  case PatchGridMagnitudeEvidenceStatus::missing_crossing_witness:
    return PatchGridEdgeScalarLineageStatus::missing_crossing_witness;
  case PatchGridMagnitudeEvidenceStatus::identity_uncertified:
    return PatchGridEdgeScalarLineageStatus::identity_uncertified;
  case PatchGridMagnitudeEvidenceStatus::root_uncertified:
    return PatchGridEdgeScalarLineageStatus::root_uncertified;
  case PatchGridMagnitudeEvidenceStatus::bounded_location_uncertified:
    return PatchGridEdgeScalarLineageStatus::bounded_location_uncertified;
  case PatchGridMagnitudeEvidenceStatus::repeated_root_ambiguous:
    return PatchGridEdgeScalarLineageStatus::repeated_root_ambiguous;
  case PatchGridMagnitudeEvidenceStatus::multi_ref_relation_uncertified:
    return PatchGridEdgeScalarLineageStatus::multi_ref_relation_uncertified;
  case PatchGridMagnitudeEvidenceStatus::primitive_simplicity_uncertified:
    return PatchGridEdgeScalarLineageStatus::primitive_simplicity_uncertified;
  case PatchGridMagnitudeEvidenceStatus::transversality_uncertified:
    return PatchGridEdgeScalarLineageStatus::transversality_uncertified;
  }
  return PatchGridEdgeScalarLineageStatus::transversality_uncertified;
}

[[nodiscard]] unsigned
scalar_lineage_rank(PatchGridEdgeScalarLineageStatus status) noexcept {
  switch (status) {
  case PatchGridEdgeScalarLineageStatus::certified:
    return 0U;
  case PatchGridEdgeScalarLineageStatus::no_analytic_source:
    return 1U;
  case PatchGridEdgeScalarLineageStatus::sparse_sample:
    return 2U;
  case PatchGridEdgeScalarLineageStatus::grid_spacing_fallback:
    return 3U;
  case PatchGridEdgeScalarLineageStatus::clamp_floor:
    return 4U;
  case PatchGridEdgeScalarLineageStatus::tolerance_tied_source:
    return 5U;
  case PatchGridEdgeScalarLineageStatus::missing_crossing_witness:
    return 6U;
  case PatchGridEdgeScalarLineageStatus::identity_uncertified:
    return 7U;
  case PatchGridEdgeScalarLineageStatus::root_uncertified:
    return 8U;
  case PatchGridEdgeScalarLineageStatus::bounded_location_uncertified:
    return 9U;
  case PatchGridEdgeScalarLineageStatus::repeated_root_ambiguous:
    return 10U;
  case PatchGridEdgeScalarLineageStatus::multi_ref_relation_uncertified:
    return 11U;
  case PatchGridEdgeScalarLineageStatus::primitive_simplicity_uncertified:
    return 12U;
  case PatchGridEdgeScalarLineageStatus::transversality_uncertified:
    return 13U;
  }
  return 13U;
}

[[nodiscard]] PatchGridEdgeScalarLineageStatus
worse_scalar_lineage(PatchGridEdgeScalarLineageStatus first,
                     PatchGridEdgeScalarLineageStatus second) noexcept {
  return scalar_lineage_rank(first) >= scalar_lineage_rank(second) ? first
                                                                   : second;
}

// scalar_lineage_rank(), rather than the enum ordinal, defines deterministic
// aggregate precedence. Endpoint arrays remain authoritative.
[[nodiscard]] PatchGridEdgeScalarLineageStatus classify_vertex_scalar_lineage(
    const PreparedPatchGridVertexMagnitudeWitnesses &prepared,
    std::size_t vertex_position) noexcept {
  const PatchGridVertexMagnitudeWitness &vertex =
      prepared.vertices[vertex_position];
  if (vertex.raw_source_certificate_complete &&
      vertex.final_scalar_source_certificate_complete)
    return PatchGridEdgeScalarLineageStatus::certified;
  PatchGridEdgeScalarLineageStatus status =
      PatchGridEdgeScalarLineageStatus::no_analytic_source;
  for (const PatchGridMagnitudeSourceCandidate &candidate :
       vertex_magnitude_candidates(prepared, vertex_position)) {
    status = worse_scalar_lineage(status, candidate_scalar_status(candidate));
  }
  return status;
}

[[nodiscard]] PatchGridEdgeTopologyStatus
aggregate_topology_status(PatchGridTopologyEvidenceStatus first,
                          PatchGridTopologyEvidenceStatus second) noexcept {
  const auto rank = [](PatchGridTopologyEvidenceStatus status) {
    switch (status) {
    case PatchGridTopologyEvidenceStatus::certified_inside_atom_union:
    case PatchGridTopologyEvidenceStatus::
        certified_outside_probe_expanded_union:
      return 0U;
    case PatchGridTopologyEvidenceStatus::unknown:
      return 1U;
    case PatchGridTopologyEvidenceStatus::
        direct_unanimous_parity_not_yet_audited:
      return 2U;
    case PatchGridTopologyEvidenceStatus::analytic_perturbed_point_parity:
      return 3U;
    case PatchGridTopologyEvidenceStatus::morphology_fallback_nonanalytic:
      return 4U;
    case PatchGridTopologyEvidenceStatus::cavity_fill_nonanalytic:
      return 5U;
    }
    return 1U;
  };
  const unsigned worst = std::max(rank(first), rank(second));
  switch (worst) {
  case 0U:
    return PatchGridEdgeTopologyStatus::exact;
  case 2U:
    return PatchGridEdgeTopologyStatus::direct_parity_not_yet_audited;
  case 3U:
    return PatchGridEdgeTopologyStatus::analytic_local_repair_perturbed;
  case 4U:
    return PatchGridEdgeTopologyStatus::morphology_fallback_nonanalytic;
  case 5U:
    return PatchGridEdgeTopologyStatus::cavity_fill_nonanalytic;
  default:
    return PatchGridEdgeTopologyStatus::topology_evidence_missing;
  }
}

[[nodiscard]] bool
exact_topology_evidence(PatchGridTopologyEvidenceStatus status) noexcept {
  return status ==
             PatchGridTopologyEvidenceStatus::certified_inside_atom_union ||
         status == PatchGridTopologyEvidenceStatus::
                       certified_outside_probe_expanded_union;
}

[[nodiscard]] bool append_endpoint_patch_refs(
    const PreparedPatchGridVertexMagnitudeWitnesses &prepared,
    std::size_t vertex_position, std::vector<PatchRefId> &scratch,
    std::vector<PatchRefId> &destination) {
  scratch.clear();
  for (const PatchGridMagnitudeSourceCandidate &candidate :
       vertex_magnitude_candidates(prepared, vertex_position)) {
    if (candidate.kind != PatchGridMagnitudeSourceKind::analytic_crossing)
      continue;
    if (candidate.refs_begin > candidate.refs_end ||
        candidate.refs_end > prepared.candidate_refs.size())
      return false;
    scratch.insert(scratch.end(),
                   prepared.candidate_refs.begin() + candidate.refs_begin,
                   prepared.candidate_refs.begin() + candidate.refs_end);
  }
  std::sort(scratch.begin(), scratch.end());
  scratch.erase(std::unique(scratch.begin(), scratch.end()), scratch.end());
  std::size_t combined = destination.size();
  if (!checked_add(combined, scratch.size()) ||
      combined > destination.max_size())
    return false;
  destination.insert(destination.end(), scratch.begin(), scratch.end());
  return true;
}

[[nodiscard]] bool grid_vertex_point(const PatchMappingGridSpec &spec,
                                     const PatchGridVertexMagnitudeWitness &v,
                                     Vec3 &point) noexcept {
  return add_scaled_coordinate(spec.lower.x, spec.grid_spacing, v.ix,
                               point.x) &&
         add_scaled_coordinate(spec.lower.y, spec.grid_spacing, v.iy,
                               point.y) &&
         add_scaled_coordinate(spec.lower.z, spec.grid_spacing, v.iz, point.z);
}

struct CartesianEdgeIdentity {
  bool cartesian{false};
  RayAxis axis{RayAxis::x};
  std::size_t row{0U};
};

[[nodiscard]] CartesianEdgeIdentity classify_cartesian_grid_edge(
    const PatchMappingGridSpec &spec,
    const PatchGridVertexMagnitudeWitness &first,
    const PatchGridVertexMagnitudeWitness &second) noexcept {
  CartesianEdgeIdentity result;
  const std::size_t dx =
      first.ix > second.ix ? first.ix - second.ix : second.ix - first.ix;
  const std::size_t dy =
      first.iy > second.iy ? first.iy - second.iy : second.iy - first.iy;
  const std::size_t dz =
      first.iz > second.iz ? first.iz - second.iz : second.iz - first.iz;
  if (dx + dy + dz != 1U)
    return result;
  PatchGridVertexMagnitudeRequest request;
  request.ix = first.ix;
  request.iy = first.iy;
  request.iz = first.iz;
  double coordinate = 0.0;
  if (dx == 1U)
    result.axis = RayAxis::x;
  else if (dy == 1U)
    result.axis = RayAxis::y;
  else
    result.axis = RayAxis::z;
  if (!vertex_axis_row(spec, request, result.axis, result.row, coordinate))
    return {};
  result.cartesian = true;
  return result;
}

[[nodiscard]] double interpolation_location_tolerance(
    const PatchMappingGridSpec &spec, const Vec3 &first, const Vec3 &second,
    const Vec3 &crossing, const Vec3 &interpolated) noexcept {
  const double scale = std::max(
      {1.0, std::abs(spec.grid_spacing), std::abs(first.x), std::abs(first.y),
       std::abs(first.z), std::abs(second.x), std::abs(second.y),
       std::abs(second.z), std::abs(crossing.x), std::abs(crossing.y),
       std::abs(crossing.z), std::abs(interpolated.x), std::abs(interpolated.y),
       std::abs(interpolated.z)});
  const auto ulp = [](double value) {
    return std::abs(
        std::nextafter(value, std::numeric_limits<double>::infinity()) - value);
  };
  const double maximum_ulp = std::max(
      {ulp(first.x), ulp(first.y), ulp(first.z), ulp(second.x), ulp(second.y),
       ulp(second.z), ulp(crossing.x), ulp(crossing.y), ulp(crossing.z),
       ulp(interpolated.x), ulp(interpolated.y), ulp(interpolated.z)});
  return std::sqrt(3.0) *
         (4096.0 * std::numeric_limits<double>::epsilon() * scale +
          8.0 * maximum_ulp);
}

struct ProductionEdgeInterpolation {
  Vec3 first_point;
  Vec3 second_point;
  double first_scalar{0.0};
  double second_scalar{0.0};
};

[[nodiscard]] ProductionEdgeInterpolation compute_production_interpolation(
    const PatchMappingGridSpec &spec, const PatchGridCanonicalCellLineage &cell,
    const PatchGridVertexMagnitudeWitness &first_vertex,
    const PatchGridVertexMagnitudeWitness &second_vertex,
    PatchGridSignChangingEdgeLineage &edge) {
  ProductionEdgeInterpolation production;
  if (!grid_vertex_point(spec, first_vertex, production.first_point) ||
      !grid_vertex_point(spec, second_vertex, production.second_point))
    throw std::length_error("edge_point");
  const bool first_inside =
      (cell.inside_mask & (std::uint8_t{1} << edge.first_cell_corner)) != 0U;
  const bool second_inside =
      (cell.inside_mask & (std::uint8_t{1} << edge.second_cell_corner)) != 0U;
  production.first_scalar = first_inside ? -first_vertex.clamped_magnitude
                                         : first_vertex.clamped_magnitude;
  production.second_scalar = second_inside ? -second_vertex.clamped_magnitude
                                           : second_vertex.clamped_magnitude;
  const double denominator = production.first_scalar - production.second_scalar;
  const double parameter =
      std::abs(denominator) > 1.0e-300
          ? std::clamp(production.first_scalar / denominator, 0.0, 1.0)
          : 0.5;
  const Vec3 interpolated{
      production.first_point.x +
          parameter * (production.second_point.x - production.first_point.x),
      production.first_point.y +
          parameter * (production.second_point.y - production.first_point.y),
      production.first_point.z +
          parameter * (production.second_point.z - production.first_point.z)};
  if (!std::isfinite(production.first_scalar) ||
      !std::isfinite(production.second_scalar) || !std::isfinite(parameter) ||
      !finite_vec(interpolated))
    throw std::length_error("edge_interpolation");
  edge.interpolation_parameter = parameter;
  edge.interpolated_point = interpolated;
  edge.production_interpolation_complete = true;
  return production;
}

void classify_axis_edge_location(
    const PatchRayGrid &grid, const PatchMappingGridSpec &spec,
    const PreparedPatchCrossingWitnesses &crossing_witnesses,
    const PreparedPatchGridVertexMagnitudeWitnesses &vertex_witnesses,
    const PatchGridVertexMagnitudeWitness &first_vertex,
    const PatchGridVertexMagnitudeWitness &second_vertex,
    std::size_t first_vertex_position, std::size_t second_vertex_position,
    const ProductionEdgeInterpolation &production,
    PatchGridSignChangingEdgeLineage &edge,
    std::vector<PatchCrossingAddress> &first_addresses,
    std::vector<PatchCrossingAddress> &common_addresses,
    std::vector<PatchRefId> &location_refs) {
  if (!edge.production_interpolation_complete)
    throw std::length_error("missing_production_interpolation");
  const CartesianEdgeIdentity cartesian =
      classify_cartesian_grid_edge(spec, first_vertex, second_vertex);
  edge.cartesian_axis_edge = cartesian.cartesian;
  edge.cartesian_axis = cartesian.axis;
  if (!cartesian.cartesian) {
    edge.location_status = PatchGridEdgeSurfaceLocationStatus::
        internal_diagonal_location_uncertified;
    return;
  }
  if (!edge.scalar_lineage_complete) {
    edge.location_status =
        PatchGridEdgeSurfaceLocationStatus::endpoint_scalar_lineage_uncertified;
    return;
  }

  const auto eligible = [](const PatchGridMagnitudeSourceCandidate &candidate) {
    return candidate.kind == PatchGridMagnitudeSourceKind::analytic_crossing &&
           candidate.raw_exact_owner && !candidate.raw_tolerance_tied &&
           candidate.final_scalar_exact_owner &&
           candidate.source_evidence_certified() && candidate.crossing.valid();
  };
  first_addresses.clear();
  common_addresses.clear();
  for (const PatchGridMagnitudeSourceCandidate &candidate :
       vertex_magnitude_candidates(vertex_witnesses, first_vertex_position)) {
    if (eligible(candidate))
      first_addresses.push_back(candidate.crossing);
  }
  std::sort(first_addresses.begin(), first_addresses.end());
  first_addresses.erase(
      std::unique(first_addresses.begin(), first_addresses.end()),
      first_addresses.end());
  for (const PatchGridMagnitudeSourceCandidate &candidate :
       vertex_magnitude_candidates(vertex_witnesses, second_vertex_position)) {
    if (eligible(candidate) &&
        std::binary_search(first_addresses.begin(), first_addresses.end(),
                           candidate.crossing))
      common_addresses.push_back(candidate.crossing);
  }
  std::sort(common_addresses.begin(), common_addresses.end());
  common_addresses.erase(
      std::unique(common_addresses.begin(), common_addresses.end()),
      common_addresses.end());
  if (common_addresses.empty()) {
    edge.location_status =
        PatchGridEdgeSurfaceLocationStatus::no_common_axis_crossing;
    return;
  }
  if (common_addresses.size() != 1U) {
    edge.location_status =
        PatchGridEdgeSurfaceLocationStatus::multiple_common_axis_crossings;
    return;
  }

  const PatchCrossingAddress address = common_addresses.front();
  edge.location_crossing = address;
  if (address.axis() != cartesian.axis || address.row() != cartesian.row) {
    edge.location_status =
        PatchGridEdgeSurfaceLocationStatus::crossing_axis_row_mismatch;
    return;
  }
  const CertifiedPatchCrossingWitness *crossing =
      find_crossing_witness(crossing_witnesses, address);
  if (crossing == nullptr || !crossing->identity_certified() ||
      !crossing->root_reproduced() || !crossing->location_certified() ||
      !crossing->transversality_certified() || crossing->ref_count != 1U ||
      crossing->total_root_multiplicity != 1U ||
      crossing->maximum_ref_multiplicity != 1U) {
    edge.location_status =
        PatchGridEdgeSurfaceLocationStatus::endpoint_scalar_lineage_uncertified;
    return;
  }

  const double dx = edge.interpolated_point.x - crossing->point.x;
  const double dy = edge.interpolated_point.y - crossing->point.y;
  const double dz = edge.interpolated_point.z - crossing->point.z;
  const double residual = std::sqrt(dx * dx + dy * dy + dz * dz);
  const double tolerance = interpolation_location_tolerance(
      spec, production.first_point, production.second_point, crossing->point,
      edge.interpolated_point);
  edge.interpolation_residual = residual;
  edge.interpolation_tolerance = tolerance;
  if (!std::isfinite(residual) || !std::isfinite(tolerance) ||
      tolerance < 0.0) {
    edge.location_status =
        PatchGridEdgeSurfaceLocationStatus::interpolation_residual_failed;
    return;
  }
  edge.analytic_location_residual_available = true;
  const std::size_t axis_index = static_cast<std::size_t>(cartesian.axis);
  const auto component = [axis_index](const Vec3 &point) {
    return axis_index == 0U ? point.x : (axis_index == 1U ? point.y : point.z);
  };
  const double interval_lower = std::min(component(production.first_point),
                                         component(production.second_point));
  const double interval_upper = std::max(component(production.first_point),
                                         component(production.second_point));
  const double crossing_coordinate = component(crossing->point);
  if (crossing_coordinate < interval_lower ||
      crossing_coordinate > interval_upper) {
    edge.location_status =
        PatchGridEdgeSurfaceLocationStatus::crossing_outside_edge;
    return;
  }
  if (residual > tolerance) {
    edge.location_status =
        PatchGridEdgeSurfaceLocationStatus::interpolation_residual_failed;
    return;
  }

  const AxisPatchProvenance &axis =
      grid.provenance.axes[static_cast<std::size_t>(address.axis())];
  const std::span<const PatchRefId> refs = axis.refs(address.crossing());
  if (refs.size() != 1U || refs.front().valid() == false)
    throw std::length_error("location_ref_contract");
  std::size_t combined = location_refs.size();
  if (!checked_add(combined, refs.size()) ||
      combined > location_refs.max_size())
    throw std::length_error("location_refs");
  location_refs.insert(location_refs.end(), refs.begin(), refs.end());
  edge.location_status =
      PatchGridEdgeSurfaceLocationStatus::certified_axis_crossing;
  edge.surface_location_certified = true;
}

[[nodiscard]] bool
cell_lineage_payload_bytes(const PreparedPatchGridCellLineage &result,
                           std::size_t &bytes) noexcept {
  bytes = 0U;
  return checked_vector_payload(result.cells.capacity(),
                                sizeof(PatchGridCanonicalCellLineage), bytes) &&
         checked_vector_payload(result.cell_tetra_offsets.capacity(),
                                sizeof(std::size_t), bytes) &&
         checked_vector_payload(result.tetrahedra.capacity(),
                                sizeof(PatchGridCanonicalTetraLineage),
                                bytes) &&
         checked_vector_payload(result.tetra_edge_offsets.capacity(),
                                sizeof(std::size_t), bytes) &&
         checked_vector_payload(result.edges.capacity(),
                                sizeof(PatchGridSignChangingEdgeLineage),
                                bytes) &&
         checked_vector_payload(result.edge_endpoint_patch_offsets.capacity(),
                                sizeof(std::size_t), bytes) &&
         checked_vector_payload(result.endpoint_patch_refs.capacity(),
                                sizeof(PatchRefId), bytes) &&
         checked_vector_payload(result.edge_relation_offsets.capacity(),
                                sizeof(std::size_t), bytes) &&
         checked_vector_payload(result.relations.capacity(),
                                sizeof(PatchGridEdgePatchRelationEvidence),
                                bytes) &&
         checked_vector_payload(result.edge_location_ref_offsets.capacity(),
                                sizeof(std::size_t), bytes) &&
         checked_vector_payload(result.location_patch_refs.capacity(),
                                sizeof(PatchRefId), bytes);
}

} // namespace

PreparedPatchGridCellLineage build_patch_grid_cell_lineage(
    const PatchComplex &complex, const LocalPatchProjectionIndex &index,
    const PatchRayGrid &grid, const PatchMappingGridSpec &spec,
    const PreparedPatchCrossingWitnesses &crossing_witnesses,
    const PreparedPatchGridVertexMagnitudeWitnesses &vertex_witnesses,
    std::span<const PatchGridActiveCellRequest> active_cells) {
  const auto begin = std::chrono::steady_clock::now();
  PreparedPatchGridCellLineage result;
  const auto stamp_time = [&]() noexcept {
    result.build_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - begin)
                          .count();
  };
  try {
    if (active_cells.empty()) {
      result.status = "empty_active_cells";
      stamp_time();
      return result;
    }

    std::vector<PatchGridVertexMagnitudeRequest> rebuilt_requests;
    rebuilt_requests.reserve(vertex_witnesses.vertices.size());
    for (const PatchGridVertexMagnitudeWitness &vertex :
         vertex_witnesses.vertices) {
      if ((!vertex.sparse_magnitude_present &&
           std::bit_cast<std::uint32_t>(vertex.sparse_magnitude) != 0U) ||
          (vertex.sparse_magnitude_present &&
           (!std::isfinite(vertex.sparse_magnitude) ||
            vertex.sparse_magnitude < 0.0F))) {
        result.status = "invalid_vertex_payload";
        stamp_time();
        return result;
      }
      PatchGridVertexMagnitudeRequest request;
      request.ix = vertex.ix;
      request.iy = vertex.iy;
      request.iz = vertex.iz;
      request.sparse_magnitude = vertex.sparse_magnitude;
      request.topology_state = vertex.topology_state;
      request.topology_source = vertex.topology_source;
      request.sparse_magnitude_present = vertex.sparse_magnitude_present;
      request.cavity_fill = vertex.cavity_fill;
      rebuilt_requests.push_back(request);
    }
    const PreparedPatchGridVertexMagnitudeWitnesses rebuilt =
        build_patch_grid_vertex_magnitude_witnesses(
            complex, index, grid, spec, crossing_witnesses,
            std::span<const PatchGridVertexMagnitudeRequest>(rebuilt_requests));
    if (!same_magnitude_semantics(vertex_witnesses, rebuilt)) {
      result.status = "invalid_vertex_payload";
      stamp_time();
      return result;
    }
    result.complex_fingerprint = rebuilt.complex_fingerprint;
    result.lattice_identity = rebuilt.lattice_identity;
    std::size_t request_payload = 0U;
    if (!checked_vector_payload(rebuilt_requests.capacity(),
                                sizeof(PatchGridVertexMagnitudeRequest),
                                request_payload))
      throw std::length_error("request_payload");
    std::size_t rebuild_phase_peak = request_payload;
    if (!checked_add(rebuild_phase_peak, rebuilt.peak_temporary_bytes))
      throw std::length_error("rebuild_peak");
    result.accounted_peak_temporary_bytes = rebuild_phase_peak;

    std::vector<CanonicalCellInput> canonical_cells;
    canonical_cells.reserve(active_cells.size());
    for (const PatchGridActiveCellRequest &request : active_cells) {
      CanonicalCellInput cell;
      cell.request = request;
      if (request.inside_mask == 0U || request.inside_mask == 0xffU ||
          !PatchGridCellAddress::try_make(spec, request.ix, request.iy,
                                          request.iz, cell.address)) {
        clear_cell_lineage_payload(result);
        result.status = "invalid_active_cell";
        stamp_time();
        return result;
      }
      canonical_cells.push_back(cell);
    }
    std::sort(canonical_cells.begin(), canonical_cells.end(),
              canonical_cell_less);
    for (std::size_t position = 1U; position < canonical_cells.size();
         ++position) {
      if (canonical_cells[position - 1U].address ==
          canonical_cells[position].address) {
        clear_cell_lineage_payload(result);
        result.status = "duplicate_active_cell";
        stamp_time();
        return result;
      }
    }

    const AnalyticPatchAdjacencyOracle oracle =
        build_analytic_patch_adjacency_oracle(complex, index);
    result.oracle_build_ms = oracle.build_ms();
    result.oracle_accounted_setup_bytes = oracle.retained_payload_bytes();
    const bool oracle_bound =
        oracle.certified() &&
        oracle.seam_certificate().source_complex == &complex &&
        result.complex_fingerprint.valid &&
        index.source_complex_fingerprint == result.complex_fingerprint;
    if (oracle_bound) {
      result.relation_oracle_binding =
          PatchGridRelationOracleBindingStatus::built_once_from_bound_batch;
    }

    std::size_t tetra_capacity = 0U;
    std::size_t edge_capacity = 0U;
    std::size_t cell_offset_capacity = canonical_cells.size();
    std::size_t tetra_offset_capacity = 0U;
    std::size_t edge_offset_capacity = 0U;
    std::size_t endpoint_offset_capacity = 0U;
    if (!checked_multiply(canonical_cells.size(),
                          patch_grid_production_tetrahedra.size(),
                          tetra_capacity) ||
        !checked_multiply(tetra_capacity, 4U, edge_capacity) ||
        !checked_add(cell_offset_capacity, 1U) ||
        (tetra_offset_capacity = tetra_capacity,
         !checked_add(tetra_offset_capacity, 1U)) ||
        (edge_offset_capacity = edge_capacity,
         !checked_add(edge_offset_capacity, 1U)) ||
        !checked_multiply(edge_capacity, 2U, endpoint_offset_capacity) ||
        !checked_add(endpoint_offset_capacity, 1U))
      throw std::length_error("cell_capacity");

    result.cells.reserve(canonical_cells.size());
    result.cell_tetra_offsets.reserve(cell_offset_capacity);
    result.tetrahedra.reserve(tetra_capacity);
    result.tetra_edge_offsets.reserve(tetra_offset_capacity);
    result.edges.reserve(edge_capacity);
    result.edge_endpoint_patch_offsets.reserve(endpoint_offset_capacity);
    result.edge_relation_offsets.reserve(edge_offset_capacity);
    result.edge_location_ref_offsets.reserve(edge_offset_capacity);
    result.cell_tetra_offsets.push_back(0U);
    result.tetra_edge_offsets.push_back(0U);
    result.edge_endpoint_patch_offsets.push_back(0U);
    result.edge_relation_offsets.push_back(0U);
    result.edge_location_ref_offsets.push_back(0U);
    std::vector<PatchRefId> endpoint_ref_scratch;
    std::vector<PatchCrossingAddress> first_address_scratch;
    std::vector<PatchCrossingAddress> common_address_scratch;
    endpoint_ref_scratch.reserve(8U);
    first_address_scratch.reserve(8U);
    common_address_scratch.reserve(8U);
    bool all_scalar_lineage = true;
    bool all_candidate_collection = true;
    bool all_relation_classification = true;
    bool any_relation = false;
    bool all_surface_locations = true;
    bool all_topology_signs = true;
    for (const CanonicalCellInput &input : canonical_cells) {
      PatchGridCanonicalCellLineage cell;
      cell.address = input.address;
      cell.ix = input.request.ix;
      cell.iy = input.request.iy;
      cell.iz = input.request.iz;
      cell.inside_mask = input.request.inside_mask;
      cell.topology_sign_evidence_complete = true;
      std::array<const PatchGridVertexMagnitudeWitness *, 8> corner_vertices{};
      for (std::size_t corner = 0U;
           corner < patch_grid_production_corners.size(); ++corner) {
        const auto &offset = patch_grid_production_corners[corner];
        const std::size_t ix = input.request.ix + offset[0];
        const std::size_t iy = input.request.iy + offset[1];
        const std::size_t iz = input.request.iz + offset[2];
        std::size_t linear = 0U;
        if (!try_patch_grid_vertex_linear_index(spec, ix, iy, iz, linear)) {
          clear_cell_lineage_payload(result);
          result.status = "corner_index_overflow";
          stamp_time();
          return result;
        }
        const PatchGridVertexMagnitudeWitness *vertex =
            find_vertex_witness(vertex_witnesses, linear);
        if (vertex == nullptr) {
          clear_cell_lineage_payload(result);
          result.status = "missing_corner_vertex";
          stamp_time();
          return result;
        }
        const bool inside =
            (input.request.inside_mask & (std::uint8_t{1} << corner)) != 0U;
        if ((vertex->topology_state == PatchGridTopologyState::inside &&
             !inside) ||
            (vertex->topology_state == PatchGridTopologyState::outside &&
             inside)) {
          clear_cell_lineage_payload(result);
          result.status = "topology_sign_mismatch";
          stamp_time();
          return result;
        }
        if (!vertex->topology_exact_vertex_certified)
          cell.topology_sign_evidence_complete = false;
        cell.corner_vertex_linear_indices[corner] = linear;
        corner_vertices[corner] = vertex;
      }
      const std::size_t cell_position = result.cells.size();
      result.cells.push_back(cell);
      for (std::size_t tetra_index = 0U;
           tetra_index < patch_grid_production_tetrahedra.size();
           ++tetra_index) {
        const auto &tetra_corners =
            patch_grid_production_tetrahedra[tetra_index];
        PatchGridCanonicalTetraLineage tetra;
        tetra.cell_position = cell_position;
        tetra.tetra_index = static_cast<std::uint8_t>(tetra_index);
        tetra.cell_corners = tetra_corners;
        tetra.topology_sign_evidence_complete = true;
        for (std::size_t local = 0U; local < tetra_corners.size(); ++local) {
          const std::uint8_t corner = tetra_corners[local];
          tetra.vertex_linear_indices[local] =
              cell.corner_vertex_linear_indices[corner];
          if ((cell.inside_mask & (std::uint8_t{1} << corner)) != 0U)
            tetra.inside_mask |= std::uint8_t{1} << local;
          if (!corner_vertices[corner]->topology_exact_vertex_certified)
            tetra.topology_sign_evidence_complete = false;
        }
        const std::size_t tetra_position = result.tetrahedra.size();
        result.tetrahedra.push_back(tetra);

        std::array<PatchGridLocalTetraEdge, 6> local_edges{};
        const std::size_t edge_count =
            enumerate_patch_grid_tetra_sign_changing_edges(tetra.inside_mask,
                                                           local_edges);
        if (edge_count == std::numeric_limits<std::size_t>::max())
          throw std::length_error("tetra_mask");
        for (std::size_t ordinal = 0U; ordinal < edge_count; ++ordinal) {
          const PatchGridLocalTetraEdge local_edge = local_edges[ordinal];
          std::uint8_t first_corner = tetra_corners[local_edge.first];
          std::uint8_t second_corner = tetra_corners[local_edge.second];
          std::size_t first_vertex =
              cell.corner_vertex_linear_indices[first_corner];
          std::size_t second_vertex =
              cell.corner_vertex_linear_indices[second_corner];
          if (second_vertex < first_vertex) {
            std::swap(first_vertex, second_vertex);
            std::swap(first_corner, second_corner);
          }
          PatchGridSignChangingEdgeLineage edge;
          edge.tetra_position = tetra_position;
          edge.edge_ordinal = static_cast<std::uint8_t>(ordinal);
          edge.first_cell_corner = first_corner;
          edge.second_cell_corner = second_corner;
          edge.first_vertex_linear_index = first_vertex;
          edge.second_vertex_linear_index = second_vertex;
          const PatchGridVertexMagnitudeWitness &first_endpoint =
              *corner_vertices[first_corner];
          const PatchGridVertexMagnitudeWitness &second_endpoint =
              *corner_vertices[second_corner];
          const std::size_t first_position =
              vertex_witness_position(vertex_witnesses, &first_endpoint);
          const std::size_t second_position =
              vertex_witness_position(vertex_witnesses, &second_endpoint);
          edge.endpoint_scalar_status[0] =
              classify_vertex_scalar_lineage(vertex_witnesses, first_position);
          edge.endpoint_scalar_status[1] =
              classify_vertex_scalar_lineage(vertex_witnesses, second_position);
          edge.scalar_status = worse_scalar_lineage(
              edge.endpoint_scalar_status[0], edge.endpoint_scalar_status[1]);
          edge.scalar_lineage_complete =
              edge.endpoint_scalar_status[0] ==
                  PatchGridEdgeScalarLineageStatus::certified &&
              edge.endpoint_scalar_status[1] ==
                  PatchGridEdgeScalarLineageStatus::certified;
          if (edge.scalar_lineage_complete) {
            if (!checked_add(result.scalar_lineage_certified_edges, 1U))
              throw std::length_error("counter");
          } else {
            all_scalar_lineage = false;
          }

          edge.endpoint_topology_status[0] =
              first_endpoint.topology_evidence_status;
          edge.endpoint_topology_status[1] =
              second_endpoint.topology_evidence_status;
          edge.topology_status =
              aggregate_topology_status(edge.endpoint_topology_status[0],
                                        edge.endpoint_topology_status[1]);
          edge.topology_sign_evidence_complete =
              exact_topology_evidence(edge.endpoint_topology_status[0]) &&
              exact_topology_evidence(edge.endpoint_topology_status[1]);
          if (edge.topology_sign_evidence_complete) {
            if (!checked_add(result.topology_sign_certified_edges, 1U))
              throw std::length_error("counter");
          } else {
            all_topology_signs = false;
          }

          edge.patch_candidate_collection_complete = append_endpoint_patch_refs(
              vertex_witnesses, first_position, endpoint_ref_scratch,
              result.endpoint_patch_refs);
          if (!edge.patch_candidate_collection_complete)
            throw std::length_error("endpoint_refs");
          result.edge_endpoint_patch_offsets.push_back(
              result.endpoint_patch_refs.size());
          if (!append_endpoint_patch_refs(vertex_witnesses, second_position,
                                          endpoint_ref_scratch,
                                          result.endpoint_patch_refs))
            throw std::length_error("endpoint_refs");
          result.edge_endpoint_patch_offsets.push_back(
              result.endpoint_patch_refs.size());

          const std::size_t edge_position = result.edges.size();
          std::size_t endpoint_offset_base = 0U;
          std::size_t endpoint_offset_end = 0U;
          if (!checked_multiply(edge_position, 2U, endpoint_offset_base) ||
              (endpoint_offset_end = endpoint_offset_base,
               !checked_add(endpoint_offset_end, 2U)) ||
              endpoint_offset_end >= result.edge_endpoint_patch_offsets.size())
            throw std::length_error("endpoint_offset");
          const std::size_t first_begin =
              result.edge_endpoint_patch_offsets[endpoint_offset_base];
          const std::size_t first_end =
              result.edge_endpoint_patch_offsets[endpoint_offset_base + 1U];
          const std::size_t second_begin = first_end;
          const std::size_t second_end =
              result.edge_endpoint_patch_offsets[endpoint_offset_end];
          const std::span<const PatchRefId> first_refs =
              first_end == first_begin
                  ? std::span<const PatchRefId>{}
                  : std::span<const PatchRefId>(
                        result.endpoint_patch_refs.data() + first_begin,
                        first_end - first_begin);
          const std::span<const PatchRefId> second_refs =
              second_end == second_begin
                  ? std::span<const PatchRefId>{}
                  : std::span<const PatchRefId>(
                        result.endpoint_patch_refs.data() + second_begin,
                        second_end - second_begin);
          edge.patch_relation_classification_complete =
              oracle_bound && !first_refs.empty() && !second_refs.empty();
          std::size_t pair_count = 0U;
          if (!checked_multiply(first_refs.size(), second_refs.size(),
                                pair_count))
            throw std::length_error("relation_count");
          std::size_t relation_total = result.relations.size();
          if (!checked_add(relation_total, pair_count) ||
              relation_total > result.relations.max_size())
            throw std::length_error("relation_count");
          for (const PatchRefId first_ref : first_refs) {
            for (const PatchRefId second_ref : second_refs) {
              PatchGridEdgePatchRelationEvidence relation;
              relation.first = first_ref;
              relation.second = second_ref;
              relation.relation = oracle_bound
                                      ? oracle.relation(first_ref, second_ref)
                                      : PatchRelation::uncertified;
              if (relation.relation == PatchRelation::uncertified)
                edge.patch_relation_classification_complete = false;
              const std::size_t relation_slot =
                  static_cast<std::size_t>(relation.relation);
              if (relation_slot >= result.relation_counts.size() ||
                  !checked_add(result.relation_counts[relation_slot], 1U))
                throw std::length_error("relation_counter");
              result.relations.push_back(relation);
              any_relation = true;
            }
          }
          result.edge_relation_offsets.push_back(result.relations.size());
          if (!edge.patch_relation_classification_complete)
            all_relation_classification = false;

          const ProductionEdgeInterpolation production =
              compute_production_interpolation(spec, cell, first_endpoint,
                                               second_endpoint, edge);
          classify_axis_edge_location(
              grid, spec, crossing_witnesses, vertex_witnesses, first_endpoint,
              second_endpoint, first_position, second_position, production,
              edge, first_address_scratch, common_address_scratch,
              result.location_patch_refs);
          if (edge.cartesian_axis_edge) {
            if (!checked_add(result.axis_edges, 1U))
              throw std::length_error("counter");
          } else if (!checked_add(result.diagonal_edges, 1U)) {
            throw std::length_error("counter");
          }
          if (edge.surface_location_certified) {
            if (!checked_add(result.surface_location_certified_edges, 1U))
              throw std::length_error("counter");
          } else {
            all_surface_locations = false;
          }
          result.edges.push_back(edge);
          result.edge_location_ref_offsets.push_back(
              result.location_patch_refs.size());
        }
        result.tetra_edge_offsets.push_back(result.edges.size());
      }
      result.cell_tetra_offsets.push_back(result.tetrahedra.size());
    }

    if (!cell_lineage_payload_bytes(result, result.retained_bytes))
      throw std::length_error("retained_bytes");
    std::size_t final_accounted = result.retained_bytes;
    std::size_t canonical_payload = 0U;
    std::size_t endpoint_scratch_payload = 0U;
    std::size_t first_address_payload = 0U;
    std::size_t common_address_payload = 0U;
    if (!checked_vector_payload(canonical_cells.capacity(),
                                sizeof(CanonicalCellInput),
                                canonical_payload) ||
        !checked_vector_payload(endpoint_ref_scratch.capacity(),
                                sizeof(PatchRefId), endpoint_scratch_payload) ||
        !checked_vector_payload(first_address_scratch.capacity(),
                                sizeof(PatchCrossingAddress),
                                first_address_payload) ||
        !checked_vector_payload(common_address_scratch.capacity(),
                                sizeof(PatchCrossingAddress),
                                common_address_payload) ||
        !checked_add(final_accounted, request_payload) ||
        !checked_add(final_accounted, rebuilt.retained_bytes) ||
        !checked_add(final_accounted, canonical_payload) ||
        !checked_add(final_accounted, endpoint_scratch_payload) ||
        !checked_add(final_accounted, first_address_payload) ||
        !checked_add(final_accounted, common_address_payload) ||
        !checked_add(final_accounted, result.oracle_accounted_setup_bytes))
      throw std::length_error("peak_bytes");
    result.accounted_peak_temporary_bytes =
        std::max(result.accounted_peak_temporary_bytes, final_accounted);

    const bool have_edges = !result.edges.empty();
    result.scalar_lineage_complete =
        have_edges && all_scalar_lineage &&
        result.scalar_lineage_certified_edges == result.edges.size();
    result.patch_candidate_collection_complete =
        have_edges && all_candidate_collection;
    result.patch_relation_classification_complete =
        have_edges && any_relation && all_relation_classification;
    result.surface_location_certificate_complete =
        have_edges && all_surface_locations &&
        result.surface_location_certified_edges == result.edges.size();
    result.topology_sign_certificate_complete =
        have_edges && all_topology_signs &&
        result.topology_sign_certified_edges == result.edges.size();
    result.status = result.scalar_lineage_complete &&
                            result.patch_candidate_collection_complete &&
                            result.patch_relation_classification_complete &&
                            result.surface_location_certificate_complete &&
                            result.topology_sign_certificate_complete
                        ? "pass"
                        : "pass_with_uncertified_edges";
    stamp_time();
    return result;
  } catch (const std::bad_alloc &) {
    clear_cell_lineage_payload(result);
    result.status = "alloc_fail";
    stamp_time();
    return result;
  } catch (const std::length_error &) {
    clear_cell_lineage_payload(result);
    result.status = "length_err";
    stamp_time();
    return result;
  }
}

namespace {

[[nodiscard]] bool c2a_offsets_valid(const std::vector<std::size_t> &offsets,
                                     std::size_t rows,
                                     std::size_t payload_size) noexcept {
  std::size_t expected = rows;
  if (!checked_add(expected, 1U) || offsets.size() != expected ||
      offsets.empty() || offsets.front() != 0U ||
      offsets.back() != payload_size)
    return false;
  for (std::size_t index = 1U; index < offsets.size(); ++index) {
    if (offsets[index - 1U] > offsets[index])
      return false;
  }
  return true;
}

[[nodiscard]] bool c2a_refs_valid(const PatchComplex &complex,
                                  std::span<const PatchRefId> refs) noexcept {
  for (std::size_t index = 0U; index < refs.size(); ++index) {
    if (!patch_ref_in_range(complex, refs[index]) ||
        (index != 0U && !(refs[index - 1U] < refs[index])))
      return false;
  }
  return true;
}

template <class Value>
[[nodiscard]] bool c2a_safe_subspan(const std::vector<Value> &values,
                                    std::size_t begin, std::size_t end,
                                    std::span<const Value> &result) noexcept {
  result = {};
  if (begin > end || end > values.size())
    return false;
  if (begin == end)
    return true;
  result = std::span<const Value>(values.data() + begin, end - begin);
  return true;
}

[[maybe_unused, nodiscard]] bool validate_c2a_vertex_payload(
    const PatchComplex &complex, const LocalPatchProjectionIndex &index,
    const PatchRayGrid &grid, const PatchMappingGridSpec &spec,
    const PreparedPatchCrossingWitnesses &crossing_witnesses,
    const PreparedPatchGridVertexMagnitudeWitnesses &vertex_witnesses,
    PatchComplexFingerprint &fingerprint,
    PatchRayProvenance::LatticeIdentity &lattice) noexcept {
  if (!validate_magnitude_bindings(complex, index, grid, spec,
                                   crossing_witnesses, fingerprint, lattice) ||
      vertex_witnesses.complex_fingerprint != fingerprint ||
      vertex_witnesses.lattice_identity != lattice ||
      !c2a_offsets_valid(vertex_witnesses.candidate_offsets,
                         vertex_witnesses.vertices.size(),
                         vertex_witnesses.candidates.size()))
    return false;

  std::size_t ref_cursor = 0U;
  std::size_t previous_linear = 0U;
  bool have_previous = false;
  for (std::size_t position = 0U; position < vertex_witnesses.vertices.size();
       ++position) {
    const PatchGridVertexMagnitudeWitness &vertex =
        vertex_witnesses.vertices[position];
    std::size_t linear = 0U;
    if (!try_patch_grid_vertex_linear_index(spec, vertex.ix, vertex.iy,
                                            vertex.iz, linear) ||
        linear != vertex.linear_index ||
        (have_previous && previous_linear >= linear) ||
        !std::isfinite(vertex.raw_magnitude) || vertex.raw_magnitude < 0.0 ||
        !std::isfinite(vertex.clamped_magnitude) ||
        vertex.clamped_magnitude < 0.0 ||
        !std::isfinite(vertex.double_arithmetic_tie_tolerance) ||
        vertex.double_arithmetic_tie_tolerance < 0.0 ||
        !std::isfinite(vertex.sparse_float_tie_tolerance) ||
        vertex.sparse_float_tie_tolerance < 0.0 ||
        !std::isfinite(vertex.source_tie_tolerance) ||
        vertex.source_tie_tolerance < 0.0 ||
        vertex.surface_location_certified ||
        (!vertex.sparse_magnitude_present &&
         std::bit_cast<std::uint32_t>(vertex.sparse_magnitude) != 0U) ||
        (vertex.sparse_magnitude_present &&
         (!std::isfinite(vertex.sparse_magnitude) ||
          vertex.sparse_magnitude < 0.0F)))
      return false;
    previous_linear = linear;
    have_previous = true;

    const std::size_t begin = vertex_witnesses.candidate_offsets[position];
    const std::size_t end = vertex_witnesses.candidate_offsets[position + 1U];
    for (std::size_t candidate_position = begin; candidate_position < end;
         ++candidate_position) {
      const PatchGridMagnitudeSourceCandidate &candidate =
          vertex_witnesses.candidates[candidate_position];
      const auto kind = static_cast<std::uint8_t>(candidate.kind);
      const auto evidence =
          static_cast<std::uint8_t>(candidate.evidence_status);
      if (kind > static_cast<std::uint8_t>(
                     PatchGridMagnitudeSourceKind::clamp_floor) ||
          evidence >
              static_cast<std::uint8_t>(PatchGridMagnitudeEvidenceStatus::
                                            transversality_uncertified) ||
          !std::isfinite(candidate.distance) || candidate.distance < 0.0 ||
          candidate.refs_begin > candidate.refs_end ||
          candidate.refs_end > vertex_witnesses.candidate_refs.size())
        return false;
      std::span<const PatchRefId> refs;
      if (!c2a_safe_subspan(vertex_witnesses.candidate_refs,
                            candidate.refs_begin, candidate.refs_end, refs))
        return false;
      if (candidate.kind == PatchGridMagnitudeSourceKind::analytic_crossing) {
        if (candidate.refs_begin != ref_cursor || !candidate.crossing.valid() ||
            refs.empty() || !c2a_refs_valid(complex, refs))
          return false;
        ref_cursor = candidate.refs_end;
      } else if (candidate.refs_begin != 0U || candidate.refs_end != 0U ||
                 !refs.empty() || candidate.crossing.valid()) {
        return false;
      }
    }
  }
  return ref_cursor == vertex_witnesses.candidate_refs.size();
}

[[maybe_unused, nodiscard]] bool validate_c2a_cell_payload(
    const PatchComplex &complex, const PatchRayGrid &grid,
    const PatchMappingGridSpec &spec,
    const PreparedPatchCrossingWitnesses &crossing_witnesses,
    const PreparedPatchGridVertexMagnitudeWitnesses &vertex_witnesses,
    const PreparedPatchGridCellLineage &cell_lineage,
    const PatchComplexFingerprint &fingerprint,
    const PatchRayProvenance::LatticeIdentity &lattice,
    std::size_t &validation_scratch_bytes) {
  validation_scratch_bytes = 0U;
  if (cell_lineage.complex_fingerprint != fingerprint ||
      cell_lineage.lattice_identity != lattice ||
      (cell_lineage.status != "pass" &&
       cell_lineage.status != "pass_with_uncertified_edges") ||
      !c2a_offsets_valid(cell_lineage.cell_tetra_offsets,
                         cell_lineage.cells.size(),
                         cell_lineage.tetrahedra.size()) ||
      !c2a_offsets_valid(cell_lineage.tetra_edge_offsets,
                         cell_lineage.tetrahedra.size(),
                         cell_lineage.edges.size()) ||
      !c2a_offsets_valid(cell_lineage.edge_relation_offsets,
                         cell_lineage.edges.size(),
                         cell_lineage.relations.size()) ||
      !c2a_offsets_valid(cell_lineage.edge_location_ref_offsets,
                         cell_lineage.edges.size(),
                         cell_lineage.location_patch_refs.size()))
    return false;
  std::size_t endpoint_rows = 0U;
  if (!checked_multiply(cell_lineage.edges.size(), 2U, endpoint_rows) ||
      !c2a_offsets_valid(cell_lineage.edge_endpoint_patch_offsets,
                         endpoint_rows,
                         cell_lineage.endpoint_patch_refs.size()))
    return false;

  for (std::size_t cell_position = 0U;
       cell_position < cell_lineage.cells.size(); ++cell_position) {
    const PatchGridCanonicalCellLineage &cell =
        cell_lineage.cells[cell_position];
    PatchGridCellAddress address;
    if (!PatchGridCellAddress::try_make(spec, cell.ix, cell.iy, cell.iz,
                                        address) ||
        address != cell.address || cell.inside_mask == 0U ||
        cell.inside_mask == 0xffU ||
        (cell_position != 0U &&
         !(cell_lineage.cells[cell_position - 1U].address < cell.address)) ||
        cell_lineage.cell_tetra_offsets[cell_position + 1U] -
                cell_lineage.cell_tetra_offsets[cell_position] !=
            patch_grid_production_tetrahedra.size())
      return false;
    for (std::size_t corner = 0U; corner < 8U; ++corner) {
      const auto &offset = patch_grid_production_corners[corner];
      std::size_t linear = 0U;
      if (!try_patch_grid_vertex_linear_index(spec, cell.ix + offset[0],
                                              cell.iy + offset[1],
                                              cell.iz + offset[2], linear) ||
          linear != cell.corner_vertex_linear_indices[corner] ||
          find_vertex_witness(vertex_witnesses, linear) == nullptr)
        return false;
    }
  }

  for (std::size_t tetra_position = 0U;
       tetra_position < cell_lineage.tetrahedra.size(); ++tetra_position) {
    const PatchGridCanonicalTetraLineage &tetra =
        cell_lineage.tetrahedra[tetra_position];
    if (tetra.cell_position >= cell_lineage.cells.size() ||
        tetra.tetra_index >= patch_grid_production_tetrahedra.size() ||
        tetra_position < cell_lineage.cell_tetra_offsets[tetra.cell_position] ||
        tetra_position >=
            cell_lineage.cell_tetra_offsets[tetra.cell_position + 1U])
      return false;
    const PatchGridCanonicalCellLineage &cell =
        cell_lineage.cells[tetra.cell_position];
    std::size_t expected_tetra_position =
        cell_lineage.cell_tetra_offsets[tetra.cell_position];
    if (!checked_add(expected_tetra_position,
                     static_cast<std::size_t>(tetra.tetra_index)) ||
        tetra_position != expected_tetra_position)
      return false;
    const auto &expected_corners =
        patch_grid_production_tetrahedra[tetra.tetra_index];
    std::uint8_t expected_mask = 0U;
    for (std::size_t local = 0U; local < 4U; ++local) {
      const std::uint8_t corner = expected_corners[local];
      if (tetra.cell_corners[local] != corner ||
          tetra.vertex_linear_indices[local] !=
              cell.corner_vertex_linear_indices[corner])
        return false;
      if ((cell.inside_mask & (std::uint8_t{1} << corner)) != 0U)
        expected_mask |= std::uint8_t{1} << local;
    }
    if (tetra.inside_mask != expected_mask)
      return false;

    std::array<PatchGridLocalTetraEdge, 6> expected_edges{};
    const std::size_t expected_count =
        enumerate_patch_grid_tetra_sign_changing_edges(expected_mask,
                                                       expected_edges);
    const std::size_t edge_begin =
        cell_lineage.tetra_edge_offsets[tetra_position];
    const std::size_t edge_end =
        cell_lineage.tetra_edge_offsets[tetra_position + 1U];
    if (expected_count == std::numeric_limits<std::size_t>::max() ||
        edge_end - edge_begin != expected_count)
      return false;
    for (std::size_t ordinal = 0U; ordinal < expected_count; ++ordinal) {
      const PatchGridSignChangingEdgeLineage &edge =
          cell_lineage.edges[edge_begin + ordinal];
      const PatchGridLocalTetraEdge local = expected_edges[ordinal];
      std::uint8_t first_corner = expected_corners[local.first];
      std::uint8_t second_corner = expected_corners[local.second];
      std::size_t first_vertex =
          cell.corner_vertex_linear_indices[first_corner];
      std::size_t second_vertex =
          cell.corner_vertex_linear_indices[second_corner];
      if (second_vertex < first_vertex) {
        std::swap(first_vertex, second_vertex);
        std::swap(first_corner, second_corner);
      }
      if (edge.tetra_position != tetra_position ||
          edge.edge_ordinal != ordinal ||
          edge.first_cell_corner != first_corner ||
          edge.second_cell_corner != second_corner ||
          edge.first_vertex_linear_index != first_vertex ||
          edge.second_vertex_linear_index != second_vertex ||
          !edge.production_interpolation_complete ||
          !std::isfinite(edge.interpolation_parameter) ||
          edge.interpolation_parameter < 0.0 ||
          edge.interpolation_parameter > 1.0 ||
          !finite_vec(edge.interpolated_point))
        return false;

      const PatchGridVertexMagnitudeWitness *first =
          find_vertex_witness(vertex_witnesses, first_vertex);
      const PatchGridVertexMagnitudeWitness *second =
          find_vertex_witness(vertex_witnesses, second_vertex);
      if (first == nullptr || second == nullptr)
        return false;
      PatchGridSignChangingEdgeLineage reproduced = edge;
      reproduced.production_interpolation_complete = false;
      const ProductionEdgeInterpolation ignored =
          compute_production_interpolation(spec, cell, *first, *second,
                                           reproduced);
      static_cast<void>(ignored);
      if (!bit_equal(reproduced.interpolation_parameter,
                     edge.interpolation_parameter) ||
          !bit_equal(reproduced.interpolated_point.x,
                     edge.interpolated_point.x) ||
          !bit_equal(reproduced.interpolated_point.y,
                     edge.interpolated_point.y) ||
          !bit_equal(reproduced.interpolated_point.z,
                     edge.interpolated_point.z))
        return false;
    }
  }

  std::vector<PatchRefId> candidate_scratch;
  std::vector<PatchRefId> expected_endpoint_refs;
  std::vector<PatchCrossingAddress> first_address_scratch;
  std::vector<PatchCrossingAddress> common_address_scratch;
  candidate_scratch.reserve(8U);
  expected_endpoint_refs.reserve(8U);
  first_address_scratch.reserve(8U);
  common_address_scratch.reserve(8U);
  std::array<std::size_t, 5> relation_counts{};
  std::size_t axis_edges = 0U;
  std::size_t diagonal_edges = 0U;
  std::size_t scalar_certified = 0U;
  std::size_t location_certified = 0U;
  std::size_t topology_certified = 0U;
  bool all_scalar = true;
  bool all_candidates = true;
  bool all_relations = true;
  bool any_relation = false;
  bool all_locations = true;
  bool all_topology = true;
  const bool oracle_bound =
      cell_lineage.relation_oracle_binding ==
      PatchGridRelationOracleBindingStatus::built_once_from_bound_batch;
  if (static_cast<std::uint8_t>(cell_lineage.relation_oracle_binding) >
          static_cast<std::uint8_t>(PatchGridRelationOracleBindingStatus::
                                        built_once_from_bound_batch) ||
      !std::isfinite(cell_lineage.oracle_build_ms) ||
      cell_lineage.oracle_build_ms < 0.0 ||
      !std::isfinite(cell_lineage.build_ms) || cell_lineage.build_ms < 0.0 ||
      cell_lineage.accounted_peak_temporary_bytes < cell_lineage.retained_bytes)
    return false;

  for (std::size_t edge_position = 0U;
       edge_position < cell_lineage.edges.size(); ++edge_position) {
    const PatchGridSignChangingEdgeLineage &edge =
        cell_lineage.edges[edge_position];
    std::size_t endpoint = 0U;
    std::size_t endpoint_after = 0U;
    std::size_t edge_after = edge_position;
    if (!checked_multiply(edge_position, 2U, endpoint) ||
        (endpoint_after = endpoint, !checked_add(endpoint_after, 2U)) ||
        endpoint_after >= cell_lineage.edge_endpoint_patch_offsets.size() ||
        !checked_add(edge_after, 1U) ||
        edge_after >= cell_lineage.edge_relation_offsets.size() ||
        edge_after >= cell_lineage.edge_location_ref_offsets.size())
      return false;

    const std::array<std::size_t, 2> endpoint_linear = {
        edge.first_vertex_linear_index, edge.second_vertex_linear_index};
    std::array<const PatchGridVertexMagnitudeWitness *, 2> endpoint_vertices{};
    std::array<std::size_t, 2> endpoint_positions{};
    std::array<std::span<const PatchRefId>, 2> endpoint_refs{};
    for (std::size_t side = 0U; side < 2U; ++side) {
      const std::size_t begin =
          cell_lineage.edge_endpoint_patch_offsets[endpoint + side];
      const std::size_t end =
          cell_lineage.edge_endpoint_patch_offsets[endpoint + side + 1U];
      if (!c2a_safe_subspan(cell_lineage.endpoint_patch_refs, begin, end,
                            endpoint_refs[side]) ||
          !c2a_refs_valid(complex, endpoint_refs[side]))
        return false;
      endpoint_vertices[side] =
          find_vertex_witness(vertex_witnesses, endpoint_linear[side]);
      if (endpoint_vertices[side] == nullptr)
        return false;
      endpoint_positions[side] =
          vertex_witness_position(vertex_witnesses, endpoint_vertices[side]);
      expected_endpoint_refs.clear();
      if (!append_endpoint_patch_refs(
              vertex_witnesses, endpoint_positions[side], candidate_scratch,
              expected_endpoint_refs) ||
          expected_endpoint_refs.size() != endpoint_refs[side].size() ||
          !std::equal(expected_endpoint_refs.begin(),
                      expected_endpoint_refs.end(),
                      endpoint_refs[side].begin()))
        return false;
    }
    if (!edge.patch_candidate_collection_complete)
      return false;

    const std::array<PatchGridEdgeScalarLineageStatus, 2> expected_scalar = {
        classify_vertex_scalar_lineage(vertex_witnesses, endpoint_positions[0]),
        classify_vertex_scalar_lineage(vertex_witnesses,
                                       endpoint_positions[1])};
    const PatchGridEdgeScalarLineageStatus expected_scalar_aggregate =
        worse_scalar_lineage(expected_scalar[0], expected_scalar[1]);
    const bool expected_scalar_complete =
        expected_scalar[0] == PatchGridEdgeScalarLineageStatus::certified &&
        expected_scalar[1] == PatchGridEdgeScalarLineageStatus::certified;
    const std::array<PatchGridTopologyEvidenceStatus, 2> expected_topology = {
        endpoint_vertices[0]->topology_evidence_status,
        endpoint_vertices[1]->topology_evidence_status};
    const PatchGridEdgeTopologyStatus expected_topology_aggregate =
        aggregate_topology_status(expected_topology[0], expected_topology[1]);
    const bool expected_topology_complete =
        exact_topology_evidence(expected_topology[0]) &&
        exact_topology_evidence(expected_topology[1]);
    if (edge.endpoint_scalar_status != expected_scalar ||
        edge.scalar_status != expected_scalar_aggregate ||
        edge.scalar_lineage_complete != expected_scalar_complete ||
        edge.endpoint_topology_status != expected_topology ||
        edge.topology_status != expected_topology_aggregate ||
        edge.topology_sign_evidence_complete != expected_topology_complete)
      return false;

    const std::size_t relation_begin =
        cell_lineage.edge_relation_offsets[edge_position];
    const std::size_t relation_end =
        cell_lineage.edge_relation_offsets[edge_after];
    std::span<const PatchGridEdgePatchRelationEvidence> relations;
    std::size_t expected_relation_count = 0U;
    if (!c2a_safe_subspan(cell_lineage.relations, relation_begin, relation_end,
                          relations) ||
        !checked_multiply(endpoint_refs[0].size(), endpoint_refs[1].size(),
                          expected_relation_count) ||
        relations.size() != expected_relation_count)
      return false;
    bool edge_relations_complete =
        oracle_bound && !endpoint_refs[0].empty() && !endpoint_refs[1].empty();
    std::size_t relation_position = 0U;
    for (const PatchRefId first : endpoint_refs[0]) {
      for (const PatchRefId second : endpoint_refs[1]) {
        const PatchGridEdgePatchRelationEvidence &relation =
            relations[relation_position++];
        const std::size_t slot = static_cast<std::size_t>(relation.relation);
        if (!patch_ref_in_range(complex, relation.first) ||
            !patch_ref_in_range(complex, relation.second) ||
            relation.first != first || relation.second != second ||
            slot >= relation_counts.size() ||
            !checked_add(relation_counts[slot], 1U))
          return false;
        if (relation.relation == PatchRelation::uncertified)
          edge_relations_complete = false;
        if (!oracle_bound && relation.relation != PatchRelation::uncertified)
          return false;
        any_relation = true;
      }
    }
    if (edge.patch_relation_classification_complete != edge_relations_complete)
      return false;
    if (!edge_relations_complete)
      all_relations = false;

    const std::size_t location_begin =
        cell_lineage.edge_location_ref_offsets[edge_position];
    const std::size_t location_end =
        cell_lineage.edge_location_ref_offsets[edge_after];
    std::span<const PatchRefId> location_refs;
    if (!c2a_safe_subspan(cell_lineage.location_patch_refs, location_begin,
                          location_end, location_refs) ||
        !c2a_refs_valid(complex, location_refs))
      return false;

    const CartesianEdgeIdentity cartesian = classify_cartesian_grid_edge(
        spec, *endpoint_vertices[0], *endpoint_vertices[1]);
    if (edge.cartesian_axis_edge != cartesian.cartesian ||
        (cartesian.cartesian && edge.cartesian_axis != cartesian.axis) ||
        ((edge.location_status ==
          PatchGridEdgeSurfaceLocationStatus::certified_axis_crossing) !=
         edge.surface_location_certified))
      return false;
    if (edge.cartesian_axis_edge) {
      if (!checked_add(axis_edges, 1U))
        return false;
    } else if (!checked_add(diagonal_edges, 1U)) {
      return false;
    }

    if (edge.surface_location_certified) {
      if (!edge.scalar_lineage_complete || !cartesian.cartesian ||
          !edge.location_crossing.valid() ||
          edge.location_crossing.axis() != cartesian.axis ||
          edge.location_crossing.row() != cartesian.row ||
          location_refs.size() != 1U ||
          !edge.analytic_location_residual_available)
        return false;
      const CertifiedPatchCrossingWitness *crossing =
          find_crossing_witness(crossing_witnesses, edge.location_crossing);
      if (crossing == nullptr || !crossing->identity_certified() ||
          !crossing->root_reproduced() || !crossing->location_certified() ||
          !crossing->transversality_certified() || crossing->ref_count != 1U ||
          crossing->total_root_multiplicity != 1U ||
          crossing->maximum_ref_multiplicity != 1U)
        return false;
      const auto eligible = [](const PatchGridMagnitudeSourceCandidate &item) {
        return item.kind == PatchGridMagnitudeSourceKind::analytic_crossing &&
               item.raw_exact_owner && !item.raw_tolerance_tied &&
               item.final_scalar_exact_owner &&
               item.source_evidence_certified() && item.crossing.valid();
      };
      first_address_scratch.clear();
      common_address_scratch.clear();
      for (const PatchGridMagnitudeSourceCandidate &candidate :
           vertex_magnitude_candidates(vertex_witnesses,
                                       endpoint_positions[0])) {
        if (eligible(candidate))
          first_address_scratch.push_back(candidate.crossing);
      }
      std::sort(first_address_scratch.begin(), first_address_scratch.end());
      first_address_scratch.erase(std::unique(first_address_scratch.begin(),
                                              first_address_scratch.end()),
                                  first_address_scratch.end());
      for (const PatchGridMagnitudeSourceCandidate &candidate :
           vertex_magnitude_candidates(vertex_witnesses,
                                       endpoint_positions[1])) {
        if (eligible(candidate) &&
            std::binary_search(first_address_scratch.begin(),
                               first_address_scratch.end(), candidate.crossing))
          common_address_scratch.push_back(candidate.crossing);
      }
      std::sort(common_address_scratch.begin(), common_address_scratch.end());
      common_address_scratch.erase(std::unique(common_address_scratch.begin(),
                                               common_address_scratch.end()),
                                   common_address_scratch.end());
      if (common_address_scratch.size() != 1U ||
          common_address_scratch.front() != edge.location_crossing)
        return false;
      const AxisPatchProvenance &axis =
          grid.provenance.axes[static_cast<std::size_t>(cartesian.axis)];
      const std::span<const PatchRefId> provenance_refs =
          axis.refs(edge.location_crossing.crossing());
      if (provenance_refs.size() != 1U ||
          provenance_refs.front() != location_refs.front())
        return false;

      Vec3 first_point;
      Vec3 second_point;
      if (!grid_vertex_point(spec, *endpoint_vertices[0], first_point) ||
          !grid_vertex_point(spec, *endpoint_vertices[1], second_point))
        return false;
      const double dx = edge.interpolated_point.x - crossing->point.x;
      const double dy = edge.interpolated_point.y - crossing->point.y;
      const double dz = edge.interpolated_point.z - crossing->point.z;
      const double residual = std::sqrt(dx * dx + dy * dy + dz * dz);
      const double tolerance = interpolation_location_tolerance(
          spec, first_point, second_point, crossing->point,
          edge.interpolated_point);
      if (!std::isfinite(residual) || !std::isfinite(tolerance) ||
          tolerance < 0.0 || residual > tolerance ||
          !bit_equal(residual, edge.interpolation_residual) ||
          !bit_equal(tolerance, edge.interpolation_tolerance))
        return false;
      const std::size_t axis_index = static_cast<std::size_t>(cartesian.axis);
      const auto component = [axis_index](const Vec3 &point) {
        return axis_index == 0U ? point.x
                                : (axis_index == 1U ? point.y : point.z);
      };
      const double lower =
          std::min(component(first_point), component(second_point));
      const double upper =
          std::max(component(first_point), component(second_point));
      const double coordinate = component(crossing->point);
      if (coordinate < lower || coordinate > upper ||
          !checked_add(location_certified, 1U))
        return false;
    } else {
      if (!location_refs.empty())
        return false;
      all_locations = false;
    }

    if (edge.scalar_lineage_complete) {
      if (!checked_add(scalar_certified, 1U))
        return false;
    } else {
      all_scalar = false;
    }
    if (edge.topology_sign_evidence_complete) {
      if (!checked_add(topology_certified, 1U))
        return false;
    } else {
      all_topology = false;
    }
  }

  std::size_t scratch_bytes = 0U;
  if (!checked_vector_payload(candidate_scratch.capacity(), sizeof(PatchRefId),
                              scratch_bytes) ||
      !checked_vector_payload(expected_endpoint_refs.capacity(),
                              sizeof(PatchRefId), scratch_bytes) ||
      !checked_vector_payload(first_address_scratch.capacity(),
                              sizeof(PatchCrossingAddress), scratch_bytes) ||
      !checked_vector_payload(common_address_scratch.capacity(),
                              sizeof(PatchCrossingAddress), scratch_bytes))
    return false;
  validation_scratch_bytes = scratch_bytes;
  const bool have_edges = !cell_lineage.edges.empty();
  const bool scalar_complete =
      have_edges && all_scalar && scalar_certified == cell_lineage.edges.size();
  const bool candidate_complete = have_edges && all_candidates;
  const bool relation_complete = have_edges && any_relation && all_relations;
  const bool location_complete =
      have_edges && all_locations &&
      location_certified == cell_lineage.edges.size();
  const bool topology_complete =
      have_edges && all_topology &&
      topology_certified == cell_lineage.edges.size();
  const bool expected_pass = scalar_complete && candidate_complete &&
                             relation_complete && location_complete &&
                             topology_complete;
  return cell_lineage.relation_counts == relation_counts &&
         cell_lineage.axis_edges == axis_edges &&
         cell_lineage.diagonal_edges == diagonal_edges &&
         cell_lineage.scalar_lineage_certified_edges == scalar_certified &&
         cell_lineage.surface_location_certified_edges == location_certified &&
         cell_lineage.topology_sign_certified_edges == topology_certified &&
         cell_lineage.scalar_lineage_complete == scalar_complete &&
         cell_lineage.patch_candidate_collection_complete ==
             candidate_complete &&
         cell_lineage.patch_relation_classification_complete ==
             relation_complete &&
         cell_lineage.surface_location_certificate_complete ==
             location_complete &&
         cell_lineage.topology_sign_certificate_complete == topology_complete &&
         cell_lineage.status ==
             (expected_pass ? "pass" : "pass_with_uncertified_edges");
}

struct C2aCanonicalOccurrence {
  PatchGridCanonicalInterpolationVertexKey key;
  std::size_t edge_occurrence{0U};
};

[[nodiscard]] bool
c2a_occurrence_less(const C2aCanonicalOccurrence &left,
                    const C2aCanonicalOccurrence &right) noexcept {
  return left.key < right.key || (!(right.key < left.key) &&
                                  left.edge_occurrence < right.edge_occurrence);
}

void clear_c2a_payload(
    PreparedPatchGridTriangleOccurrenceLineage &result) noexcept {
  std::vector<std::size_t>().swap(result.edge_occurrence_to_canonical_vertex);
  std::vector<PatchGridCanonicalInterpolationVertexLineage>().swap(
      result.canonical_vertices);
  std::vector<std::size_t>().swap(result.canonical_vertex_occurrence_offsets);
  std::vector<std::size_t>().swap(result.canonical_vertex_edge_occurrences);
  std::vector<std::size_t>().swap(
      result.canonical_vertex_candidate_ref_offsets);
  std::vector<PatchRefId>().swap(result.canonical_vertex_candidate_refs);
  std::vector<std::size_t>().swap(result.canonical_vertex_relation_offsets);
  std::vector<PatchGridCanonicalVertexPatchRelationEvidence>().swap(
      result.canonical_vertex_relations);
  std::vector<std::size_t>().swap(
      result.canonical_vertex_exact_location_ref_offsets);
  std::vector<PatchRefId>().swap(result.canonical_vertex_exact_location_refs);
  std::vector<PatchGridPreOrientationTriangleOccurrenceLineage>().swap(
      result.triangles);
  std::vector<std::size_t>().swap(result.triangle_candidate_ref_offsets);
  std::vector<PatchRefId>().swap(result.triangle_candidate_refs);
  result.complex_fingerprint = {};
  result.lattice_identity = {};
  result.relation_counts = {};
  result.interpolation_occurrences = 0U;
  result.canonical_interpolation_vertices = 0U;
  result.bitwise_interpolation_conflicts = 0U;
  result.exact_location_vertices = 0U;
  result.candidate_refs_without_exact_location_vertices = 0U;
  result.triangle_occurrences = 0U;
  result.finite_triangle_occurrences = 0U;
  result.computed_zero_area_triangle_occurrences = 0U;
  result.nonfinite_triangle_occurrences = 0U;
  result.single_common_candidate_triangles = 0U;
  result.vertex_triplet_patch_identity_certified_triangles = 0U;
  result.retained_bytes = 0U;
  result.accounted_peak_temporary_bytes = 0U;
  result.interpolation_identity_complete = false;
  result.patch_candidate_attribution_complete = false;
  result.triangle_occurrence_lineage_complete = false;
  result.vertex_triplet_patch_identity_classification_complete = false;
  result.planar_face_surface_location_certified = false;
  result.final_mesh_orientation_certified = false;
}

template <class Value>
[[nodiscard]] bool c2a_append_checked(std::vector<Value> &destination,
                                      std::span<const Value> values) {
  std::size_t combined = destination.size();
  if (!checked_add(combined, values.size()) ||
      combined > destination.max_size())
    return false;
  destination.insert(destination.end(), values.begin(), values.end());
  return true;
}

[[nodiscard]] bool
c2a_collect_candidate_refs(const PreparedPatchGridCellLineage &cell_lineage,
                           std::size_t edge_position,
                           std::vector<PatchRefId> &refs) {
  refs.clear();
  std::size_t base = 0U;
  std::size_t end_offset = 0U;
  if (!checked_multiply(edge_position, 2U, base) ||
      (end_offset = base, !checked_add(end_offset, 2U)) ||
      end_offset >= cell_lineage.edge_endpoint_patch_offsets.size())
    return false;
  for (std::size_t side = 0U; side < 2U; ++side) {
    std::span<const PatchRefId> endpoint_refs;
    if (!c2a_safe_subspan(
            cell_lineage.endpoint_patch_refs,
            cell_lineage.edge_endpoint_patch_offsets[base + side],
            cell_lineage.edge_endpoint_patch_offsets[base + side + 1U],
            endpoint_refs) ||
        !c2a_append_checked(refs, endpoint_refs))
      return false;
  }
  std::sort(refs.begin(), refs.end());
  refs.erase(std::unique(refs.begin(), refs.end()), refs.end());
  return true;
}

[[nodiscard]] bool
c2a_collect_exact_refs(const PreparedPatchGridCellLineage &cell_lineage,
                       std::size_t edge_position,
                       std::vector<PatchRefId> &refs) {
  refs.clear();
  std::size_t after = edge_position;
  if (!checked_add(after, 1U) ||
      after >= cell_lineage.edge_location_ref_offsets.size())
    return false;
  std::span<const PatchRefId> exact_refs;
  return c2a_safe_subspan(cell_lineage.location_patch_refs,
                          cell_lineage.edge_location_ref_offsets[edge_position],
                          cell_lineage.edge_location_ref_offsets[after],
                          exact_refs) &&
         c2a_append_checked(refs, exact_refs);
}

[[nodiscard]] bool c2a_relation_less(
    const PatchGridCanonicalVertexPatchRelationEvidence &left,
    const PatchGridCanonicalVertexPatchRelationEvidence &right) noexcept {
  if (left.first != right.first)
    return left.first < right.first;
  if (left.second != right.second)
    return left.second < right.second;
  return static_cast<std::uint8_t>(left.relation) <
         static_cast<std::uint8_t>(right.relation);
}

[[nodiscard]] bool c2a_relation_equal(
    const PatchGridCanonicalVertexPatchRelationEvidence &left,
    const PatchGridCanonicalVertexPatchRelationEvidence &right) noexcept {
  return left.first == right.first && left.second == right.second &&
         left.relation == right.relation;
}

[[nodiscard]] bool c2a_collect_relations(
    const PreparedPatchGridCellLineage &cell_lineage, std::size_t edge_position,
    std::vector<PatchGridCanonicalVertexPatchRelationEvidence> &relations) {
  relations.clear();
  std::size_t after = edge_position;
  if (!checked_add(after, 1U) ||
      after >= cell_lineage.edge_relation_offsets.size())
    return false;
  std::span<const PatchGridEdgePatchRelationEvidence> source;
  if (!c2a_safe_subspan(cell_lineage.relations,
                        cell_lineage.edge_relation_offsets[edge_position],
                        cell_lineage.edge_relation_offsets[after], source) ||
      source.size() > relations.max_size())
    return false;
  relations.reserve(source.size());
  for (const PatchGridEdgePatchRelationEvidence &item : source)
    relations.push_back({item.first, item.second, item.relation});
  std::sort(relations.begin(), relations.end(), c2a_relation_less);
  relations.erase(
      std::unique(relations.begin(), relations.end(), c2a_relation_equal),
      relations.end());
  return true;
}

[[nodiscard]] bool c2a_relations_equal(
    const std::vector<PatchGridCanonicalVertexPatchRelationEvidence> &left,
    const std::vector<PatchGridCanonicalVertexPatchRelationEvidence>
        &right) noexcept {
  if (left.size() != right.size())
    return false;
  for (std::size_t index = 0U; index < left.size(); ++index) {
    if (!c2a_relation_equal(left[index], right[index]))
      return false;
  }
  return true;
}

[[nodiscard]] bool
c2a_payload_bytes(const PreparedPatchGridTriangleOccurrenceLineage &result,
                  std::size_t &bytes) noexcept {
  bytes = 0U;
  return checked_vector_payload(
             result.edge_occurrence_to_canonical_vertex.capacity(),
             sizeof(std::size_t), bytes) &&
         checked_vector_payload(
             result.canonical_vertices.capacity(),
             sizeof(PatchGridCanonicalInterpolationVertexLineage), bytes) &&
         checked_vector_payload(
             result.canonical_vertex_occurrence_offsets.capacity(),
             sizeof(std::size_t), bytes) &&
         checked_vector_payload(
             result.canonical_vertex_edge_occurrences.capacity(),
             sizeof(std::size_t), bytes) &&
         checked_vector_payload(
             result.canonical_vertex_candidate_ref_offsets.capacity(),
             sizeof(std::size_t), bytes) &&
         checked_vector_payload(
             result.canonical_vertex_candidate_refs.capacity(),
             sizeof(PatchRefId), bytes) &&
         checked_vector_payload(
             result.canonical_vertex_relation_offsets.capacity(),
             sizeof(std::size_t), bytes) &&
         checked_vector_payload(
             result.canonical_vertex_relations.capacity(),
             sizeof(PatchGridCanonicalVertexPatchRelationEvidence), bytes) &&
         checked_vector_payload(
             result.canonical_vertex_exact_location_ref_offsets.capacity(),
             sizeof(std::size_t), bytes) &&
         checked_vector_payload(
             result.canonical_vertex_exact_location_refs.capacity(),
             sizeof(PatchRefId), bytes) &&
         checked_vector_payload(
             result.triangles.capacity(),
             sizeof(PatchGridPreOrientationTriangleOccurrenceLineage), bytes) &&
         checked_vector_payload(
             result.triangle_candidate_ref_offsets.capacity(),
             sizeof(std::size_t), bytes) &&
         checked_vector_payload(result.triangle_candidate_refs.capacity(),
                                sizeof(PatchRefId), bytes);
}

[[nodiscard]] bool c2a_empty_cell_payload(
    const PreparedPatchGridCellLineage &cell_lineage) noexcept {
  const auto empty_offsets = [](const std::vector<std::size_t> &offsets) {
    return offsets.empty() || (offsets.size() == 1U && offsets.front() == 0U);
  };
  return cell_lineage.complex_fingerprint == PatchComplexFingerprint{} &&
         cell_lineage.lattice_identity ==
             PatchRayProvenance::LatticeIdentity{} &&
         cell_lineage.cells.empty() && cell_lineage.tetrahedra.empty() &&
         cell_lineage.edges.empty() &&
         cell_lineage.endpoint_patch_refs.empty() &&
         cell_lineage.relations.empty() &&
         cell_lineage.location_patch_refs.empty() &&
         empty_offsets(cell_lineage.cell_tetra_offsets) &&
         empty_offsets(cell_lineage.tetra_edge_offsets) &&
         empty_offsets(cell_lineage.edge_endpoint_patch_offsets) &&
         empty_offsets(cell_lineage.edge_relation_offsets) &&
         empty_offsets(cell_lineage.edge_location_ref_offsets) &&
         std::all_of(cell_lineage.relation_counts.begin(),
                     cell_lineage.relation_counts.end(),
                     [](std::size_t count) { return count == 0U; }) &&
         cell_lineage.axis_edges == 0U && cell_lineage.diagonal_edges == 0U &&
         cell_lineage.scalar_lineage_certified_edges == 0U &&
         cell_lineage.surface_location_certified_edges == 0U &&
         cell_lineage.topology_sign_certified_edges == 0U &&
         cell_lineage.relation_oracle_binding ==
             PatchGridRelationOracleBindingStatus::unavailable &&
         cell_lineage.oracle_accounted_setup_bytes == 0U &&
         cell_lineage.oracle_build_ms == 0.0 &&
         cell_lineage.retained_bytes == 0U &&
         cell_lineage.accounted_peak_temporary_bytes == 0U &&
         !cell_lineage.scalar_lineage_complete &&
         !cell_lineage.patch_candidate_collection_complete &&
         !cell_lineage.patch_relation_classification_complete &&
         !cell_lineage.surface_location_certificate_complete &&
         !cell_lineage.topology_sign_certificate_complete;
}

[[nodiscard]] bool
c2a_find_tetra_edge_occurrence(const PreparedPatchGridCellLineage &cell_lineage,
                               std::size_t tetra_position,
                               PatchGridLocalTetraEdge local_edge,
                               std::size_t &edge_position) noexcept {
  edge_position = std::numeric_limits<std::size_t>::max();
  std::size_t after = tetra_position;
  if (tetra_position >= cell_lineage.tetrahedra.size() ||
      local_edge.first >= 4U || local_edge.second >= 4U ||
      local_edge.first == local_edge.second || !checked_add(after, 1U) ||
      after >= cell_lineage.tetra_edge_offsets.size())
    return false;
  const PatchGridCanonicalTetraLineage &tetra =
      cell_lineage.tetrahedra[tetra_position];
  PatchGridCanonicalInterpolationVertexKey expected;
  if (!PatchGridCanonicalInterpolationVertexKey::try_make(
          tetra.vertex_linear_indices[local_edge.first],
          tetra.vertex_linear_indices[local_edge.second], expected))
    return false;
  for (std::size_t candidate = cell_lineage.tetra_edge_offsets[tetra_position];
       candidate < cell_lineage.tetra_edge_offsets[after]; ++candidate) {
    const PatchGridSignChangingEdgeLineage &edge =
        cell_lineage.edges[candidate];
    PatchGridCanonicalInterpolationVertexKey actual;
    if (!PatchGridCanonicalInterpolationVertexKey::try_make(
            edge.first_vertex_linear_index, edge.second_vertex_linear_index,
            actual))
      return false;
    if (actual == expected) {
      if (edge_position != std::numeric_limits<std::size_t>::max())
        return false;
      edge_position = candidate;
    }
  }
  return edge_position != std::numeric_limits<std::size_t>::max();
}

[[nodiscard]] bool c2a_triangle_area(const Vec3 &a, const Vec3 &b,
                                     const Vec3 &c,
                                     double &twice_area) noexcept {
  const double abx = b.x - a.x;
  const double aby = b.y - a.y;
  const double abz = b.z - a.z;
  const double acx = c.x - a.x;
  const double acy = c.y - a.y;
  const double acz = c.z - a.z;
  const double nx = aby * acz - abz * acy;
  const double ny = abz * acx - abx * acz;
  const double nz = abx * acy - aby * acx;
  twice_area = std::hypot(nx, ny, nz);
  return finite_vec(a) && finite_vec(b) && finite_vec(c) &&
         std::isfinite(twice_area);
}

[[nodiscard]] bool c2a_intersect_refs(std::span<const PatchRefId> first,
                                      std::span<const PatchRefId> second,
                                      std::vector<PatchRefId> &result) {
  result.clear();
  const std::size_t maximum = std::min(first.size(), second.size());
  if (maximum > result.max_size())
    return false;
  result.reserve(maximum);
  std::set_intersection(first.begin(), first.end(), second.begin(),
                        second.end(), std::back_inserter(result));
  return true;
}

[[nodiscard]] bool c2a_account_peak(
    PreparedPatchGridTriangleOccurrenceLineage &result,
    std::span<const std::pair<std::size_t, std::size_t>> payloads) noexcept {
  std::size_t bytes = 0U;
  if (!c2a_payload_bytes(result, bytes))
    return false;
  for (const auto [capacity, element_size] : payloads) {
    if (!checked_vector_payload(capacity, element_size, bytes))
      return false;
  }
  result.accounted_peak_temporary_bytes =
      std::max(result.accounted_peak_temporary_bytes, bytes);
  return true;
}

} // namespace

PreparedPatchGridTriangleOccurrenceLineage
build_patch_grid_pre_orientation_triangle_lineage(
    const PatchComplex &complex, const LocalPatchProjectionIndex &index,
    const PatchRayGrid &grid, const PatchMappingGridSpec &spec,
    const PreparedPatchCrossingWitnesses &crossing_witnesses,
    const PreparedPatchGridVertexMagnitudeWitnesses &vertex_witnesses,
    const PreparedPatchGridCellLineage &cell_lineage) {
  const auto begin = std::chrono::steady_clock::now();
  PreparedPatchGridTriangleOccurrenceLineage result;
  const auto stamp_time = [&]() noexcept {
    result.build_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - begin)
                          .count();
  };
  const auto fail = [&](const char *status,
                        std::size_t geometry_conflicts = 0U) {
    clear_c2a_payload(result);
    result.bitwise_interpolation_conflicts = geometry_conflicts;
    result.status = status;
    stamp_time();
    return result;
  };

  try {
    PatchComplexFingerprint fingerprint;
    PatchRayProvenance::LatticeIdentity lattice;
    if (!validate_c2a_vertex_payload(complex, index, grid, spec,
                                     crossing_witnesses, vertex_witnesses,
                                     fingerprint, lattice))
      return fail("invalid_vertex_payload");
    if (cell_lineage.cells.empty()) {
      if (cell_lineage.status != "empty_active_cells" ||
          !c2a_empty_cell_payload(cell_lineage))
        return fail("invalid_cell_payload");
      result.complex_fingerprint = fingerprint;
      result.lattice_identity = lattice;
      result.status = "empty_input";
      stamp_time();
      return result;
    }

    std::size_t validation_scratch_bytes = 0U;
    if (!validate_c2a_cell_payload(complex, grid, spec, crossing_witnesses,
                                   vertex_witnesses, cell_lineage, fingerprint,
                                   lattice, validation_scratch_bytes))
      return fail("invalid_cell_payload");
    result.complex_fingerprint = fingerprint;
    result.lattice_identity = lattice;
    result.accounted_peak_temporary_bytes = validation_scratch_bytes;

    const std::size_t edge_count = cell_lineage.edges.size();
    std::size_t edge_plus_one = edge_count;
    std::size_t triangle_capacity = 0U;
    std::size_t triangle_plus_one = 0U;
    if (!checked_add(edge_plus_one, 1U) ||
        !checked_multiply(cell_lineage.tetrahedra.size(), 2U,
                          triangle_capacity) ||
        (triangle_plus_one = triangle_capacity,
         !checked_add(triangle_plus_one, 1U)))
      throw std::length_error("capacity");

    std::vector<C2aCanonicalOccurrence> occurrences;
    occurrences.reserve(edge_count);
    result.edge_occurrence_to_canonical_vertex.assign(
        edge_count, std::numeric_limits<std::size_t>::max());
    result.canonical_vertices.reserve(edge_count);
    result.canonical_vertex_occurrence_offsets.reserve(edge_plus_one);
    result.canonical_vertex_edge_occurrences.reserve(edge_count);
    result.canonical_vertex_candidate_ref_offsets.reserve(edge_plus_one);
    result.canonical_vertex_candidate_refs.reserve(
        cell_lineage.endpoint_patch_refs.size());
    result.canonical_vertex_relation_offsets.reserve(edge_plus_one);
    result.canonical_vertex_relations.reserve(cell_lineage.relations.size());
    result.canonical_vertex_exact_location_ref_offsets.reserve(edge_plus_one);
    result.canonical_vertex_exact_location_refs.reserve(
        cell_lineage.location_patch_refs.size());
    result.canonical_vertex_occurrence_offsets.push_back(0U);
    result.canonical_vertex_candidate_ref_offsets.push_back(0U);
    result.canonical_vertex_relation_offsets.push_back(0U);
    result.canonical_vertex_exact_location_ref_offsets.push_back(0U);

    for (std::size_t edge_position = 0U; edge_position < edge_count;
         ++edge_position) {
      const PatchGridSignChangingEdgeLineage &edge =
          cell_lineage.edges[edge_position];
      C2aCanonicalOccurrence occurrence;
      if (!PatchGridCanonicalInterpolationVertexKey::try_make(
              edge.first_vertex_linear_index, edge.second_vertex_linear_index,
              occurrence.key))
        return fail("invalid_canonical_key");
      occurrence.edge_occurrence = edge_position;
      occurrences.push_back(occurrence);
    }
    std::sort(occurrences.begin(), occurrences.end(), c2a_occurrence_less);
    occurrences.erase(std::unique(occurrences.begin(), occurrences.end(),
                                  [](const C2aCanonicalOccurrence &left,
                                     const C2aCanonicalOccurrence &right) {
                                    return left.key == right.key &&
                                           left.edge_occurrence ==
                                               right.edge_occurrence;
                                  }),
                      occurrences.end());
    if (occurrences.size() != edge_count)
      return fail("duplicate_occurrence");

    std::vector<PatchRefId> baseline_candidates;
    std::vector<PatchRefId> current_candidates;
    std::vector<PatchRefId> baseline_exact_refs;
    std::vector<PatchRefId> current_exact_refs;
    std::vector<PatchGridCanonicalVertexPatchRelationEvidence>
        baseline_relations;
    std::vector<PatchGridCanonicalVertexPatchRelationEvidence>
        current_relations;
    baseline_candidates.reserve(8U);
    current_candidates.reserve(8U);
    baseline_exact_refs.reserve(4U);
    current_exact_refs.reserve(4U);
    baseline_relations.reserve(16U);
    current_relations.reserve(16U);

    for (std::size_t group_begin = 0U; group_begin < occurrences.size();) {
      std::size_t group_end = group_begin + 1U;
      while (group_end < occurrences.size() &&
             occurrences[group_end].key == occurrences[group_begin].key)
        ++group_end;
      const std::size_t first_edge_position =
          occurrences[group_begin].edge_occurrence;
      const PatchGridSignChangingEdgeLineage &first_edge =
          cell_lineage.edges[first_edge_position];
      if (!c2a_collect_candidate_refs(cell_lineage, first_edge_position,
                                      baseline_candidates) ||
          !c2a_collect_exact_refs(cell_lineage, first_edge_position,
                                  baseline_exact_refs) ||
          !c2a_collect_relations(cell_lineage, first_edge_position,
                                 baseline_relations))
        throw std::length_error("evidence");

      for (std::size_t position = group_begin + 1U; position < group_end;
           ++position) {
        const std::size_t edge_position = occurrences[position].edge_occurrence;
        const PatchGridSignChangingEdgeLineage &edge =
            cell_lineage.edges[edge_position];
        if (!bit_equal(edge.interpolation_parameter,
                       first_edge.interpolation_parameter) ||
            !bit_equal(edge.interpolated_point.x,
                       first_edge.interpolated_point.x) ||
            !bit_equal(edge.interpolated_point.y,
                       first_edge.interpolated_point.y) ||
            !bit_equal(edge.interpolated_point.z,
                       first_edge.interpolated_point.z))
          return fail("canonical_geometry_conflict", 1U);
        if (!c2a_collect_candidate_refs(cell_lineage, edge_position,
                                        current_candidates) ||
            !c2a_collect_exact_refs(cell_lineage, edge_position,
                                    current_exact_refs) ||
            !c2a_collect_relations(cell_lineage, edge_position,
                                   current_relations))
          throw std::length_error("evidence");
        if (current_candidates != baseline_candidates ||
            current_exact_refs != baseline_exact_refs ||
            !c2a_relations_equal(current_relations, baseline_relations) ||
            edge.location_crossing != first_edge.location_crossing ||
            edge.location_status != first_edge.location_status ||
            edge.surface_location_certified !=
                first_edge.surface_location_certified ||
            edge.analytic_location_residual_available !=
                first_edge.analytic_location_residual_available ||
            !bit_equal(edge.interpolation_residual,
                       first_edge.interpolation_residual) ||
            !bit_equal(edge.interpolation_tolerance,
                       first_edge.interpolation_tolerance) ||
            edge.cartesian_axis_edge != first_edge.cartesian_axis_edge ||
            edge.cartesian_axis != first_edge.cartesian_axis ||
            edge.patch_candidate_collection_complete !=
                first_edge.patch_candidate_collection_complete ||
            edge.patch_relation_classification_complete !=
                first_edge.patch_relation_classification_complete)
          return fail("canonical_evidence_conflict");
      }

      const std::size_t canonical_position = result.canonical_vertices.size();
      PatchGridCanonicalInterpolationVertexLineage vertex;
      vertex.key = occurrences[group_begin].key;
      vertex.canonical_interpolation_point = first_edge.interpolated_point;
      vertex.canonical_interpolation_parameter =
          first_edge.interpolation_parameter;
      vertex.exact_location_crossing = first_edge.location_crossing;
      vertex.canonical_interpolation_complete = true;
      vertex.canonical_edge_occurrence_geometry_bitwise_consistent = true;
      vertex.directed_production_geometry_bitwise_certified = false;
      vertex.patch_candidate_collection_complete =
          first_edge.patch_candidate_collection_complete;
      vertex.patch_relation_classification_complete =
          first_edge.patch_relation_classification_complete;
      vertex.exact_analytic_location_certified =
          first_edge.surface_location_certified &&
          first_edge.location_crossing.valid() && !baseline_exact_refs.empty();
      vertex.candidate_refs_present_without_exact_location =
          !baseline_candidates.empty() &&
          !vertex.exact_analytic_location_certified;
      result.canonical_vertices.push_back(vertex);
      if (vertex.exact_analytic_location_certified) {
        if (!checked_add(result.exact_location_vertices, 1U))
          throw std::length_error("counter");
      }
      if (vertex.candidate_refs_present_without_exact_location &&
          !checked_add(result.candidate_refs_without_exact_location_vertices,
                       1U))
        throw std::length_error("counter");

      for (std::size_t position = group_begin; position < group_end;
           ++position) {
        const std::size_t edge_position = occurrences[position].edge_occurrence;
        if (edge_position >= result.edge_occurrence_to_canonical_vertex.size())
          throw std::length_error("edge_position");
        result.edge_occurrence_to_canonical_vertex[edge_position] =
            canonical_position;
        result.canonical_vertex_edge_occurrences.push_back(edge_position);
      }
      result.canonical_vertex_occurrence_offsets.push_back(
          result.canonical_vertex_edge_occurrences.size());
      if (!c2a_append_checked(
              result.canonical_vertex_candidate_refs,
              std::span<const PatchRefId>(baseline_candidates)) ||
          !c2a_append_checked(
              result.canonical_vertex_exact_location_refs,
              std::span<const PatchRefId>(baseline_exact_refs)) ||
          !c2a_append_checked(
              result.canonical_vertex_relations,
              std::span<const PatchGridCanonicalVertexPatchRelationEvidence>(
                  baseline_relations)))
        throw std::length_error("canonical_payload");
      for (const PatchGridCanonicalVertexPatchRelationEvidence &relation :
           baseline_relations) {
        const std::size_t slot = static_cast<std::size_t>(relation.relation);
        if (slot >= result.relation_counts.size() ||
            !checked_add(result.relation_counts[slot], 1U))
          throw std::length_error("relation_counter");
      }
      result.canonical_vertex_candidate_ref_offsets.push_back(
          result.canonical_vertex_candidate_refs.size());
      result.canonical_vertex_relation_offsets.push_back(
          result.canonical_vertex_relations.size());
      result.canonical_vertex_exact_location_ref_offsets.push_back(
          result.canonical_vertex_exact_location_refs.size());
      group_begin = group_end;
    }

    result.interpolation_occurrences = edge_count;
    result.canonical_interpolation_vertices = result.canonical_vertices.size();
    const std::array<std::pair<std::size_t, std::size_t>, 7> canonical_temps = {
        std::pair{occurrences.capacity(), sizeof(C2aCanonicalOccurrence)},
        std::pair{baseline_candidates.capacity(), sizeof(PatchRefId)},
        std::pair{current_candidates.capacity(), sizeof(PatchRefId)},
        std::pair{baseline_exact_refs.capacity(), sizeof(PatchRefId)},
        std::pair{current_exact_refs.capacity(), sizeof(PatchRefId)},
        std::pair{baseline_relations.capacity(),
                  sizeof(PatchGridCanonicalVertexPatchRelationEvidence)},
        std::pair{current_relations.capacity(),
                  sizeof(PatchGridCanonicalVertexPatchRelationEvidence)}};
    if (!c2a_account_peak(result, canonical_temps))
      throw std::length_error("peak");

    result.triangles.reserve(triangle_capacity);
    result.triangle_candidate_ref_offsets.reserve(triangle_plus_one);
    result.triangle_candidate_ref_offsets.push_back(0U);
    std::vector<PatchRefId> first_intersection;
    std::vector<PatchRefId> triangle_intersection;
    first_intersection.reserve(8U);
    triangle_intersection.reserve(8U);
    bool all_candidate_summaries = true;
    for (std::size_t tetra_position = 0U;
         tetra_position < cell_lineage.tetrahedra.size(); ++tetra_position) {
      const PatchGridCanonicalTetraLineage &tetra =
          cell_lineage.tetrahedra[tetra_position];
      std::array<PatchGridPreOrientationTriangleEdges, 2> local_triangles{};
      const std::size_t triangle_count =
          enumerate_patch_grid_pre_orientation_triangles(tetra.inside_mask,
                                                         local_triangles);
      if (triangle_count == std::numeric_limits<std::size_t>::max())
        throw std::length_error("tetra_mask");
      for (std::size_t triangle_ordinal = 0U; triangle_ordinal < triangle_count;
           ++triangle_ordinal) {
        PatchGridPreOrientationTriangleOccurrenceLineage triangle;
        triangle.tetra_position = tetra_position;
        triangle.triangle_ordinal = static_cast<std::uint8_t>(triangle_ordinal);
        for (std::size_t vertex = 0U; vertex < 3U; ++vertex) {
          std::size_t edge_position = 0U;
          if (!c2a_find_tetra_edge_occurrence(
                  cell_lineage, tetra_position,
                  local_triangles[triangle_ordinal].edges[vertex],
                  edge_position) ||
              edge_position >=
                  result.edge_occurrence_to_canonical_vertex.size())
            return fail("missing_tetra_edge");
          const std::size_t canonical_position =
              result.edge_occurrence_to_canonical_vertex[edge_position];
          if (canonical_position >= result.canonical_vertices.size())
            throw std::length_error("canonical_position");
          triangle.source_edge_occurrence_positions[vertex] = edge_position;
          triangle.canonical_vertex_positions[vertex] = canonical_position;
        }

        const Vec3 &a =
            result.canonical_vertices[triangle.canonical_vertex_positions[0]]
                .canonical_interpolation_point;
        const Vec3 &b =
            result.canonical_vertices[triangle.canonical_vertex_positions[1]]
                .canonical_interpolation_point;
        const Vec3 &c =
            result.canonical_vertices[triangle.canonical_vertex_positions[2]]
                .canonical_interpolation_point;
        triangle.finite_geometry_observed =
            c2a_triangle_area(a, b, c, triangle.canonical_twice_area);
        if (triangle.finite_geometry_observed) {
          if (!checked_add(result.finite_triangle_occurrences, 1U))
            throw std::length_error("counter");
          triangle.computed_zero_area_observed =
              triangle.canonical_twice_area == 0.0;
          if (triangle.computed_zero_area_observed &&
              !checked_add(result.computed_zero_area_triangle_occurrences, 1U))
            throw std::length_error("counter");
        } else if (!checked_add(result.nonfinite_triangle_occurrences, 1U)) {
          throw std::length_error("counter");
        }

        std::array<std::span<const PatchRefId>, 3> candidate_sets{};
        std::array<std::span<const PatchRefId>, 3> exact_sets{};
        triangle.patch_candidate_summary_complete = true;
        for (std::size_t vertex = 0U; vertex < 3U; ++vertex) {
          const std::size_t canonical =
              triangle.canonical_vertex_positions[vertex];
          std::size_t canonical_after = canonical;
          if (!checked_add(canonical_after, 1U) ||
              canonical_after >=
                  result.canonical_vertex_candidate_ref_offsets.size() ||
              canonical_after >=
                  result.canonical_vertex_exact_location_ref_offsets.size())
            throw std::length_error("triangle_ref_offset");
          if (!c2a_safe_subspan(
                  result.canonical_vertex_candidate_refs,
                  result.canonical_vertex_candidate_ref_offsets[canonical],
                  result
                      .canonical_vertex_candidate_ref_offsets[canonical_after],
                  candidate_sets[vertex]) ||
              !c2a_safe_subspan(
                  result.canonical_vertex_exact_location_refs,
                  result.canonical_vertex_exact_location_ref_offsets[canonical],
                  result.canonical_vertex_exact_location_ref_offsets
                      [canonical_after],
                  exact_sets[vertex]))
            throw std::length_error("triangle_refs");
          triangle.patch_candidate_summary_complete &=
              result.canonical_vertices[canonical]
                  .patch_candidate_collection_complete;
        }
        if (!triangle.patch_candidate_summary_complete)
          all_candidate_summaries = false;
        if (!c2a_intersect_refs(candidate_sets[0], candidate_sets[1],
                                first_intersection) ||
            !c2a_intersect_refs(first_intersection, candidate_sets[2],
                                triangle_intersection) ||
            !c2a_append_checked(
                result.triangle_candidate_refs,
                std::span<const PatchRefId>(triangle_intersection)))
          throw std::length_error("triangle_candidates");
        result.triangle_candidate_ref_offsets.push_back(
            result.triangle_candidate_refs.size());
        triangle.single_common_candidate_ref_available =
            triangle.patch_candidate_summary_complete &&
            triangle_intersection.size() == 1U;
        if (triangle.single_common_candidate_ref_available) {
          triangle.single_common_candidate_ref = triangle_intersection.front();
          if (!checked_add(result.single_common_candidate_triangles, 1U))
            throw std::length_error("counter");
        }

        const auto &v0 =
            result.canonical_vertices[triangle.canonical_vertex_positions[0]];
        const auto &v1 =
            result.canonical_vertices[triangle.canonical_vertex_positions[1]];
        const auto &v2 =
            result.canonical_vertices[triangle.canonical_vertex_positions[2]];
        triangle.vertex_triplet_patch_identity_certified =
            v0.exact_analytic_location_certified &&
            v1.exact_analytic_location_certified &&
            v2.exact_analytic_location_certified &&
            exact_sets[0].size() == 1U && exact_sets[1].size() == 1U &&
            exact_sets[2].size() == 1U &&
            exact_sets[0].front() == exact_sets[1].front() &&
            exact_sets[0].front() == exact_sets[2].front();
        if (triangle.vertex_triplet_patch_identity_certified &&
            !checked_add(
                result.vertex_triplet_patch_identity_certified_triangles, 1U))
          throw std::length_error("counter");
        triangle.planar_face_surface_location_certified = false;
        triangle.final_mesh_orientation_certified = false;
        result.triangles.push_back(triangle);
      }
    }

    result.triangle_occurrences = result.triangles.size();
    std::size_t classified_triangle_occurrences =
        result.finite_triangle_occurrences;
    if (!checked_add(classified_triangle_occurrences,
                     result.nonfinite_triangle_occurrences) ||
        classified_triangle_occurrences != result.triangle_occurrences ||
        result.computed_zero_area_triangle_occurrences >
            result.finite_triangle_occurrences)
      throw std::length_error("triangle_counters");
    result.interpolation_identity_complete =
        !result.canonical_vertices.empty() &&
        result.bitwise_interpolation_conflicts == 0U;
    result.patch_candidate_attribution_complete =
        !result.triangles.empty() && all_candidate_summaries;
    result.triangle_occurrence_lineage_complete = !result.triangles.empty();
    result.vertex_triplet_patch_identity_classification_complete =
        !result.triangles.empty();
    result.planar_face_surface_location_certified = false;
    result.final_mesh_orientation_certified = false;

    const std::array<std::pair<std::size_t, std::size_t>, 9> final_temps = {
        std::pair{occurrences.capacity(), sizeof(C2aCanonicalOccurrence)},
        std::pair{baseline_candidates.capacity(), sizeof(PatchRefId)},
        std::pair{current_candidates.capacity(), sizeof(PatchRefId)},
        std::pair{baseline_exact_refs.capacity(), sizeof(PatchRefId)},
        std::pair{current_exact_refs.capacity(), sizeof(PatchRefId)},
        std::pair{baseline_relations.capacity(),
                  sizeof(PatchGridCanonicalVertexPatchRelationEvidence)},
        std::pair{current_relations.capacity(),
                  sizeof(PatchGridCanonicalVertexPatchRelationEvidence)},
        std::pair{first_intersection.capacity(), sizeof(PatchRefId)},
        std::pair{triangle_intersection.capacity(), sizeof(PatchRefId)}};
    if (!c2a_payload_bytes(result, result.retained_bytes) ||
        !c2a_account_peak(result, final_temps))
      throw std::length_error("bytes");
    if (result.nonfinite_triangle_occurrences != 0U) {
      result.status = "pass_with_nonfinite_canonical_geometry";
    } else if (result.computed_zero_area_triangle_occurrences != 0U) {
      result.status = "pass_with_computed_zero_canonical_area";
    } else if (!result.patch_candidate_attribution_complete) {
      result.status = "pass_with_candidate_attribution_uncertified";
    } else {
      result.status = "pass";
    }
    stamp_time();
    return result;
  } catch (const std::bad_alloc &) {
    return fail("alloc_fail");
  } catch (const std::length_error &) {
    return fail("length_err");
  }
}

} // namespace TitanPBE::Tabi::analytic_ses
