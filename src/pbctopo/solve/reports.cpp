#include "pbctopo/solve/solve_common.h"

namespace pbctopo_solve {

void reset_pbctopo_log_file() {
  titan_pbctopo::reset_log_file();
}

void append_pbctopo_log_line(const string &message) {
  titan_pbctopo::append_log_line(message);
}

ofstream open_pbctopo_frame_report() {
  namespace fs = std::filesystem;
  static bool initialized = false;
  static fs::path initialized_cwd;

  std::error_code ec;
  const fs::path cwd = fs::current_path(ec);
  const bool reset_report =
      !initialized || ec || initialized_cwd.empty() || initialized_cwd != cwd;

  ofstream out("pbctopo_frame_report.csv",
               ios::out | (reset_report ? ios::trunc : ios::app));
  if (out.is_open()) {
    if (reset_report)
      write_pbctopo_frame_report_header(out);
    initialized = true;
    initialized_cwd = cwd;
  }
  return out;
}

ofstream open_pbctopo_assignment_set_report() {
  namespace fs = std::filesystem;
  static bool initialized = false;
  static fs::path initialized_cwd;

  std::error_code ec;
  const fs::path cwd = fs::current_path(ec);
  const bool reset_report =
      !initialized || ec || initialized_cwd.empty() || initialized_cwd != cwd;

  ofstream out("pbctopo_assignment_candidates.csv",
               ios::out | (reset_report ? ios::trunc : ios::app));
  if (out.is_open()) {
    if (reset_report)
      write_pbctopo_assignment_set_header(out);
    initialized = true;
    initialized_cwd = cwd;
  }
  return out;
}

ofstream open_pbctopo_rejected_assignment_report() {
  namespace fs = std::filesystem;
  static bool initialized = false;
  static fs::path initialized_cwd;
  std::error_code ec;
  const fs::path cwd = fs::current_path(ec);
  const bool reset_report =
      !initialized || ec || initialized_cwd.empty() || initialized_cwd != cwd;
  ofstream out("pbctopo_rejected_candidates.csv",
               ios::out | (reset_report ? ios::trunc : ios::app));
  if (out.is_open()) {
    if (reset_report)
      titan_pbctopo::write_pbctopo_rejected_assignment_header(out);
    initialized = true;
    initialized_cwd = cwd;
  }
  return out;
}

ofstream open_pbctopo_candidate_evidence_summary_report() {
  namespace fs = std::filesystem;
  static bool initialized = false;
  static fs::path initialized_cwd;

  std::error_code ec;
  const fs::path cwd = fs::current_path(ec);
  const bool reset_report =
      !initialized || ec || initialized_cwd.empty() || initialized_cwd != cwd;

  ofstream out("pbctopo_candidate_evidence_summary.csv",
               ios::out | (reset_report ? ios::trunc : ios::app));
  if (out.is_open()) {
    if (reset_report)
      write_pbctopo_candidate_evidence_summary_header(out);
    initialized = true;
    initialized_cwd = cwd;
  }
  return out;
}

ofstream open_pbctopo_candidate_evidence_relation_report() {
  namespace fs = std::filesystem;
  static bool initialized = false;
  static fs::path initialized_cwd;

  std::error_code ec;
  const fs::path cwd = fs::current_path(ec);
  const bool reset_report =
      !initialized || ec || initialized_cwd.empty() || initialized_cwd != cwd;

  ofstream out("pbctopo_candidate_evidence_relations.csv",
               ios::out | (reset_report ? ios::trunc : ios::app));
  if (out.is_open()) {
    if (reset_report)
      write_pbctopo_candidate_evidence_relation_header(out);
    initialized = true;
    initialized_cwd = cwd;
  }
  return out;
}

PbctopoComponentLayoutReportState &
pbctopo_component_layout_report_state() {
  static PbctopoComponentLayoutReportState state;
  return state;
}

ofstream open_pbctopo_component_layout_report() {
  namespace fs = std::filesystem;
  PbctopoComponentLayoutReportState &state =
      pbctopo_component_layout_report_state();

  std::error_code ec;
  const fs::path cwd = fs::current_path(ec);
  const bool reset_report =
      !state.initialized || ec || state.initialized_cwd.empty() ||
      state.initialized_cwd != cwd;

  ofstream out("pbctopo_component_layout.csv",
               ios::out | (reset_report ? ios::trunc : ios::app));
  if (out.is_open()) {
    if (reset_report) {
      write_pbctopo_component_layout_header(out);
      state.emitted_layouts.clear();
    }
    state.initialized = true;
    state.initialized_cwd = cwd;
  }
  return out;
}

ofstream open_pbctopo_hard_edge_report() {
  namespace fs = std::filesystem;
  static bool initialized = false;
  static fs::path initialized_cwd;

  std::error_code ec;
  const fs::path cwd = fs::current_path(ec);
  const bool reset_report =
      !initialized || ec || initialized_cwd.empty() || initialized_cwd != cwd;

  ofstream out("pbctopo_hard_edges.csv",
               ios::out | (reset_report ? ios::trunc : ios::app));
  if (out.is_open()) {
    if (reset_report)
      write_pbctopo_hard_edge_header(out);
    initialized = true;
    initialized_cwd = cwd;
  }
  return out;
}

template <typename HeaderWriter>
ofstream open_pbctopo_h9_report(const char *filename,
                                PbctopoH9ReportState &state,
                                HeaderWriter &&write_header) {
  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::path cwd = fs::current_path(ec);
  const bool reset_report =
      !state.initialized || ec || state.initialized_cwd.empty() ||
      state.initialized_cwd != cwd;
  ofstream out(filename, ios::out | (reset_report ? ios::trunc : ios::app));
  if (out.is_open()) {
    out.imbue(locale::classic());
    out << setprecision(numeric_limits<double>::max_digits10);
    if (reset_report)
      write_header(out);
    state.initialized = true;
    state.initialized_cwd = cwd;
  }
  return out;
}

ofstream open_pbctopo_h9_component_representative_report() {
  static PbctopoH9ReportState state;
  return open_pbctopo_h9_report(
      "pbctopo_h9_component_representatives.csv", state,
      [](ofstream &out) {
        out << "schema_version,frame,time_ps,component_index,component_id,"
               "atom_count,mass_policy,total_mass,representative_valid,"
               "box_b00,box_b01,box_b02,box_b10,box_b11,box_b12,"
               "box_b20,box_b21,box_b22,sx,sy,sz,x_A,y_A,z_A\n";
      });
}

ofstream open_pbctopo_h9_evidence_component_report() {
  static PbctopoH9ReportState state;
  return open_pbctopo_h9_report(
      "pbctopo_h9_evidence_components.csv", state, [](ofstream &out) {
        out << "schema_version,frame,time_ps,component_index,component_id,"
               "atom_count,evidence_component_label,is_canonical_root,"
               "evidence_connected_component_count,"
               "relative_lattice_degrees_of_freedom\n";
      });
}

ofstream open_pbctopo_h9_orbit_report() {
  static PbctopoH9ReportState state;
  return open_pbctopo_h9_report(
      "pbctopo_h9_orbit.csv", state, [](ofstream &out) {
        out << "schema_version,frame,time_ps,qx,qy,qz,root_label,"
               "shifted_label,within_radius,hard_certified,"
               "evidence_evaluated,evidence_noncontradicted,accepted,"
               "assignment_signature,evidence_state,"
               "evidence_connected_component_count,"
               "relative_lattice_degrees_of_freedom,evidence_edge_count,"
               "compatible_hypothesis_count,compatible_contact_count\n";
      });
}

ofstream open_pbctopo_temporal_path_report() {
  namespace fs = std::filesystem;
  static bool initialized = false;
  static fs::path initialized_cwd;

  std::error_code ec;
  const fs::path cwd = fs::current_path(ec);
  const bool reset_report =
      !initialized || ec || initialized_cwd.empty() || initialized_cwd != cwd;

  ofstream out("pbctopo_temporal_path.csv",
               ios::out | (reset_report ? ios::trunc : ios::app));
  if (out.is_open()) {
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (reset_report) {
      out << "schema_version,producer_version,coordinate_unit,time_unit,"
             "endianness,floating_format,identity_hash_algorithm,"
             "binding_hash_algorithm,payload_checksum_algorithm,binding_mode,"
             "negative_zero_policy,"
             "nonfinite_policy,observation_scope,temporal_policy_status,frame,"
             "time_ps,selected_rank,cut_x,cut_y,cut_z,emission_cost,"
             "assignment_signature,changed_from_greedy,replay_applied\n";
    }
    initialized = true;
    initialized_cwd = cwd;
  }
  return out;
}

void merge_pbctopo_search_domain_provenance(
    PbctopoCutCandidate &target, const PbctopoCutCandidate &source) {
  const bool target_audit_known = target.search_domain_audit_known;
  const size_t target_radius = target.component_search_shell_radius;
  const bool target_complete = target.search_domain_evidence_complete;
  const bool target_boundary =
      target.selected_assignment_touches_shell_boundary;
  const bool target_expansion_exhausted = target.domain_expansion_exhausted;
  if (source.search_domain_audit_known) {
    const size_t merged_radius =
        max(target_audit_known ? target_radius : 0,
            source.component_search_shell_radius);
    const bool target_is_max_domain =
        target_audit_known && target_radius == merged_radius;
    const bool source_is_max_domain =
        source.component_search_shell_radius == merged_radius;
    target.component_search_shell_radius = merged_radius;
    target.search_domain_evidence_complete =
        (target_is_max_domain && target_complete) ||
        (source_is_max_domain && source.search_domain_evidence_complete);
    target.selected_assignment_touches_shell_boundary =
        (target_is_max_domain && target_boundary) ||
        (source_is_max_domain &&
         source.selected_assignment_touches_shell_boundary);
    target.domain_expansion_exhausted =
        (target_is_max_domain && target_expansion_exhausted) ||
        (source_is_max_domain && source.domain_expansion_exhausted);
    target.search_domain_audit_known = true;
  } else if (!target_audit_known) {
    target.search_domain_evidence_complete = false;
  }
  target.domain_expansion_attempted =
      target.domain_expansion_attempted || source.domain_expansion_attempted;
  target.bounded_domain_only =
      target.bounded_domain_only || source.bounded_domain_only;
}

using PbctopoGyrationShapeMetrics = GyrationShapeMetrics;

string format_int3(const Int3 &value) {
  return "(" + to_string(value.x) + "," + to_string(value.y) + "," +
         to_string(value.z) + ")";
}

string format_vec3(const Vec3 &value) {
  return "(" + to_string(value.x) + "," + to_string(value.y) + "," +
         to_string(value.z) + ")";
}

string format_component_offsets(const vector<Int3> &offsets) {
  string formatted = "[";
  for (size_t idx = 0; idx < offsets.size(); ++idx) {
    if (idx > 0)
      formatted += " ";
    formatted += to_string(idx) + ":" + format_int3(offsets[idx]);
  }
  formatted += "]";
  return formatted;
}

string format_graph_edges(const vector<PbctopoGraphEdge> &edges) {
  string formatted = "[";
  for (size_t idx = 0; idx < edges.size(); ++idx) {
    if (idx > 0)
      formatted += " ";
    const auto &edge = edges[idx];
    formatted += to_string(edge.a) + "-" + to_string(edge.b) + ":" +
                 format_int3({edge.dx, edge.dy, edge.dz}) + "/min=" +
                 to_string(edge.min_pair_d2) + "/bbox=" +
                 to_string(edge.bbox_d2) + "/conf=" +
                 to_string(edge.confidence);
  }
  formatted += "]";
  return formatted;
}

} // namespace pbctopo_solve
