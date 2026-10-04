#include "tabi_node_patch_curved.h"
#include <cstdlib>
#include "../titan_executor.h"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <thread>
namespace TitanPBE::Tabi {
namespace {
using V = std::array<double, 3>;
using B = std::array<double, 4>;
constexpr double pi = std::numbers::pi;
V add(V a, V b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
V sub(V a, V b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
V mul(V a, double b) { return {a[0] * b, a[1] * b, a[2] * b}; }
double dot(V a, V b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
double norm(V a) { return std::sqrt(dot(a, a)); }
V cross(V a, V b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}
std::vector<std::array<double, 2>> gauss(int n) {
  std::vector<std::array<double, 2>> rule;
  for (int i = 0; i < n; ++i) {
    double z = std::cos(pi * (i + .75) / (n + .5)), d = 0;
    for (int it = 0; it < 64; ++it) {
      double p = 1, prev = 0;
      for (int k = 1; k <= n; ++k) {
        double old = p;
        p = ((2 * k - 1) * z * p - (k - 1) * prev) / k;
        prev = old;
      }
      d = n * (z * p - prev) / (z * z - 1);
      double step = p / d;
      z -= step;
      if (std::abs(step) < 2e-15)
        break;
    }
    rule.push_back({(z + 1) / 2, 1 / ((1 - z * z) * d * d)});
  }
  return rule;
}
std::vector<CurvedDualSample> samples(const std::vector<Panel> &triangles,
                                      const NodePatchGeometry &g,
                                      std::size_t node, int order) {
  const auto rule = gauss(order);
  std::vector<CurvedDualSample> out;
  for (auto pos = g.incident_offsets[node]; pos < g.incident_offsets[node + 1];
       ++pos) {
    const auto ti = g.incident_triangles[pos];
    const auto &t = g.triangles[ti];
    const auto &s = sphere_patch_sphere(triangles[ti]);
    V origin{s[0], s[1], s[2]};
    int local = 0;
    while (t.vertices[local] != node)
      ++local;
    V a = sub(g.vertex_xyz[node], origin),
      b = sub(g.vertex_xyz[t.vertices[(local + 1) % 3]], origin),
      c = sub(g.vertex_xyz[t.vertices[(local + 2) % 3]], origin);
    V middle = mul(add(add(a, b), c), 1. / 3);
    for (const auto &edge : std::array<std::array<V, 2>, 2>{
             {{{mul(add(a, b), .5), middle}}, {{middle, mul(add(a, c), .5)}}}})
      for (auto u : rule)
        for (auto v : rule) {
          V du = sub(add(mul(edge[0], 1 - v[0]), mul(edge[1], v[0])), a),
            dv = mul(sub(edge[1], edge[0]), u[0]), p = add(a, mul(du, u[0]));
          double len = norm(p);
          if (!(len > 0))
            throw std::runtime_error("curved_dual_radial_map_singular");
          V normal = mul(p, 1 / len);
          double w = u[1] * v[1] * s[3] * s[3] *
                     std::abs(dot(p, cross(du, dv))) / (len * len * len);
          if (!(w > 0) || !std::isfinite(w))
            throw std::runtime_error("curved_dual_invalid_weight");
          out.push_back({add(origin, mul(normal, s[3])), normal, w});
        }
  }
  return out;
}
// Curved PN triangle (Vlachos, Peters, Boyd, Mitchell 2001): the cubic Bezier
// patch that interpolates three vertex positions and is tangent to their
// normals.  The edge control points are the flat edge thirds projected onto
// the tangent planes at their nearer vertex; the centre point is pulled off
// the flat centroid by half the edge-point offset.
struct PnPatch {
  std::array<V, 10> b; // 300 030 003 210 120 021 012 102 201 111
};
PnPatch pn_patch(const V &p1, const V &p2, const V &p3, const V &n1,
                 const V &n2, const V &n3) {
  auto edge = [](const V &pa, const V &pb, const V &na) {
    // (2 pa + pb - w na) / 3 with w = (pb - pa).na projects the flat third
    // point onto the tangent plane at pa.
    const double w = dot(sub(pb, pa), na);
    return mul(sub(add(mul(pa, 2.0), pb), mul(na, w)), 1.0 / 3.0);
  };
  PnPatch t;
  t.b[0] = p1; t.b[1] = p2; t.b[2] = p3;
  t.b[3] = edge(p1, p2, n1); t.b[4] = edge(p2, p1, n2);
  t.b[5] = edge(p2, p3, n2); t.b[6] = edge(p3, p2, n3);
  t.b[7] = edge(p3, p1, n3); t.b[8] = edge(p1, p3, n1);
  V e{};
  for (int k = 3; k < 9; ++k) e = add(e, t.b[k]);
  e = mul(e, 1.0 / 6.0);
  const V v = mul(add(add(p1, p2), p3), 1.0 / 3.0);
  t.b[9] = add(e, mul(sub(e, v), 0.5));
  return t;
}
// Position and the three barycentric partials of the cubic patch.
void pn_evaluate(const PnPatch &t, double l1, double l2, double l3, V &pos,
                 V &d1, V &d2, V &d3) {
  const auto &b = t.b;
  // Bernstein cubic: sum b_ijk * 3!/(i!j!k!) l1^i l2^j l3^k.
  pos = add(add(add(mul(b[0], l1 * l1 * l1), mul(b[1], l2 * l2 * l2)),
                add(mul(b[2], l3 * l3 * l3), mul(b[3], 3 * l1 * l1 * l2))),
            add(add(add(mul(b[4], 3 * l1 * l2 * l2), mul(b[5], 3 * l2 * l2 * l3)),
                    add(mul(b[6], 3 * l2 * l3 * l3), mul(b[7], 3 * l1 * l3 * l3))),
                add(mul(b[8], 3 * l1 * l1 * l3), mul(b[9], 6 * l1 * l2 * l3))));
  d1 = add(add(add(mul(b[0], 3 * l1 * l1), mul(b[3], 6 * l1 * l2)),
               add(mul(b[4], 3 * l2 * l2), mul(b[7], 3 * l3 * l3))),
           add(mul(b[8], 6 * l1 * l3), mul(b[9], 6 * l2 * l3)));
  d2 = add(add(add(mul(b[1], 3 * l2 * l2), mul(b[3], 3 * l1 * l1)),
               add(mul(b[4], 6 * l1 * l2), mul(b[5], 6 * l2 * l3))),
           add(mul(b[6], 3 * l3 * l3), mul(b[9], 6 * l1 * l3)));
  d3 = add(add(add(mul(b[2], 3 * l3 * l3), mul(b[5], 3 * l2 * l2)),
               add(mul(b[6], 6 * l2 * l3), mul(b[7], 6 * l1 * l3))),
           add(mul(b[8], 3 * l1 * l1), mul(b[9], 6 * l1 * l2)));
}
// Same dual-cell decomposition and Gauss layout as samples(), but the piece is
// mapped onto the PN patch of its parent triangle instead of onto an exact
// sphere.  The area element is |dP/du x dP/dv| through the chain rule, and the
// normal is the patch's own geometric normal, oriented by the flat triangle.
std::vector<CurvedDualSample> pn_samples(const std::vector<Panel> &nodes,
                                         const NodePatchGeometry &g,
                                         std::size_t node, int order) {
  const auto rule = gauss(order);
  std::vector<CurvedDualSample> out;
  for (auto pos = g.incident_offsets[node]; pos < g.incident_offsets[node + 1];
       ++pos) {
    const auto ti = g.incident_triangles[pos];
    const auto &t = g.triangles[ti];
    int local = 0;
    while (t.vertices[local] != node)
      ++local;
    const std::size_t ia = t.vertices[local],
                      ib = t.vertices[(local + 1) % 3],
                      ic = t.vertices[(local + 2) % 3];
    const V pa{g.vertex_xyz[ia][0], g.vertex_xyz[ia][1], g.vertex_xyz[ia][2]},
        pb{g.vertex_xyz[ib][0], g.vertex_xyz[ib][1], g.vertex_xyz[ib][2]},
        pc{g.vertex_xyz[ic][0], g.vertex_xyz[ic][1], g.vertex_xyz[ic][2]};
    const V na{nodes[ia].nx, nodes[ia].ny, nodes[ia].nz},
        nb{nodes[ib].nx, nodes[ib].ny, nodes[ib].nz},
        nc{nodes[ic].nx, nodes[ic].ny, nodes[ic].nz};
    const V flat{t.nx, t.ny, t.nz};
    const PnPatch patch = pn_patch(pa, pb, pc, na, nb, nc);
    // Barycentric coordinates of the fan corners: vertex a, then the two
    // edge midpoints and the centroid, matching samples() exactly.
    const std::array<double, 3> La{1, 0, 0}, Mab{.5, .5, 0}, Mac{.5, 0, .5},
        C{1. / 3, 1. / 3, 1. / 3};
    for (const auto &edge : std::array<std::array<std::array<double, 3>, 2>, 2>{
             {{{Mab, C}}, {{C, Mac}}}})
      for (auto u : rule)
        for (auto v : rule) {
          // lambda(u,v) = a + u * ((1-v) e0 + v e1 - a): linear in (u,v).
          std::array<double, 3> lam{}, dlu{}, dlv{};
          for (int k = 0; k < 3; ++k) {
            const double ev = (1 - v[0]) * edge[0][k] + v[0] * edge[1][k];
            lam[k] = La[k] + u[0] * (ev - La[k]);
            dlu[k] = ev - La[k];
            dlv[k] = u[0] * (edge[1][k] - edge[0][k]);
          }
          V P, d1, d2, d3;
          pn_evaluate(patch, lam[0], lam[1], lam[2], P, d1, d2, d3);
          const V Pu = add(add(mul(d1, dlu[0]), mul(d2, dlu[1])), mul(d3, dlu[2]));
          const V Pv = add(add(mul(d1, dlv[0]), mul(d2, dlv[1])), mul(d3, dlv[2]));
          V nrm = cross(Pu, Pv);
          const double area = norm(nrm);
          if (!(area > 0) || !std::isfinite(area))
            throw std::runtime_error("curved_dual_pn_degenerate_area");
          nrm = mul(nrm, 1 / area);
          if (dot(nrm, flat) < 0)
            nrm = mul(nrm, -1.0);
          const double w = u[1] * v[1] * area;
          out.push_back({P, nrm, w});
        }
  }
  return out;
}
B integrate(const Panel &target, const std::vector<CurvedDualSample> &q,
            double eps, double kappa) {
  B sum{};
  const V x{target.x, target.y, target.z}, nx{target.nx, target.ny, target.nz};
  for (const auto &s : q) {
    const V delta = sub(x, s.xyz);
    const double r = norm(delta);
    if (!(r > 1e-14))
      throw std::runtime_error("curved_dual_quadrature_collision");
    const double z = kappa * r, e = std::exp(-z);
    // Analytic series only where subtraction would lose relative precision.
    // At the switch its omitted absolute term is below 3e-23.
    const double e1 =
        z < .01
            ? z * z *
                  (.5 +
                   z * (-1. / 3 +
                        z * (1. / 8 +
                             z * (-1. / 30 + z * (1. / 144 + z * (-1. / 840 +
                                                                  z / 5760))))))
            : -std::expm1(-z) - z * e;
    const double a = dot(nx, delta) / r, b = dot(s.normal, delta) / r,
                 c = dot(nx, s.normal);
    const double scale = 1 / (4 * pi * r * r), src = b * scale,
                 tgt = -a * scale;
    const double gp = e1 * scale, gpp = (z * z * e - 2 * e1) * scale / r;
    const B value{static_cast<double>(-std::expm1(-z) / (4 * pi * r)),
                  static_cast<double>((eps - 1 - eps * e1) * src),
                  static_cast<double>((1 - 1 / eps + e1 / eps) * tgt),
                  static_cast<double>(-gpp * a * b - gp / r * (c - a * b))};
    for (int k = 0; k < 4; ++k)
      sum[k] += s.weight * value[k];
  }
  for (double v : sum)
    if (!std::isfinite(v))
      throw std::runtime_error("curved_dual_nonfinite_kernel");
  return sum;
}
} // namespace
bool prepare_curved_node_patches(const std::vector<Panel> &triangles,
                                 std::vector<Panel> &nodes, double eps,
                                 double kappa, std::size_t budget,
                                 TitanExecutor &executor, bool eager) {
  if (nodes.empty())
    return false;
  // Exact spherical parents when every triangle carries sphere-patch
  // quadrature; otherwise a PN-triangle lift from the mesh itself, which
  // needs only the vertex positions and the node normals that
  // build_node_patch_panels has already averaged onto each node.
  // TITAN_CURVED_NODE_PATCH = auto (default) | exact | pn | off selects the
  // representation for A/B work: "off" returns false so the flat legacy cell
  // runs, "pn" forces the mesh lift even where exact spheres exist, which is
  // how the lift is checked against a known answer.
  static const std::string mode = [] {
    const char *raw = std::getenv("TITAN_CURVED_NODE_PATCH");
    return std::string(raw ? raw : "auto");
  }();
  if (mode == "off")
    return false;
  const bool have_spheres =
      std::all_of(triangles.begin(), triangles.end(), [](const Panel &p) {
        return bool(p.sphere_patch_quadrature);
      });
  if (mode == "exact" && !have_spheres)
    return false;
  const bool exact = have_spheres && mode != "pn";
  if (!exact)
    for (const Panel &p : nodes)
      if (!std::isfinite(p.nx + p.ny + p.nz) ||
          !(p.nx * p.nx + p.ny * p.ny + p.nz * p.nz > 0.5))
        return false; // no usable node normal: keep the flat legacy cell
  auto g =
      std::make_shared<NodePatchGeometry>(*nodes.front().node_patch_geometry);
  auto curved = std::make_shared<CurvedNodePatchGeometry>();
  curved->representation = exact ? "curved_dual" : "curved_pn";
  const std::size_t n = nodes.size();
  curved->cells.resize(n);
  curved->eps_ratio = eps;
  curved->kappa = kappa;
  // Reconstruct collocation normals from the exact parent surfaces. A crease
  // does not have a unique smooth-surface normal and must not be averaged away.
  // PN mode skips this entirely and keeps the averaged node normals.
  for (std::size_t i = 0; exact && i < n; ++i) {
    V normal{};
    bool first = true;
    for (auto pos = g->incident_offsets[i]; pos < g->incident_offsets[i + 1];
         ++pos) {
      const auto ti = g->incident_triangles[pos];
      const auto &s = sphere_patch_sphere(triangles[ti]);
      const V v = mul(sub(g->vertex_xyz[i], {s[0], s[1], s[2]}), 1 / s[3]);
      if (!first && norm(sub(normal, v)) > 1e-7)
        throw std::runtime_error("curved_dual_nonunique_vertex_normal");
      normal = v;
      first = false;
      const auto &t = g->triangles[ti];
      int local = 0;
      while (t.vertices[local] != i)
        ++local;
      V origin{s[0], s[1], s[2]},
          a = sub(g->vertex_xyz[i], origin),
          b = sub(g->vertex_xyz[t.vertices[(local + 1) % 3]], origin),
          c = sub(g->vertex_xyz[t.vertices[(local + 2) % 3]], origin);
      for (const auto &corner :
           std::array<V, 3>{add(a, b), add(a, c), add(add(a, b), c)}) {
        V mapped = mul(corner, s[3] / norm(corner));
        // Normalized positive combinations stay in this cap: its cosine is
        // positive, so the maximum angular distance is attained at a corner.
        if (dot(mapped, a) <= 0)
          throw std::runtime_error("curved_dual_cap_exceeds_hemisphere");
        curved->cells[i].radius = std::max(curved->cells[i].radius,
                                           norm(sub(mapped, a)) * (1 + 1e-12));
      }
    }
    nodes[i].nx = normal[0];
    nodes[i].ny = normal[1];
    nodes[i].nz = normal[2];
  }
  executor.parallel_for(
      n, std::min<std::size_t>(8, executor.capacity()), 1,
      [&](std::size_t, std::size_t begin, std::size_t end) {
        for (auto i = begin; i < end; ++i) {
          auto &cell = curved->cells[i];
          cell.samples = exact ? samples(triangles, *g, i, 4)
                               : pn_samples(nodes, *g, i, 4);
          if (!exact) {
            // Bounding radius for the treecode AABB: the exact path took it
            // from the parent sphere; here it is the farthest sample.
            const V x{nodes[i].x, nodes[i].y, nodes[i].z};
            for (const auto &q : cell.samples)
              cell.radius = std::max(cell.radius, norm(sub(q.xyz, x)) * (1 + 1e-12));
          }
          for (const auto &q : cell.samples) {
            cell.area += q.weight;
            cell.center = add(cell.center, mul(q.xyz, q.weight));
            cell.normal_integral =
                add(cell.normal_integral, mul(q.normal, q.weight));
          }
          cell.center = mul(cell.center, 1 / cell.area);
          for (const auto &q : cell.samples) {
            V d = sub(q.xyz, cell.center);
            double px[6]{1}, py[6]{1}, pz[6]{1};
            for (int k = 1; k < 6; ++k) {
              px[k] = px[k - 1] * d[0];
              py[k] = py[k - 1] * d[1];
              pz[k] = pz[k - 1] * d[2];
            }
            for (int a = 0; a < 6; ++a)
              for (int b = 0; b + a < 6; ++b)
                for (int c = 0; c + a + b < 6; ++c) {
                  const auto index = kMomentSlot[a * 36 + b * 6 + c];
                  double w = q.weight * px[a] * py[b] * pz[c];
                  cell.moments[0][index] += w;
                  for (int j = 0; j < 3; ++j)
                    cell.moments[j + 1][index] += w * q.normal[j];
                }
          }
          cell.near_targets.assign(
              g->adjacent_nodes.begin() + g->adjacent_offsets[i],
              g->adjacent_nodes.begin() + g->adjacent_offsets[i + 1]);
          cell.near_targets.push_back(i);
          std::sort(cell.near_targets.begin(), cell.near_targets.end());
          auto high = exact ? samples(triangles, *g, i, 16)
                            : pn_samples(nodes, *g, i, 16);
          for (auto target : cell.near_targets)
            cell.near_blocks.push_back(
                integrate(nodes[target], high, eps, kappa));
        }
      });
  for (std::size_t i = 0; i < n; ++i) {
    auto &p = nodes[i];
    auto &c = curved->cells[i];
    p.area = c.area;
    p.source_x = c.center[0];
    p.source_y = c.center[1];
    p.source_z = c.center[2];
    p.has_source_quadrature_center = true;
    p.node_patch_geometry = g;
  }
  g->curved = curved;
  if (eager && n && n <= budget / sizeof(B) / n) {
    curved->pair_blocks.resize(n * n);
    executor.parallel_for(
        n, std::min<std::size_t>(8, executor.capacity()), 1,
        [&](std::size_t, std::size_t begin, std::size_t end) {
          for (auto i = begin; i < end; ++i)
            for (std::size_t j = 0; j < n; ++j) {
              const auto &c = curved->cells[j];
              auto found = std::lower_bound(c.near_targets.begin(),
                                            c.near_targets.end(), i);
              curved->pair_blocks[i * n + j] =
                  found != c.near_targets.end() && *found == i
                      ? c.near_blocks[found - c.near_targets.begin()]
                      : integrate(nodes[i], c.samples, eps, kappa);
            }
        });
  } else if (budget && n) {
    // Cache only accessed pairs. Bounded probing may decline admission when
    // crowded; that changes cost, never the evaluated kernel or accuracy.
    curved->sparse_capacity = std::min(budget / sizeof(CurvedKernelCacheEntry),
                                       n * std::min<std::size_t>(n, 256));
    if (curved->sparse_capacity)
      curved->sparse_blocks =
          std::make_unique<CurvedKernelCacheEntry[]>(curved->sparse_capacity);
  }
  curved->storage_bytes =
      sizeof(*curved) + curved->cells.capacity() * sizeof(CurvedDualCell) +
      curved->pair_blocks.capacity() * sizeof(B) +
      curved->sparse_capacity * sizeof(CurvedKernelCacheEntry);
  for (const auto &c : curved->cells)
    curved->storage_bytes += c.samples.capacity() * sizeof(CurvedDualSample) +
                             c.near_targets.capacity() * sizeof(std::size_t) +
                             c.near_blocks.capacity() * sizeof(B);
  return true;
}
std::array<double, 4> curved_node_pair(const Panel &target, std::size_t i,
                                       const Panel &source, std::size_t j,
                                       double eps, double kappa) {
  (void)i;
  (void)j;
  i = target.node_patch_index;
  j = source.node_patch_index;
  const auto &g = *source.node_patch_geometry->curved;
  if (g.eps_ratio != eps || g.kappa != kappa)
    throw std::runtime_error("curved_dual_kernel_cache_parameter_mismatch");
  if (target.node_patch_geometry.get() == source.node_patch_geometry.get()) {
    if (!g.pair_blocks.empty())
      return g.pair_blocks.at(i * g.cells.size() + j);
    const auto &c = g.cells.at(j);
    auto found =
        std::lower_bound(c.near_targets.begin(), c.near_targets.end(), i);
    if (found != c.near_targets.end() && *found == i)
      return c.near_blocks[found - c.near_targets.begin()];
    if (g.sparse_capacity) {
      const std::size_t key = i * g.cells.size() + j + 1;
      std::size_t hash = key;
      hash ^= hash >> 30;
      hash *= 0xbf58476d1ce4e5b9ULL;
      hash ^= hash >> 27;
      hash *= 0x94d049bb133111ebULL;
      hash ^= hash >> 31;
      for (std::size_t probe = 0; probe < 16; ++probe) {
        auto &slot = g.sparse_blocks[(hash + probe) % g.sparse_capacity];
        auto observed = slot.key.load(std::memory_order_acquire);
        if (observed == 0 && slot.key.compare_exchange_strong(
                                 observed, key, std::memory_order_acq_rel)) {
          try {
            slot.value = integrate(target, c.samples, eps, kappa);
            slot.ready.store(1, std::memory_order_release);
            return slot.value;
          } catch (...) {
            slot.ready.store(2, std::memory_order_release);
            throw;
          }
        }
        if (observed == key) {
          unsigned char ready;
          while ((ready = slot.ready.load(std::memory_order_acquire)) == 0)
            std::this_thread::yield();
          if (ready != 1)
            throw std::runtime_error("curved_dual_cached_quadrature_failed");
          return slot.value;
        }
      }
    }
  }
  return integrate(target, g.cells.at(j).samples, eps, kappa);
}
std::size_t curved_node_storage_bytes(const std::vector<Panel> &p) noexcept {
  return p.empty() || !p.front().node_patch_geometry ||
                 !p.front().node_patch_geometry->curved
             ? 0
             : p.front().node_patch_geometry->curved->storage_bytes;
}
} // namespace TitanPBE::Tabi
