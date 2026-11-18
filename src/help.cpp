#include "arguments.h"
#include <iostream>

int arguments::help_print() {
  if (help == false)
    return 1;
  else {
    cout << R"MANUAL(
================================================================================
                       TITAN ENGINE - USER MANUAL
                 author:nguyenhoanglinh9@duytan.edu.vn
================================================================================

SYNOPSIS
    titan.exe [OPTIONS]
    titan.exe -a arg.in

  DESCRIPTION
    Titan is a high-performance molecular analysis engine for computing
    electrostatic and van der Waals interactions, GBSA/ALPB solvation,
    hydrogen bonds, DCCM network analysis, PCA/FES, allostery, corr-kNN,
    TICA, MSM,
    ZTS minimum-energy paths, DHAM/MBAR reweighting, TPT pathway analysis,
    LJ condensation, LLPS, GIST water thermodynamics, IFP occupancy,
    LSP-MD, and related observables from PDB/XTC/TRR trajectories.

================================================================================
1. CORE INPUT / OUTPUT
================================================================================

SUPPORTED INPUT MODES

  1) Single structure / single-PDB trajectory
     Use:
       -f <file>
       -1pdb
     Typical case:
       all frames are stored in one PDB/GRO-like text file.

  2) XTC/TRR trajectory with topology reference
     Use:
       -s <structure.pdb>
       -traj <traj.xtc|traj.trr>
     Notes:
       -s sets the reference structure and enables XDR mode.
       Coordinates from XTC/TRR are converted from nm to Angstrom.

  3) Multi-trajectory batch mode
     Use:
       -i <input.txt>
     Notes:
       each line in input.txt is one trajectory file. Titan can process
       multiple replicas or multiple trajectory chunks in one run.

  4) Argument-file mode
     Use:
       -a arg.in
     Notes:
       value options must use -option=value; '=' is mandatory for them.
       Boolean/presence switches may stand alone. Examples:
         -1pdb
         -traj=md.xtc
         -ff=AMBER99SB
         -g=1&2

  SYNTAX CONTRACT
     Terminal: value options use -option value; switches may stand alone.
     arg.in: value options use -option=value; switches may stand alone.
     Explicit boolean values remain valid: terminal `-gbsa false`, arg.in
     `-gbsa=false`.
     Unknown or obsolete options, conflicts, and missing values are reported
     before analysis starts; close misspellings include suggested options.

CORE DEPENDENCY CHAIN

  -pca        implies -network
  -allostery  implies -network
  -corr-knn   implies -network
  -tica       implies -network
  -zts        requires -pca or -tica; if both are active, set -zts-domain
  -umap       implies -tica and -network
  -msm        implies -tica and -network
  -dham       implies -msm and -tica; for protein-like traces it also uses
              -network, while LJ jobs switch to the LJ phase-2 feature backend
  -mbar       standalone REMD/HREMD temperature-ladder reweighting;
              domain=tica implies -msm/-tica/-network, domain=pca implies
              -pca/-network, domain=both enables both paths
  -tpt        implies -msm and -tica; phase 0 uses the built MSM TPM +
              stationary distribution directly
  -community=louvain implies -network
  -graph-source=dccm implies -network
  -graph-source=lspmd implies -lspmd
  -graph-source=allostery-gc / allostery-te imply -allostery
  -graph-source=tpt implies -tpt
  -lspmd      runs as its own network-style module and does not need -network

GENERAL OUTPUT BEHAVIOR

  Common files:
    log.txt               Main runtime log
    cal.txt / cal.xvg     Total interaction energy summaries/time-series
    out_*.pdb             Optional coordinate output from -opdb / -opdbsep

  Module-specific output files are created only when the corresponding module
  is enabled. See Chapter 2 for module input/output details.

  Shared performance observability:
    -perf-report=<off|summary|detail>
      Default: off. When enabled, Titan writes cross-module performance
      summaries without replacing existing module-specific performance files.
      summary writes titan_performance_summary.dat and
      titan_performance_modules.csv; detail also writes
      titan_performance_frames.csv.

  Shared uncertainty / convergence diagnostics:
    -uncertainty=<off|block|bootstrap>
      Default: off. When enabled, Titan writes opt-in sidecar diagnostics and
      leaves all point-estimate output schemas unchanged.
    -uncertainty-blocks=<auto|N>
      Default: auto. auto uses deterministic contiguous blocks: insufficient
      below 10 samples, 5 blocks for 10..49, and min(20,sqrt(N)) from 50 up.
    -uncertainty-bootstrap=<N>
      Default: 200. Used only for bootstrap mode, resampling block means.
    -uncertainty-seed=<N>
      Default: 42. Deterministic bootstrap seed.
    Output sidecars include uncertainty_summary.dat and module files such as
      cmap_uncertainty.csv, qnative_uncertainty.csv,
      dihedral_uncertainty.csv, msd_uncertainty.csv, s2_uncertainty.csv,
      membrane_uncertainty.csv, and edecomp_uncertainty.csv.

  Output manifest / dry-run:
    -output-manifest=<true|false>
      Default: false. Writes titan_outputs_manifest.csv and
      titan_run_config_summary.dat for the resolved configuration.
    -dry-run=<true|false>
      Default: false. Implies -output-manifest=true, writes manifest/schema
      sidecars, then exits before coordinate loading and trajectory processing.
    -schema-export=<off|json>
      Default: off. json writes titan_schema.json with the public option
      schema used by the manifest/dry-run layer.

IMPORTANT NOTES

  - When DHAM is not enabled, or DHAM does not converge, the rest of the
    pipeline still works normally. PCA, FES, basin extraction, TICA, MSM,
    LLPS, GIST, and other active modules continue in their standard mode.

  - Default equilibrium weighting uses `-eq-weight-source auto`:
    if DHAM converges, PCA/FES uses DHAM frame weights at the requested
    target temperature; otherwise TITAN falls back to uniform weights.

  - MBAR v1 is standalone-only. It writes `mbar_*` outputs, reweights to a
    target temperature from REMD/HREMD energy ladders, and does not overwrite
    the standard PCA/TICA/DHAM outputs unless you explicitly opt in with
    `-eq-weight-source mbar`.

  - PCA basin extraction writes an auto-generated shell script:
      extract_basins.sh
    Titan does not check whether GROMACS is installed. Please make sure
    `gmx` is available in your environment before running that script.

  - Representative basin structures are chosen as the frames closest to the
    FES sink in PCA space. They are not guaranteed to be the lowest-energy
    frames inside the basin.

================================================================================
2. MODULE INPUTS / OUTPUTS
================================================================================

2.1 Interaction Energy / Force Analysis

  Purpose:
    Compute electrostatic and van der Waals interaction energies, force maps,
    residue decomposition, COM-distance projections, and related observables.

  Main inputs:
    -f / -s / -traj / -i
    -g <group1&group2>
    -ff <forcefield>
    -topo / -topo2 / -common-topo when needed

  Typical outputs:
    cal.txt
    cal.xvg
    edecomp_mm_residue_matrix_total.csv  (with -edecomp=true)
    edecomp_mm_residue_edges.csv         (with -edecomp=true)
    edecomp_mm_residue_ranking.csv       (with -edecomp=true)
    out_dis_com_energy.dat
    out_dis_com.dat

2.2 GBSA / ALPB

  Purpose:
    Add implicit-solvent free-energy terms on top of interaction analysis.

  Main inputs:
    -gbsa or -alpb
    -igb, -ext_diel, -int_diel, -saltcon, -gbsa-temp, -epsilon_in
    -gbsa-sasa-points
    -gbsa-rgbmax <A>|off
    -gbsa-surften <kcal/mol/A^2>
    -gbsa-perf off|summary|detail
    -gbsa-ie-block, -gbsa-ie-min-samples, -gbsa-ie-filter-sigma

  Typical outputs:
    gbsa_snapshots.dat
    gbsa_timeseries_avg.dat
    gbsa_summary.dat
    gbsa_decomposition.dat
    gbsa_performance.dat
    gbsa_performance_frames.csv   (with -gbsa-perf=detail)
    gbsa_ie_diagnostics.csv
    gbsa_alpb_sizes.csv                 (with -alpb)
    edecomp_mmgbsa_residue_map.csv       (with -edecomp-source=gbsa|all)
    edecomp_mmgbsa_residue_ranking.csv   (with -edecomp-source=gbsa|all)
    out_entropy.dat   (auto-enabled by -gbsa for free-energy completeness)

  Notes:
    - GBSA/ALPB are energy modules and are ignored in -nocal mode; remove
      -nocal to compute implicit-solvent free energies.
    - GBSA interaction entropy uses E_MM = E_elec + E_vdw; the reported
      energy kernel timing remains fused with GB pair-polar work.
    - HCT/OBC use analytical pair descreening, Amber intrinsic radii and
      element screen factors. Salt concentration is converted to Amber's
      temperature/dielectric-aware Debye kappa.
    - IE reports status, sample count, lag-1 ESS and block diagnostics.
      Final_dG is nan until the configured minimum sample gate is satisfied;
      filter sigma 0 (default) disables hard outlier filtering.
    - ALPB uses the salt-screened Sigalov expression with dimensionless
      alpha=0.571412 and a per-frame determinant electrostatic size (A_Det)
      for the complex, receptor, and ligand endpoints. It is not a fixed
      empirical length added to the GB denominator. Summary/log provenance
      and all three endpoint sizes are exported for audit.
    - When IE sigma filtering removes samples, the retained ensemble is
      re-centered and the minimum-sample gate is applied again.

2.2b PBE

  Purpose:
    Solve Poisson-Boltzmann electrostatics on each processed frame.
    The v1 implementation uses CPU double precision, a two-dielectric model,
    VDW/SAS/SES/DelPhi-like SES/DelPhi-like VDW surface classification,
    Stern-aware linear ion screening, selectable Coulombic/dipolar/zero
    boundaries, linear PB by default, optional nonlinear PB (damped Newton
    by default, Picard/SOR on request) on the fine grid, and reaction-field
    energy reporting.
    `-pbsa` wraps this path for production two-molecule MM/PBSA: it enables
    PBE deltaG_solv endpoint solves, force-field elec/vdw interaction energy,
    SASA nonpolar energy, optional interaction entropy, and per-frame binding
    energy CSV output. The `TitanSurface` workspace is shared by GBSA/PBSA
    SASA and BIE exposed-panel construction; `TitanSASA` remains the
    compatibility wrapper for per-atom SASA/SA nonpolar energy. Each workflow
    keeps its own residue grouping and output files.
    Backend selection is explicit: bare -pbe, terminal -pbe grid, and
    arg.in -pbe=grid; compatibility
    aliases -pbe=delphi, -pbe=pydelphi, and -pbe=fd use the established
    finite-difference grid implementation; -pbe=bie uses the separate
    clean-room boundary-integral equation (BIE) backend for linear PB only.
    The historical TABI/JCP name denotes the formulation lineage; `tabi` and
    `-pbe-tabi-*` remain accepted as compatibility aliases for archived input
    files. The BIE backend defaults to the JCP 2013 coupled formulation with a
    clean-room triangle surface. Self-panel analytic singular integration is
    available as a hidden validation path; one-ring Duffy remains
    experimental/report-only, and `auto` remains centroid until analytic gates
    pass. The legacy scalar `titan-v1`
    formulation is retained only as a hidden debug path. `jcp2013-centroid` is the paper-style
    direct centroid-collocation path. The public production selection
    `-pbe=bie -pbe-bie-fast-operator=p1` uses the coupled JCP 2013 BIE path,
    hierarchical Cartesian multipole acceleration, and sparse near-field
    ILU(0) by default. Lower-level sum and operator controls remain available
    for regression and diagnostics; `tabi2-nodepatch` remains a hidden report
    path.
    `pbe_timing.csv` reports BIE mesh, diagonal, RHS, GMRES, matvec,
    BIE-formulation, and tree counters; `-pbe-timing=summary|detail` also
    writes `pbe_memory_profile.csv`, while `detail` writes
    `pbe_tabi_profile.csv`.
    The hidden `-pbe-bie-surface-only=true` feasibility mode builds and
    profiles the BIE surface, then exits before RHS, operator,
    preconditioner, or GMRES construction; it does not report an energy.

  Main inputs:
    -pbe[=grid|bie]
    -pbe-bie-fast-operator=auto|direct|p1
    -pbe-scale, -pbe-margin, -pbe-gridbox
    -pbe-model, -pbe-biomodel, -pbe-dielectric
    -pbe-int-diel, -pbe-ext-diel, -pbe-salt, -pbe-temp
    -pbe-surface, -pbe-probe, -pbe-boundary, -pbe-charge-assignment,
    -pbe-radius-convention
    -pbe-ion-access, -pbe-ion-radius
    -pbe-dgsolv
    -pbsa
    -pbsa-components, -pbsa-decomp, -pbsa-residue-frames
    -pbsa-ie, -pbsa-ie-block, -pbsa-ie-min-samples
    -pbsa-ie-filter-sigma, -pbsa-perf, -pbsa-grid-reuse
    -pbsa-sasa-points
    -pbe-max-iter, -pbe-rms, -pbe-dphi, -pbe-omega
    -pbe-nl-max-iter, -pbe-nl-inner-iter, -pbe-nl-stagnation
    -pbe-nl-block, -pbe-nl-dphi, -pbe-nl-relax, -pbe-nl-clip
    -pbe-nl-continuation, -pbe-nl-line-search, -pbe-nl-residual
    -pbe-nl-solver, -pbe-nl-inner, -pbe-nl-multigrid
    -pbe-gaussian-exp, -pbe-gaussian-sigma, -pbe-density-cutoff,
    -pbe-surface-cutoff, -pbe-gap-diel
    -pbe-focus, -pbe-multigrid, -pbe-focus-levels, -pbe-focus-margin
    -pbe-mg-levels, -pbe-mg-pre, -pbe-mg-post, -pbe-mg-coarse-iter,
    -pbe-mg-accel
    -pbe-map, -pbe-map-frames, -pbe-max-grid, -pbe-kernel
    BIE advanced/debug: -pbe-bie-sum=auto|direct|treecode,
      -pbe-bie-geometry=triangle|point-panel,
      -pbe-bie-diagonal=solid-angle|flat|off,
      -pbe-bie-surface-points=<8..4096>,
      -pbe-bie-sphere-mesh=auto|latlong|icosphere,
        auto: icosphere for triangle/vdw, legacy latlong for SAS.
      -pbe-bie-sphere-patch-clip=on|off (default on): clip each atom's
        sphere-patch triangles at its neighbours' caps; off keeps each
        whole triangle unless its centroid or all three corners are buried
        (the v0.5 rule).
        Default 96 is coarse; check energy convergence with resolution.
        Latlong saturates by 3969 points (16128 triangles/atom), even with
        -pbe-bie-highres-surface=true. Use icosphere for further refinement.
      -pbe-bie-surface-only=true|false,
      -pbe-bie-ses-max-grid-vertices=<4000000..100000000>,
      -pbe-bie-taylor-order=3|4,
      -pbe-bie-derivatives=analytic|finite-diff-debug,
      -pbe-bie-discretization=centroid|node-patch,
      -pbe-bie-formulation=titan-v1|jcp2013|jcp2013-centroid|
        jcp2013-treecode|tabi2-nodepatch|calibrated-debug,
      -pbe-bie-energy-projection=boundary|born-check|charge-potential-debug,
      -pbe-bie-fmm-m2l=auto|off|single-matvec-analytic,
      -pbe-bie-fmm-engine=cartesian|cartesian-full-experimental|kifmm-lite|
        kifmm-local|kifmm-lite-experimental|
        kifmm-channel-hybrid-experimental,
      -pbe-bie-fmm-cartesian-order=1|4 (required by cartesian-full-experimental),
      -pbe-bie-fmm-cartesian-error-audit=off|channel-shadow|
        channel-shadow-retry-global-theta015 (opt-in O1 node-patch),
      -pbe-bie-precond=auto|none|diagonal|block|block-budgeted|
        coarse-galerkin-probe|fmm-block-jacobi|fmm-overlap-schwarz|
        fmm-nearfield-ilu0|fmm-nearfield-ilu0-coarse|
        distance-knn-overlap-schwarz|
        distance-knn-ilu0|classic-nearfield-ilu0,
        classic-nearfield-ilu0 is ILU(0) on the classic treecode's own near
        list, capped by -pbe-bie-precond-ilu-neighbor-cap. It needs
        -pbe-bie-sum=treecode and salt, but not the FMM M2L path, so it is
        the ILU for curved node-patch.
      -pbe-bie-energy-far-eta=<0|2..1000> (default 32): curved node-patch
        reaction energy sums a cell farther than eta cell radii from an atom
        as one point (centroid, area, mean normal) instead of all its
        samples. 0 keeps the exact sample sum. The error falls as
        (1/eta)^2; report the value used.
      -pbe-bie-curved-sample-order=<2|3|4> (default 2): Gauss order per
        direction of the samples each curved dual cell stores, about
        12*order^2 per cell. They dominate curved node-patch memory.
      -pbe-bie-ses-charge-refine=<off|alpha in (0,4]> (default off):
        patch-ray-resolution remeshing grades the target edge towards
        shallow charges, edge(x) = clamp(alpha * distance to the nearest
        charge with |q| >= 0.1 e, min edge, uniform edge). A charge d below
        the surface needs edges of order d near its foot point.
      -pbe-bie-ses-charge-refine-min-edge=<0.05..2.0> (default 0.2 A):
        the smallest target edge the charge refinement may ask for.
      -pbe-bie-ses-charge-refine-min-charge=<0..2> (default 0.1 e): charges
        smaller in magnitude do not shape the mesh.
      -pbe-bie-curved-near-zone=<1..16> (default 8): p1 FMM route only.
        Self, one-ring and pairs within this many cell radii of the source
        vertex are integrated over the curved node-patch cell; every other
        pair uses one point per cell, as the multipole far field does.
      -pbe-bie-precond-ilu-neighbor-cap=<32..1024>,
      -pbe-bie-precond-ilu-drop-tol=<0..1e-2>,
      -pbe-bie-precond-ilu-regularization=adaptive|off (default adaptive),
      -pbe-bie-precond-coarse-aggregate-cap=<4..64>,
      -pbe-bie-precond-coarse-verify=off|full-small,
      -pbe-bie-precond-ilu-regularization-max=<0..1e-2>,
      -pbe-bie-precond-memory-budget-mb=<16..8192>,
      -pbe-bie-near-integration=auto|centroid|analytic_singular|
        duffy_adaptive,
        Salt-free sphere-patch/triangle/vdw: auto uses curved self/one-ring
        Poisson quadrature. Explicit centroid retains the legacy rule.
      -pbe-bie-quad-buffer=auto|off|gauss7|adaptive,
      -pbe-bie-leaf-near-policy=auto|exact|smooth-node|large-protein,
      -pbe-bie-leaf-panel-kernel=auto|scalar|avx2-gather|tile,
      -pbe-bie-interaction-mode=auto|cached|onthefly,
      -pbe-bie-expansion-center=legacy|area-centroid|aabb-center,
      -pbe-bie-fragmentation-safety=off|audit|split,
        -pbe-bie-mac-safety=off|audit|physical-gap,
        -pbe-bie-mac-gap-factor=<0.5..4.0>,
        -pbe-bie-moment-audit=off|sampled|full-small,
        -pbe-bie-convergence-safety=off|audit|split,
        -pbe-bie-convergence-rho-cap=<0.0..2.0>,
        -pbe-bie-mac-aspect-cap=<0.0..50.0>,
        -pbe-bie-convergence-estimator=off|moment-tail|hybrid,
        -pbe-bie-convergence-tail-threshold=<1..1e12>,
        -pbe-bie-convergence-estimator-error-cap=<1e-8..1.0>,
        -pbe-bie-convergence-unstable-rho-cap=<0.05..2.0>,
        -pbe-bie-fmm-local-plan-builder=rescan|cached-metrics,
        -pbe-bie-convergence-calibration=calibrated-tail-v1|consecutive|aspect|consecutive-aspect,
        -pbe-bie-convergence-estimator-model=calibrated-estimator-v1|channel-tail|channel-tail-eps,
        -pbe-bie-expansion-audit=off|sampled|full-small,
        -pbe-bie-expansion-audit-sample-count=<64..2048>,
        -pbe-bie-expansion-audit-max-panels=<128..8192>,
        -pbe-bie-matvec-audit=off|sampled|sampled-detail|targets|full-small,
        -pbe-bie-matvec-audit-sample-count=<128..4096>,
        -pbe-bie-node-patch-operator-audit=on|off,
      -pbe-bie-progress=off|auto|detail,
      -pbe-bie-progress-interval=<1..300>,
      -pbe-bie-gmres-restart=<8..128>,
      -pbe-bie-true-residual-probe=restart|interval|off,
      -pbe-bie-true-residual-probe-interval=<1..128>
      -pbe-bie-primary-surface=sphere-patch|grid-ses|analytic-ses,
      -pbe-bie-analytic-ses-engine=patch-ray-experimental|
        patch-ray-resolution|current|current-dense
        (default patch-ray-experimental),
      -pbe-bie-ses-grid-spacing=<0.25..4.0> (default 0.75)
    grid-ses is an opt-in watertight solvent-excluded surface. It applies a
    probe-radius morphological closing to the VDW volume on a Cartesian grid,
    extracts a welded triangle mesh, and rejects open/nonmanifold output before
    the BIE solve. The default remains sphere-patch.
    analytic-ses independently constructs exact contact, toroidal, and
    reentrant probe patches from expanded atom spheres, intersects grid rays
    with those patches, and topology-audits the welded mesh before solving.
    The engines differ in where the inside/outside sign comes from, and
    that decides where the mesh converges. current takes the sign from a
    26-direction sampling of the probe ball, a direction set that does not
    grow as the spacing falls, so its vertices stay outside the true surface
    by up to 0.09 A at probe 1.4 A at every spacing and are never checked
    against the analytic patches; its status certifies a closed mesh only.
    current-dense is the same engine sampling 98 directions instead of 26.
    That is a finer angular resolution, not a finer grid, so it lowers the
    floor rather than removing it: on an isolated 1.5 A sphere the mean
    radial error falls from 0.023 to 0.005 A at h=0.3 and from 0.025 to
    0.007 A at h=0.15. The sign test costs about twice as much, which is a
    small share of the build.
    The two cross over near h=0.7 and current-dense is the worse choice above
    it. current's sampling error points outward and the grid's points inward,
    so on a 1.7 A sphere its signed mean error is +0.025 A at h=0.3, +0.001 A
    at h=0.8 where the two cancel, and -0.051 A at h=1.25. current-dense
    removes the outward half, leaving nothing to cancel the inward half: its
    mean error is 0.006 A against 0.025 at h=0.3 but 0.094 A against 0.063 at
    h=1.25. Use current-dense at h<=0.6 and current above it. Neither has
    certified vertices.
    patch-ray-experimental takes the sign from certified ray parity and
    projects every vertex onto a bounded analytic patch, so its vertices are
    on the surface to rounding. Keep current to reproduce earlier runs.
    patch-ray-experimental is a clean-room analytic SES extractor and the
    default. It visits patches, builds compact three-axis ray crossings, and
    leaves the older current engine reachable and unchanged. Any
    odd ray row invalidates only the vertices that use that row. Certified,
    unanimous three-axis parity is authoritative in the compact solvent shell;
    exact seam/tangency vertices use deterministic analytic symbolic
    perturbation. Any remaining morphology fallback, unresolved odd or
    uncertified quartic row, or failed all-vertex analytic projection makes the
    experimental engine fail closed. Unequal-radius torus arcs use
    the minor contact arc, and spindle-torus quartic roots must also pass a
    geometric probe-radius residual. Boundary magnitude and projection use
    sparse active-cell storage plus an atom-local patch CSR. Patch events are
    written to CSR by two merge passes without a full merged-event copy, and
    final vertex projection uses the TITAN executor and a vertex-only
    transaction. Short-edge collapse builds an immutable-source candidate
    without copying the full panel mesh. Projection-created zero-area faces
    and locally inverted faces are repaired transactionally; ULP-scale seam
    duplicates are welded by a local-scale tolerance. Every final face must
    pass a minimum analytic-normal orientation certificate, and all
    broad-phase non-adjacent triangle AABB pairs are checked and counted for
    self-intersection;
    generic diagonal flips are disabled because planar quality alone cannot
    certify a curved analytic-patch seam. Final gates report connected
    components, Euler characteristic, genus, filled cavities, sampled analytic
    distance/residual/normal error, the all-vertex and all-face certificates,
    repair scratch/event peak bytes, and deterministic repair rollback. A
    coarse mesh that reports analytic_self_intersection must be rerun with a
    smaller SES grid spacing; the certificate cannot be disabled. No lazy
    ray-bucket index is built. SES grid spacing controls geometric
    discretization independently of the GMRES tolerance. The 0.75 A default
    is a throughput setting, not a universal energy-accuracy guarantee;
    quantitative work should report an h-refinement check and reduce the
    spacing until the observable is stable.
    patch-ray-resolution uses the same certified ray-parity signs and
    marching tetrahedra, then accepts the surface at the resolution of the
    grid instead of certifying every patch. Odd ray rows and fallback signs
    are reported, not refused. Each vertex moves onto its closest analytic
    patch point unless that turns or flattens an incident triangle; any
    triangle pair that still intersects has its vertices returned to their
    marching positions, which cannot intersect. Solvent slabs and molecular
    plates one grid cell thick are merged, so features thinner than h do not
    reach the solver as two facing sheets. The mesh is then remeshed
    isotropically onto the analytic SES (edge split, collapse, Delaunay flip
    and tangential relaxation; any operation behind an intersecting pair is
    undone), which gives well-shaped elements, about half the nodes at the
    same spacing, and much faster GMRES convergence on large proteins. The
    result stays closed, oriented and free of self-intersections.
      -pbe-bie-ses-remesh-edge=off|0|<0.25..4.0> (default 0 = the SES grid
        spacing): target edge of that remeshing in A; off keeps the
        marching-tetrahedra mesh.
      -pbe-bie-ses-remesh-iterations=<1..12> (default 6): passes of split,
        collapse, flip and tangential relaxation in that remeshing.
    Large complexes that patch-ray-experimental refuses should use this
    engine and report an h-refinement check.
    Node-patch treecode runs a fail-closed, spatially stratified pre-GMRES
    operator audit by default. It starts at 64 targets and expands to 256 or
    512 only when the sampled error is ambiguous. Advanced production runs
    may set -pbe-bie-node-patch-operator-audit=off to remove this guard;
    convergence then no longer certifies agreement with sampled direct rows.
    -pbe-bie-matvec-audit=targets compares the solved operator with direct
    rows after GMRES, on every surface and cell: m = min(sample count, N)
    targets at the panel indices floor((2k+1)N/(2m)), both equations (2m
    rows), giving the sampled operator discrepancy ||[(A_acc - A_dir) u]_S|| /
    ||(A_dir u)_S|| and the sampled direct residual ||b_S - (A_dir u)_S|| /
    ||b_S|| of the solution u.
    fmm-nearfield-ilu0-coarse is an opt-in balanced two-level
    preconditioner. It reuses adaptive occupied octree nodes for an
    area-weighted coarse space and combines the coarse correction with the
    existing near-field ILU(0). Its setup captures A_near*P while assembling
    ILU, adds spatial-octant far monopoles, and audits the hybrid operator with
    one full FMM probe. A failed audit falls back to exact Galerkin
    setup; a setup or budget failure falls back to ILU(0). Auto and
    fmm-nearfield-ilu0 keep the current single-level pipeline. Setting
    coarse-verify=full-small deliberately requires exact fallback and repeats
    the exact coarse columns for verification.
    hidden PQR parity only: -pbe-parity-dump, -pbe-parity-map-dump,
    -pbe-test-charge-grid, -pbe-test-charge-grid-diff,
    -pbe-test-surface-map,
    -pbe-test-midpoint-epsilon, -pbe-test-midpoint-diff,
    -pbe-test-vdwms-remap,
    -pbe-test-boundary-map, -pbe-test-boundary-convention,
    -pbe-test-field-units, -pbe-test-operator-convention,
    -pbe-test-rhs-map, -pbe-test-screen-map, -pbe-test-denom-map,
    -pbe-test-nl-source, -pbe-test-newton-derivative,
    -pbe-test-energy-convention
    In arg files use -flag=value; on the CLI use -flag value.
    v1 requires normal charge/radius initialization; do not combine with -nocal
    Zero-radius polar hydrogens from supported force fields use a small
    Bondi-style PBE surface fallback radius while charges remain mandatory.
    BIE v1 uses an explicit linear-BIE convention with Angstrom kernels,
    outward normals, solid-angle diagonal terms, and final Coulomb conversion.
    BIE writes `pbe_tabi_formulation_audit.csv` for sign/scale, kappa,
    dielectric contrast, energy projection, and fallback diagnostics.
    BIE audit/profile rows also report surface_generator, discretization,
    requested/effective preconditioner, near_integration/self/one-ring ASI
    counters, and paper-mode guard status when an
    experimental path is pinned.
    It rejects nonlinear, RPB, Gaussian dielectric/surface, focus, multigrid,
    and grid-map modes. Linear MM/PBSA is supported through three BIE
    reaction-field endpoint solves (complex, receptor, and ligand), including
    molecule, residue, atom, and hybrid decomposition outputs.

  Typical outputs:
    pbe_energy.csv
    pbe_summary.dat
    mm_pbsa_energy.csv, mm_pbsa_summary.dat, mm_pbsa_by_trajectory.csv,
    mm_pbsa_ie_diagnostics.csv, and mm_pbsa_performance.dat when -pbsa is
    enabled
    optional: mm_pbsa_decomposition.dat and mm_pbsa_residue_*.csv when
    -pbsa-decomp is molecule/residue; mm_pbsa_unit_*.csv for hybrid/auto
    protein-ligand decomposition
    energies are reported in kJ/mol and kcal/mol; kT is retained for continuity
    optional: pbe_phi_*.cube, pbe_epsilon_*.cube, and pbe_ion_*.cube
    optional: pbe_probe_sites.csv when DelPhi focus/probe fixture knobs are used
    optional: pbe_timing.csv and pbe_memory_profile.csv with
    -pbe-timing=summary|detail; memory rows include both nonnegative and
    signed RSS deltas. pbe_tabi_profile.csv is written with detail mode.
    hidden PQR parity only: pbe_parity_stage.csv and pbe_parity_points.csv
    barnase PQR forensic mode additionally writes pbe_barnase_forensic_inputs.csv,
    pbe_barnase_surface_audit.csv, pbe_barnase_operator_audit.csv, and
    pbe_atom_phi_audit.csv; ISCM parity with -pbe-parity-dump=all writes
    pbe_iscm_surface_charge_samples.csv and pbe_iscm_energy_terms.csv;
    -pbe-parity-map-dump=samples writes
    pbe_field_samples.csv, pbe_boundary_samples.csv,
    pbe_operator_stage.csv, pbe_charge_grid_audit.csv,
    pbe_nwt_trace.csv, and pbe_nwt_solver_progress.csv for DelPhi NWT
    nonlinear parity rows. The hidden
    DelPhi map extractor script writes
    oracle-side pbe_oracle_field_samples.csv, pbe_oracle_boundary_samples.csv,
    pbe_oracle_boundary_map.csv, pbe_oracle_boundary_inputs.csv,
    pbe_oracle_nwt_trace.csv, pbe_oracle_nwt_solver_summary.csv,
    pbe_oracle_midpoint_epsilon_samples.csv,
    pbe_oracle_atom_radius_audit.csv, a replayable sparse
    pbe_oracle_surface_map.csv with inside semantics, and
    barnase_stage_oracle.tsv under test/tmp for stage comparison. With
    -FullMidpointEpsilon it writes a full replayable midpoint-epsilon oracle
    for one salt only, default 0.15 M, selectable with -FullMidpointSalt.
    With -FullIscmEnergyOracle it also writes
    pbe_oracle_energy_components.csv, pbe_oracle_iscm_full.csv, and
    pbe_oracle_iscm_energy_terms.csv for final DelPhi ISCM convention checks.
    Phase 6d extends the hidden exporter with -FullGaussianOracle,
    -FullRpbOracle, -FullFocusProbeOracle, and -Fixture. The Gaussian density
    oracle path is strict-capable for the 5tif Gaussian-density fixture; RPB
    source comparison is report-first until the remaining grad-epsilon/source
    convention is locked. test/pbe_timing_compare.ps1 writes
    pbe_timing_compare.csv for TITAN vs pyDelPhi wall-clock/profile rows.
    Oracle CSVs, pyDelPhi/DelPhi runtimes, and the oracle provenance manifest
    are test-only artifacts; production PBE does not read or call them.
    DelPhi-named modes are compatibility labels, not imported implementation.
    test/pbe_nonlinear_nwt_trace_compare.ps1 compares TITAN NWT rows against
    the pyDelPhi oracle in fixed stage order, and
    test/pbe_nwt_solver_progress_smoke.ps1 compares block RMS/max-dphi progress
    before Gaussian/RPB/focus strict.

2.3 Hydrogen Bonds

  Purpose:
    Detect hydrogen bonds between the selected molecular groups.

  Main inputs:
    -hb
    optional: -hb side-chain, -ahb, -dhb

  Typical outputs:
    stat_hydrogenbonds.dat
    stat_average_HB.dat
    out_list_res_HB_<group>.dat
    out_hydrogen_bonds.dat
    out_hydrogen_bonds_receptor.dat
    out_hydrogen_bonds_ligand.dat

2.3k HBond / Salt / Water Kinetics

  Purpose:
    Track primitive hydrogen bonds, salt bridges, and water bridges as
    time-resolved on/off events, then export residence / lifetime statistics
    plus continuous and intermittent correlation functions.

  Main inputs:
    -hbk
    -hbk-kind <all|hbond|salt|water>
    -hbk-domain <inter|intra|both>
    -hbk-maxlag <N>
    -hbk-hbond-cutoff <A>
    -hbk-angle <deg>
    -hbk-salt-cutoff <A>
    -hbk-water-cutoff <A>

  Typical outputs:
    hbond_kinetics_summary.dat
    hbond_kinetics_hbond_pairs.csv
    hbond_kinetics_salt_pairs.csv
    hbond_kinetics_water_pairs.csv
    hbond_kinetics_hbond_ct.dat
    hbond_kinetics_salt_ct.dat
    hbond_kinetics_water_ct.dat
    hbond_kinetics_hbond_lifetimes.dat
    hbond_kinetics_salt_lifetimes.dat
    hbond_kinetics_water_lifetimes.dat

  Notes:
    - `-hbk` is a new kinetics module. It does not replace legacy `-hb`,
      `-ifp`, or the classic `global_water_bridges_*` outputs.
    - V1 supports `inter`, `intra`, and `both`; the default is `both`.
    - `inter` means molecule 1 <-> molecule 2.
    - `intra` means molecule 1, plus molecule 2 only outside `-nocal`.
    - Primitive keys are not collapsed to residue-pairs:
      hbond = donor / hydrogen / acceptor
      salt = acidic atom / basic atom
      water = water index / solute atom A / solute atom B
    - Water-bridge kinetics are exported only when a water trajectory is
      available. If water is requested but unavailable, TITAN warns and writes
      header-only water files instead of failing the whole run.
    - Canonical syntax:
      arg.in uses `-hbk-kind=water`
      CLI uses `-hbk-kind water`

2.3a DSSP Secondary Structure

  Purpose:
    Compute DSSP-like secondary-structure states per residue and aggregate
    them across trajectories as mean +/- SEM, plus a consensus text string
    suitable for martinize2-style workflows.

  Main inputs:
    -dssp                enable DSSP output
    -no-dssp             disable DSSP output
    -dssp-hbond-cutoff <val>
    -segment-config <file>
    protein/protein-user molecule types

  Typical outputs:
    dssp_mol1.dat
    dssp_mol1_content.dat
    dssp_mol1_segments.csv
    dssp_mol1_bridges.csv
    dssp_mol1_ladders.csv
    dssp_mol2.dat        only outside -nocal and only if molecule 2 is
                         protein/protein-user
    dssp_mol2_content.dat
                         only outside -nocal and only if molecule 2 is
                         protein/protein-user

  Notes:
    - In -nocal mode, molecule 2 DSSP is evaluated only when another enabled
      analysis explicitly loads molecule 2 coordinates.
    - TITAN auto-skips DSSP if topology lacks a DSSP-compatible protein
      backbone signature, which avoids meaningless output on LJ/monoatomic jobs.
    - For each residue, Titan writes the dominant state, state fractions,
      SEM across input trajectories, and one `martinize2_dssp` consensus
      string built from the highest-population state per residue.
    - The *_content.dat file reports, for each secondary-structure state,
      1. protein-average content
      2. per-residue occupancy +/- SEM
    - Trajectory aggregation is two-stage:
      1. frame fractions are averaged within each input trajectory
      2. Titan reports mean +/- SEM across trajectories
    - A segment config adds entity/segment state content plus bridge
      orientation, registry, and ladder-length distributions.

2.3b RMSD / RMSF

  Purpose:
    Compute a PBC-safe trace-atom RMSD time series and per-trace-atom RMSF
    profile after Kabsch alignment.

  Main inputs:
    -rms                enable RMSD + RMSF
    aliases: -rmsd, -rmsf

  Trace atoms:
    atomistic protein/protein-user: CA
    atomistic RNA/DNA: C4' (fallback C4*)
    MARTINI: BB (fallback BB2)

  Typical outputs:
    rmsd.dat            one file per trajectory; indexed if multiple inputs
                        share one output folder
    rmsf.dat            one file per trajectory; RMSD is combined over all
                        active trace atoms, RMSF is split by molecule

  Notes:
    - In -nocal mode, only molecule 1 is analyzed.
    - Outside -nocal, Titan builds one joint fit set from all supported
      trace atoms in molecule 1 + molecule 2.
    - RMSD is reported for the combined fit set.
    - RMSF is reported separately for molecule 1 and molecule 2 in the same
      RMSF file when both contribute trace atoms.
    - If no explicit PBC mode is requested, Titan auto-enables -pbci 0 for
      RMSD/RMSF and logs that decision.

2.3c Rg2

  Purpose:
    Compute a PBC-safe trace-atom radius of gyration time series for the same
    trace atom set used by RMSD/RMSF.

  Main inputs:
    -rg2                enable Rg2 analysis
    -rg2-hist-bins <N>  histogram bin count for rg2_hist.dat. Default: 100

  Outputs:
    rg2.dat             per-frame frame/time_ps/Rg^2/Rg time series
    rg2_summary.dat     mean/std/min/max for Rg^2 and Rg
    rg2_hist.dat        histogram over Rg^2
    rg2_all.dat         aggregated per-frame time series across multi-input runs
    rg2_summary_all.dat per-trajectory summaries plus a global ALL row
    rg2_hist_all.dat    histogram over Rg^2 using all frames from all inputs

  Units:
    - Rg^2 is reported in A^2.
    - Rg is reported in A.

  Notes:
    - In -nocal mode, only molecule 1 is analyzed.
    - Outside -nocal, Titan builds one joint trace set from molecule 1 +
      molecule 2, matching RMSD/RMSF trace selection.
    - If no explicit PBC mode is requested, Titan auto-enables -pbci 0 for
      Rg2 and logs that decision.

2.3d LJ Condensation / Cluster Density

  Purpose:
    Analyze monoatomic or generic Lennard-Jones-like systems through
    cluster detection on molecule 1, then report condensation observables
    such as largest-cluster Rg and radial density profile rho(r).

  Main inputs:
    -lj
    -lj-cut <val>          cluster cutoff in Angstrom; default 5.0
    -lj-min-cluster <N>    minimum size counted as a cluster; default 2
    -lj-rho-bin <val>      rho(r) bin width in Angstrom; default 1.0
    -lj-rho-max <val>      rho(r) max radius in Angstrom; default auto
    -lj-solid-q6 <val>     q6 threshold for solid-like atoms; default 0.40
    -lj-solid-min-cn <N>   minimum CN for dense/solid-like atoms; default 6
    -lj-crystal-tol <val>  q4_bar/q6_bar crystal-class tolerance; default 0.08

  Typical outputs:
    lj_timeseries.dat      one file per trajectory; indexed if multiple inputs
                           share one output folder
    lj_cluster_rho_r.dat   trajectory-averaged radial density profile around
                           the largest-cluster centroid
    lj_features.dat        phase-2 feature matrix used by TICA/MSM for LJ jobs
    lj_summary.dat         trajectory summary with mean +/- SEM observables

  Notes:
    - LJ analysis always reads molecule 1 only, both in -nocal and outside it.
    - All atoms of molecule 1 are used; no protein/RNA trace-atom filter is
      applied.
    - TITAN auto-skips DSSP in LJ/monoatomic jobs when no protein backbone
      signature is found in the topology.
    - Timeseries columns include bulk density, number of clusters,
      largest-cluster size/fraction, Rg, max cluster radius, dense-region
      q4/q6, Lechner-Dellago q4_bar/q6_bar, solid-like fraction, largest
      solid-cluster fraction, and FCC/HCP/BCC/ICO-like fractions.
    - rho(r) is reported in atoms / nm^3 as a function of radius in Angstrom.
    - If -lj is combined with -tica/-msm and no DCCM trace atoms exist, TITAN
      automatically switches TICA input to LJ phase-2 features:
      bulk density, largest-cluster fraction, normalized Rg/Rmax, CN
      statistics, dense-region q4/q6 + q4_bar/q6_bar, solid-like fraction,
      FCC/HCP fractions, and near-core shell enrichments.
    - If no explicit PBC mode is requested, Titan auto-enables -pbci 0 for LJ
      analysis and logs that decision.

  2.3e Structure Factor / Pair Correlation

  Purpose:
    Build streaming RDF / pair-correlation histograms on molecule 1 and derive
    isotropic structure factors S(q) at finalize time.

  Main inputs:
    -sq
    -sq-channel <all|trace|backbone|sticker>
    -sq-rdf-bin <A>
    -sq-rdf-max <A|auto>
    -sq-q-bin <A^-1>
    -sq-q-max <A^-1|auto>

  Typical outputs:
    sq_summary.dat
    sq_rdf_<channel>.dat
    sq_structure_factor_<channel>.dat
    sq_rdf_<channelA>__<channelB>.dat
    sq_structure_factor_<channelA>__<channelB>.dat

  Notes:
    - SQ analysis always reads molecule 1 only.
    - `all` uses all molecule-1 atoms/beads.
    - `trace` reuses the RMS/Rg trace-selection rules.
    - `backbone` uses CA, C4', C4*, BB, BB1, BB2, and BB3 atoms/beads.
    - `sticker` reuses the -llps-sticker bead list.
    - TITAN only exports RDF / S(q) from valid periodic condensed-phase frames.
      Invalid / non-periodic boxes are skipped with a warning.
    - S(q) is derived from RDF using an isotropic Fourier transform with a
      Lorch window to reduce finite-cutoff ripple.
    - If no explicit PBC mode is requested, Titan auto-enables -pbci 0 for SQ
      analysis and logs that decision.

  2.4 DCCM Network

  Purpose:
    Build an aligned CA/backbone trajectory and compute dynamic
    cross-correlation networks.

  Main inputs:
    -network
    a protein-like trajectory with stable residue mapping

  Typical outputs:
    network_edges.csv
    network_nodes.csv
    dccm_equilibrium_context.dat     (when -eq-weight-source mbar resolves)
    allostery_equilibrium_ranking.csv (when -allostery and -eq-weight-source mbar resolve)

  Integration note:
    `-eq-weight-source mbar` only adds equilibrium/context sidecars for DCCM
    and allostery. It does not alter correlation, MI, GC, or TE calculations.

2.4a Community Detection

  Purpose:
    Partition the DCCM graph into residue communities using Louvain modularity
    optimization.

  Main inputs:
    -community=louvain
    implies -network

  Typical outputs:
    network_communities.csv
    network_community_summary.csv
    network_supergraph.csv
    network_community_id_bfactor.pdb
    network_participation_coeff_bfactor.pdb

2.4b Unified Sparse Graph Analytics

  Purpose:
    Run one shared sparse-graph analytics layer on top of internal TITAN graph
    sources from DCCM, LSP-MD, thresholded allostery GC/TE, and TPT.

  Main inputs:
    -graph
    -graph-source <dccm|lspmd|allostery-gc|allostery-te|tpt>
    -graph-community <auto|off|louvain>
    -graph-filter-topk <N>
    -graph-filter-component <all|largest>

  Typical outputs:
    graph_summary.dat
    graph_<source>_edges.csv
    graph_<source>_nodes.csv
    graph_<source>_communities.csv
    graph_<source>_community_summary.csv
    graph_<source>_supergraph.csv
    graph_<source>_bottleneck_edges.csv
    graph_<source>_bottleneck_nodes.csv

  Notes:
    - V1 consumes only in-memory graph data already produced inside TITAN.
      It does not read standalone CSV edge lists.
    - `dccm`, `lspmd`, and `allostery-gc` can run Louvain communities.
    - Directed sources (`allostery-te`, `tpt`) skip community detection in V1
      and export bottleneck-style node/edge rankings instead.
    - `-community=louvain` is treated as a compatibility alias onto the new
      graph core for the DCCM source.
    - Canonical syntax:
      arg.in uses `-graph-source=dccm`
      CLI uses `-graph-source dccm`

2.4c CMap Side-chain Contact Map

  Purpose:
    Export residue contact-probability matrices and sparse edge summaries from
    trajectory contact occupancies.

  Main inputs:
    -cmap
    -cmap-scope <mol1|interface>
    -cmap-mode <sidechain-com|sidechain-heavy|cb|ca|bb>
    -cmap-cut <A>
    -cmap-domain <intra|inter|both>
    -cmap-min-seqsep <N>

  Typical outputs:
    contact_probability_matrix.csv
    contact_probability_edges.csv
    contact_probability_summary.dat
    contact_count_timeseries.csv
    contact_residue_occupancy.csv

  Notes:
    - sidechain-com is the default mode: mass-weighted side-chain COM at
      distance <= 6.5 A; glycine (no side chain) uses its CA.
    - sidechain-heavy remains an explicit atom-pair mode with default 4.5 A.
    - Distances use the minimum image when the frame has a valid periodic
      box, plain distances otherwise (counted in the summary as
      frames_without_box_plain_distances).
    - interface produces a rectangular mol1 x mol2 matrix and works in -nocal.
    - backbone is accepted as an alias for bb.
    - Canonical syntax:
      arg.in uses `-cmap-scope=interface`
      CLI uses `-cmap-scope interface`

2.4c-i Configurable Interface-State Analysis

  Purpose:
    Quantify contacts among two core entities and an optional third entity,
    without embedding entity names or segment indices in TITAN.

  Main inputs:
    -interface-analysis
    -interface-config <file>
    -segment-config <file>
    -interface-contact-mode <sidechain-com|sidechain-heavy|cb|ca|bb>
    -interface-cut <A>
    -interface-state-min-contacts <N>
    -interface-sasa-stride <N>
    -interface-sasa-points <N>

  Config grammar:
    version 1
    entity NAME mol1|mol2 chain|chainnum VALUE
    role core_a ENTITY
    role core_b ENTITY
    role partner ENTITY
    segment ENTITY NAME ordinal|resnr|internal BEGIN END

  Typical outputs:
    interface_state_timeseries.csv
    interface_state_populations.csv
    interface_segment_contacts.csv
    interface_segment_timeseries.csv
    interface_metrics.csv
    interface_sasa_timeseries.csv

  Notes:
    - The partner role is optional for two-entity controls.
    - State populations and segment contacts use the global uncertainty
      settings. Use -uncertainty=block for equilibrium ensemble estimates.
    - Buried surface is reported both as total loss of accessible area and
      one-half of that value under the per-side convention.
    - Whole-entity coordinates are required for SASA; PBCTOPO is recommended.

2.4d Native Contacts Q

  Purpose:
    Compute the fraction of native contacts Q against the loaded reference or
    first frame. The default native-contact definition matches CMap:
    mass-weighted side-chain residue COM distance <= 6.5 A.

  Main inputs:
    -qnative
    -qnative-ref <pdb|gro>
    -qnative-scope <mol1|interface>
    -qnative-mode <sidechain-com|sidechain-heavy|cb|ca|bb>
    -qnative-cut <A>
    -qnative-domain <intra|inter|both>
    -qnative-min-seqsep <N>
    -qnative-soft <true|false>
    -qnative-beta <value>

  Typical outputs:
    qnative_timeseries.csv
    qnative_native_contacts.csv
    qnative_contact_occupancy.csv
    qnative_summary.dat

  Notes:
    - sidechain-com is the default mode: mass-weighted side-chain COM at
      distance <= 6.5 A (glycine: CA), as CMap.
    - sidechain-heavy remains an explicit atom-pair mode with default 4.5 A.
    - -qnative-soft=true uses the Best-Hummer switch per native pair,
      1/(1+exp(beta*(r - lambda*r0))) with r0 its reference distance,
      beta = 5 A^-1 (50 nm^-1) by default, lambda = 1.5 for ca mode and 1.8
      for the other modes; the hard count (r <= cutoff) is unchanged.
    - Frames without a valid periodic box use plain distances (summary:
      frames_without_box_plain_distances).
    - interface produces a bipartite mol1 x mol2 native-contact set and works
      in -nocal.
    - TPT accepts qnative_range:min,max, qnative_lt:x, and qnative_gt:x.

2.4e Dihedral Analysis

  Purpose:
    Export backbone phi/psi/omega and side-chain chi1/chi2 torsion
    observables for standard protein MD.

  Main inputs:
    -dihedral
    -dihedral-kind <backbone|sidechain|both>
    -dihedral-residues <all|list/range>
    -dihedral-stride <N>
    -dihedral-export-tica

  Typical outputs:
    dihedral_backbone_timeseries.csv
    dihedral_sidechain_timeseries.csv
    dihedral_ramachandran.csv
    dihedral_rotamer_populations.csv
    dihedral_rotamer_transitions.csv
    dihedral_summary.dat
    dihedral_feature_matrix.csv      (only with -dihedral-export-tica)

  Notes:
    - `-dihedral-kind` defaults to backbone; sidechain emits chi1/chi2
      rotamer populations and frame-to-frame transition counts.
    - `-dihedral-residues` defaults to all and accepts values such as
      `10,12-20,A:30,A:35-40`.
    - The feature matrix stores sin/cos columns for enabled torsions.
    - AVX2 accelerates MIC/bond-vector preparation; final atan2 remains scalar
      per torsion for portability and accuracy.
    - Missing terminal or incomplete torsions are written as `nan`.

2.4f Standalone Translational / Rotational MSD and Diffusion

  Purpose:
    Compute translational mean-squared displacement and diffusion summaries
    for a selected molecule, atom set, backbone/trace, or residue range. This
    is separate from LLPS dense-phase MSD outputs. With -msd-rot=true, also
    compute rigid-body rotational MSD and diffusion from the same selection.

  Main inputs:
    -msd
    -msd-selection <mol1|mol2|trace|backbone|atom:N,CA|residue:10,12-20,A:30>
    -msd-mode <com|atom>
    -msd-maxlag <N|auto>
    -msd-fit-window <auto|tmin,tmax>
    -msd-pbc <auto|unwrap|mic>
    -msd-rot <true|false>
    -msd-rot-fit-window <auto|tmin,tmax>
    -msd-memory-limit <auto|MB>

  Typical outputs:
    msd_timeseries.dat
    msd_diffusion_summary.dat
    msd_alpha_fit.dat
    rotational_msd.dat
    rotational_quaternion_timeseries.csv
    rotational_covariance_fit.csv
    rotational_diffusion_summary.dat

  Notes:
    - `-msd-mode` defaults to com; atom mode averages MSD over selected atoms.
    - `-msd-maxlag=auto` uses min(frames - 1, 1000).
    - `-msd-fit-window=auto` fits 20-80 % of the lags that keep at least half
      of the frames as time origins (lag <= frames/2; at least the first
      three lags), so the fit ends at <= 40 % of the trajectory. gmx msd
      fits 10-90 % of all lags by default.
    - Every frame of the selection is kept (multiple time origins, time
      blocks): 24 B per atom and frame in atom mode (and again for
      -msd-rot). `-msd-memory-limit=<auto|MB>` stops the run with a message
      before the storage passes the limit; auto = half the free RAM at
      start, at least 1024 MB. The limit in frames is printed at start.
    - `-msd-pbc=auto` unwraps by consecutive-frame MIC when all used frames
      have valid boxes; otherwise it reports raw-coordinate policy.
    - `-msd-rot=true` is supported with `-msd=true`.
    - Rotational analysis uses the first analyzed frame as its Kabsch
      reference and derives diffusion from the Favro quaternion covariance
      tensor.
    - The rotational summary records `reference_note` and a finite-size note
      for downstream interpretation.
    - arg.in examples should use equals syntax, for example
      `-msd-selection=atom:CA`.

2.4g Membrane Analysis (Phase 5)

  Status:
    Implemented. Phase 5A geometry and Phase 5B dynamics run in streaming mode
    with a dedicated lipid coordinate target. -nocal=true does not enable
    energy calculations.

  Purpose:
    Analyze lipid-bilayer leaflets, APL, thickness, order parameters,
    interdigitation, and persistent flip-flop events.

  Main arg.in inputs:
    -membrane=true|false
    -membrane-group=<auto|N>
    -membrane-lipids=<auto|resname:list>
    -membrane-head=<auto|atomname:list>
    -membrane-tail=<auto|atomname:list>
    -membrane-normal=<auto|x|y|z>
    -membrane-grid=<auto|Nx,Ny>
    -membrane-leaflet=<auto|upper-lower|user>
    -membrane-leaflet-file=<csv>
    -membrane-flipflop-persistence=<N>
    -membrane-protein-contacts=<true|false>
    -membrane-contact-cut=<A>
    -membrane-enrichment=<true|false>
    -membrane-curvature=<off|grid>

  Per-trajectory outputs:
    membrane_leaflets.csv
    membrane_apl_timeseries.csv
    membrane_apl_per_lipid.csv
    membrane_thickness_timeseries.csv
    membrane_thickness_map.csv
    membrane_summary.dat
    membrane_order_parameters.csv
    membrane_interdigitation.dat
    membrane_flipflop_events.csv
    membrane_protein_lipid_contacts.csv
    membrane_lipid_enrichment.csv
    membrane_curvature_map.csv
    membrane_proximity_coupling.csv
    membrane_coupling_summary.dat

  Multi-input aggregate outputs:
    membrane_summary_average.dat
    membrane_thickness_map_average.csv
    membrane_order_parameters_average.csv
    membrane_lipid_enrichment_average.csv
    membrane_curvature_map_average.csv
    membrane_proximity_coupling_average.csv
    membrane_coupling_summary_average.dat

  Notes:
    - Membrane analysis defaults off; selector fields default to auto.
    - Flip-flop persistence defaults to 3 frames and must be >= 1.
    - `-membrane-leaflet-file` is required exactly when leaflet mode is user.
    - Auto grid sizing is aspect-aware, targets about 8 lipids per leaflet per
      cell, and clamps each axis to 4..64 bins.
    - Auto normal uses PCA with sign continuity. Auto leaflets use periodic
      local-neighborhood PCA midplanes for undulating bilayers.
    - `-membrane-protein-contacts`, `-membrane-enrichment`, and
      `-membrane-curvature=grid` auto-enable membrane analysis.
    - Protein-lipid contacts are structural contacts between mol1 residue
      side-chain COMs and MARTINI lipid head/tail beads, not energy terms.
    - membrane_proximity_coupling.csv reports near-vs-far APL, thickness,
      order, curvature, contact-probability, and leaflet-composition deltas
      using per-frame protein-lipid contacts.
    - membrane_summary.dat records a reference note for interpretation.
    - membrane_order_parameters.csv: mean_P2_bond = <0.5*(3cos^2(theta)-1)>
      of the vector between consecutive tail sites and the bilayer normal.
      For MARTINI beads this is the usual bond order parameter; for all-atom
      tail carbons (-membrane-tail=atomname:...) it is S_CC of the C-C bonds,
      not the NMR S_CD of the C-H bonds.
    - membrane_curvature_map.csv (-membrane-curvature=grid):
      height_dev_over_cell_edge = (mean head height in the cell - leaflet
      mean height)/sqrt(cell area), averaged over the leaflets in the cell;
      a dimensionless height map, not a mean (H) or Gaussian (K) curvature.
    - arg.in uses the equals syntax shown above; terminal `-membrane true`
      enables the analyzer.

2.4h NMR Order Parameters S2 (Phase 6)

  Status:
    Implemented in this branch. Phase 6 standalone S2 runtime integration is
    complete.

  Purpose:
    Export Lipari-Szabo model-free S2 order parameters from backbone NH and/or
    CA-C vectors for residue-level rigidity and flexibility analysis.

  Main arg.in inputs:
    -s2=true|false
    -s2-vector=<nh|ca-c|both>
    -s2-align=<rms|none>
    -s2-maxlag=<auto|N>
    -s2-fit-window=<auto|tmin,tmax>
    -s2-residues=<all|list/range>
    -s2-bfactor-pdb=<true|false>

  Per-trajectory outputs:
    s2_order_parameters.csv
      chain, residue_number, residue_name, vector_type, plateau_C_raw,
      S2 (clipped), tau_ps, fit_R2, fit_window_start_ps, fit_window_end_ps,
      samples, fit_status
    s2_correlation_functions.dat
      chain, residue_number, residue_name, vector_type, lag_index,
      lag_time_ps, C_t, samples
    s2_summary.dat
    s2_bfactor.pdb                 optional

  Multi-input aggregate outputs:
    s2_order_parameters_average.csv
    s2_summary_average.dat
    s2_bfactor_average.pdb         optional

  Notes:
    - S2 analysis defaults off. Terminal `-s2 true` enables it.
    - N-H rows require exact backbone H or HN atoms. PRO and terminal
      residues without a backbone hydrogen are skipped with a warning.
    - CA-C rows use the exact vector and are emitted independently of N-H rows.
    - `-s2-align=rms` uses Kabsch alignment to the first analyzed frame.
    - `-s2-align=provided` is rejected in v1 with a first-frame note.
    - The legacy `-s2-plateau-window` spelling is rejected; use
      `-s2-fit-window`.
    - `-s2-maxlag` is `auto` or an integer >= 1.
    - `-s2-fit-window` is `auto` or tmin,tmax with tmax > tmin.
    - TitanExecutor uses aligned SoA buffers with AVX2/FMA kernels and a
      scalar fallback.
    - Fits run per trajectory. Multi-input aggregates report mean and SEM;
      independent trajectories are never pooled into one C(t).

2.5 PCA / FES / Basin Extraction

  Purpose:
    Project aligned CA motion onto principal components, build a 2D free
    energy surface from PC1 vs PC2, cluster FES basins, and export
    representative structures metadata.

  Main inputs:
    -pca
    optional: -fes-grid=<N>
    optional: -pca-solver=<full|partial>, -pca-topk=<N>, -pca-max-k-full=<N>
    optional: -pca-symmetry-aware
    optional DHAM weights if -dham converges

  Typical outputs:
    pca_fel.csv
    pca_fes_2d.dat
    cluster_representatives.dat
    extract_basins.sh
    pca_zts_path.csv        (when -zts -zts-domain pca is enabled)

  Notes:
    - FES files (pca/tica/mbar/umap *_fes_2d.dat) write "nan" for bins
      without samples; sampled bins hold -kT ln(P/P_max) in the output unit.
    - Without DHAM, FES is built from uniform frame counts.
    - With converged DHAM, FES and basin populations are reweighted to
      -dham-target-temp automatically.
    - `-pca-symmetry-aware` canonicalizes equivalent chain blocks before
      global PCA alignment. This helps symmetric dimers/multimers whose
      subunits exchange roles across the trajectory.
    - `extract_basins.sh` uses stored trajectory metadata
      (replica id + time in ps) to help recover full-atom basin structures.
    - `-zts` on PCA exports a minimum-energy path between two FES basins.

2.6 Allostery

  Purpose:
    Estimate nonlinear coupling and directed information flow between residues
    using mutual-information-derived Generalized Correlation (GC) and
    Transfer Entropy (TE).

  Main inputs:
    -allostery
    optional: -allostery-tau=<N>, -allostery-bins=<N>
    optional: -allostery-knn=<N>, -allostery-knn-prior=<frame|contact>
    optional: -allostery-knn-contact-cut=<A>
    optional: -allostery-cut=<A>, -allostery-prune-window=<avg|first>
    optional: -allostery-no-te
    optional: -allostery-estimator=<hist|knn|kde>
    optional: -allostery-estimator-k=<N>
    optional: -allostery-kde-bandwidth=<val>

  Typical outputs:
    raw_mi_edges.csv
    network_gc_edges.csv
    network_te_edges.csv

  Notes:
    - TITAN computes pairwise mutual information (MI) internally and exports it
      to `raw_mi_edges.csv` for every candidate pair that survives the pair
      filter.
    - GC is derived from MI using:
        GC = sqrt(1 - exp(-2 * MI))
    - `network_gc_edges.csv` remains a thresholded coupling graph, while
      `raw_mi_edges.csv` stores the unthresholded MI values for the same
      candidate-pair set.
    - `-allostery-estimator=knn` routes through the exact continuous 3D
      corr-kNN backend. `-allostery-estimator-k` controls the estimator
      neighborhood size, while `-allostery-knn` still means pair-pruning top-k.
    - `-allostery-bins` and `-allostery-kde-bandwidth` are ignored in
      knn mode and TITAN logs that decision.
    - hist/kde: every MI and TE entropy term is the plug-in estimate of the
      binned counts with the Miller-Madow correction (m - 1)/(2N) (m occupied
      bins, N samples); kde smooths the MI joint histogram and then uses
      plug-in entropies. Residual bias remains when the B^3 bins of the TE
      histogram are sparsely filled (N not >> B^3).

2.6a Mutual Information / Generalized Correlation

  Interpretation:
    - `raw_mi_edges.csv` reports the raw pairwise MI estimated from the
      discretized allostery projections.
    - `network_gc_edges.csv` reports the MI-derived GC transform, which maps
      nonlinear dependence onto the bounded interval [0, 1).
    - `network_te_edges.csv` remains the directed lagged-information export.

  2.6b Corr-kNN

  Purpose:
    Export raw continuous-estimator MI / GC / TE tables from aligned
    residue-trace fluctuations using exact kNN estimators.

  Main inputs:
    -corr-knn
    optional: -corr-knn-k=<N>, -corr-knn-tau=<N>
    optional: -corr-knn-pair-cut=<A>, -corr-knn-pair-topk=<N>
    optional: -corr-knn-pair-prior=<frame|contact>
    optional: -corr-knn-pair-contact-cut=<A>
    optional: -corr-knn-prune-window=<avg|first>
    optional: -corr-knn-no-te

  Typical outputs:
    corr_knn_summary.dat
    corr_knn_mi_pairs.csv
    corr_knn_gc_pairs.csv
    corr_knn_te_pairs.csv
    corr_knn_equilibrium_ranking.csv   (when -eq-weight-source mbar resolves)

  Notes:
    - `-corr-knn` implies `-network` and reuses the aligned DCCM trace cache.
    - MI uses exact KSG type-1 on 3D fluctuation vectors.
    - TE uses continuous conditional-MI (Frenzel-Pompe style) on
      (source[t-tau], target[t], target[t-tau]).
    - GC is derived from MI using the 3D vector form:
        GC = sqrt(1 - exp(-2 * MI / 3))
    - Outputs are raw tables only; thresholded graph policy stays with the
      legacy allostery/network path.

  2.7 TICA

  Purpose:
    Extract slow collective coordinates from time-lagged covariance and
    optionally build TICA-space FES / ZTS pathways.

  Main inputs:
    -tica
    optional: -tica-lag=<N>, -tica-components=<N>
    optional: -tica-backend=<auto|exact_dense|exact_streaming|exact_dual|randomized>
    optional: -tica-rank=<N>
    optional: -feature-export=<true|false>
    optional: -feature-source=<auto|dccm|lj|dihedral|qnative|cmap|mixed>

  Typical outputs:
    tica_eigenvalues.dat
    tica_eigenvectors.dat
    tica_projections.csv
    tica_fes_2d.dat         (when TICA FES postproc runs)
    tica_cluster_representatives.dat
    tica_zts_path.csv       (when -zts -zts-domain tica is enabled)
    features_by_frame.csv   (when -feature-export is enabled)
    features_metadata.csv
    feature_export_summary.dat

  Notes:
    - `-feature-source auto` preserves current behavior: protein-like jobs use
      DCCM/pairwise C-alpha distance features, while LJ phase-2 jobs use the
      LJ feature backend when no DCCM trace exists.
    - Explicit `dihedral`, `qnative`, or `cmap` sources auto-enable that
      producer with its normal defaults and route those features into TICA.
    - `mixed` combines DCCM distances, QNative scalar Q, dihedral sin/cos
      torsions, and CMap scalar contact count when CMap is active.
    - TICA-space ZTS uses the projection-cache capable backend path.
    - TITAN auto-promotes `-tica-backend exact_dense` when projection caching
      is required by MSM, DHAM, UMAP, or ZTS.

2.8 MSM

  Purpose:
    Build a Markov State Model on top of TICA projections via K-means
    clustering and lagged transition counting.

  Main inputs:
    -msm
    optional:
      -msm-clusters
      -msm-lag
      -msm-nts
      -msm-kmeans-iter
      -msm-seed
      -msm-macrostates
      -msm-no-boundary-split
      -msm-no-pcca
      -msm-no-its
      -msm-its-lags
      -msm-its-solver
      -msm-its-topk
      -msm-its-max-k-full

  Typical outputs:
    msm_tpm.dat
    msm_discrete_traj.dat
    msm_timescales.dat
    msm_centroids.dat
    msm_its_scan.dat
    msm_macrostates.dat
    msm_macrostate_traj.dat
    msm_count_matrix.dat
    msm_stationary.dat
    msm_equilibrium_summary.dat       (when -eq-weight-source mbar resolves)
    msm_its_tpm/

2.8a Validation

  Purpose:
    Run Chapman-Kolmogorov checks, VAMP-2 candidate ranking, and optional
    bootstrap uncertainty on the current TICA -> MSM -> TPT workflow.

  Main inputs:
    -validation
    optional:
      -validation-ck-multiples <m1,m2,...>
      -validation-vamp-components <c1,c2,...>
      -validation-vamp-lags <l1,l2,...>
      -validation-vamp-clusters <k1,k2,...>
      -validation-vamp-folds <N>
      -validation-auto-pick
      -validation-bootstrap <N>
      -validation-bootstrap-seed <N>

  Typical outputs:
    validation_summary.dat
    validation_ck_summary.dat
    validation_ck_states.dat
    validation_vamp2_ranking.csv
    validation_selected_model.dat
    validation_msm_bootstrap_summary.dat
    validation_msm_bootstrap_states.dat
    validation_tpt_bootstrap_summary.dat
    validation_tpt_bootstrap_states.dat

  Notes:
    - `-validation` implies `-msm` and reuses the solved TICA embedding plus
      cached MSM segments.
    - VAMP-2 candidate sweeps fit K-means centroids on training trajectories
      only and score held-out trajectory folds. Component-prefix, MSM lag, and
      cluster count vary inside the already solved TICA space.
    - `-validation-auto-pick` is opt-in; when off, TITAN only reports ranking
      and leaves the current MSM/TPT pipeline unchanged.
    - `-validation-bootstrap` covers MSM uncertainty and, when `-tpt` is also
      active, TPT uncertainty. Legacy `-tpt-bootstrap` files remain available.

2.9 UMAP

  Purpose:
    Build a nonlinear 2D/3D embedding from cached TICA projections using
    Titan's internal UMAP solver.

  Main inputs:
    -umap
    optional:
      -umap-neighbors=<N>
      -umap-components=<2|3>
      -umap-epochs=<N>
      -umap-min-dist=<val>
      -umap-spread=<val>
      -umap-seed=<N>
      -umap-knn-mode=<exact|approx|auto>
      -umap-knn-benchmark
      -umap-verbose

  Required context:
    UMAP depends on network -> TICA and reuses the same cached TICA
    projection matrix used by MSM/DHAM workflows. Enabling -umap does not
    automatically enable MSM clustering.

  Typical outputs:
    titan_umap_embedding.csv
    umap_fes_2d.dat                  (when -eq-weight-source mbar resolves)
    umap_cluster_representatives.csv (when -eq-weight-source mbar resolves)
    umap_equilibrium_summary.dat     (when -eq-weight-source mbar resolves)

  Integration note:
    `-eq-weight-source mbar` adds equilibrium-only UMAP sidecars without
    changing the embedding fit itself. KNN mode defaults to `auto` so large
    embeddings can select the approximate path.

2.10 DHAM Reweighting (TRAM estimator)

  Purpose:
    Combine MSM trajectories run at several temperatures into the stationary
    distribution, free energies, frame weights and (when a replica runs at
    the target temperature) the transition matrix at -dham-target-temp.
    The estimator is TRAM, the transition-based reweighting analysis method:
    every frame with an energy is a sample whose own energy (U, or U + P V
    in NPT) sets its weight at each temperature, and transitions are counted
    inside one replica. The option keeps its name; the earlier small-bias
    DHAM approximation with state-mean energies is not used any more, since
    for temperature ensembles the bias depends on each configuration's
    energy, not on the Markov state (KIN-13).
    Please cite: Wu, Paul, Wehmeyer, Noe, Proc. Natl. Acad. Sci. USA 113,
    E3221 (2016), doi:10.1073/pnas.1525092113 (TRAM); Wu, Mey, Rosta, Noe,
    J. Chem. Phys. 141, 214106 (2014), doi:10.1063/1.4902240 (dTRAM);
    Shirts, Chodera, J. Chem. Phys. 129, 124105 (2008),
    doi:10.1063/1.2978177 (MBAR); for replica exchange also Stelzl, Hummer,
    J. Chem. Theory Comput. 13, 3927 (2017), doi:10.1021/acs.jctc.7b00372.
    The same lines are written into every -dham output.

  Main inputs:
    -dham
    -dham-temps <temps.txt>
    -dham-segment-map <metadata.txt>
      Audited segment/exchange metadata, TITAN_DHAM_SEGMENTS_V1 format, for
      replica exchange. Mutually exclusive with -dham-ener. Samples are the
      frames with an exact energy; transitions are the lag windows inside one
      segment with no accepted exchange and exact energies at both ends
      (dham_segment_counts.csv, dham_segment_decisions.csv). Dropping the
      windows that hold an exchange is a selection TRAM does not correct, so
      rates need lag times well below the time between accepted exchanges.
    -dham-ener <energy_list.txt>
      One .xvg per replica; each replica trajectory is taken as continuous at
      its temperature (independent runs at several temperatures). Replica-
      exchange data need -dham-segment-map instead.
    optional:
      -dham-target-temp <K>
      -dham-ensemble <nvt|npt>
      -dham-pressure-bar <bar>
      -dham-max-iter=<N>
      -dham-tol=<val>
      -dham-n-replicas=<N>
      -dham-dt-energy <ps>
      -dham-window-match <auto|off|strict>

  Required context:
    DHAM depends on TICA -> MSM and requires a valid MSM lag.
    Protein-like jobs use network -> TICA -> MSM.
    LJ jobs use the LJ phase-2 feature backend -> TICA -> MSM.
    Energy list files should point to GROMACS .xvg energy trajectories.
    In NPT mode, TITAN combines the first numeric .xvg column with trajectory
    box volumes using H = E_col1 + P*V.
    The pressure term is the ensemble set-point, not instantaneous pressure(t).
    By default, TITAN trims each replica to the common trajectory/energy
    overlap before DHAM (`-dham-window-match auto`).

  Typical outputs:
    dham_stationary_distribution.dat
    dham_free_energy_profile.dat
    dham_unbiased_tpm.dat   (transition matrix of the replica at the target
                             temperature; none if no replica runs there)
    dham_tram_summary.dat   (counts, convergence, notes and references)
    dham_window_audit.dat

  Integration note:
    `-eq-weight-source auto` prefers converged DHAM frame weights for the
    default PCA/TICA FES outputs.
    If DHAM does not converge, Titan prints a warning and PCA/FES falls back
    to uniform weights while other active modules continue normally.
    NVT mode uses energy-based reweighting.
    NPT mode uses an enthalpy-like observable assembled from the aligned .xvg
    column plus trajectory volume metadata.

2.10a MBAR Reweighting

  Purpose:
    Solve temperature-ladder MBAR for REMD/HREMD-like energy series, export
    per-frame weights, and generate standalone weighted FES outputs in TICA
    and/or PCA space at a chosen target temperature.

  Main inputs:
    -mbar
    -mbar-temps <temps.txt>
    -mbar-ener <energy_list.txt>
    optional:
      -mbar-target-temp <K>
      -mbar-domain <tica|pca|both>
      -mbar-ensemble <nvt|npt>
      -mbar-window-match <auto|off|strict>
      -mbar-temp-mode <fixed|per_frame>
      -mbar-temp-col <col>
      -mbar-demux-map <file>
      -mbar-pressure-bar <bar>
      -mbar-dt-energy <ps>
      -mbar-bootstrap <N>
      -mbar-scan-points <N>

  Required context:
    MBAR v1 is temperature-ladder only and expects one GROMACS .xvg
    observable trajectory per replica/window.
    `-mbar-temp-mode fixed` keeps the original ladder-per-file workflow and
    expects `-mbar-temps`.
    `-mbar-temp-mode per_frame` supports raw GROMACS T-REMD-style XVG files
    where temperature changes with time inside each series.
    In `per_frame` mode, `-mbar-temps` is optional: TITAN can infer the
    ladder from replica temperatures directly, or from `-mbar-demux-map`
    plus the XVG temperature column.
    `-mbar-demux-map` is the authoritative state-assignment route for raw
    replica outputs; if it is omitted, TITAN maps each frame to the nearest
    inferred ladder temperature using the XVG temperature column.
    Demux map rows: `time_ps r_0 r_1 ...` (GROMACS replica_index.xvg; r_s is
    the replica at state s). States are resolved by time from the nearest
    row (within half the row spacing), never by row or frame index. Without
    the time column the map must have exactly one row per energy-file row,
    all energy files sharing one time axis. Samples the map cannot resolve
    stop MBAR.
    `-mbar-ensemble nvt` treats the XVG observable as energy.
    `-mbar-ensemble npt` reweights with beta (U + p V): p from
    `-mbar-pressure-bar` (default 1 bar), V from the trajectory boxes, U the
    XVG column (the potential energy); <H> and Cp are then those of
    H = U + p V. Without box volumes (energy-only input) or with a pressure
    <= 0 the p V term is not added and the XVG column is used as it is;
    mbar_summary.dat states which ("MBAR pV Term").
    `-mbar-domain tica` reuses the TICA/MSM frame metadata path.
    `-mbar-domain pca` reuses PCA trajectory metadata and the standard
    `pca_fel.csv` projection export.
    `-mbar-domain both` runs both standalone weighted exports.
    MBAR uses DHAM-style window alignment controls, but does not replace DHAM
    automatically.
    `-eq-weight-source mbar` can opt MBAR weights into the default PCA/TICA
    FES path plus equilibrium-only MSM/TPT summaries and UMAP/DCCM/LSP-MD
    context sidecars.
    It does not reweight MSM kinetics, DHAM, or TPT core TPM/pi.
    If a requested projection domain is unavailable, TITAN warns and skips it.

  Typical outputs:
    mbar_frame_weights.dat
    mbar_summary.dat
    mbar_window_audit.dat
    mbar_cv.dat                       (when -mbar-scan-points > 1)
    mbar_tica_fes_2d.dat              (when domain includes tica)
    mbar_tica_cluster_representatives.csv
    mbar_pca_fes_2d.dat               (when domain includes pca)
    mbar_pca_cluster_representatives.csv

  Integration note:
    MBAR outputs are intentionally standalone in v1.
    Standard `tica_fes_2d.dat`, `pca_fes_2d.dat`, and DHAM/TPT outputs keep
    their original behavior unless `-eq-weight-source mbar` is requested.

2.10b TPT

  Purpose:
    Run Transition Path Theory on the built MSM to estimate committors,
    reactive flux edges, and dominant pathways between user-defined source
    and sink microstate basins.

  Main inputs:
    -tpt
    -tpt-source <micro:list>
    -tpt-sink <micro:list>
    optional:
      -tpt-mode <ensemble|kinetic>
      -tpt-top-paths <N>
      -tpt-flux-cut <val>

  Required context:
    TPT phase 0 depends on a completed TICA -> MSM workflow.
    Source/sink basins currently support:
      `micro:0,1-3`
      `macro:0`
      `tica_rect:x1,y1,x2,y2`
      `lj:q6_bar>=0.172`
      `ljpreset:gas`
      `protein:major_macro`
    `lj:` basin specs are only meaningful for LJ jobs. Protein jobs
    normally use `micro:`, `macro:`, `tica_rect:`, or Q-native selectors
    (`qnative_range:min,max`, `qnative_lt:x`, `qnative_gt:x`).
    Named LJ presets currently include `gas`, `dense`, `liquid`, and `solid`.
    Protein presets currently include `major_macro`, `minor_macro`,
    `low_pc1`, `high_pc1`, `low_pc2`, `high_pc2`,
    `rmsd_low`, `rmsd_high`, `helical`, `strand`, `coil`, `turn`, and `bend`.
    `protein:rmsd_*` presets require `-rms`; DSSP-informed presets require
    `-dssp`.
    Phase 0 uses the MSM TPM + stationary distribution directly.
    In ensemble mode, TITAN can use the converged DHAM-reweighted TPM/pi
    when `-tpt-use-dham yes` (default). If DHAM does not converge, TITAN
    falls back to the built MSM model automatically.
    `-eq-weight-source mbar` only adds an equilibrium overlay summary from
    MBAR populations. It does not change committors, rates, MFPTs, dominant
    paths, or the TPT model source.
    In kinetic mode, TITAN always uses the built MSM model and writes
    `tpt_kinetics.dat` diagnostics; it still does not certify kinetic
    validity automatically. If `-tpt-demux-map <file>` is supplied, TITAN
    rebuilds a kinetic MSM from the demuxed microstate segments before TPT.
    Kinetic diagnostics now include CK / Markov sanity metrics from the shared
    validation core.
    `-tpt-bootstrap <N>` enables bootstrap uncertainty for both MSM-backed and
    DHAM-backed TPT models. When `-validation-bootstrap` matches the legacy
    TPT bootstrap config, TITAN computes once and writes both file sets.

  Typical outputs:
    tpt_summary.dat
    tpt_committor.dat
    tpt_flux_edges.dat
    tpt_paths.dat
    tpt_state_labels.dat
    tpt_kinetics.dat
    tpt_reactive_fes_2d.dat
    tpt_flux_map_2d.dat
    tpt_bootstrap_summary.dat
    tpt_bootstrap_states.dat
    tpt_all_pairs_kinetics.dat
    tpt_path_clusters.dat
    tpt_bottleneck_edges.dat
    tpt_bottleneck_states.dat
    tpt_equilibrium_overlay.dat       (when -eq-weight-source mbar resolves)
    `tpt_summary.dat` now appends a compact top-bottleneck table.
    `tpt_state_labels.dat` always contains generic state columns.
    For LJ jobs, TITAN appends LJ microstate-mean order parameters.
    If TICA/MSM centroids are at least 2D, TITAN also exports a smoothed
    reactive free-energy surface and a 2D centroid-to-centroid net-flux map.
    Bottleneck tables rank reactive edges and microstates by flux throughput
    near the transition region; dominant-path support is reported as an
    auxiliary signal.

2.11 LLPS

  Purpose:
    Detect dense and dilute phases, percolation, spatial clustering, and
    time-autocorrelation for condensate-like systems.
    The current LLPS stack also covers contact dynamics, dense-phase mobility,
    protein-RNA coupling, RNA compaction/information metrics, and RNA topology
    signatures such as bridges and clamps. Paper-facing sidecars include
    sticker hotspot ranking, droplet shape anisotropy/sphericity proxies,
    droplet genealogy summaries, and FRAP-like residence/recovery proxies.
    `-llps-paper` is an opt-in Martini protein/RNA paper bundle that reuses
    DBSCAN dense labels to export phase partitioning, composition profiles,
    convergence checks, and manuscript-friendly summaries without changing
    default `-llps` output volume.
    Run metadata and diagnostics are written into the central TITAN log
    (`log.txt`) under LLPS Run Context / Output Audit sections; TITAN does not
    create a separate LLPS manifest sidecar.
    LLPS runs in the energy frame loop: `-llps` cannot be combined with
    `-nocal` and needs two groups in `-g`.

  Main inputs:
    -llps
    optional:
      -llps-rcut
      -llps-perc
      -llps-fuzzy
      -llps-dynamics
      -llps-dynamics-detailed
      -llps-maxlag
      -llps-dimer-size
      -llps-contacts
      -llps-sticker
      -llps-voxel
      -llps-dbscan-eps
      -llps-dbscan-minpts
      -llps-lineage-overlap
      -llps-prna-rcut
      -llps-msd-fit-start-ns
      -llps-msd-fit-end-ns
      -llps-rna-compaction
      -llps-rna-information
      -llps-rna-topology
      -llps-rna-topology-detailed
      -llps-rna-intra-rcut
      -llps-rna-bridge-geom-min
      -llps-rna-patch-gap-max
      -llps-rna-clamp-require-loop
      -llps-rna-info-rg-bins
      -llps-rna-info-longrange-bins
      -llps-rna-longrange-min
      -llps-energy-correlation
      -llps-energy-detailed
      -llps-paper
      -llps-paper-profile-bin
      -llps-paper-blocks
      -llps-paper-geometry

  Typical outputs:
    llps_timeseries.dat
    llps_autocorr.dat
    llps_degree_dist.dat
    llps_contact_map.dat
    llps_contact_map_chain_residue.csv
    llps_sticker_hotspots.csv
    llps_performance_report.dat
    llps_summary.dat
    llps_density_protein.dx
    llps_density_rna.dx
    polymer_scaling.dat

  C(t) note (llps_autocorr.dat):
    C(k) = sum_t |d(t) and d(t+k)| / sum_t |d(t)|, d(t) = macro-node
    adjacency xor the persistent pattern (pairs in contact in at least half of
    the window). The window is every frame from the percolation frame on
    (the whole run without percolation), for the pattern and for the time
    origins; lags run to min(-llps-maxlag, window/2) frames; the lag time
    uses the median frame step. When the stored window exceeds a quarter of
    the free RAM (at least 256 MB), one frame in 2, 4, ... is kept
    (llps_summary.dat: ct_frame_stride).

  Contact-map note:
    llps_contact_map.dat keeps the legacy residue-index matrix schema.
    llps_contact_map_chain_residue.csv is a sparse long-form map with chain,
    residue, contact class, same-dimer status, contact-frame count, and
    contact frequency. Protein contacts are classified as intra-chain,
    inter-chain within one dimer, or inter-chain between different dimers;
    protein-RNA contacts are reported separately.
    llps_dbscan_droplets.pdb
    llps_dbscan_timeseries.csv
    llps_droplet_shape_timeseries.csv
    llps_membership_autocorr_protein.dat
    llps_membership_autocorr_rna.dat
    llps_residence_times.csv
    llps_entry_exit_rates.dat
    llps_cluster_lineage.csv
    llps_fusion_fission.dat
    llps_droplet_genealogy_summary.dat
    llps_frap_recovery_proxy.dat
    llps_sticker_contact_lifetimes.dat
    llps_protein_rna_contact_lifetimes.dat
    llps_neighbor_rewiring.dat
    llps_interface_turnover.dat
    llps_dense_msd_protein.dat
    llps_dense_msd_rna.dat
    llps_liquidity_summary.dat
    llps_rna_compaction_binding_summary.dat
    llps_rna_compaction_binding_binned.csv
    llps_rna_chain_compaction_summary.csv
    llps_rna_intrachain_contacts_summary.csv
    llps_rna_compaction_trace.csv (with -llps-dynamics-detailed=true)
    llps_rna_energy_compaction_summary.dat
    llps_rna_energy_compaction_binned.csv
    llps_rna_binding_energy_trace.csv
    llps_prna_pair_energy_trace.csv (with -llps-energy-detailed=true)
    llps_rna_information_summary.dat
    llps_rna_information_per_chain.csv
    llps_rna_compaction_state_histogram.csv
    llps_rna_partner_state_histogram.csv
    llps_rna_transition_information.csv
    llps_rna_binding_compaction_information.csv
    llps_rna_partner_state_trace.csv (with -llps-dynamics-detailed=true)
    llps_rna_topology_summary.dat
    llps_rna_topology_per_chain.csv
    llps_rna_topology_binding_binned.csv
    llps_rna_topology_state_histogram.csv
    llps_rna_bridge_lifetimes.dat
    llps_rna_clamp_lifetimes.dat
    llps_rna_loop_closure_efficiency.dat
    llps_rna_local_competition_summary.dat
    llps_rna_topology_trace.csv
    llps_rna_bridge_trace.csv
    llps_rna_clamp_trace.csv
    with -llps-paper:
      llps_phase_partition_summary.dat
      llps_radial_profiles.csv or llps_slab_profiles.csv
      llps_composition_timeseries.csv
      llps_contact_taxonomy.csv
      llps_contact_taxonomy_summary.dat
      llps_convergence_report.dat
      llps_paper_fractal_timeseries.csv
      llps_paper_fractal_scales.csv
      llps_paper_fractal_summary.dat
      llps_paper_fractal_readme.md
      llps_paper_summary.md
      llps_paper_summary.json
    log.txt sections:
      LLPS Run Context
      LLPS Macro Summary
      LLPS Output Audit
      LLPS DBSCAN Output Audit
      LLPS Dynamics Output Audit
      LLPS Paper Output Audit

2.12 GIST

  Purpose:
    Map density, cutoff solute-water and unreferenced water-water energy,
    plus first-order SO(3) nearest-neighbor orientational entropy. Pair E_ww is
    half allocated per water and includes off-grid partners from the input.
    Bulk-referenced total enthalpy and hydration free energy are unavailable.
    RT-log-density remains a proxy; orientational disorder is not entropy.
    NN entropy needs decorrelated, converged sampling. Duplicate orientations
    give NaN; storage stops at 4000000 visits without hidden subsampling.
    Solute alignment atoms must be made whole upstream. Periodic grids/cutoffs
    must fit the conservative single-image domain; O,H,H waters are registered
    before rotation. Supported models: TIP3P and SPC/E (SPCE).
    Supply -n_wat <water group index>, or a unique Water/SOL/WAT/HOH group.
    The group must contain ordered O,H,H triplets and exclude solute atoms.
    GIST uses the standard analysis path; do not combine it with -nocal.

  Main inputs:
    -gist
    optional:
      -gist-resolution=<A>
      -gist-padding=<A>
      -gist-temperature=<K>
      -gist-water-model=<name>
      -gist-min-count=<N>
      -gist-cutoff=<A>

  Typical outputs:
    gist_*.dx
    (density, E_sw, E_ww_unref, minus_TdS_orient_nn,
    minus_TdS_density_proxy, orientational_disorder DX maps;
    gist_observables.json declares unavailable quantities and frame counts)

2.12 IFP (Interaction Fingerprint)

  Purpose:
    Compute per-contact occupancy fingerprints for H-bond, salt bridge,
    hydrophobic, pi-pi, cation-pi, and halogen contacts.
    Supports protein-protein and protein-RNA/DNA systems while keeping
    ifp_occupancy.csv backward compatible.

  Main inputs:
    -ifp
    optional:
      -ifp-hbond
      -ifp-salt
      -ifp-hphob
      -ifp-pipi
      -ifp-cpi
      -ifp-hal
      -ifp-out <file>

  Typical outputs:
    ifp_occupancy.csv
    ifp_rna_annotations.csv   (only when nucleic-acid contacts are detected)
    nucleic_acid_interactions.csv
      coarse RNA/DNA recognition summary, separate from residue-pair IFP

2.12a AWH RC-Binned IFP

  Purpose:
    For GROMACS AWH trajectories, read walker-aligned RC time series from
    .xvg, bin snapshots along the biased reaction coordinate, then report
    IFP patterns and representative structures along that coordinate.
    This workflow is intended for structural mechanism mapping on a biased
    ensemble, not for absolute kinetics or equilibrium populations.

  Main inputs:
    -awh-ifp
    -awh-xvg <filelist>
    optional:
      -awh-xvg-col <N>
      -awh-bins <N>
      -awh-out-prefix <prefix>

  Typical outputs:
    awh_rc_bin_summary.dat
    awh_walker_diagnostics.dat
    awh_rc_overlap_diagnostics.dat
    awh_walker_stability_by_bin.dat
    awh_walker_pair_overlap.dat
    awh_ifp_by_bin.csv
    awh_ifp_heatmap.dat
    awh_bin_structural_summary.dat
    awh_bin_top_contacts.csv
    awh_rc_frame_map.dat
    awh_frame_assignments.dat
    awh_exploratory_context.dat
    awh_rep_bin_*.pdb

2.13 LSP-MD

  Purpose:
    Build a Local Spatial Pattern network from residue geometry without
    requiring the full DCCM workflow.

  Main inputs:
    -lspmd
    optional: -lspmd_cut=<A>

  Typical outputs:
    lspmd_edges.csv
    lspmd_degree_centrality.csv
    lspmd_betweenness_centrality.csv
    dccm_equilibrium_context.dat     (when -eq-weight-source mbar resolves)
    lspmd_equilibrium_ranking.csv    (when -eq-weight-source mbar resolves)

  Integration note:
    `-eq-weight-source mbar` only adds contextual ranking/overlay sidecars.
    TITAN does not rewrite core LSP-MD graph construction or centrality values.

2.14 Other Modules

  CAS:
    GBSA alanine-like side-chain neutralization hotspot analysis.
    `-cas=true` auto-enables GBSA and cannot run with `-nocal=true`.
    CAS does not relax or repack mutated structures.
    Outputs include:
      cas_receptor.dat
      cas_ligand.dat
      cas_hotspots_b_factor.pdb
      cas_summary.dat
      cas_performance.dat

  Hotspot report:
    `-hotspot-report=true` writes a residue-level CAS/EDecomp consensus
    ranking. The combined score is dimensionless rank consensus, not a new
    physical energy; raw CAS and EDecomp components remain in the CSV.
    Outputs include:
      hotspot_residue_ranking.csv
      hotspot_summary.dat
      hotspot_bfactor.pdb

  TDA:
    Topological data analysis of structure/geometry.

  ML:
    Simple ML affinity prediction.
    Outputs include:
      titan_ml_features.csv
      titan_rf_prediction.txt

================================================================================
3. EXAMPLES
================================================================================

3.1 Basic energy decomposition from a PDB

  titan.exe -f complex.pdb -g 1&2 -ff AMBER99SB

3.2 GBSA / ALPB free-energy style analysis

  titan.exe -f complex.pdb -g 1&2 -gbsa -igb 5

  titan.exe -f complex.pdb -g 1&2 -alpb -igb 5

3.3 Single XTC trajectory with topology reference

  titan.exe -s reference.pdb -traj md.xtc -g 1&2 -ff AMBER99SB

3.4 Multi-trajectory batch mode from input list

  titan.exe -s reference.pdb -i input.txt -g 1&2 -ff AMBER99SB

  Example input.txt:
    rep0.xtc
    rep1.xtc
    rep2.xtc

3.5 DCCM network + PCA/FES

  titan.exe -s reference.pdb -traj md.xtc -network -community louvain \
      -pca -fes-grid 80

3.6 Allostery analysis

  titan.exe -s reference.pdb -traj md.xtc -allostery \
      -allostery-tau 10 -allostery-bins 48

  titan.exe -s reference.pdb -traj md.xtc -allostery \
      -allostery-cut 12 -allostery-prune-window avg -allostery-no-te

  titan.exe -s reference.pdb -traj md.xtc -allostery \
      -allostery-cut 12 -allostery-knn 4

  titan.exe -s reference.pdb -traj md.xtc -allostery \
      -allostery-knn 4 -allostery-knn-prior contact \
      -allostery-knn-contact-cut 8 -allostery-no-te

  titan.exe -s reference.pdb -traj md.xtc -allostery \
      -allostery-estimator knn -allostery-estimator-k 4 \
      -allostery-knn 6 -allostery-cut 12

  Inspect:
    raw_mi_edges.csv for unthresholded mutual information
    network_gc_edges.csv for the derived GC network
    network_te_edges.csv for directed transfer entropy

  3.6a Corr-kNN raw information export

  titan.exe -s reference.pdb -traj md.xtc -corr-knn \
      -corr-knn-k 4 -corr-knn-tau 10 -corr-knn-pair-cut 12

  titan.exe -s reference.pdb -traj md.xtc -corr-knn \
      -corr-knn-pair-topk 6 -corr-knn-pair-prior contact \
      -corr-knn-pair-contact-cut 8 -corr-knn-no-te

  3.6b HBond kinetics

  CLI:
    titan.exe -s reference.pdb -traj md.xtc -hbk \
        -hbk-kind hbond -hbk-kind salt -hbk-domain both -hbk-maxlag 250

  arg.in:
    -hbk=true
    -hbk-kind=hbond
    -hbk-kind=salt
    -hbk-domain=both
    -hbk-maxlag=250
    -hbk-hbond-cutoff=3.5
    -hbk-angle=135
    -hbk-salt-cutoff=4.0

  3.7 TICA + MSM workflow

  titan.exe -s reference.pdb -i input.txt -tica -tica-lag 10 \
      -tica-components 5 -msm -msm-clusters 150 -msm-lag 10

3.8 DHAM + weighted FES from REMD-style data

  titan.exe -s reference.pdb -i xtc_list.txt -network -tica -msm -dham \
      -msm-clusters 150 -msm-lag 10 \
      -dham-temps temps.txt \
      -dham-ener energy_xvg_list.txt \
      -dham-target-temp 300 \
      -dham-max-iter 5000 \
      -dham-tol 1e-6

  Example temps.txt:
    300
    320
    340
    360

  Example energy_xvg_list.txt:
    rep0_energy.xvg
    rep1_energy.xvg
    rep2_energy.xvg
    rep3_energy.xvg

 Result:
    if DHAM converges, PCA/FES is reweighted automatically to the requested
    target temperature. Basin extraction still works even if DHAM is off.

3.8a Protein NPT DHAM workflow

  titan.exe -s reference.pdb -i xtc_list.txt -network -tica -msm -dham \
      -dham-temps temps.txt \
      -dham-ener potential_xvg_list.txt \
      -dham-target-temp 300 \
      -dham-ensemble npt \
      -dham-pressure-bar 1.0

  Notes:
    Use the ensemble set pressure with -dham-pressure-bar.
    TITAN does not need instantaneous pressure(t) extracted from .edr.
    In NPT mode, TITAN builds H = E_col1 + P*V from the .xvg column and
    per-frame trajectory box volumes.

3.8b LJ REMD / Argon DHAM workflow

  titan.exe -s conf.pdb -i xtc_list.txt -nocal true -lj -tica -msm -dham \
      -common-topo topol.top \
      -dham-temps temps.txt \
      -dham-ener energy_xvg_list.txt \
      -dham-target-temp 94 \
      -dham-ensemble npt \
      -dham-pressure-bar 1.0

  Notes:
    Each trajectory file is treated as one replica.
    TITAN uses LJ phase-2 features instead of DCCM trace atoms.
    `-dham-ensemble nvt` keeps the REMD-style energy workflow.
    `-dham-ensemble npt` uses H = E_col1 + P*V with volume taken from the
    trajectory box at each frame.

3.9 MBAR standalone weighted TICA/PCA FES from REMD/HREMD data

  titan.exe -s reference.pdb -i xtc_list.txt -network -tica -msm -pca -mbar \
      -mbar-temps temps.txt \
      -mbar-ener energy_xvg_list.txt \
      -mbar-target-temp 300 \
      -mbar-domain both \
      -mbar-temp-mode per_frame \
      -mbar-window-match auto \
      -mbar-scan-points 24

  Notes:
    MBAR writes separate `mbar_*` outputs and keeps the default TICA/PCA FES
    files untouched.
    Raw GROMACS T-REMD NVT/NPT energy files with a fluctuating temperature
    column can be handled via `-mbar-temp-mode per_frame`.
    If you have a demux/state-assignment map, prefer `-mbar-demux-map`;
    otherwise TITAN falls back to nearest-ladder assignment from the XVG
    temperature column.
    `-mbar-domain tica` is the lightest-weight integration path.
    `-mbar-domain pca` reuses PCA projection exports and is useful when you
    want a target-temperature PCA free-energy map without changing the default
    PCA analysis.

3.9 PCA basin extraction after FES clustering

  titan.exe -s reference.pdb -traj md.xtc -pca

  After the run:
    1. inspect cluster_representatives.dat
    2. make sure GROMACS `gmx` is available in your shell
    3. run:
         bash extract_basins.sh

  PCA ZTS example:
    titan.exe -s reference.pdb -traj md.xtc -pca -zts -zts-domain pca \
        -zts-images 64 -zts-source 0 -zts-sink 1

  TICA ZTS example:
    titan.exe -s reference.pdb -traj md.xtc -tica -zts -zts-domain tica \
        -zts-images 64 -zts-source 0 -zts-sink 1

3.10 LLPS analysis (Martini)

  titan.exe -s system.pdb -traj md.xtc -ff MARTINI -llps true

  titan.exe -s system.pdb -traj md.xtc -ff MARTINI -llps true \
      -llps-dbscan-eps 12 -llps-dbscan-minpts 8 -llps-voxel 0.6 \
      -llps-fuzzy true

3.11 GIST water thermodynamics

  titan.exe -s complex.pdb -traj md.xtc -gist \
      -gist-resolution 0.5 \
      -gist-padding 8.0 \
      -gist-temperature 300 \
      -gist-water-model TIP3P

3.12 IFP occupancy analysis

  titan.exe -s complex.pdb -traj md.xtc -ifp \
      -ifp-hbond 3.5 -ifp-salt 4.0 -ifp-hphob 4.0

  Protein-RNA/DNA runs keep ifp_occupancy.csv unchanged and may also
  write ifp_rna_annotations.csv with base/sugar/phosphate contact bias.

3.13 LSP-MD network analysis

  titan.exe -s reference.pdb -traj md.xtc -lspmd -lspmd_cut 12.0

3.14 Using arg.in

  titan.exe -a arg.in

  Example arg.in:
    -s=reference.pdb
    -traj=md.xtc
    -g=1&2
    -ff=AMBER99SB
    -network=true
    -pca=true
    -tica=true
    -tica-lag=10
    -msm=true
    -msm-lag=10

================================================================================
4. OPTION REFERENCE
================================================================================

4.1 Input / Output

  -f <file>             Single PDB/GRO input file (reference structure).
                        Default: complex.pdb

  -s <file>             Structure file for XTC/TRR mode.
                        Sets the reference structure and enables XDR mode.

  -traj <file>          XTC or TRR trajectory file.
                        Auto-detects format from extension.

  -i <file>             List file with multiple trajectory filenames.
                        Default: input.txt

  -1pdb                 Use single-PDB trajectory mode.

  -a <file>             Read arguments from file. Default: arg.in

  -opdb=<name.pdb>      Write output PDB.
  -opdbsep=<name.pdb>   Write per-trajectory/separated output PDB.

  -n <file>             Index file. Default: index.ndx

  -topo <file>          Topology for molecule 1. Supports .itp/.top/.parm7
  -topo2 <file>         Topology for molecule 2
  -common-topo <file>   Use one topology for both molecules
  -common_topo <file>   Alias of -common-topo

  -ff <name>            Force field. Default: AMBER99SB
                        Supported:
                        AMBER99SB
                        AMBER99SB-ILDN
                        AMBER99SB-BSC1
                        AMBER99SB-DISP
                        AMBER14SB
                        AMBER19SB
                        GROMOS43A1
                        GROMOS53A6
                        GROMOS54A7
                        CHARMM27
                        CHARMM36
                        OPLSAA
                        MARTINI
  -fflib-auto on|off    Auto-assign charge/VDW from lib_ff when no topology is
                        provided. Default: on
  -fflib-missing warn|strict
                        Missing lib_ff atom/template policy. Default: warn
  -fflib-hydrogens reference|none
                        Add missing hydrogens for PDB/reference inputs only.
                        Default: reference

  -pattern-input <pat> [b] [e] [step]
                        Pattern input mode; <pat> must contain '*'

4.2 Groups / Molecules / Time

  -g <G1>&<G2>          Interaction groups
  -sg <SG1>-<SG2>       Terminal sub-groups; e.g. -sg 1 2-3 4
  -sg=<SG1>-<SG2>       arg.in form; e.g. -sg= 1 2-3 4
  -sg-mode <mode>       auto, stream, or cache
                        Sub-group energy mode. stream avoids dense pair cache.
  -sg-cache <mode>      auto, off, or on; dense pair-cache preference for -sg
  -sg-memory-gb <v>     auto or N; RAM budget for -sg auto mode
  -sg-timeseries <mode> auto, full, or summary-only
                        Control -sg time-series output volume
  -sg-kernel <mode>     auto, scalar, or avx2
                        Streaming -sg compute kernel preference
  -sg-cutoff <value>    auto, VALUE, or none
                        Streaming -sg cutoff in nm; auto uses energy cutoff
  -mol-1-type <type>    protein, complex, lig, rna, dna
  -mol-2-type <type>    protein, protein-user, complex, lig, rna, dna
  -mol2 <N>             Starting residue/atom index for molecule 2
  -n_wat <N>            Zero-based water index-group number; -1 disables

  -b <time>             Begin time (ps, or ns with -tu=ns)
  -e <time>             End time (ps, or ns with -tu=ns)
  -dt <N>               Time between analysed frames (ps, or whole ns with
                        -tu=ns)
  -tu <ps|ns>           Time unit of -b/-e/-dt and of the times written in
                        the outputs. Default: ps
  -nt <N>               Number of threads

4.3 Energy / Solvation / Entropy

  -gbsa                 Enable GBSA solvation analysis
                        With -common_topo, the Coulomb/LJ part applies the
                        topology's exclusions and listed (1-4) pairs between
                        the two groups (groups joined by covalent bonds); the
                        GB polar term keeps every pair, as in Amber. Bonded
                        terms across such a junction are not computed.
                        Without a topology, covalently joined groups are
                        refused (invalid_bonded_groups).
  -alpb                 Enable Sigalov ALPB (per-frame A_Det) and imply -gbsa
  -gbsa-perf <mode>     off, summary, or detail. Default: summary.
                        Reports a fused GBSA energy-kernel timing bucket.
  -gbsa-sasa-points <N> Shrake-Rupley points/atom. Default: 96; perform a
                        convergence check (for example 96, 384, 1536).
                        The SASA (and the ALPB size) uses each atom's LJ
                        radius r_min/2, which is 0 for LJ-less polar H in
                        AMBER/OPLS; the Born radii use the GB intrinsic
                        radii (mbondi/mbondi2/mbondi3, Bondi for igb 66).
  -gbsa-rgbmax <A>|off  Cutoff in Angstrom on the HCT descreening integral for
                        the effective Born radii. Default: off (uncut sum).
                        25 matches Amber's egb default and is much faster on
                        large systems. Any finite value changes DeltaG_polar,
                        so re-baseline before comparing against uncut runs.
                        The value used is recorded in gbsa_summary.dat.
  -gbsa-surften <val>   Nonpolar surface tension in kcal/mol/A^2 applied to the
                        buried SASA. Default: 0.0072, which is MMPBSA.py's GB
                        default; sander's own gbsa=1 default is 0.005. The
                        value used is recorded in gbsa_summary.dat.
  -igb <N>              GB model: 1 HCT, 2 OBC-I, 5 OBC-II, 8 GBn2,
                        66 GBNSR6. Default: 5.
                        GBn2 uses mbondi3, neck tables, and Amber protein/RNA
                        parameters. Nonstandard ligand radii require care;
                        this path infers H partners from coordinates/types.
  -gbnsr6               Enable experimental native SES-R6 GBSA (igb=66).
                        Bondi radii, 1.4 A probe, B=0.028 A^-1, ALPB energy.
                        Uses certified patch-ray SES; errors stop the frame.
                        Different surface quadrature from Amber gbnsr6.
                        Physical PB salt kappa; direct Coulomb is unscaled.
                        SASA nonpolar energy retains TITAN's existing model.
                        No CAS or finite rgbmax. Not an MD force model.
  -gbnsr6-spacing <A>   SES mesh spacing, [0.1,1.0]. Default: 0.5.
                        Check convergence at smaller spacing before use.
  -ext_diel <val>       External dielectric. Default: 78.5
  -int_diel <val>       Internal dielectric. Default: 1.0
  -saltcon <val>        Salt concentration (M). Default: 0.15
  -temperature <val>    Analysis temperature (K) for Debye screening, GB salt
                        and interaction entropy (alias -gbsa-temp). Default: 298.15
  -gbsa-ie-block <N>    auto or positive block size. Default: auto
  -gbsa-ie-min-samples <N>
                        Minimum frames before IE is valid. Default: 10
  -gbsa-ie-filter-sigma <val>
                        Hard sigma filter; 0 disables it. Default: 0
  -epsilon_in <val>     Internal epsilon for solvation models

  -pbe <backend>        Enable frame-based PB electrostatics. backend is grid
                        or bie. Bare -pbe defaults to grid; delphi, pydelphi,
                        and fd are grid aliases. tabi is a legacy BIE alias.
  -pbe-bie-fast-operator <mode>
                        auto, direct, or p1. The p1 preset selects the
                        manuscript production fast operator. Default: auto
  -pbe-model <mode>     linear or nonlinear. Default: linear
                        nonlinear uses the current linear off/focus/vcycle
                        solve as a seed, then runs damped Newton (default) or
                        Picard/SOR on the final fine grid. Vacuum endpoints
                        remain linear.
  -pbe-biomodel <mode>  pbe or rpbe. Default: pbe
  -pbe-dielectric <m>   two-dielectric or gaussian. Default: two-dielectric
  -pbe-scale <val>      Grid scale in points/A. Default: 2.0
  -pbe-margin <val>     Solvent margin around VDW surface (A). Default: 15.0
  -pbe-gridbox <mode>   cuboidal or cubic. Default: cuboidal
  -pbe-int-diel <val>   PBE internal dielectric. Default: 1.0
  -pbe-ext-diel <val>   PBE external dielectric. Default: 80.0
  -pbe-salt <val>       PBE salt concentration (M). Default: 0.15
  -pbe-temp <val>       PBE temperature (K). Default: 298.15
  -pbe-surface <mode>   vdw, sas, ses, delphi-ses, delphi-vdw, gaussian, or
                        gaussian-cutoff. Default: vdw
                        ses applies a Euclidean rolling-probe closing to the
                        VDW mask, then fills only solvent-inaccessible cavities.
                        delphi-ses uses the same SES mask and refines dielectric
                        faces by 2x2 sub-grid sampling of that mask.
                        delphi-vdw uses DelPhi-compatible discrete media
                        indices at positive x/y/z half-grid faces: each
                        midpoint is classified by VDW atom radius, then the
                        PQR DelPhi-parity path applies a deterministic
                        VDW-to-MS boundary remap before producing final
                        midpoint epsilon. VDW media assignment uses explicit
                        PQR/SIZ radii without small-atom floors; Bondi floors
                        remain a fallback for non-PQR delphi-bondi paths.
                        DelPhi-parity diagnostics append phase RF energies
                        (solvent/vacuum), atom/grid corrected RF energy, and
                        analytic Coulomb energy to pbe_energy.csv.
  -pbe-boundary <mode>  coulombic, dipolar, or zero. Default: coulombic
                        DelPhi parity harnesses can switch boundary mode
                        without changing legacy default behavior.
  -pbe-charge-assignment <mode>
                        trilinear, delphi, or gaussian. Default: trilinear
                        delphi enables strict-grid charge diagnostics for
                        hidden PQR/DelPhi regression fixtures.
  -pbe-radius-convention <mode>
                        auto, native, or delphi-bondi. Default: auto
  -pbe-probe <val>      SAS/SES probe radius (A). Default: 1.4
  -pbe-ion-access <m>   dielectric or stern. Default: dielectric
                        stern applies Debye screening only in ion-accessible
                        solvent after excluding atom radius + ion radius.
  -pbe-ion-radius <val> Ion probe radius for Stern accessibility (A). Default: 2.0
  -pbe-dgsolv [bool]    Compute deltaG_solv. For a single BIE or Grid PQR this is
                        the reaction-field energy 0.5*sum(q_i*phi_reac_i)
                        from the selected ionic-solvent solve. Native Grid
                        two-molecule systems use complex, mol1, and mol2
                        solvent/vacuum endpoint solves.
                        Arg file: -pbe-dgsolv=true; CLI: -pbe-dgsolv true
  -pbsa                 Enable MM/PBSA summary for two-molecule systems.
                        Implies -pbe and -pbe-dgsolv; writes
                        mm_pbsa_energy.csv and mm_pbsa_summary.dat with PB
                        endpoint polar, SASA nonpolar, elec/vdw,
                        interaction entropy, and final dG.
                        -pbe=grid uses solvent/vacuum grid endpoints;
                        -pbe=bie uses three boundary-integral reaction-field
                        endpoints (complex, receptor, ligand).
                        Use bare -pbsa in both arg.in and the command line.
  -pbsa-components <m>  auto, complex,ligand,receptor, or the equivalent
                        ligand-receptor-complex order. All three components
                        are required when selection is explicit. Default: auto
  -pbsa-decomp <mode>   off, molecule, residue, hybrid, or auto. Default: auto
                        auto selects hybrid only for protein-ligand input;
                        hybrid writes protein residues and ligand atoms
                        without re-solving PB per unit.
  -pbsa-ie <bool>       Enable interaction entropy correction. Default: true
                        Uses the same Duan-Zhang interaction entropy core as
                        TITAN's -entropy module from E_MM = E_elec + E_vdw,
                        with PBSA diagnostics.
                        Arg file: -pbsa-ie=true; CLI: -pbsa-ie true
  -pbsa-residue-frames <mode>
                        off or all. Default: off. all writes
                        mm_pbsa_residue_frame.csv.
  -pbsa-ie-block <N>    auto or positive block size. Default: auto
  -pbsa-ie-min-samples <N>
                        Minimum samples before IE correction is applied.
                        Default: 10
  -pbsa-grid-reuse <m> off, validate, or auto. Default: auto
                        validate compares same-endpoint solvent/vacuum
                        geometry/operator reuse against the baseline grid
                        solve; auto remains gated until validation passes.
                        BIE reports this grid-only optimization as
                        not_applicable and rejects validate mode.
  -pbsa-sasa-points <N>
                        Shrake-Rupley points/atom. Default: 96; perform a
                        convergence check (for example 96, 384, 1536).
  -pbsa-ie-filter-sigma <val>
                        Sigma filter for IE samples. Default: 3.0 (the
                        cutoff of Sun et al., J. Chem. Phys. 2017; was 2.0).
                        Samples with |dE - <dE>| > val*sigma(dE) are dropped
                        before the exponential average (truncated IE). This
                        lowers -TdS for broad distributions: Gaussian dE at
                        298 K, sigma 5 kJ/mol gives 5.04 untruncated, 4.61
                        at 3.0, 3.41 at 2.0 kJ/mol. 0 = untruncated
                        estimator (Duan et al., JACS 2016). Ekberg and Ryde,
                        JCTC 2021, discuss the bias of truncation.
                        mm_pbsa_summary.dat records the value and these notes.
  -pbsa-perf <mode>     off, summary, or detail. Default: summary
  -pbe-max-iter <N>     SOR iteration cap. Default: 10000
  -pbe-rms <val>        SOR RMS convergence tolerance. Default: 1e-6
  -pbe-dphi <val>       SOR max delta-phi tolerance. Default: 1e-5
  -pbe-omega <val>      auto or SOR relaxation value in (0,2). Default: auto
  -pbe-nl-max-iter <N>  Nonlinear Picard/SOR sweep cap or Newton work budget.
                        Default: 80
  -pbe-nl-inner-iter <N>
                        Newton correction SOR sweeps or V-cycle cap per outer
                        step.
                        Default: 40
  -pbe-nl-stagnation <N>
                        Newton residual stagnation window. Default: 8
  -pbe-nl-block <N>     Nonlinear diagnostic block size. Default: 5
                        In DelPhi NWT parity rows this is the true block-stop
                        size used for RMS/max-dphi checks.
  -pbe-nl-dphi <val>    Nonlinear max delta-phi tolerance. Default: 1e-4
  -pbe-nl-relax <val>   auto or Picard damping in [0.05,1.0]. Default: auto
  -pbe-nl-clip <val>    Clip alpha*phi before sinh. Default: 20.0
  -pbe-nl-continuation <mode>
                        off or auto. Default: auto
  -pbe-nl-line-search <mode>
                        off or auto. Default: auto
  -pbe-nl-residual <val>
                        Nonlinear residual tolerance. Default: 1e-5
                        A nonlinear solve reports status ok only when its
                        final RMS nonlinear residual is at or below this
                        value (DelPhi NWT parity rows keep the DelPhi block
                        stop); otherwise not_converged.
  -pbe-nl-solver <mode> auto, picard or newton. Default: auto (= newton)
                        picard stops when the potential stops changing and
                        stalls above the residual tolerance on proteins
                        (1A63, 0.5 A: dG_solv 9.3 kJ/mol off), so it reports
                        not_converged there. The DelPhi NWT parity path needs
                        an explicit newton.
  -pbe-nl-inner <mode>  auto, sor, or vcycle. Default: auto (vcycle; sor for
                        the DelPhi NWT fixture: test PQR with delphi-vdw)
                        newton solves a stencil Jacobian correction with SOR
                        or Galerkin V-cycle and appends correction residual,
                        inner levels/cycles, Jacobian, lambda, and stop-reason
                        diagnostics. In DelPhi PQR parity rows, auto keeps the
                        strict Newton path on SOR first; if line search rejects
                        a correction, TITAN falls back to a DelPhi NWT SOR
                        block with denominator += screen*cosh(phi) and RHS
                        += screen*(phi*cosh(phi)-sinh(phi)). PQR parity dump
                        mode writes pbe_nwt_trace.csv and
                        pbe_nwt_solver_progress.csv; pbe_energy.csv appends
                        nonlinear stress, osmotic, and total correction columns.
  -pbe-nl-multigrid <m> off or fas. Default: off
  -pbe-gaussian-exp <v> Gaussian density exponent. Default: 2
  -pbe-gaussian-sigma <v>
                        Gaussian density sigma scale. Default: 1.0
  -pbe-density-cutoff <v>
                        Gaussian cutoff density. Default: 0.759
                        Gaussian dielectric now uses DelPhi-style
                        multiplicative density union at centers and positive
                        midpoint faces instead of additive clamp.
  -pbe-surface-cutoff <v>
                        auto or explicit Gaussian surface cutoff. Default: auto
  -pbe-gap-diel <v>     Gaussian-cutoff gap dielectric. Default: 22.0
  -pbe-max-grid <N>     Maximum grid points before preflight rejection
  -pbe-map <mode>       off, phi, epsilon, ion, or all. Default: off
  -pbe-map-frames <m>   none, first, last, or all. Default: none
  -pbe-atom-potentials <m>
                        off or csv. Default: off. csv writes per-atom reaction
                        potentials in kJ mol^-1 e^-1 and 0.5*q*phi reaction-
                        energy contributions for grid and BIE.
  -pbe-binding-cycle <m>
                        off or pqr-components. The latter solves complex,
                        receptor, and ligand PQRs in TITAN and writes polar and
                        total continuum-electrostatic binding contributions.
                        The intercomponent Coulomb term uses -pbe-int-diel on
                        the fixed bound geometry; this is not total binding
                        free energy.
  -pbe-binding-receptor-pqr <file>
                        Receptor PQR for pqr-components binding cycle.
  -pbe-binding-ligand-pqr <file>
                        Ligand PQR for pqr-components binding cycle.
  -pbe-bie-surface-output <m>
                        off, csv, vtk, or all. Default: off. For -pbe=bie,
                        writes surface coordinates, normals, area, the raw
                        boundary unknown, and physical total, Coulomb, and
                        reaction potentials in e/A, kcal/mol/e, and kT/e as
                        pbe_tabi_surface_solution.csv/.vtk.
                        Node-patch analytic/grid SES output retains connected
                        triangles and stores solution fields as VTK POINT_DATA;
                        centroid output remains cell-associated.
  -pbe-kernel <mode>    auto, scalar, or avx2. Default: auto
                        v1 records the request but uses scalar-threaded SOR
                        Arg file form: -pbe-kernel=auto
                        CLI form:      -pbe-kernel auto
  -pbe-focus <mode>     off, auto, or delphi. Default: off
                        auto runs nested-grid focusing: coarse solve first,
                        then interpolated coarse potential seeds fine-grid
                        boundary/initial values.
                        delphi uses the parent phimap as the child boundary
                        path and exposes hidden probe/focus fixture
                        diagnostics.
  -pbe-multigrid <mode> off, focus, auto, or vcycle. Default: off
                        focus/auto keep the nested focusing path; vcycle
                        solves the fine grid with Galerkin multigrid: coarse
                        operators P^T A P of the fine operator (no coarse
                        re-discretisation), Gauss-Seidel smoothing, CG
                        acceleration; fine-grid SOR polishes a solve that
                        has not converged within -pbe-max-iter. The nonlinear
                        Newton correction (-pbe-nl-inner=vcycle, the auto
                        choice) uses the same
                        hierarchy. References: Alcouffe, Brandt, Dendy,
                        Painter, SIAM J. Sci. Stat. Comput. 2, 430 (1981);
                        Holst, Saied, J. Comput. Chem. 14, 105 (1993).
  -pbe-focus-levels <N> Number of nested grids, 1..4. Values >1 imply auto.
                        Default effective value: 1 when off, 2 when auto.
  -pbe-focus-margin <A> Extra coarse-grid outer margin in Angstrom. Default: 8.0
  -pbe-mg-levels <N>    auto or 1..16. Default: auto (coarsen until the
                        coarsest level has at most 1000 unknowns, which are
                        then solved directly)
  -pbe-mg-pre <N>       V-cycle pre-smoothing sweeps. Default: 2
  -pbe-mg-post <N>      V-cycle post-smoothing sweeps. Default: 2. CG needs
                        -pbe-mg-pre equal to -pbe-mg-post.
  -pbe-mg-coarse-iter <N>
                        Symmetric Gauss-Seidel sweeps on a coarsest level
                        too large for the direct solve (only when
                        -pbe-mg-levels caps the hierarchy). Default: 200
  -pbe-mg-accel <m>     cg or none. Default: cg (CG preconditioned by one
                        V-cycle per iteration); none repeats V-cycles.
  Hidden PQR parity diagnostics:
  -pbe-parity-dump <m>  off, summary, points, or all. Requires -pbe-test-pqr.
                        Writes pbe_parity_stage.csv and optionally
                        pbe_parity_points.csv for DelPhi root-cause reports.
                        Barnase PQR rows also emit hidden forensic input,
                        surface, operator, and per-atom phi audits so linear
                        field/source/energy drift can be isolated before
                        nonlinear Newton tuning.
  -pbe-parity-map-dump <m>
                        off, samples, or full. Requires -pbe-test-pqr.
                        samples writes pbe_field_samples.csv,
                        pbe_boundary_samples.csv, and
                        pbe_charge_grid_audit.csv for barnase field-oracle
                        comparison. full is intentionally large. Use
                        test/pbe_delphi_export_maps.ps1 to generate optional
                        DelPhi oracle CSVs outside production paths. The
                        extractor normalizes pyDelPhi VDW surfmap semantics
                        as 1=solvent and 0=solute and writes a sparse
                        pbe_oracle_surface_map.csv containing inside=1 rows
                        for replay. It also writes midpoint-epsilon samples by
                        default, or a full pbe_oracle_midpoint_epsilon.csv for
                        one salt (default 0.15 M, set with -FullMidpointSalt)
                        when called with -FullMidpointEpsilon. If local pyDelPhi
                        cannot export maps, the script remains report-only and
                        strict barnase parity stays disabled.
                        test/pbe_midpoint_oracle_roundtrip_smoke.ps1 verifies
                        the full midpoint oracle layout, axis order, positive
                        face convention, and midpoint six-face sum before
                        native remap changes are trusted.
                        test/pbe_barnase_field_oracle_smoke.ps1
                        writes pbe_barnase_root_cause_report.csv, and
                        test/pbe_barnase_timing_compare.ps1 compares TITAN and
                        pyDelPhi wall-clock timing on the same barnase rows.
  -pbe-test-charge-grid <m>
                        native or oracle. Requires -pbe-test-pqr. oracle reads
                        pbe_oracle_charge_grid_full.csv or
                        pbe_oracle_charge_grid.csv from the run directory.
  -pbe-test-charge-grid-diff <m>
                        off, samples, or full. Requires -pbe-test-pqr and a
                        pyDelPhi charge-grid oracle. Writes
                        pbe_charge_grid_diff.csv and
                        pbe_charge_grid_diff_summary.csv to isolate local
                        charge assignment before SOR update debugging.
  -pbe-test-surface-map <m>
                        native or oracle. Requires -pbe-test-pqr. oracle reads
                        pbe_oracle_surface_map.csv with i,j,k,inside columns.
  -pbe-test-midpoint-epsilon <m>
                        native or oracle. Requires -pbe-test-pqr. oracle reads
                        pbe_oracle_midpoint_epsilon.csv with
                        i,j,k,eps_x,eps_y,eps_z columns and overrides
                        positive-face epsilon before denominator/solver setup.
  -pbe-test-midpoint-diff <m>
                        off, samples, or full. Requires -pbe-test-pqr and
                        pbe_oracle_midpoint_epsilon.csv. Writes only native
                        vs oracle positive-face mismatches to
                        pbe_midpoint_face_diff.csv with remap action, raw face
                        state, signed encoded media/entity state, owner, and
                        nearest atom/exposed-point metrics. full also writes
                        pbe_native_midpoint_state.csv and
                        pbe_vdwms_action_histogram.csv for VDWMS stage
                        root-cause analysis.
  -pbe-test-vdwms-remap <m>
                        auto or off. Requires -pbe-test-pqr. auto enables the
                        DelPhi-parity VDW-to-MS midpoint-media remap; off keeps
                        the raw VDW half-grid media map for forensic runs.
                        pbe_parity_stage.csv and
                        pbe_barnase_surface_audit.csv append
                        midpoint_eps_sum_* columns to distinguish solver
                        denominator audits from raw midpoint-epsilon sums.
  -pbe-test-boundary-map <m>
                        native or oracle. Requires -pbe-test-pqr. oracle reads
                        a complete pbe_oracle_boundary_map.csv boundary phi
                        map and fails clearly if any boundary point is absent.
                        Oracle CSVs are hidden PQR parity inputs only; arg
                        files use -flag=value, while direct CLI usage uses
                        -flag value.
  -pbe-test-boundary-convention <m>
                        native-real, delphi-grid, or oracle. Requires
                        -pbe-test-pqr. delphi-grid samples boundary distances
                        from DelPhi atom-grid coordinates and uses DelPhi
                        exdi/indi scaled by epkt; oracle implies boundary-map
                        replay.
  -pbe-test-field-units <m>
                        native or delphi-dimensionless. Requires
                        -pbe-test-pqr. delphi-dimensionless stores the field
                        in pyDelPhi raw units, scales RHS by epkt, and writes
                        raw/native phi columns in field, boundary, and
                        operator-stage diagnostics.
  -pbe-test-operator-convention <m>
                        native or delphi. Requires -pbe-test-pqr. delphi keeps
                        native field units but uses DelPhi's solvent screening
                        convention in the linear operator audit path.
  -pbe-test-rhs-map <m>, -pbe-test-screen-map <m>, -pbe-test-denom-map <m>
                        native or oracle. Requires -pbe-test-pqr and a full
                        pbe_oracle_operator_stage.csv. These replay RHS,
                        screen, or denominator maps to isolate water_phi
                        mismatch before changing the solver.
  -pbe-test-nl-source <m>
                        titan or delphi-dimensionless. Requires -pbe-test-pqr.
                        delphi-dimensionless solves psi=alpha*phi with alpha-
                        scaled RHS/source, then converts maps/energy back to
                        physical phi for parity diagnostics.
  -pbe-test-newton-derivative [bool]
                        Finite-difference check for the Newton Jacobian.
                        Requires -pbe-test-pqr.
  -pbe-test-energy-convention <m>
                        auto, atom, grid, delphi-corrected, or delphi-iscm.
                        Requires -pbe-test-pqr. Production default remains auto.
                        delphi-iscm is a hidden DelPhi two-dielectric parity
                        convention for induced-surface-charge water RF energy;
                        its final corrected RF follows the DelPhi VDW
                        water-phase ISCM convention while keeping vacuum grid
                        RF as an audit field.
                        TITAN now builds DelPhi-style scaled surface-charge
                        positions natively for delphi-iscm/delphi-corrected.
                        When pbe_oracle_iscm_full.csv is present in a hidden
                        PQR parity run, it is used only for compare/audit
                        fields such as iscm_position_rms_vs_oracle, not for
                        final ISCM accumulation.
                        native/PBSA defaults still use the normal atom/grid
                        convention unless this test path is selected.
                        Use the DelPhi stage-oracle smoke scripts, not this
                        hidden knob alone, to decide whether an energy mismatch
                        is field/operator or reaction-energy convention drift.

  -edecomp=<bool>       Enable unified energy decomposition. Default: false
  -edecomp-source=<m>   mm, gbsa, pbsa, or all. Default: mm
                        gbsa requires -gbsa; pbsa requires -pbsa=true.
                        all does not auto-enable GBSA or PBSA.
  -edecomp-level=<m>    residue, atom, or both. Default: residue
  -edecomp-edge-top=<N> Positive edge count or all. Default: 200
  -edecomp-rank-by=<m>  abs-mean or mean-abs. Default: abs-mean
  -edecomp-frame-output=<m>
                        off or all. Default: off
  -edecomp-perf=<m>     off, summary, or detail. Default: summary
                        MM writes residue pair matrices plus sparse atom edges.
                        MM/GBSA and MM/PBSA write endpoint entity maps/rankings only:
                        edecomp_mmgbsa_{residue|atom}_map.csv and ranking.csv.
                        edecomp_mmpbsa_{residue|atom}_map.csv and ranking.csv.
                        Legacy -emap, -emap-level, -energy-res, -new-emap,
                        and -useForceOptionforEnergy now fail; migrate to
                        -edecomp=true and -edecomp-*.
  -hotspot-report=<b>   CAS + EDecomp residue hotspot report. Default: false
  -hotspot-source=<m>   cas, edecomp, or both. Default: both
                        cas auto-enables -cas=true; edecomp auto-enables
                        -edecomp=true and requires residue or both level.
  -hotspot-rank-by=<m>  cas-ddg, edecomp-abs-mean, or combined.
                        Defaults to combined for source=both, cas-ddg for
                        source=cas, and edecomp-abs-mean for source=edecomp.
  -fmap [type]          Force map type
  -fmap-level <level>   per-res or all
  -force-res            Per-residue force decomposition
  -com_energy <axis>    x, y, z, all, or default euclidean
  -interchain-energy    Pairwise inter-chain energy analysis
  -denfo <val>          Distance-energy cutoff
  -dsc <val>            Standard distance cutoff. Default: 6.5 A

  -entropy              Interaction entropy (Duan-Zhang), -TdS = kT ln
                        <exp(beta (E - <E>))>, E = E_elec + E_vdW, at the
                        -temperature (default 298.15 K)
  -entropy-filter-sigma <x>
                        Drop samples farther than x standard deviations from
                        the mean before the exponential average. Default: 0
                        (no trimming; a 2-sigma cut removes the tail that
                        decides the average: -33 % for Gaussian sigma = 5
                        kJ/mol). Written to entropy_result.dat.
  -entropy-quasi <com>,<lig>,<rec>
  -qha-temp=<K>                 Quasi-harmonic temperature (default: 300 K).
  -ml-predict-experimental=true  No effect. The bundled affinity model is closed
                        pending a model card (training set, feature units,
                        cross-validation score, applicability domain).
                        -ml still exports the five descriptors.
                        Quasi-harmonic entropy

4.4 PBC

  -pbc                  Basic PBC correction
  -pbci <chain_ref>     Internal-reference PBC
  -pbci2 <chain_ref>    More aggressive internal-reference PBC
  -pbci3 <chain_ref>    Most aggressive internal-reference PBC
  -pbca <chain_ref>     Aggressive PBC mode
  -pbcn                 FixBox-style PBC mode
  -pbctopo              pbctopo (VIBE): Virtual Intact Block Extraction PBC mode
                        (short reference: -vibe-help)
  -pbctopo-mode=<name>  vibe | llps-local. Default: vibe
  -pbctopo-llps-local   Alias for -pbctopo -pbctopo-mode=llps-local
  -pbctopo-search-policy=<name>
                        fast-certified | audit-exact. Default: fast-certified
                        audit-exact runs exact search after a certified graph seed.
  -pbctopo-audit-max-states=<N>
                        Maximum exact-search nodes per audit frame. Default: 10000000
  -pbctopo-audit-max-seconds=<value>
                        Maximum exact-search wall time per audit frame. Default: 60
  -pbctopo-audit-max-memory-mb=<N>
                        Maximum estimated audit workspace per frame. Default: 1024
  -pbctopo-anchor-backend=<name>
                        auto | prim | spatial-all. Default: auto
  -pbctopo-anchor-metrics=<name>
                        full | mst-only | no-order | no-contact. Default: full
  -pbctopo-temporal=<name>
                        greedy | viterbi | hololift. Default: greedy
                        viterbi is experimental/diagnostic and is not part of
                        the framewise VIBE observation contract.
)MANUAL"
#ifdef TITAN_WITHOUT_HOLOLIFT
        R"MANUAL(                        hololift is not available in this build (HOLOLIFT=0).
)MANUAL"
#else
        R"MANUAL(                        hololift runs HoloLift after VIBE (short reference:
                        -hololift-help): the relative
                        periodic images of the components are selected for the
                        whole trajectory by exact dynamic programming, with a
                        charge on selections that lack contact evidence, and
                        written beside the VIBE package as hololift_*.csv and
                        hololift_coordinates.{xtc,pdb,csv}. The coordinates keep
                        the selected relative images and unwrap the common
                        translation with the displacement-preserving (toroidal)
                        scheme, so they serve transport under constant pressure.
                        Requires -traj <file.xtc>; uses -nt threads.
)MANUAL"
#endif
        R"MANUAL(  -pbctopo-temporal-k=<N>
                        Top-K virtual candidates for experimental Viterbi.
                        Range: 1..32; default: 4
  -pbctopo-temporal-penalty=<value>
                        Center-jump penalty for experimental Viterbi. Default: 2.0
  -pbctopo-evidence-export[=<bool>]
                        Opt-in Paper-2 candidate evidence summary/relation
                        sidecars. Default: false
  -pbctopo-steric=<report|filter>
                        report (default): overlaps between parts do not affect the
                        placement; the frame report counts the overlaps in the input
                        (input_overlap_*). filter: the 1.0.x rule, which rejects a
                        candidate whose placed parts overlap.
  -pbctopo-vdw-clash-scale=<value>
                        Severe-overlap threshold lambda*(ri+rj). Range: (0,1]; default: 0.55
                        Explicit zero-LJ sites have no LJ hard core.
  -pbctopo-bond-cutoff=<A>
                        Local topology bond cutoff. Default: 2.15 A; MARTINI: 5.5 A
  -pbctopo-clash-cutoff=<A>
                        Overlap cutoff for atoms without radii (input scan; candidate
                        rejection with -pbctopo-steric=filter). Default: 1.05 A; MARTINI: 3.5 A
  -pbctopo-contact-cutoff=<A>
                        Owner/contact cutoff for split guards. Default: 4.5 A; MARTINI: 8.0 A
  -pbctopo-required-contacts=<file>
                        Hard interface contacts: atom_a atom_b max_A [weight].
                        Atom indices are 1-based in the selected PBCTOPO atom set.
  -pbctopo-continuity-cutoff=<A>
                        Continuity graph cutoff. Default: 20.0 A
  -pbctopo-exact-component-limit=<N>
                        Max raw/effective components for exact search.
                        Range: 1..16. Default: 6
  -pbctopo-beam-width=<N>
                        Large-component beam width. Range: 1..4096. Default: 12
  -pbctopo-component-shell-radius=<N>
                        Component-image search radius in lattice cells.
                        Range: 1..2. Default: auto (1; 2 for two components,
                        with exact-mode adaptive expansion to 2).
  -pbctopo-h9-orbit=<bool>
                        Export topology-whole component representatives and
                        audit the prespecified 27-state local orbit when the
                        selected evidence graph has k=2 and d=3. Default: false
  -pbcclust <axes>      Clustered PBC mode; axes like x, xy, xyz
  -pbcm [chain_ref]     Membrane-oriented PBC mode
  -refpbc <file>        Separate reference PDB for PBC correction
  -pbc-envi <chains...> Environment chains for PBC handling
  -no-center            Disable COM centering during PBC correction

4.5 LJ, RMS, Network, PCA, TICA, MSM, DHAM, LLPS, GIST, IFP, LSP-MD

  -network              DCCM network analysis
  -community=louvain    Louvain community detection on the DCCM graph

  -pca                  PCA and FES analysis
  -fes-grid=<N>         FES grid size. Default: 60
  -pca-solver=<mode>    PCA spectral solver: full | partial (default: full)
  -pca-topk=<N>         Top spectrum size for partial solver; 0 = auto from FES needs
  -pca-max-k-full=<N>   Full solver cutoff K; if K > cutoff, auto-switch full -> partial
  -pca-symmetry-aware   Canonicalize equivalent chain blocks before PCA alignment

  -allostery            MI-derived GC + Transfer Entropy analysis
  -allostery-tau=<N>    TE lag in frames. Default: 5
  -allostery-bins=<N>   Histogram bins. Default: auto (-1)
  -allostery-knn=<N>    Keep top-N ranked partners per residue. Default: 0
  -allostery-knn-prior=<frame|contact>
                        frame = nearest-neighbor votes; contact = Cbeta/N contact frequency
  -allostery-knn-contact-cut=<A>
                        Anchor-contact cutoff for contact prior. Default: 8 A
  -allostery-cut=<A>    Distance-prune candidate residue pairs. Default: off
  -allostery-prune-window=<avg|first>
                        avg scans all stored frames; first only frame 0 for pruning
  -allostery-cut-mode=<avg|first>
                        Deprecated alias for -allostery-prune-window
  -allostery-no-te      Skip Transfer Entropy stage and emit header-only TE CSV
  -allostery-estimator=<mode>
                        hist | knn | kde. Default: hist
  -allostery-estimator-k=<N>
                        Exact kNN estimator neighborhood size. Default: 4
  -allostery-kde-bandwidth=<val>
                        KDE bandwidth override for estimator=kde. Default: 0 (auto)

  -mem-budget-gb=<auto|N>
                        Shared post-processing RAM budget. Default: auto
                        (60% currently available RAM; lower in low-memory mode)
  -low-memory=<true|false>
                        Prefer RAM-saving fallbacks when estimates are large
  -network-mode=<auto|dense|sparse>
                        DCCM network memory policy. Default: auto
  -network-sparse-cut=<A>
                        Sparse candidate cutoff in Angstrom. Default: 30
  -network-sparse-k=<N>
                        Sparse top-N nearest candidates per node. Default: 24
  -network-cache-precision=<auto|fp32|fp64>
                        Cache precision policy metadata for large network runs
  -graph                Unified sparse graph analytics
  -graph-source <name>  dccm | lspmd | allostery-gc | allostery-te | tpt
                        Repeatable. Default: auto (attach to active producers)
  -graph-community <m>  auto | off | louvain. Default: auto
  -graph-filter-topk <N>
                        Keep only top-N weighted edges per node after source
                        construction. Default: 0
  -graph-filter-component <m>
                        all | largest. Default: all

  -cmap                 Side-chain COM contact probability map analysis
  -cmap-scope <mode>    mol1 | interface. Default: mol1
  -cmap-mode <mode>     sidechain-com | sidechain-heavy | cb | ca | bb.
                        Default: sidechain-com
  -cmap-cut <A>         Default: 6.5 for sidechain-com; 4.5 for atom modes
  -cmap-domain <mode>   intra | inter | both. Default: both
  -cmap-min-seqsep <N>  Minimum intra-chain sequence separation. Default: 3
  -segment-config <f>   Generic entity/role/segment definition file
  -interface-analysis   Configurable two/three-entity contact-state analysis
  -interface-config <f> Alias that sets segment config and enables analysis
  -interface-contact-mode <mode>
                        sidechain-com | sidechain-heavy | cb | ca | bb
  -interface-cut <A>    Contact cutoff. Default: 6.5
  -interface-state-min-contacts <N>
                        Contact-pair threshold for state labels. Default: 1
  -interface-sasa-stride <N>
                        SASA/BSA sampling stride; 0 disables. Default: 10
  -interface-sasa-points <N>
                        Sphere points per atom. Default: 96
  -qnative              Native-contact fraction Q analysis
  -qnative-ref <file>   Optional reference; default: loaded reference/first frame
  -qnative-scope <mode> mol1 | interface. Default: mol1
  -qnative-mode <mode>  sidechain-com | sidechain-heavy | cb | ca | bb.
                        Default: sidechain-com
  -qnative-cut <A>      Default: 6.5 for sidechain-com; 4.5 for atom modes
  -qnative-domain <m>   intra | inter | both. Default: both
  -qnative-min-seqsep <N>
                        Minimum intra-chain sequence separation. Default: 3
  -qnative-soft <bool>  Best-Hummer soft contacts 1/(1+exp(beta(r-lambda r0))),
                        lambda 1.5 (ca) or 1.8. Default: false
  -qnative-beta <val>   Soft-contact beta in 1/A (5 = 50 nm^-1); only with
                        soft=true. Default: 5
  -dihedral             Dihedral analysis: phi/psi/omega and chi1/chi2
  -dihedral-kind <mode> backbone | sidechain | both. Default: backbone
  -dihedral-residues <s>
                        all or list/ranges like 10,12-20,A:30,A:35-40
  -dihedral-stride <N>  Analyze every Nth frame. Default: 1
  -dihedral-export-tica Export enabled torsions as sin/cos feature matrix
  -msd                  Translational MSD and diffusion analysis
  -msd-selection <sel>  mol1, mol2, trace, backbone, atom:N,CA, or
                        residue:10,12-20,A:30. Default: mol1
  -msd-mode <mode>      com or atom. Default: com
  -msd-maxlag <N|auto>  Maximum lag index. Default: auto=min(frames-1,1000)
  -msd-fit-window <w>   auto or tmin,tmax in ps. Default: auto (20-80 % of
                        the lags <= frames/2)
  -msd-pbc <policy>     auto, unwrap, or mic. Default: auto
  -msd-memory-limit <auto|MB>
                        Stop before the kept frames pass this size.
                        Default: auto (half the free RAM, >= 1024 MB)
  -msd-rot <bool>       Enable rotational MSD with -msd=true. Default: false
  -msd-rot-fit-window <w>
                        Rotational fit window: auto or tmin,tmax in ps

  -membrane             Enable Phase 5 MARTINI membrane analysis
  -membrane=<bool>      true or false. Default: false
  -membrane-group=<g>   auto or group index >= 0. Default: auto
  -membrane-lipids=<s>  auto or resname:list. Default: auto
  -membrane-head=<s>    auto or atomname:list. Default: auto
  -membrane-tail=<s>    auto or atomname:list. Default: auto
  -membrane-normal=<n>  auto, x, y, or z. Default: auto
  -membrane-grid=<g>    auto or Nx,Ny positive bins. Default: auto
  -membrane-leaflet=<m> auto, upper-lower, or user. Default: auto
  -membrane-leaflet-file=<csv>
                        Required exactly when -membrane-leaflet=user
  -membrane-flipflop-persistence=<N>
                        Persistent reassignment frames. Default: 3
  -membrane-protein-contacts=<bool>
                        Protein residue side-chain COM to lipid bead contacts
  -membrane-contact-cut=<A>
                        Protein-lipid contact cutoff. Default: 6.5
  -membrane-enrichment=<bool>
                        Lipid resname/leaflet enrichment near mol1 protein
  -membrane-curvature=<off|grid>
                        Headgroup-grid height-deviation map (not H or K).
                        Default: off

  -s2                   Enable Phase 6 NMR order-parameter analysis
  -s2=<bool>            true or false. Default: false
  -s2-vector=<v>        nh, ca-c, or both. Default: both
  -s2-align=<mode>      rms or none. Default: rms first-frame alignment
  -s2-maxlag=<N|auto>   Maximum lag index. Default: auto
  -s2-fit-window=<w>    auto or tmin,tmax in ps. Default: auto
  -s2-residues=<s>      all or a non-empty residue list/range. Default: all
  -s2-bfactor-pdb=<bool>
                        Export S2 B-factor PDB. Default: false

  -corr-knn             Raw continuous kNN MI / GC / TE export
  -corr-knn-k=<N>       Exact kNN estimator neighborhood size. Default: 4
  -corr-knn-tau=<N>     TE lag in frames. Default: 5
  -corr-knn-pair-cut=<A>
                        Distance-prune candidate residue pairs. Default: off
  -corr-knn-pair-topk=<N>
                        Keep top-N ranked partners per residue. Default: 0
  -corr-knn-pair-prior=<frame|contact>
                        frame = nearest-neighbor votes; contact = Cbeta/N contact frequency
  -corr-knn-pair-contact-cut=<A>
                        Anchor-contact cutoff for contact prior. Default: 8 A
  -corr-knn-prune-window=<avg|first>
                         avg scans all stored frames; first only frame 0 for pruning
  -corr-knn-no-te       Skip Transfer Entropy stage and emit header-only TE CSV

  -hbk                  HBond / salt / water kinetics analysis
  -hbk-kind <mode>      all | hbond | salt | water. Repeatable. Default: all
  -hbk-domain <mode>    inter | intra | both. Default: both
  -hbk-maxlag <N>       Max lag in frames for C(t). Default: 500
  -hbk-hbond-cutoff <A>
                        HBond donor-acceptor cutoff. Default: effective -dhb or 3.5
  -hbk-angle <deg>      HBond angle cutoff. Default: effective -ahb or 135
  -hbk-salt-cutoff <A>  Salt-bridge cutoff. Default: 4.0
  -hbk-water-cutoff <A>
                        Water-bridge O...solute cutoff. Default: effective
                        -dis-water or 3.5

  -tica                 TICA analysis
  -tica-lag=<N>         TICA lag. Default: 10
  -tica-components=<N>  Components kept. Default: 5
  -tica-backend=<name>  auto | exact_dense | exact_streaming | exact_dual | randomized
  -tica-rank=<N>        Reduced-rank target for large-system TICA. Default: 24
  -feature-export=<b>   Export routed observable features by frame. Default: false
  -feature-source=<m>   auto | dccm | lj | dihedral | qnative | cmap | mixed
  -perf-report=<mode>   off | summary | detail. Default: off
                        Shared cross-module timing/counter report. It keeps
                        existing module-specific performance files unchanged.
  -uncertainty=<mode>   off | block | bootstrap. Default: off
  -uncertainty-blocks=<auto|N>
  -uncertainty-bootstrap=<N>
  -uncertainty-seed=<N>
                        Opt-in block/bootstrap sidecars; point estimates and
                        existing output schemas are unchanged.
  -output-manifest=<b>  Write titan_outputs_manifest.csv and
                        titan_run_config_summary.dat. Default: false
  -dry-run=<b>          Validate and write manifest/schema/config sidecars,
                        then exit before trajectory processing. Default: false
  -schema-export=<m>    off | json. Default: off

  -zts                  Zero-Temperature String minimum-energy path on PCA/TICA FES
  -zts-domain=<mode>    pca | tica. Default: pca unless both domains are active
  -zts-images=<N>       Number of string images. Default: 50
  -zts-iter=<N>         Maximum ZTS iterations. Default: 2000
  -zts-step=<val>       Gradient-descent step size. Default: 0.01
  -zts-tol=<val>        Convergence tolerance. Default: 1e-5
  -zts-source=<id>      Source basin sink_id. Default: auto lowest basin
  -zts-sink=<id>        Sink basin sink_id. Default: auto second-lowest basin
  -zts-sigma=<val>      Gaussian smoothing radius in grid units. Default: 1.0
  -zts-no-smooth        Disable pre-smoothing of the FES grid
  -zts-init=<mode>      Initialization mode. V1 supports: linear

  -msm                  MSM analysis
  -msm-clusters <N>     K-means cluster count. Default: 25
  -msm-lag <N>          MSM lag. Default: 10
  -msm-nts <N>          Number of implied timescales. Default: 3
  -msm-kmeans-iter <N>  K-means iterations. Default: 300
  -msm-seed <N>         Random seed. Default: 42
                        Seeds k-means++, the partial ITS and partial PCA
                        start blocks and the randomized TICA sketch, so
                        repeated runs give identical results.
  -msm-macrostates <N>  Number of macrostates. Default: -1 (auto)
  -msm-no-boundary-split
                        Disable boundary splitting
  -msm-no-pcca          Disable PCCA
  -msm-no-its           Disable implied-timescale scan
  -msm-its-lags <list>  Comma-separated lag list
  -msm-its-solver <m>   ITS spectral solver: full | schur | partial (default: full)
  -msm-its-topk <N>     Top spectrum size for partial solver; 0 = auto from -msm-nts
  -msm-its-max-k-full <N>
                         Full solver cutoff K; if K > cutoff, auto-switch full -> schur
  -eq-weight-source <mode>
                        auto | uniform | dham | mbar. Default: auto
                        Controls which equilibrium frame weights feed the
                        default PCA/TICA FES path and equilibrium-only
                        MSM/TPT summaries. `auto` preserves legacy behavior.
                        An explicitly selected DHAM/MBAR source is required to
                        converge; otherwise TITAN exits with failure.
  -trajectory-semantics <mode>
                        continuous | remd-state. Default: continuous
                        `remd-state` means fixed-temperature REMD streams with
                        configuration swaps. Use fixed-ladder MBAR for
                        equilibrium reweighting. DHAM/dTRAM requires explicit
                        exchange segmentation and is rejected on these raw
                        streams. TICA/MSM may be used only as descriptive
                        state-space models; physical timescales, ITS, CK/VAMP
                        validation, HBK, lagged allostery, and corr-kNN are
                        disabled. Kinetic TPT additionally needs a demux map.

  -dham                 DHAM reweighting
  -dham-temps <file>    Replica temperatures in K
  -dham-ener <file>     File containing list of GROMACS .xvg energy files
  -dham-target-temp <K> Target reweighting temperature. Default: 300
  -dham-ensemble <mode> NVT or NPT DHAM observable. Default: nvt
  -dham-pressure-bar <bar>
                        Target pressure for NPT enthalpy term. Default: 1.0
  -dham-max-iter=<N>    Maximum DHAM iterations. Default: 5000
  -dham-tol=<val>       DHAM convergence tolerance. Default: 1e-6
  -dham-n-replicas=<N>  Explicit replica count validation
  -dham-dt-energy <ps>  Energy sampling interval in ps; 0 = auto
  -dham-window-match <mode>
                        auto | off | strict. Default: auto
                        Raw DHAM .xvg input also honors global -b/-e/-dt

  -mbar                 MBAR reweighting for REMD/HREMD temperature ladders
-mbar-temps <file>    Optional ladder temperatures in K; required for fixed mode,
                      optional override for per-frame mode
  -mbar-ener <file>     File containing list of GROMACS .xvg energy files
  -mbar-target-temp <K> Target reweighting temperature. Default: 300
  -mbar-domain <mode>   tica | pca | both. Default: tica
  -mbar-ensemble <mode> nvt | npt. Default: nvt
  -mbar-window-match <mode>
                         auto | off | strict. Default: auto
  -mbar-temp-mode <mode>
                        fixed | per_frame. Default: fixed
  -mbar-temp-col <col>  XVG temperature-column index for raw per-frame mode.
                        Default: 2
  -mbar-demux-map <file>
                        Optional per-frame state-assignment map for raw
                        replica outputs: rows `time_ps r_0 r_1 ...`
                        (replica_index.xvg), resolved by time
  -mbar-pressure-bar <bar>
                        Pressure of the NPT p V term. Default: 1.0
  -mbar-dt-energy <ps>  Energy sampling interval in ps; 0 = auto
  -mbar-bootstrap <N>   Bootstrap replicates for Cv/<E> uncertainty. Default: 0
                        Moving-block resampling per replica series, block
                        length ceil(4 g) (g = statistical inefficiency);
                        replicates whose MBAR solve fails are dropped and
                        counted in mbar_summary.dat
  -mbar-scan-points <N> Number of temperature scan points; >1 writes mbar_cv.dat
                        Raw MBAR .xvg input also honors global -b/-e/-dt

  -tpt                  Transition Path Theory on the built MSM
  -tpt-mode <mode>      ensemble | kinetic. Default: ensemble
  -tpt-source <list>    Basin spec, e.g. micro:0,1-3 | macro:0 |
                        tica_rect:-1,-1,1,1 | lj:q6_bar>=0.172 |
                        ljpreset:gas | protein:major_macro |
                        protein:rmsd_low | protein:helical
  -tpt-sink <list>      Basin spec, same syntax as -tpt-source
  -tpt-top-paths <N>    Dominant pathways to export. Default: 5
  -tpt-flux-cut <val>   Absolute flux cutoff. Default: 0.0
  -tpt-use-dham <yes/no>
                        Use converged DHAM TPM/pi when available.
                        Default: yes
  -tpt-demux-map <file>
                        Optional demux replica-map for kinetic-TPT. Each row
                        is one time step; each column is one demuxed segment.
  -tpt-bootstrap <N>    Segment bootstrap replicates for uncertainty. Default: 0
  -tpt-bootstrap-seed <N>
                        RNG seed for TPT bootstrap. Default: 42
                        `-eq-weight-source mbar` writes
                        tpt_equilibrium_overlay.dat without changing kinetics

  -llps                 LLPS analysis
  -llps-rcut <val>      Contact cutoff (nm). Default: 0.7
  -llps-perc <val>      Percolation threshold. Default: 0.5
  -llps-fuzzy           Fuzzy EM refinement after DBSCAN
  -llps-dynamics        Enable liquid-like dynamics outputs
  -llps-dynamics-detailed
                        Export detailed chain/pair transition traces
  -llps-maxlag <N>      Max lag for C(t). Default: 500
  -llps-dimer-size <N>  Chains per macro-node. Default: 1
  -llps-contacts        Export contact map data
  -llps-sticker <beads> Sticker bead names
  -llps-voxel <val>     Voxel size for acceleration. Default: 0.5
  -llps-lineage-overlap <val>
                        Lineage containment share: a previous droplet belongs
                        to a new one when >= val of its chains are in it
                        (2+ = fusion); a new droplet descends from a previous
                        one when >= val of its chains were in it (2+ =
                        fission). Default: 0.5
  -llps-prna-rcut <val> Protein-RNA chain contact cutoff (nm). Default: llps-rcut
  -llps-density=<auto|full|projection|off>
                        Density accumulation policy. Default: auto
  -llps-phase-geometry=<none|droplet|slab>
                        Explicit density windows; default none produces NaN phase densities.
  -llps-phase-axis=<a|b|c>
                        Slab normal to the corresponding lattice face (default c).
  -llps-phase-dense-nm=<value>
                        Dense ball radius or slab half-width, measured from GCC center.
  -llps-phase-dilute-inner-nm=<value>
  -llps-phase-dilute-outer-nm=<value>
                        Dilute annulus or paired slab windows. Require
                        0 < dense < dilute-inner < dilute-outer < half cell height.
                        Counts use one BB/CA representative per macro-node.
                        Ambiguous centers/winding droplet embeddings produce NaN + status.
  -llps-dbscan=<true|false>
                        Optional DBSCAN/paper/dynamics companion (default true).
                        DBSCAN and paper profiles support triclinic cells; select paper geometry explicitly.
  -llps-kernel=<auto|scalar|avx2>
                        LLPS hot-kernel policy for benchmark/regression. Default: auto
  -llps-rna-compaction=<auto|true|false>
                        RNA compaction vs protein-binding metrics. Default: auto
  -llps-energy-correlation=<auto|true|false>
                        Contact-gated protein-RNA energy vs RNA compaction
  -llps-energy-detailed=<true|false>
                        Export protein-chain/RNA-chain energy trace. Default: false
  -llps-paper=<true|false>
                        Opt-in Martini protein/RNA paper bundle. Enables LLPS,
                        contacts, dynamics, DBSCAN paper summaries, composition
                        profiles, partition proxies, and convergence report.
                        Default: false
  -llps-paper-profile-bin <nm>
                        Profile bin width for radial/slab paper profiles.
                        Default: 0.2
  -llps-paper-blocks <N|auto>
                        Block count for paper SEM/convergence summaries.
                        Default: auto
  -llps-paper-geometry=<auto|droplet|slab>
                        Profile geometry policy. Default: auto
  -llps-paper-fractal=<auto|true|false>
                        Dense-phase bead/residue fractal descriptors for
                        citeable LLPS paper outputs. Default: auto
                        Morphology-only; not standalone LLPS, binodal, or
                        spinodal evidence.
  -llps-rna-information=<auto|true|false>
                        RNA information-theoretic entropy and coupling metrics
  -llps-rna-topology=<auto|true|false>
                        RNA bridge/clamp/topology metrics in dense phase. Default: auto
  -llps-rna-topology-detailed=<true|false>
                        Export detailed topology/bridge/clamp traces. Default: true
  -llps-rna-intra-rcut <val>
                        RNA intra-chain contact cutoff (nm). Default: llps-rcut
  -llps-rna-longrange-min <N>
                        Minimum residue span counted as long-range RNA contact. Default: 10
  -llps-rna-bridge-geom-min <val|auto>
                        Minimum geometric bridge span (nm). Default: auto
  -llps-rna-patch-gap-max <N>
                        Max residue gap merged into an RNA patch. Default: 1
  -llps-rna-clamp-require-loop <true|false>
                        Require loop support when calling RNA clamps. Default: true
  -llps-rna-info-rg-bins <N>
                        Number of Rg bins for RNA information states. Default: 16
  -llps-rna-info-longrange-bins <N>
                        Number of long-range-contact bins for RNA information states. Default: 16
  -llps-rna-segment-beads <n|none>
        Backbone beads per RNA membership/MSD segment (default 150 = 50 nt at
        3 Martini backbone beads per nucleotide). A long RNA is one
        moleculetype, so a whole-molecule dense/dilute majority vote flickers
        and its COM is not the coordinate of a mobile body. 'none' restores
        whole-molecule behaviour.
  -llps-msd-drift-correct <true|false>
        Subtract the cumulative translation of the dense phase before the
        single-chain MSD (default true).
  -llps-mobile-displacement-nm <val>
        Displacement threshold (nm) for the FRAP-comparable mobile fraction
        (default 2.0).
  -llps-mobile-lag-ns <val|auto>
        Lag at which the displacement mobile fraction is reported
        (default auto = 0.25 x analysed window).
  -llps-msd-fit-start-ns <val|auto>
                        Dense-phase MSD fit window start. Default: auto
  -llps-msd-fit-end-ns <val|auto>
                        Dense-phase MSD fit window end. Default: auto
  -llps-dbscan-eps <val>
                        DBSCAN epsilon in Angstrom. Default: 15.0
  -llps-dbscan-minpts <N>
                        DBSCAN min neighbours. Default: 5

  -gist                 GIST water thermodynamics
  -gist-resolution=<val>
                        Grid resolution in Angstrom. Default: 0.5
  -gist-padding=<val>   Solute padding in Angstrom. Default: 8.0
  -gist-temperature=<val>
                        Temperature in K. Default: 300.0
  -gist-water-model=<name>
                        Water model. Default: TIP3P
  -gist-min-count=<N>   Min voxel count. Default: 10
  -gist-cutoff=<val>    Interaction cutoff in Angstrom. Default: 9.0

  -ifp                  Interaction fingerprint analysis for protein-protein
                        and protein-RNA/DNA systems
  -ifp-hbond <val>      H-bond cutoff. Default: 3.5
  -ifp-salt <val>       Salt-bridge cutoff. Default: 4.0
  -ifp-hphob <val>      Hydrophobic cutoff. Default: 4.0
  -ifp-pipi <val>       Pi-pi cutoff. Default: 5.5
  -ifp-cpi <val>        Cation-pi cutoff. Default: 6.0
  -ifp-hal <val>        Halogen cutoff. Default: 4.0
  -ifp-out <file>       Custom IFP output filename
                        Extra RNA/DNA annotations still go to
                        ifp_rna_annotations.csv when applicable
  -ifp-pbctopo-certified-only[=<bool>]
                        Include only frames whose PBCTOPO status is certified;
                        writes frame-level and summary audit CSV files
  -awh-ifp              RC-binned IFP analysis for GROMACS AWH trajectories
                        Auto-enables -ifp and keeps classic IFP output intact
                        Writes awh_frame_assignments.dat and
                        awh_exploratory_context.dat for exploratory-only
                        biased-ensemble overlays when downstream context exists
  -awh-xvg <file>       Text file listing one walker RC .xvg path per line
  -awh-xvg-col <N>      RC value column index in the .xvg file. Default: 1
  -awh-bins <N>         Number of equal-width RC bins. Default: 20
  -awh-out-prefix <p>   Prefix for AWH representative PDB filenames

  -dssp                 DSSP-like secondary-structure analysis
                        Default: ON
                        Protein/protein-user only
                        In -nocal mode, only molecule 1 is analyzed
  -no-dssp              Disable DSSP analysis
  -dssp-hbond-cutoff <val>
                        DSSP H-bond energy threshold. Default: -0.5

  -lj                   LJ condensation / cluster-density analysis
                        Uses all atoms of molecule 1
                        If no PBC mode is set, Titan auto-enables -pbci 0
  -lj-cut <val>         Cluster cutoff in Angstrom. Default: 5.0
  -lj-min-cluster <N>   Minimum size counted as a cluster. Default: 2
  -lj-rho-bin <val>     rho(r) bin width in Angstrom. Default: 1.0
  -lj-rho-max <val>     rho(r) max radius in Angstrom. Default: auto
  -lj-solid-q6 <val>    q6 threshold for solid-like atoms. Default: 0.40
  -lj-solid-min-cn <N>  Minimum CN for dense/solid-like atoms. Default: 6
  -lj-crystal-tol <val> Crystal classification tolerance. Default: 0.08
  -sq                   Structure-factor / RDF analysis on molecule 1
                        Channels: all, trace, backbone, sticker
                        If no PBC mode is set, Titan auto-enables -pbci 0
  -sq-channel <name>    Repeated selection channel. Default: all
  -sq-rdf-bin <val>     RDF bin width in Angstrom. Default: 1.0
  -sq-rdf-max <val>     RDF max radius in Angstrom. Default: auto
  -sq-q-bin <val>       S(q) bin width in A^-1. Default: 0.05
  -sq-q-max <val>       S(q) max q in A^-1. Default: auto

  -rms                  RMSD + RMSF analysis
                        Atomistic protein/protein-user uses CA
                        Atomistic RNA/DNA uses C4' (fallback C4*)
                        MARTINI uses BB (fallback BB2)
                        In -nocal mode, only molecule 1 is analyzed
                        If no PBC mode is set, Titan auto-enables -pbci 0
  -rmsd                 Alias for -rms
  -rmsf                 Alias for -rms
  -rg2                  Radius-of-gyration analysis on the RMS trace set
                        Outputs Rg^2 in A^2 and Rg in A
                        In -nocal mode, only molecule 1 is analyzed
                        If no PBC mode is set, Titan auto-enables -pbci 0
  -rg2-hist-bins <N>    Histogram bin count for rg2_hist.dat. Default: 100

  -lspmd                Local Spatial Pattern MD analysis
  -lspmd_cut=<val>      C-alpha cutoff in Angstrom. Default: 12.0

  -cas[=true|false]     GBSA alanine-like side-chain neutralization scan.
                        Auto-enables -gbsa; incompatible with -nocal=true.
  -hotspot-report=<bool>
                        Export CAS/EDecomp hotspot_residue_ranking.csv,
                        hotspot_summary.dat, and hotspot_bfactor.pdb when
                        a B-factor template is available.
  -hotspot-source=<m>   cas, edecomp, or both. Default: both
  -hotspot-rank-by=<m>  cas-ddg, edecomp-abs-mean, or combined
  -tda                  Topological data analysis
  -tda-max-points=<N>   Reservoir cap for TDA point-cloud CSV. Default: 200000
  -ml                   ML feature export (affinity prediction closed)

4.6 Hydrogen Bonds / Water / Geometry

  -hb                   Hydrogen-bond analysis
  -hb side-chain        Side-chain-only hydrogen bonds
  -ahb <val>            H-bond angle cutoff. Default: 135.0
  -dhb <val>            H-bond distance cutoff. Default: 3.5

  -water-type <type>    Water mode: surface or sphere
  -dis-water <val>      Water distance cutoff. Default: 3.5

  -charged_atom_dis [N] -nocal: distribution of the distances from mol1's
                        centre of mass to its charged atoms, per input
                        (<input>_distributionChargedAtoms.dat); N bins (400)
  -charged_distances_min <val>
                        Min charged distance
  -charged_distances_max <val>
                        Max charged distance
  -angle_dipole_efield [N]
                        -nocal: distribution of the cosine between each water
                        dipole and the solute field there, per input; N bins
  -cal-mindis           Minimum distance per frame
  -resdis2com           -nocal: distance from each residue's centre of mass
                        to mol1's, mean and SD per input
  -com-dis <axis>       mol1-mol2 centre-of-mass distance per frame
                        (out_dis_com.dat): x, y, z, or xyz (default)

4.7 Miscellaneous

  -nocal                No-energy mode; useful for PBC/utility workflows
                        Explicit -g=<mol1>-<mol2> keeps both molecules for PBC
                        and -opdb/-pqr output without enabling energy modules
                        DSSP in this mode analyzes molecule 1 only
                        LJ in this mode analyzes molecule 1 only
                        RMSD/RMSF in this mode analyzes molecule 1 only
                        GBSA/ALPB are ignored in this mode
  -cscd                 Side-chain centre-of-mass distance from each mol1
                        residue to the nearest mol2 residue, one file per
                        residue (out_cal_dis_com_<chain><resnr>.dat);
                        -act gives one column per mol2 residue. Glycine,
                        non-amino-acid residues and -side_chain_contact
                        full-res use whole-residue centres; the file header
                        says which
  -pqr                  Write PQR-style output
  -cal-force            Calculate atomic forces
  -cal-volume           -nocal: box volume per frame (volume.dat) and its
                        mean, SD and 95% CI (stat_volumes.dat)
  -max-force            Report maximum force
  -cache                Enable cache
  -cache -no            Disable cache
  -use-CI               Use confidence intervals
  -elec-screening       Debye-Huckel screening for aqueous 1:1 electrolyte at
                        298.15 K (kappa=3.291*sqrt(I) nm^-1).
  -dipole-moment        -nocal: mol1 dipole moment per frame
                        (<input>_dipole_moment.dat)
  -split                Split results per trajectory
  -all-subfolders       Process all subfolders and split outputs
  -cat-result           Concatenate results from multiple inputs
  -all-respair          Output all residue pairs
  -contact-map yes      Deprecated alias for interface sidechain-com CMap
  -contact-map no       Disable the deprecated alias only
  -act                  -cscd: every mol2 residue, not only the nearest
  -new_al               New alignment in single-PDB mode
  -new-al               Alias of -new_al
  -side_chain_contact <mode>
                        full-res or other
  -noprint-side_chain_contact
                        Suppress side-chain contact printing

4.8 Pull Direction Optimizers

  -pull-direct          Pulling-direction search on the first input only.
                        Uses the last snapshot of the first pdb/xtc/trr input.
                        If -e is set, uses the last snapshot with time <= -e.
                        TITAN exits after this module finishes and ignores
                        remaining inputs for pull-direct.
  -pull-opt <mode>      Optimizer mode. Current build supports: de, de+pattern,
                        de-shade, de-l-shade, de-jso, de-jso+pattern,
                        de-nl-shade-lbc, de-nl-shade-lbc+pattern, direct,
                        direct+pattern, corridor-grid (explicit pathway only)
  -pull-max-evals <N>   Global evaluation budget. Default: 0 (unlimited)
  -pull-seed <N>        Deterministic seed for pull-direction search. Default: 42
  -de-diff-weight <val> Differential weight for legacy de only. Default: 0.9
  -de-cross-prob <val>  Crossover probability for legacy de only. Default: 0.9
  -de-F <val>           Scaling factor for legacy de only. Default: 0.5
  -de-scan-radius <val> Scan radius. Default: 2.0
  -de-min-scan-radius <val>
                        Minimum scan radius. Default: 0.5
  -de-score-function <N>
                        Score function index
  -de-pop-num <N>       Population size. Default: 6480
  -de-ite-num <N>       Maximum iterations. Default: 100
  -de-algorithm <N>     Legacy de mutation variant
  -de-angle <val>       Restrain angle. Default: 90.0
  -de-auto-term         Auto-termination
  -de-adaptive-F        Adaptive F scaling for legacy de only
  -de-stagnation-limit <N>
                        Stagnation limit. Default: 20
  -de-memory-size <N>   SHADE-family memory size. Default: 10
  -de-p-best-rate <val> SHADE-family p-best rate. Defaults: 0.2 (de-shade,
                        de-l-shade, de-nl-shade-lbc), 0.11 (de-jso,
                        de-jso+pattern)
  -de-archive-scale <v> SHADE-family archive size scale. Default: 1.0
  -de-min-pop-num <N>   Minimum population for de-l-shade / de-jso /
                        de-jso+pattern / de-nl-shade-lbc /
                        de-nl-shade-lbc+pattern. Default: 4
  -pathway-file <file>  Ordered TITAN_PATH v1 Cartesian route. This is the
                        publication path input and is mutually exclusive with
                        -de-pathway-group. Relative paths in arg files resolve
                        from the arg-file directory.
  -pathway-route-id <id>
                        Require an exact route ID from -pathway-file.
  -pathway-graph-edge-cutoff <A>
                        Legacy group auto-builder edge cutoff. Default: 4.0
  -pathway-previous-status <file>
                        Explicit prior CONTINUE status, bound to the same route.
                        Restores mode and prior progress between MD segments.
  -pathway-continuity-window <A>
                        Required positive half-window with previous-status.
                        Rejects artificial-edge clipping; never auto-widens.
  -pathway-recovery-enter <A>
                        Enter route-recovery mode above this distance. Default: 4.0
  -pathway-recovery-exit <A>
                        Return to advance mode below this distance. Default: 3.0
  -pathway-recovery-max-turn-angle <deg>
                        Maximum direction rotation for a Recovery micro-step.
                        Default: 60
  -pathway-corridor-spacing <A>
                        Arc-length spacing for corridor geometry. Default: 0.5
  -pathway-corridor-radial-samples <N>
                        Deterministic disk samples per longitudinal point,
                        in addition to the center. Default: 16
  -pathway-registration-max-rmsd <A>
                        Maximum rigid-anchor fit RMSD. Default: 2.0
  -pathway-completion-tolerance <A>
                        Terminal progress tolerance. Default: 1.0
  -pathway-completion-min-separation <A>
                        Require this minimum receptor-ligand atom separation
                        at the terminal; 0 disables the physical gate. Default: 0
  -pathway-max-turn-angle <deg>
                        Maximum controller direction rotation per handoff. Default: 20
  -pathway-energy-key-epsilon <val>
                        Near-optimal energy-key window before maximizing progress.
                        Default: 0.05
  -pathway-min-clearance <A>
                        Reject translated ligand targets below this minimum
                        receptor-ligand atom distance. Default: 1.5
  -pathway-probe-length <A>
                        Evaluate each selected direction at this fixed applied
                        displacement; 0 keeps legacy target-length evaluation.
                        Default: 0
  -pathway-probe-lengths <A,A,...>
                        Strictly decreasing trust-region ladder. Each direction
                        uses its largest mode-eligible hard-valid probe.
                        Example: 1,.5,.25
  -pathway-navigation    Enable NAV-P1 algorithmic navigation cost and
                        deterministic guided direction seeds. Requires an
                        explicit route and fixed probe; default: disabled
  -pathway-navigation-tube-weight <val>
                        Dimensionless flat-bottom tube cost weight. Default: 1
  -pathway-navigation-backtrack-weight <val>
                        Dimensionless backtracking cost weight. Default: 1
  -pathway-navigation-cost-epsilon <val>
                        Near-minimum navigation-cost window applied after the
                        physical energy-key window in Advance. Default: 0
  -pathway-navigation-seed-radial-samples <N>
                        Deterministic guided-direction normal-ring samples.
                        Range: 0..64. Default: 8
  -pathway-navigation-seed-amplitude <val>
                        Dimensionless normal-ring perturbation amplitude.
                        Default: 0.35
  -pathway-clearance-mode <mode>
                        endpoint-center (legacy/default) or
                        swept-linear-center (exact continuous hard clearance
                        for a rigid translation). Swept mode requires a fixed
                        probe plus -pbcn or -pbctopo upstream reconstruction;
                        it uses direct distances in that reconstructed Cartesian
                        frame and does not add a soft steric score.
  -pathway-geometry-only
                        Ablation: rank feasible targets by geometry/progress only.
  -de-pathway-group <N> Legacy residue-group auto-builder (subset of g1)
  -de-pathway-radius <A>
                        Compatibility alias: graph edge cutoff = A, recovery
                        enter = A, recovery exit = 0.75 x enter (the default
                        3/4 ratio, so the hysteresis band keeps a width).
                        Explicit -pathway-graph-edge-cutoff and
                        -pathway-recovery-enter/-exit win in any order.
                        Prefer the split options.
                        In pathway mode TITAN writes pull_direction_status.txt
                        for the external TITAN/GROMACS controller.
                        Pathway mode supports DE-family plus direct and
                        direct+pattern in the current build.
                        Recovery mode prioritizes reattaching mol2 to the
                        nearest reachable forward point on the pathway before
                        energy. Advance mode first excludes non-forward
                        candidates, then maximizes forward progress within the
                        energy-epsilon pool.
                        Legacy outputs DE_out_vector.dat / DE_log.dat /
                        DE_report.dat / DE_vector.pdb / best_conf.pdb stay in
                        the original input frame. Only
                        pull_direction_aligned_snapshot.pdb and
                        pull_direction_system_info.txt are rotated so that the
                        best pull vector points to +Z.
                        de-nl-shade-lbc is the recommended DE-family branch in
                        the current build. direct+pattern remains the strongest
                        score-focused competitor on the benchmark case.
                        direct can be faster than de on small/medium cases
                        because it often reaches a good answer with far fewer
                        energy evaluations.

4.9 Help / Version

  -h, --help            Print help message
  -vibe-help            Short reference for VIBE (-pbctopo) and its frame report
  -hololift-help        Short reference for HoloLift (-pbctopo-temporal=hololift)
  -v, --version         Print build number

================================================================================
                               END OF MANUAL
================================================================================
)MANUAL";
    return 0;
  }
}

void arguments::vibe_help_print() {
  cout << R"VIBE(
TITAN-VIBE quick reference                  (all TITAN options: titan.exe -h)

VIBE places every part of a periodic frame (a molecule, or a chain of a
complex) at the lattice image that keeps the contacts it makes in the
simulation, and labels how far the contacts determine that placement.
Run it with an argument file, one option per line:  titan.exe -a arg.in

Input
  -f=frames.pdb             multi-model PDB, a CRYST1 record in every model
  -s=ref.pdb                reference PDB, with
  -traj=run.xtc             an XTC or TRR trajectory
  -1pdb=true                read -f as one multi-model PDB
  -n=index.ndx -g=0-0       index file; group 0 is reconstructed
  -common_topo=topol.top    GROMACS topology; its bonds define the parts
  -ff=<name>                force field of the topology (CHARMM36, AMBER99SB,
                            MARTINI, ...); -path-a/-path-n point at lib_ff
  -b=<ps> -e=<ps> -dt=<ps>  frame window and stride of a trajectory
  -nt=<N>                   threads

VIBE
  -pbctopo=true             turn VIBE on
  -nocal=true               coordinates only, no energy terms
  -opdb=out.pdb             output coordinates, one model per frame
  -pbctopo-mode=llps-local  phase-separated systems: make each molecule whole
                            where it is, without placing molecules relative to
                            each other
  -pbctopo-search-policy=fast-certified|audit-exact
  -pbctopo-contact-cutoff=<A>
                            contact distance; 4.5 A, MARTINI 8.0 A
  -pbctopo-steric=report|filter
                            report (default): overlaps between parts do not
                            change the placement, they are counted in the input;
                            filter: the 1.0.x candidate rejection

Output files, in the working directory
  out.pdb                   placed coordinates
  pbctopo_frame_report.csv  one row per frame

Frame report: evidence_state
  supported_connected  the contacts connect all parts and every cycle of
                       contacts closes: the frame fixes the placement up to
                       one common lattice translation
  weak_disconnected    some parts are joined to the rest by no contact path;
                       the frame does not fix their relative images
  extended_periodic    the contact network closes through the boundary (a pair
                       in contact at two images, a part touching its own image,
                       or a longer ring): no finite placement keeps every
                       contact.  contact_cycle_rank is 1 for a filament, 2 for
                       a sheet, 3 for a network; contact_cycle_b* are its
                       generators in cell vectors
  weak_ambiguous       the contacts close, but the selected placement leaves
                       a contact relation unsupported
  contradicted         the selected placement loses an observed contact
  not_evaluated        no placement was certified (status fallback)
  not_applicable       llps-local frames

Frame report: other columns
  status               certified | weak_observation | contradicted | fallback
                       | local_unwrap
  input_overlap_pairs  atom pairs of two parts closer than lambda*(ri+rj) in
                       the input; under periodic boundary conditions they are
                       present for every placement
  self_image_units     parts within the contact distance of their own image
  bonded_winding_rank  > 0 when a bonded cycle itself runs through the boundary
                       (fallback_reason bonded_winding)
)VIBE";
}

void arguments::hololift_help_print() {
#ifdef TITAN_WITHOUT_HOLOLIFT
  cout << "HoloLift is not part of this build (HOLOLIFT=0).\n";
#else
  cout << R"HOLOLIFT(
HoloLift quick reference                    (all TITAN options: titan.exe -h)

What it does
  In a periodic simulation every molecule has many copies, one per cell.  For
  each frame VIBE lists the placements of the molecules that the frame allows,
  ranked, and keeps its first choice; that choice is made for one frame at a
  time.  HoloLift chooses one placement per frame for the whole trajectory at
  once, so that the molecules move continuously: it moves a molecule to
  another copy of its partner only where the contacts in the frames call for
  it.  The answer is exact on VIBE's candidate lists.

Run it
  Write the options into a file, one per line, and run
      titan.exe -a arg.in
  in the folder that should receive the output.

    -s=reference.pdb        a structure in the atom order of the trajectory;
                            its atom records name the atoms of the output PDB
    -traj=run.xtc           the trajectory (XTC)
    -n=index.ndx  -g=0-0    index file; group 0 holds the molecules to follow
    -common_topo=topol.top  GROMACS topology of the system; its bonds define
                            the molecules
    -ff=charmm36            force field of the topology, with
    -path-a=<folder>        -path-a and -path-n pointing at the forcefields
    -path-n=<folder>        folder of the release
    -pbctopo=true           VIBE, the reconstruction of each frame
    -pbctopo-temporal=hololift
                            HoloLift after VIBE
    -nocal=true             coordinates only, no energy terms
    -no-dssp=true  -noprint-side_chain_contact=true
                            skip TITAN's structure analyses
    -nt=8                   threads (no thread count changes a result)
    -b=<ps> -e=<ps> -dt=<ps>
                            optional: first and last time, and stride

  A worked example with known answer: examples/01_separation, explained step
  by step in docs/TUTORIAL.md.

The line to look at
  A run ends with one line on the screen; for examples/01_separation
    PASS system=hololift frames=250 ... evidence_unsupported=0 ...
         changed=60 gaps=0 ...
  frames                frames read
  changed               frames where HoloLift chose differently from VIBE's
                        first choice for that frame
  evidence_unsupported  frames where HoloLift kept a placement without contact
                        evidence although another placement had it; 0 means
                        every frame keeps the contacts it shows
  gaps                  frames VIBE could not reconstruct; the path is solved
                        separately on each side of a gap
  Any line other than PASS, and any exit code other than 0, is an error; its
  class (TRAJECTORY_ERROR, SOLVE_ERROR, ...) is printed on standard error.

Output files, in the working directory
  hololift_coordinates.xtc  the reconstructed trajectory of the -g group (nm)
  hololift_coordinates.pdb  its first frame: load it, then the XTC, in VMD,
                            PyMOL or ChimeraX
  hololift_frame_results.csv
                            one row per frame:
                              framewise_* = VIBE's first choice
                              temporal_*  = HoloLift's choice
                              *_component_images = the cell shift (a:b:c) of
                                each molecule, separated by |
                              changed_from_framewise = 1 where they differ
  hololift_system_summary.csv
                            the totals of the run and the settings used
  hololift_segment_results.csv, hololift_coordinates.csv,
  hololift_policy_admissions.csv
                            details: see docs/TUTORIAL.md
  pbctopo_*.csv             the VIBE results HoloLift read (see -vibe-help)

Settings
  titan.exe runs HoloLift with its default settings: only placements that VIBE
  certified are used, rank weight 1, transition weight 1, and a charge of 1000
  on a placement without contact evidence when another one has it.  The
  written coordinates keep HoloLift's relative placements and follow the
  motion of the whole assembly across cells, also when the box changes size.
  The standalone runner (tools/hololift_temporal_value_runner.cpp) takes other
  settings as environment variables; see README.md.
)HOLOLIFT";
#endif
}
