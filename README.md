# HoloLift-MD

HoloLift-MD chooses, for every frame of a molecular-dynamics trajectory, which periodic image of each molecule to use,
so that an assembly of intact molecules stays continuous in time. A single wrapped frame does not always decide this:
two molecules can be out of contact, or one can touch two periodic copies of its partner at once. HoloLift-MD takes the
image candidates that a framewise reconstruction proposes for each frame and selects one path through them for the
whole trajectory.

- **Input.** Per-frame candidate lists and their contact evidence from VIBE, the framewise reconstruction of TITAN
  ([VIBE paper software](https://github.com/hoanglinh221191/titan-vibe)), and the trajectory that VIBE read (`.xtc`).
- **Objective.** An ordinal rank cost, a transition cost (the weighted squared displacement between frames after the
  common lattice translation is removed by an exact closest-vector search), and a charge on any selection that lacks
  contact evidence when another candidate has it.
- **Solver.** Dynamic programming, exact on the supplied candidate graph.
- **Report.** The selected path, whether each selection is supported by contact evidence, local geometric
  certificates for each transition, and whether the candidate domain is complete. These are reported separately.
- **Coordinates.** The selected path written as a trajectory: HoloLift's relative images, with the whole assembly
  translated by displacement-preserving (toroidal) unwrapping, so that constant-pressure box rescaling does not enter
  the absolute motion.

Since version 1.1.0 this repository holds the TITAN source tree that runs the whole chain in one command: TITAN reads
the trajectory, VIBE reconstructs each frame and writes its candidate package, and HoloLift-MD selects the path and
writes it as coordinates. TITAN's other analyses are part of the same source. The standalone runner of version 1.0.0,
which starts from an existing VIBE package, is still included.

## Build TITAN

Tested on Windows 11 with MSYS2 UCRT64 (Clang 22.1.4, GNU make, Python 3). From a shell in the repository root:

```powershell
C:\msys64\ucrt64\bin\mingw32-make.exe all -j8
```

This writes `titan.exe`. The default `PROFILE=portable` needs a CPU with AVX2 and FMA; `PROFILE=native` builds for
the CPU at hand. `make all` on Linux with Clang 20 or later, or g++ 13 or later, follows the same Makefile; that route
has not been tested for this release.

## Run: from a trajectory to HoloLift coordinates

Write the options into a file, for example `arg.in`:

```text
-s=protein.pdb
-traj=traj.xtc
-n=index.ndx
-g=0-0
-common_topo=topol.top
-ff=charmm36
-pbctopo=true
-pbctopo-temporal=hololift
-nocal=true
-nt=8
```

and run `titan.exe -a arg.in` in the folder that should receive the output.

| Option | Meaning |
| --- | --- |
| `-s` | Structure whose atom order matches the trajectory; its atom records name the atoms of the output PDB |
| `-traj` | The trajectory, as `.xtc` |
| `-n`, `-g` | Index file and the group that holds the molecules of the assembly (`-g=0-0`: group 0 as one system) |
| `-common_topo` | GROMACS topology of the selection; its bonds define the molecules, its atom types the steric radii |
| `-ff` | Force field of the topology |
| `-pbctopo=true` | VIBE framewise reconstruction (`-pbctopo-mode=vibe`, the default) |
| `-pbctopo-temporal=hololift` | HoloLift-MD after VIBE, with the default settings (strict admission, w_r = w_p = 1, w_s = 1000) |
| `-nocal=true` | No energy calculation |
| `-nt` | Threads, used by VIBE and by HoloLift-MD |
| `-b`, `-e`, `-dt` | Optional: begin and end time (ps) and frame stride |

`titan.exe -h` lists every option; `docs/TITAN_MANUAL.txt` is the full manual. The run writes:

| File | Content |
| --- | --- |
| `pbctopo_*.csv` | The VIBE package: per-frame reports, candidate lists, component layout and hard edges |
| `hololift_frame_results.csv` | One row per frame: framewise and selected (`temporal_*`) candidates and component images, emission and transition costs, the certificate counters of the transition into the frame, and `selected_evidence_unsupported` |
| `hololift_segment_results.csv` | One row per valid segment: objective, second-best objective and gap, number of optimal paths (capped at 2), and the certificate flags |
| `hololift_system_summary.csv` | The settings used, the objective, `changed_from_framewise_frames`, `evidence_unsupported_selections`, observation gaps and `audit_digest_sha256` |
| `hololift_policy_admissions.csv` | Candidates admitted by the `carry` or `restore` policies (empty under `strict`) |
| `hololift_coordinates.xtc` | The selected path: the selected atoms in the bound atom order, nm |
| `hololift_coordinates.pdb` | Its first frame, with the atom records of `-s` |
| `hololift_coordinates.csv` | Per frame: the anchor component's centre on the lattice and on the written path, and the translation between them |

In the written coordinates the relative images are HoloLift's, and the centre of the anchor component (the largest
one) advances by the nearest-image displacement between its wrapped centres in consecutive cells (Bullerjahn et al.,
J. Chem. Theory Comput. 2023). A frame after an observation gap restarts the anchor at its lattice position, and gap
frames are not written. Where the selected relative image changes, one molecule necessarily moves by a cell vector in
absolute coordinates; which one depends on the anchor.

The trajectory is read in blocks, as TITAN's other modules read it, never held whole: once to bind and prepare every
frame, and once after the solve to audit the all-atom reconstruction of the selected path and write it out. Frames
within a block are processed on `-nt` threads; neither the block size nor the thread count changes any output.

## Run: the standalone runner on an existing VIBE package

```powershell
.\tools\build_hololift_temporal_value_runner.ps1 -Output build\hololift_temporal_value_runner.exe
```

The script compiles `tools/hololift_temporal_value_runner.cpp` with the sources listed in it, by default with
`clang++ -O3 -march=native`. Every frame is bound to the VIBE package by digests of its wrapped coordinates, compared
bit for bit, so build the runner with the compiler and flags of the `titan.exe` that wrote the package. For a
portable TITAN build:

```powershell
.\tools\build_hololift_temporal_value_runner.ps1 -Output build\hololift_temporal_value_runner.exe `
    -OptimizationFlags '-O3','-mavx2','-mfma','-mtune=generic'
```

A mismatch shows as `TRAJECTORY_ERROR wrapped XTC coordinates do not match VIBE binding` on a triclinic cell.

```text
hololift_temporal_value_runner <system-label> <vibe-package-dir> <trajectory.xtc> <output-dir> [--transport-radius <N>]
```

| Argument | Meaning |
| --- | --- |
| `system-label` | Free text written into the `system` column of every output |
| `vibe-package-dir` | Output folder of a VIBE run: `pbctopo_frame_report.csv`, `pbctopo_assignment_candidates.csv`, `pbctopo_component_layout.csv`, `pbctopo_hard_edges.csv`; `pbctopo_rejected_candidates.csv` and `pbctopo_temporal_path.csv` are read when present |
| `trajectory.xtc` | The trajectory VIBE read |
| `output-dir` | Where the results are written |
| `--transport-radius <N>` | Optional: widen each frame's candidate list by relative-image translations up to radius `N` |

Settings are read from environment variables; unset variables keep the defaults.

| Variable | Default | Meaning |
| --- | --- | --- |
| `HOLOLIFT_EVIDENCE_SET_WEIGHT` | `1000` | w_s, the charge on a selection without contact evidence when another candidate has it |
| `HOLOLIFT_RANK_WEIGHT` | `1` | w_r, the ordinal rank weight |
| `HOLOLIFT_TRANSITION_WEIGHT` | `1` | w_p, the transition weight |
| `HOLOLIFT_RANK_COST` | `linear` | `linear` or `saturated` rank cost |
| `HOLOLIFT_EVIDENCE_ABSENT` | `strict` | Candidate admission: `strict` (provider-certified candidates only), `carry` or `restore` |
| `HOLOLIFT_EVIDENCE_WEIGHT` | `0` | w_e, an optional charge on every candidate other than a unique supported one |
| `HOLOLIFT_DP_WORKERS` | `1` | Threads for the dynamic program (1-64) |
| `HOLOLIFT_TRAJECTORY_WORKERS` | `HOLOLIFT_DP_WORKERS`, else all hardware threads | Threads for binding, preparation, audit and coordinates (1-64) |
| `HOLOLIFT_TRAJECTORY_BLOCK` | automatic | Frames per trajectory block |
| `HOLOLIFT_COORDINATES` | `none` | `lattice` or `toroidal`: also write `hololift_coordinates.{xtc,pdb,csv}` |
| `HOLOLIFT_PDB` | `first` when coordinates are written | `none`, `first` or `all` frames as PDB models |
| `HOLOLIFT_REFERENCE_PDB` | none | Structure whose atom records name the PDB atoms |
| `HOLOLIFT_TIMING` | unset | Any value: wall time of each stage on standard error |

No thread count or block size changes a result. The program prints one summary line and exits with 0 on success;
any other exit code comes with an error class (`TRAJECTORY_ERROR`, `PREPARATION_ERROR`, `SOLVE_ERROR`, ...) on
standard error.

## Tests

```powershell
.\test\hololift_phase0_smoke.ps1 -Compiler clang++
.\test\hololift_phase1_smoke.ps1 -Compiler clang++
```

Each script compiles a self-contained test against the sources and runs it; a nonzero exit code is a failure.

## Citation

If you use HoloLift-MD, please cite the paper (see `CITATION.cff`):

My Na O, Ngo Thi Chinh, Tam V.-T. Mai, Quang Linh Huynh, Mai Suan Li, Hoang Linh Nguyen.
*HoloLift-MD: Temporal Selection of Periodic Images for Molecular Assemblies in Simulation Trajectories.* Submitted.

## License

MIT (see `LICENSE`). The XTC reader in `src/gmxtraj`, the Eigen library in `src/eigen` and the GBn2 tables in
`src/titan_gbn2_tables.h` have their own terms; see `THIRD_PARTY_NOTICES.md`. `SOURCE_PROVENANCE.json` names the
TITAN commit this source tree was exported from.
