# HoloLift-MD

HoloLift-MD chooses, for every frame of a molecular-dynamics trajectory, which periodic image of each molecule to use,
so that an assembly of intact molecules stays continuous in time. A single wrapped frame does not always decide this:
two molecules can be out of contact, or one can touch two periodic copies of its partner at once. HoloLift-MD takes the
image candidates that a framewise reconstruction proposes for each frame and selects one path through them for the
whole trajectory.

- **Input.** Per-frame candidate lists and their contact evidence from [VIBE](https://github.com/hoanglinh221191/titan-vibe),
  and the trajectory that VIBE read (`.xtc`).
- **Objective.** An ordinal rank cost, a transition cost (the weighted squared displacement between frames after the
  common lattice translation is removed by an exact closest-vector search), and a charge on any selection that lacks
  contact evidence when another candidate has it.
- **Solver.** Dynamic programming, exact on the supplied candidate graph.
- **Report.** The selected path, whether each selection is supported by contact evidence, local geometric
  certificates for each transition, and whether the candidate domain is complete. These are reported separately.

## Build

Tested on Windows 11 with MSYS2 UCRT64 and Clang 22.1.4 (C++23). From PowerShell in the repository root:

```powershell
.\tools\build_hololift_temporal_value_runner.ps1 -Compiler clang++ -Output build\hololift_temporal_value_runner.exe `
    -ExtraCompilerFlags '-O3','-march=native'
```

The script compiles `tools/hololift_temporal_value_runner.cpp` with the sources listed in it. On other platforms the
same file list can be passed to any C++23 compiler with `-Isrc -Isrc/gmxtraj/include -pthread`; that route has not been
tested.

## Run

```text
hololift_temporal_value_runner <system-label> <vibe-package-dir> <trajectory.xtc> <output-dir> [--transport-radius <N>]
```

| Argument | Meaning |
| --- | --- |
| `system-label` | Free text written into the `system` column of every output |
| `vibe-package-dir` | Output folder of a VIBE run with frame reports: `pbctopo_frame_report.csv`, `pbctopo_assignment_candidates.csv`, `pbctopo_component_layout.csv`, `pbctopo_hard_edges.csv`, `pbctopo_temporal_path.csv`; `pbctopo_rejected_candidates.csv` is read when present |
| `trajectory.xtc` | The trajectory VIBE read. Every frame is bound to the VIBE package by its coordinate and box digests, and a mismatch stops the run |
| `output-dir` | Where the results are written |
| `--transport-radius <N>` | Optional: widen each frame's candidate list by relative-image translations up to radius `N` |

Settings are read from environment variables. Unset variables keep the defaults.

| Variable | Default | Meaning |
| --- | --- | --- |
| `HOLOLIFT_EVIDENCE_SET_WEIGHT` | `1000` | w_s, the charge on a selection without contact evidence when another candidate has it |
| `HOLOLIFT_RANK_WEIGHT` | `1` | w_r, the ordinal rank weight |
| `HOLOLIFT_TRANSITION_WEIGHT` | `1` | w_p, the transition weight |
| `HOLOLIFT_RANK_COST` | `linear` | `linear` or `saturated` rank cost |
| `HOLOLIFT_EVIDENCE_ABSENT` | `strict` | Candidate admission: `strict` (provider-certified candidates only), `carry` or `restore` |
| `HOLOLIFT_EVIDENCE_WEIGHT` | `0` | w_e, an optional charge on every candidate other than a unique supported one |
| `HOLOLIFT_DP_WORKERS` | `1` | Threads for the dynamic program (1-64); the result does not depend on it |

Example (PowerShell):

```powershell
$env:HOLOLIFT_EVIDENCE_SET_WEIGHT = '1000'
.\build\hololift_temporal_value_runner.exe dimer .\vibe_out .\traj.xtc .\hololift_out
```

## Outputs

| File | Content |
| --- | --- |
| `hololift_frame_results.csv` | One row per frame: framewise and selected (`temporal_*`) candidates and component images, emission and transition costs, the certificate counters of the transition into the frame, and `selected_evidence_unsupported` |
| `hololift_segment_results.csv` | One row per valid segment: objective, second-best objective and gap, number of optimal paths (capped at 2), and the certificate flags |
| `hololift_system_summary.csv` | The settings used (weights, rank form, admission policy), the objective, `changed_from_framewise_frames`, `evidence_unsupported_selections`, observation gaps and `audit_digest_sha256` |
| `hololift_policy_admissions.csv` | Candidates admitted by `carry` or `restore` |
| `hololift_transport_band_candidates.csv` | Candidates added by `--transport-radius` |

The program prints one summary line and exits with 0 on success; any other exit code comes with an error class
(`TRAJECTORY_ERROR`, `PREPARATION_ERROR`, `SOLVE_ERROR`, ...) on standard error.

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

MIT (see `LICENSE`). The XTC reader in `src/gmxtraj` has its own terms; see `THIRD_PARTY_NOTICES.md`.
