# HoloLift 1.2.0

## New

- **A Windows binary.** The release page has `HoloLift-1.2.0-windows-x64.zip` with `titan.exe` (portable profile:
  any x86-64 processor with AVX2 and FMA; statically linked), the force-field tables, the examples and the
  documents. It runs without a compiler or any installation. `SHA256SUMS.txt` in the zip covers its files.
- **An example with a known answer** (`examples/01_separation`): two peptides that part, drift around the point
  halfway between two copies of each other, and meet again. `check_example.py` compares the result with the design.
- **A tutorial** (`docs/TUTORIAL.md`): the example step by step, how to read the output, how to view the
  trajectories, and how to set up and check a run on your own trajectory.
- **`titan.exe -hololift-help`**: a one-page reference of the options, the PASS line and the output files.
- **The force-field tables** (`forcefields/`), so that `-path-a` and `-path-n` have something to point at.

## Faster, same output

VIBE's reconstruction of each frame and HoloLift's selection are faster. These changes leave every output file the
same byte for byte (checked against the source before them, at 1 and 6 threads); only the run times change.

- The minimum spanning tree that scores a candidate placement over many molecules is computed in one pass that
  updates and selects at once, with the same arithmetic and tie order as before: 2.5 times faster per call.
- The final rescoring of VIBE's beam search runs on the `-nt` threads, in order.
- The steric scan skips atoms that cannot meet another molecule; the nearest images of each frame's bonds are
  computed once.
- HoloLift's dynamic program computes each distinct transition once per frame instead of once per pair of
  candidates.

On the 21-chain protein-RNA assembly of the paper (151 frames, 6 threads) a run takes 19 min instead of 39 min; on a
201-frame window of the amyloid dimer at 1 thread it takes 17 s instead of 43 s, and the selection 0.6 s instead of
3.8 s.

## Changed

- VIBE is that of [TITAN-VIBE 1.1.0](https://github.com/hoanglinh221191/titan-vibe) (see its release notes):
  - steric overlaps between molecules are counted in the frame report instead of removing candidates;
    `-pbctopo-steric=filter` restores the rule of 1.1.0;
  - a new evidence state, `extended_periodic`, marks contact networks that close through the boundary; HoloLift
    reads it as a state of the earlier report (`weak_ambiguous`, `supported_connected` or `weak_disconnected`,
    from the frame's relations);
  - the `-common_topo` reader follows `[ molecules ]` over every molecule type of every included file.
- The method is called HoloLift (1.1.0: HoloLift-MD). The repository keeps its name.
- The README is shorter and starts from the binary; the full description of the output files and of the standalone
  runner is kept.

## Source

The source tree is exported from TITAN commit 8758139ee163 (`SOURCE_PROVENANCE.json`). `titan.exe -v` names that
commit.
