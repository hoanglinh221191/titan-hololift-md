# HoloLift 1.2.1

## Fixed

- **A run with several threads could stop and never end.** In the temporal selection, the thread that waits for
  the workers of the dynamic program could miss their last "done" signal and then wait for good, with no error
  message. It was seen with `-nt=6` on six processor cores. The results of a run that ended were never affected.
  The signal is now sent under the lock that the waiting thread holds.

## Faster, same output

VIBE's reconstruction of an assembly of many molecules is faster. These changes leave every output file the same
byte for byte (checked against the source of 1.2.0 on the paper's protein-RNA window at 1, 4 and 6 threads and on
the amyloid dimer window); only the run times change.

- A candidate placement moves whole molecules and leaves their shape as it is. The minimum spanning tree that
  scores a placement now keeps what the moves leave unchanged (the tree edges inside each molecule, and the
  shortest links between two molecules at a given relative position) and recomputes only the rest, with the same
  arithmetic and tie order as before.
- In the beam search, the bond terms of a molecule are computed once per periodic shift instead of once per shift
  and beam state; every state still adds them in the same order.
- The expansion of the beam and the first stage of the local refinement run on the `-nt` threads and are committed
  in the order of the serial code.

On the 21-chain protein-RNA assembly of the paper (151 frames, 6 threads) a run takes 3.3 min instead of
15 min. Runs on two molecules, such as the amyloid dimer, take as long as before.

## Source

The source tree is exported from TITAN commit d859c145c36b (`SOURCE_PROVENANCE.json`). `titan.exe -v` names that
commit.
