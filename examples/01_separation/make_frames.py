#!/usr/bin/env python3
"""Build the frames of example 01, two peptides that part and meet again, with a known answer.

    python make_frames.py

reads peptides.pdb (two amyloid-beta(1-42) chains in contact) and writes, in this folder:
  frames.xtc      250 frames, 100 ps apart; every atom wrapped into the cell, as an MD engine writes them
  reference.pdb   frame 0 with both chains whole: the -s structure of arg.in
  index.ndx       group 0: both chains
  truth.csv       the designed answer, frame by frame
The release already contains these files; this script shows how they were made and rebuilds them (it needs numpy
and MDAnalysis, the latter for the XTC writer). The random numbers come from a fixed seed.

The cell is a rhombic dodecahedron with 110 A between neighbouring copies.  Both chains move as rigid bodies, and
while bound they keep the contact of the starting structure:
  frames   0- 39  B is bound to A
  frames  40- 69  B leaves A towards the point halfway between A and one neighbouring copy of A
  frames  70-179  B drifts back and forth across that halfway point, out of contact with every copy of A
  frames 180-209  B comes back to the copy of A it left
  frames 210-249  B is bound to A again
The pair also drifts through the cell, so the chains cross the cell faces and the wrapping cuts them.
B never changes partner: the right reconstruction keeps B next to the same copy of A in every frame.  While B is
past the halfway point it is nearer to the other copy of A, and a choice made one frame at a time follows it there.
"""
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CELL = 110.0                                     # A, distance between neighbouring copies
DIMENSIONS = (CELL, CELL, CELL, 60.0, 60.0, 90.0)  # a, b, c, alpha, beta, gamma
FRAMES = 250
DT_PS = 100.0
SEED = 1729
CONTACT = 4.5                                    # A, contact distance of VIBE
PHASES = [(0, 'bound'), (40, 'leaving'), (70, 'apart'), (180, 'returning'), (210, 'bound')]


def box_vectors(a, b, c, alpha, beta, gamma):
    """Rows a, b, c of the cell in the GROMACS orientation (a along x, b in the xy plane)."""
    al, be, ga = np.radians([alpha, beta, gamma])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    return np.array([[a, 0.0, 0.0], [b * np.cos(ga), b * np.sin(ga), 0.0],
                     [cx, cy, np.sqrt(c * c - cx * cx - cy * cy)]])


def rotation(w):
    """Rotation matrix of the rotation vector w (axis times angle in radians)."""
    angle = np.linalg.norm(w)
    if angle < 1e-12:
        return np.eye(3)
    k = w / angle
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * K @ K


def wander(rng, sd, keep, dim=3):
    """Mean-reverting random walk with standard deviation sd; keep sets how slowly it forgets."""
    out, x = np.zeros((FRAMES, dim)), np.zeros(dim)
    for t in range(FRAMES):
        x = keep * x + rng.normal(0.0, sd * np.sqrt(1.0 - keep * keep), dim)
        out[t] = x
    return out


def smooth(s):
    return s * s * (3.0 - 2.0 * s)


def phase(t):
    return [name for start, name in PHASES if t >= start][-1]


def read_pdb(path):
    records = [line for line in path.read_text().splitlines() if line.startswith(('ATOM', 'HETATM'))]
    xyz = np.array([[float(r[30:38]), float(r[38:46]), float(r[46:54])] for r in records])
    chains = np.array([r[21] for r in records])
    return records, xyz, chains


def min_distance(p, q):
    return np.sqrt(((p[:, None, :] - q[None, :, :]) ** 2).sum(axis=2).min())


def main():
    import MDAnalysis as mda

    records, xyz, chains = read_pdb(HERE / 'peptides.pdb')
    ia, ib = np.flatnonzero(chains == 'A'), np.flatnonzero(chains == 'B')
    box = box_vectors(*DIMENSIONS)
    inv = np.linalg.inv(box)
    a0, b0 = xyz[ia], xyz[ib]
    ca, cb = a0.mean(axis=0), b0.mean(axis=0)
    r0 = cb - ca                                             # B's place relative to A when bound
    shifts = np.array([(i, j, k) for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)])
    copies = shifts @ box                                    # copies of A, the original one at index 62
    same = int(np.flatnonzero((shifts == 0).all(axis=1))[0])
    near = copies[np.isclose(np.linalg.norm(copies, axis=1), CELL)]
    v = near[np.argmax(near @ r0)]                           # the neighbouring copy B moves towards
    e = v / CELL

    rng = np.random.default_rng(SEED)
    side = wander(rng, 1.5, 0.95)                            # B sideways while apart
    along = wander(rng, 0.7, 0.8)[:, 0]                      # B along e while apart
    turn = wander(rng, np.radians(1.5), 0.95)                # the whole pair
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)                             # B's own turn while apart
    heading = np.array([1.0, 0.45, 0.2]) / np.linalg.norm([1.0, 0.45, 0.2])
    drift = np.arange(FRAMES)[:, None] * 0.8 * heading + wander(rng, 0.5, 0.95)

    def bound(t):
        return r0          # the contact of the starting structure, unchanged while bound

    def apart(t):
        s = side[t] - (side[t] @ e) * e
        return v / 2.0 + e * (5.0 * np.sin(2.0 * np.pi * (t - 70) / 18.0) + along[t]) + s

    whole, wrapped, rows = [], [], []
    for t in range(FRAMES):
        name = phase(t)
        if name == 'bound':
            r = bound(t)
        elif name == 'leaving':
            s = smooth((t - 40) / 30.0)
            r = (1.0 - s) * bound(t) + s * apart(t)
        elif name == 'apart':
            r = apart(t)
        else:
            s = smooth((t - 180) / 30.0)
            r = (1.0 - s) * apart(t) + s * bound(t)
        own = rotation(axis * np.radians(15.0) * np.sin(np.pi * (t - 40) / 170.0) ** 2) if 40 <= t < 210 \
            else np.eye(3)
        pair = rotation(turn[t])
        x = np.empty_like(xyz)
        x[ia] = (a0 - ca) @ pair.T + ca + drift[t]
        x[ib] = ((b0 - cb) @ own.T + r) @ pair.T + ca + drift[t]
        whole.append(x)
        frac = x @ inv
        wrapped.append((frac - np.floor(frac)) @ box)

        # the answer: B's centre relative to A's centre, and how far B is from each copy of A
        a, b = x[ia], x[ib]
        ba = b.mean(axis=0) - a.mean(axis=0)
        centre = np.linalg.norm(ba - copies, axis=1)              # B centre to the centre of each copy of A
        others = [int(k) for k in np.argsort(centre) if k != same][:3]   # the three nearest other copies
        dist = {k: min_distance(a + copies[k], b) for k in [same] + others}
        other = min(d for k, d in dist.items() if k != same)
        rows.append({'frame': t, 'time_ps': f'{t * DT_PS:.0f}', 'phase': name,
                     'b_minus_a_x_A': f'{ba[0]:.3f}', 'b_minus_a_y_A': f'{ba[1]:.3f}', 'b_minus_a_z_A': f'{ba[2]:.3f}',
                     'min_distance_to_partner_A': f'{dist[same]:.2f}',
                     'min_distance_to_other_copies_A': f'{other:.2f}',
                     'in_contact': int(dist[same] < CONTACT),
                     'nearer_copy': 'partner' if dist[same] <= other else 'other'})
        assert other > CONTACT, f'frame {t}: B touches another copy of A'

    with open(HERE / 'truth.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    with open(HERE / 'reference.pdb', 'w', newline='\n') as f:
        f.write('REMARK    HoloLift example 01, frame 0, both chains whole (see make_frames.py)\n')
        f.write('CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1\n' % DIMENSIONS)
        for k, (line, (x, y, z)) in enumerate(zip(records, whole[0])):
            f.write(f'{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}\n')
            if k == ia[-1]:
                f.write('TER\n')
        f.write('TER\nEND\n')

    with open(HERE / 'index.ndx', 'w', newline='\n') as f:
        f.write('[ PEPTIDES ]\n')
        ids = [str(k + 1) for k in range(len(xyz))]
        for k in range(0, len(ids), 15):
            f.write(' '.join(ids[k:k + 15]) + '\n')

    u = mda.Universe.empty(len(xyz), trajectory=True)
    with mda.Writer(str(HERE / 'frames.xtc'), len(xyz)) as out:
        for t, x in enumerate(wrapped):
            u.atoms.positions = x
            u.dimensions = DIMENSIONS
            u.trajectory.ts.time = t * DT_PS
            u.trajectory.ts.data['step'] = t * 50000
            out.write(u.atoms)

    apart_frames = sum(r['phase'] == 'apart' for r in rows)
    other_frames = sum(r['nearer_copy'] == 'other' for r in rows)
    print(f'{FRAMES} frames; B in contact with A in {sum(r["in_contact"] for r in rows)}, apart in {apart_frames}; '
          f'nearer to the other copy of A in {other_frames}')


if __name__ == '__main__':
    main()
