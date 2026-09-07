"""Opt-in distance and periodic-connectivity checks for slab candidates."""

from itertools import combinations
from math import gcd

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list

from .slab import SlabConfig, validate_slab


def _check_periodic_connectivity(natoms, i, j, shifts):
    """Require one connected infinite xy net from symmetric directed bonds.

    A spanning tree assigns each atom an integer image offset. Non-tree bonds
    generate cycle translations. Their 2x2 determinant gcd is the index of
    the reachable translation lattice in Z^2: one means a single 2D net.
    Rank alone would also accept multiple interpenetrating nets.
    """
    neighbors = [[] for _ in range(natoms)]
    for source, target, shift in zip(i, j, shifts):
        neighbors[source].append((target, (int(shift[0]), int(shift[1]))))
    offsets = [None] * natoms
    offsets[0] = (0, 0)
    pending = [0]
    cycles = set()
    while pending:
        source = pending.pop()
        x, y = offsets[source]
        for target, (dx, dy) in neighbors[source]:
            image = (x + dx, y + dy)
            if offsets[target] is None:
                offsets[target] = image
                pending.append(target)
            else:
                cycle = (image[0] - offsets[target][0], image[1] - offsets[target][1])
                if cycle != (0, 0):
                    cycles.add(cycle)
    if any(offset is None for offset in offsets):
        raise ValueError("Slab has disconnected bond components")

    index = 0
    for (ax, ay), (bx, by) in combinations(cycles, 2):
        index = gcd(index, abs(ax * by - ay * bx))
        if index == 1:
            return
    if index == 0:
        rank = 1 if cycles else 0
        raise ValueError(f"Slab bond network is {rank}D; expected one connected 2D sheet")
    raise ValueError(f"Slab bond network contains {index} disconnected periodic nets")


def validate_slab_candidate(atoms: Atoms, config: SlabConfig, *,
                            min_dist_ang: float, bond_scale: float = 1.2) -> None:
    """Reject invalid geometry, short contacts, or a disconnected/non-2D net.

    min_dist_ang is an explicit minimum separation for every species pair,
    including periodic self-images. Bonds satisfy the strict distance cutoff
    bond_scale * (covalent_radius_i + covalent_radius_j), using ASE radii.
    These are configurable geometric heuristics, not a stability test.

    All atoms and their xy repeats must form one connected 2D bond network.
    Uses physical mixed PBC and full xyz distances, without wrapping z or
    adding vacuum images. Returns None on success, raises ValueError with a
    rejection reason otherwise. Does not mutate atoms, evaluate a calculator,
    check energy/convergence, or insert candidates into the search/archive.
    """
    if not np.isfinite(min_dist_ang) or min_dist_ang <= 0:
        raise ValueError("min_dist_ang must be finite and positive")
    if not np.isfinite(bond_scale) or bond_scale <= 0:
        raise ValueError("bond_scale must be finite and positive")
    validate_slab(atoms, config)
    if np.any(atoms.numbers <= 0) or np.any(atoms.numbers >= len(covalent_radii)):
        raise ValueError("Slab bonding requires known chemical elements")

    distances = neighbor_list("d", atoms, min_dist_ang)
    if len(distances):
        raise ValueError(
            f"Slab minimum distance {distances.min():.6g} Angstrom "
            f"is below min_dist_ang {min_dist_ang:.6g}")
    cutoffs = bond_scale * covalent_radii[atoms.numbers]
    if not np.isfinite(cutoffs).all():
        raise ValueError("Scaled covalent radii must be finite")
    i, j, shifts = neighbor_list("ijS", atoms, cutoffs)
    _check_periodic_connectivity(len(atoms), i, j, shifts)
