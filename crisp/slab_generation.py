"""Seeded PyXtal layer-group generation, separate from the bulk search."""

from collections import Counter
from numbers import Integral

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.neighborlist import neighbor_list

from .slab import SlabConfig, prepare_slab


class SlabGenerationError(RuntimeError):
    """A valid generation request exhausted its bounded attempts."""


def _sample_cell(group, area, thickness, rng):
    """Sample a compatible in-plane metric, with no out-of-plane tilt."""
    gamma = np.pi / 2
    if group.lattice_type in ("hexagonal", "trigonal"):
        gamma = 2 * np.pi / 3
    elif group.number <= 7:  # Oblique layer groups (unique c for monoclinic).
        gamma = rng.uniform(np.pi / 3, 2 * np.pi / 3)
    ratio = (1.0 if group.lattice_type in ("tetragonal", "hexagonal", "trigonal")
             else rng.uniform(0.5, 2.0))
    a = np.sqrt(area * ratio / np.sin(gamma))
    b = np.sqrt(area / (ratio * np.sin(gamma)))
    # A planar request still needs a nonsingular cell inside PyXtal.
    height = thickness if thickness > 0 else 1.0
    return np.array([[a, 0.0, 0.0], [b * np.cos(gamma), b * np.sin(gamma), 0.0],
                     [0.0, 0.0, height]])


def generate_slabs(composition: dict[str, int], config: SlabConfig, n: int, *,
                   seed=None, layer_groups=None, min_dist_ang: float = 1.5,
                   max_attempts: int | None = None) -> list[Atoms]:
    """Return exactly n random slabs or raise a descriptive generation error.

    Uses a local NumPy RNG (or advances a supplied Generator), deterministic
    layer-group compatibility, and explicit area sampling. initial_thickness
    is a sampling scale, not a bound; max_thickness limits the actual extent.
    Zero initial_thickness requests planar candidates. The area/aspect prior
    uses ratios in [0.5, 2] and oblique angles in [60, 120] degrees.

    max_attempts counts PyXtal calls (default 20*n); each call also has its own
    bounded internal trials. Incompatible requests raise ValueError. Exhausted
    attempts raise RuntimeError rather than silently returning a partial batch.
    This generates candidates only; it does not run or modify CRISPSearch.
    """
    if (not composition or any(
            s not in atomic_numbers or atomic_numbers[s] == 0
            or isinstance(count, bool) or not isinstance(count, Integral) or count <= 0
            for s, count in composition.items())):
        raise ValueError("composition must map chemical symbols to positive integer counts")
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    if not np.isfinite(min_dist_ang) or min_dist_ang <= 0:
        raise ValueError("min_dist_ang must be finite and positive")
    if max_attempts is None:
        max_attempts = 20 * n
    if (isinstance(max_attempts, bool) or not isinstance(max_attempts, Integral)
            or max_attempts < n):
        raise ValueError("max_attempts must be an integer at least n")
    numbers = tuple(range(1, 81) if layer_groups is None else layer_groups)
    if not numbers or any(isinstance(g, bool) or not isinstance(g, Integral)
                          or not 1 <= g <= 80 for g in numbers):
        raise ValueError("layer_groups must contain integers from 1 through 80")
    if n == 0:
        return []

    try:
        from pyxtal import pyxtal, __version__ as pyxtal_version
        from pyxtal.lattice import Lattice
        from pyxtal.msg import Error as PyXtalError
        from pyxtal.symmetry import Group
        from pyxtal.tolerance import Tol_matrix
    except ImportError as exc:
        raise ImportError("2D generation requires PyXtal; install CRISP's [search] extra") from exc

    species = sorted(composition)
    counts = [int(composition[s]) for s in species]
    groups = [Group(int(g), dim=2) for g in sorted(set(numbers))]
    groups = [g for g in groups if g.check_compatible(counts)[0]]
    if not groups:
        raise ValueError("No requested layer group is compatible with composition")
    tm = Tol_matrix(prototype="atomic")
    for i, s1 in enumerate(species):
        for s2 in species[i:]:
            tm.set_tol(s1, s2, min_dist_ang)

    rng = np.random.default_rng(seed)
    structures = []
    last_error = None
    for _ in range(max_attempts):
        group = groups[int(rng.integers(len(groups)))]
        area = rng.uniform(*config.area_per_atom_range) * sum(counts)
        cell = _sample_cell(group, area, config.initial_thickness, rng)
        attempt_seed = int(rng.integers(2**31))
        try:
            # Seed both RNGs: a supplied PyXtal lattice retains its own stream.
            lattice = Lattice.from_matrix(
                cell, reset=False, ltype=group.lattice_type, PBC=[1, 1, 0],
                random_state=attempt_seed)
            xtal = pyxtal(random_state=attempt_seed)
            xtal.from_random(
                dim=2, group=group, species=species, numIons=counts,
                lattice=lattice, thickness=config.initial_thickness,
                conventional=True, tm=tm, random_state=attempt_seed, max_count=1)
            # to_ase wraps z, even with add_vaccum=False. Export the unwrapped
            # site coordinates to preserve finite-sheet symmetry and thickness.
            atoms = Atoms(
                [site.specie for site in xtal.atom_sites for _ in site.coords],
                scaled_positions=np.concatenate([site.coords for site in xtal.atom_sites]),
                cell=xtal.lattice.matrix, pbc=(True, True, False))
            if Counter(atoms.get_chemical_symbols()) != composition:
                raise ValueError("Generated composition differs from request")
            if config.initial_thickness == 0:
                atoms.positions[:, 2] = 0.0
            atoms = prepare_slab(atoms, config)
            # Includes nonzero periodic self-images, even for one-atom cells.
            if len(neighbor_list("d", atoms, min_dist_ang)):
                raise ValueError("Generated slab violates min_dist_ang")
        except (ValueError, RuntimeError, PyXtalError) as exc:
            last_error = exc
            continue
        atoms.info.update(origin="random", layer_group=group.number,
                          generation_seed=attempt_seed, pyxtal_version=pyxtal_version)
        structures.append(atoms)
        if len(structures) == n:
            return structures

    raise SlabGenerationError(
        f"Generated {len(structures)}/{n} slabs after {max_attempts} attempts; "
        f"last rejection: {last_error}") from last_error
