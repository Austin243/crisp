"""Bounded, calculator-free mutations of finite-thickness slab parents."""

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from .slab import SlabConfig, prepare_slab, validate_slab
from .slab_generation import SlabGenerationError


@dataclass(frozen=True)
class SlabMutations:
    """Mutation amplitudes and budgets; lengths are in Angstrom.

    max_strain bounds the Frobenius norm of the symmetric in-plane strain E
    in the deformation I + E. random_every controls random injection in the
    search driver; max_attempts bounds proposals from one selected parent.
    """

    max_displacement: float = 0.2
    max_strain: float = 0.05
    random_every: int = 5
    max_attempts: int = 20

    def __post_init__(self):
        for name in ("max_displacement", "max_strain"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, float(value))
        if self.max_strain >= 1:
            raise ValueError("max_strain must be less than 1")
        if self.max_displacement == self.max_strain == 0:
            raise ValueError("At least one mutation amplitude must be positive")
        for name in ("random_every", "max_attempts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))


class SlabMutationError(SlabGenerationError):
    """A valid parent exhausted its bounded mutation attempts."""


def mutate_slab(parent: Atoms, config: SlabConfig, mutations: SlabMutations, *,
                seed=None, min_dist_ang: float = 1.5) -> Atoms:
    """Return an independent mutated slab, or reject after bounded attempts.

    Apply a symmetric strain to the xy cell and atomic xy coordinates, leaving
    z unscaled. Then move each atom uniformly inside a ball of radius
    max_displacement (a disk when max_thickness is zero). The displacement
    bound applies after the affine strain, before xy wrapping and rigid z
    centering. Those changes of origin can make coordinate differences larger.

    Geometry and minimum distances, including periodic self-images, must pass.
    Connectivity is left to the search's post-quench validator. Parent geometry,
    distances, constraints and options are checked before sampling. The parent,
    its calculator, arrays and nested metadata are untouched; the returned copy
    has no calculator or inherited generation/relaxation success claims.
    """
    if not isinstance(mutations, SlabMutations):
        raise TypeError("mutations must be SlabMutations")
    if (isinstance(min_dist_ang, bool) or not isinstance(min_dist_ang, Real)
            or not np.isfinite(min_dist_ang) or min_dist_ang <= 0):
        raise ValueError("min_dist_ang must be finite and positive")
    validate_slab(parent, config)
    if parent.constraints:
        raise ValueError("Slab mutation does not support constrained parents")
    if len(neighbor_list("d", parent, min_dist_ang)):
        raise ValueError("Parent slab violates min_dist_ang")
    rng = np.random.default_rng(seed)
    dimension = 2 if config.max_thickness == 0 else 3
    last_error = None
    for attempt in range(1, mutations.max_attempts + 1):
        # Each symmetric component is at most max_strain / 2, so ||E||_F
        # is at most max_strain < 1 and I + E remains positive definite.
        strain = rng.uniform(-0.5, 0.5, size=(2, 2))
        strain = mutations.max_strain * (strain + strain.T) / 2
        deformation = np.eye(2) + strain
        candidate = parent.copy()
        candidate.cell[:2, :2] = parent.cell[:2, :2] @ deformation
        candidate.positions[:, :2] = parent.positions[:, :2] @ deformation
        directions = rng.normal(size=(len(parent), dimension))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions /= np.where(norms == 0, 1, norms)
        radii = mutations.max_displacement * rng.random((len(parent), 1)) ** (1 / dimension)
        candidate.positions[:, :dimension] += directions * radii
        try:
            candidate = prepare_slab(candidate, config)
        except ValueError as exc:
            last_error = exc
            continue
        if len(neighbor_list("d", candidate, min_dist_ang)):
            last_error = ValueError("Mutated slab violates min_dist_ang")
            continue
        candidate.info = deepcopy(parent.info)
        for key in list(candidate.info):
            if (key in ("layer_group", "generation_seed", "pyxtal_version")
                    or isinstance(key, str) and key.startswith("slab_quench_")):
                del candidate.info[key]
        candidate.info.update(origin="mutation", mutation_attempt=attempt)
        return candidate
    raise SlabMutationError(
        f"No valid slab mutation after {mutations.max_attempts} attempts; "
        f"last rejection: {last_error}") from last_error
