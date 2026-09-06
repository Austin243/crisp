"""Opt-in geometry helpers for future 2D searches.

These helpers are not used by CRISPSearch yet. They describe a finite-thickness
sheet in the xy plane; they do not establish bonding or fingerprint safety.
"""

from dataclasses import dataclass

import numpy as np
from ase import Atoms


_ATOL = 1e-8


@dataclass(frozen=True)
class SlabConfig:
    """Geometry bounds for a zero-pressure, free-standing sheet.

    Lengths are in Angstrom; area_per_atom_range is in Angstrom squared.
    initial_thickness is reserved for generation, not imposed on atomic z.
    min_vacuum is the total empty gap between repeated sheets, not per face.
    Geometry settings are explicit because useful bounds depend on chemistry.
    """

    area_per_atom_range: tuple[float, float]
    initial_thickness: float
    max_thickness: float
    cell_height: float
    min_vacuum: float
    pressure_GPa: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "area_per_atom_range", tuple(self.area_per_atom_range))
        if len(self.area_per_atom_range) != 2:
            raise ValueError("area_per_atom_range must contain two bounds")
        low, high = self.area_per_atom_range
        values = (low, high, self.initial_thickness, self.max_thickness,
                  self.cell_height, self.min_vacuum, self.pressure_GPa)
        if not np.isfinite(values).all():
            raise ValueError("Slab settings must be finite")
        if not 0 < low <= high:
            raise ValueError("area_per_atom_range must satisfy 0 < min <= max")
        if not 0 <= self.initial_thickness <= self.max_thickness:
            raise ValueError("Thickness must satisfy 0 <= initial <= max")
        if self.cell_height <= 0 or self.min_vacuum <= 0:
            raise ValueError("cell_height and min_vacuum must be positive")
        if self.max_thickness + self.min_vacuum > self.cell_height:
            raise ValueError(
                "cell_height must accommodate max_thickness + min_vacuum")
        if self.pressure_GPa != 0:
            raise ValueError("Slab geometry currently supports only zero pressure")


def _geometry(atoms: Atoms) -> tuple[float, float]:
    """Check the input frame and return area per atom and slab thickness."""
    if len(atoms) == 0:
        raise ValueError("A slab must contain at least one atom")
    cell = atoms.cell.array
    if not np.isfinite(cell).all() or not np.isfinite(atoms.positions).all():
        raise ValueError("Slab cell and positions must be finite")
    if (np.any(np.abs(cell[:2, 2]) > _ATOL)
            or np.any(np.abs(cell[2, :2]) > _ATOL)):
        raise ValueError("Slab requires a and b in xy and c perpendicular to xy")
    area = float(np.linalg.det(cell[:2, :2]))
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Slab requires a nondegenerate, right-handed xy cell")
    if cell[2, 2] < 0:
        raise ValueError("Slab cell height cannot be negative")
    return area / len(atoms), float(np.ptp(atoms.positions[:, 2]))


def validate_slab(atoms: Atoms, config: SlabConfig) -> None:
    """Raise ValueError when slab geometry is outside the configured bounds.

    This checks the cell, physical PBC, area, thickness and image gap. It does
    not check connectivity, interatomic distances, calculator support or the
    fingerprint cutoff. A uniform z translation is allowed.
    """
    area_per_atom, thickness = _geometry(atoms)
    if not np.array_equal(atoms.pbc, (True, True, False)):
        raise ValueError("Slab PBC must be (True, True, False)")
    if not np.isclose(atoms.cell[2, 2], config.cell_height, atol=_ATOL, rtol=0):
        raise ValueError("Slab cell height differs from configured cell_height")
    low, high = config.area_per_atom_range
    if not low - _ATOL <= area_per_atom <= high + _ATOL:
        raise ValueError("Slab area per atom is outside area_per_atom_range")
    if thickness > config.max_thickness + _ATOL:
        raise ValueError("Slab thickness exceeds max_thickness")
    if atoms.cell[2, 2] - thickness < config.min_vacuum - _ATOL:
        raise ValueError("Slab image gap is smaller than min_vacuum")


def prepare_slab(atoms: Atoms, config: SlabConfig) -> Atoms:
    """Return a centered slab copy without scaling atomic coordinates.

    The input must already have a right-handed xy cell and contiguous atomic
    z coordinates. A zero initial c is allowed. This sets the configured c,
    wraps xy, and centers the z extent; it never wraps z or rotates a cell.
    The returned Atoms follows ASE copy semantics (no attached calculator).
    """
    _geometry(atoms)
    slab = atoms.copy()
    cell = slab.cell.array.copy()
    cell[:2, 2] = 0.0  # Remove numerical roundoff from the xy plane.
    cell[2] = (0.0, 0.0, config.cell_height)
    slab.set_cell(cell, scale_atoms=False, apply_constraint=False)
    slab.set_pbc((True, True, False))
    slab.wrap()
    z = slab.positions[:, 2]
    slab.positions[:, 2] += config.cell_height / 2 - (z.min() + z.max()) / 2
    validate_slab(slab, config)
    return slab
