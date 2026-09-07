"""Opt-in physical relaxation of a slab's in-plane lattice."""

from dataclasses import dataclass
from numbers import Real

import numpy as np

try:
    from ase.filters import UnitCellFilter
except ImportError:  # ASE 3.22 keeps filters in constraints.
    from ase.constraints import UnitCellFilter


@dataclass(frozen=True)
class SlabCellRelaxation:
    """Require max(abs(c * stress[xx, yy, xy])) < stress_max, in eV/Angstrom².

    The calculator supplies ASE's volume-normalized stress in eV/Angstrom³.
    Multiplication by the fixed cell height removes the arbitrary vacuum
    normalization. This is a zero-pressure criterion alongside the physical
    atomic force threshold; it adds no energy or force bias.
    """

    stress_max: float = 0.01

    def __post_init__(self):
        value = self.stress_max
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not np.isfinite(value) or value <= 0):
            raise ValueError("stress_max must be finite and positive")
        object.__setattr__(self, "stress_max", float(value))


class _SlabCellFilter(UnitCellFilter):
    """Energy-consistent Cartesian and xy deformation coordinates.

    The owned atoms must already have an exactly planar cell. UnitCellFilter
    supplies the deformation-gradient chain rule for both atoms and virial;
    its N scaling is independent of vacuum. Projecting proposed coordinates
    also enforces the mask on ASE 3.22, which only masks the returned forces.
    """

    def __init__(self, atoms):
        super().__init__(atoms, mask=[True, True, False, False, False, True],
                         cell_factor=float(len(atoms)))

    def set_positions(self, new, **kwargs):
        projected = np.array(new, copy=True)
        deformation = projected[len(self.atoms):]
        deformation[2, :] = 0.0
        deformation[:, 2] = 0.0
        deformation[2, 2] = self.cell_factor
        super().set_positions(projected, **kwargs)

    def get_forces(self, **kwargs):
        forces = super().get_forces(**kwargs)
        if (forces.shape != (len(self.atoms) + 3, 3)
                or np.iscomplexobj(forces) or not np.isfinite(forces).all()):
            raise RuntimeError("Slab cell filter returned nonfinite generalized forces")
        return forces
