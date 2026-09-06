"""Opt-in fingerprints for slabs using a guarded, padded 3D backend."""

import numpy as np

from .fingerprint import FingerprintCalculator
from .slab import SlabConfig, prepare_slab, validate_slab


_IMAGE_MARGIN = 1e-6  # Angstrom, beyond the backend's inclusive cutoff.


class SlabFingerprintCalculator(FingerprintCalculator):
    """Reuse CRISP's differentiable fingerprints on isolated xy slabs.

    The backend searches periodic images in all three directions. Requiring
    an empty z gap greater than cutoff + 1e-6 Angstrom excludes every z image.
    Inputs must already satisfy config; each evaluation wraps xy and centers z
    on a copy so the backend also handles unwrapped in-plane coordinates.

    All Cartesian force components and all six ASE stress components are
    retained. Restricting cell strain belongs to the relaxation code, not the
    descriptor. This calculator does not enable 2D searches by itself.
    """

    def __init__(self, config: SlabConfig, cutoff: float = 6.0,
                 natx: int = 100, orbital: str = "s"):
        if not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("Slab fingerprint cutoff must be finite and positive")
        super().__init__(cutoff=cutoff, natx=natx, orbital=orbital)
        self.config = config

    def _prepare_atoms(self, atoms):
        validate_slab(atoms, self.config)
        gap = atoms.cell[2, 2] - np.ptp(atoms.positions[:, 2])
        if gap <= self.cutoff + _IMAGE_MARGIN:
            raise ValueError(
                "Slab image gap must exceed fingerprint cutoff + 1e-6 Angstrom")
        return prepare_slab(atoms, self.config)

    # Keep these wrappers on every numerical entry point: the complete base
    # method must see one prepared copy, including its scaled positions and
    # volume. Distance methods already dispatch through get_fingerprints.
    def get_fingerprints(self, atoms):
        """Compute slab fingerprints; see the base method for the return shape."""
        return super().get_fingerprints(self._prepare_atoms(atoms))

    def project_forces(self, atoms, dL_dfp):
        """Project the fingerprint gradient to all Cartesian force components."""
        return super().project_forces(self._prepare_atoms(atoms), dL_dfp)

    def project_forces_and_stress(self, atoms, dL_dfp):
        """Return Cartesian forces and all six volume-normalized ASE stresses."""
        return super().project_forces_and_stress(
            self._prepare_atoms(atoms), dL_dfp)

    def get_fingerprints_and_jacobian(self, atoms):
        """Compute slab fingerprints and the full xyz position Jacobian."""
        return super().get_fingerprints_and_jacobian(self._prepare_atoms(atoms))

    def get_fingerprints_jacobian_strain(self, atoms):
        """Compute slab fingerprints, xyz Jacobian and six strain derivatives."""
        return super().get_fingerprints_jacobian_strain(self._prepare_atoms(atoms))
