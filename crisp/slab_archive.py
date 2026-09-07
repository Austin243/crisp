"""Opt-in, fixed-composition slab archive with species-aware identity."""

from copy import deepcopy
from numbers import Integral

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

from .archive import ArchiveEntry, StructureArchive
from .slab import prepare_slab
from .slab_fingerprint import SlabFingerprintCalculator
from .slab_validation import validate_slab_candidate


class SlabArchive(StructureArchive):
    """Store validated slabs at zero pressure without changing bulk archives.

    One archive has one exact composition and one geometry/fingerprint setup.
    Input copies are centered, wrapped in xy, and sorted by atomic number so
    the inherited species-aware Hungarian matcher sees aligned species blocks.
    A duplicate requires both fingerprint and energy differences below their
    thresholds. This finite-cutoff comparison is approximate phase identity.

    Ranking, diversity selection, and feature/energy accessors are inherited.
    Treat exposed entries and cached arrays as read-only. Persistence is not
    supported until a slab-specific checkpoint contract is implemented.
    """

    def __init__(self, fp_calc: SlabFingerprintCalculator, composition: dict[str, int], *,
                 min_dist_ang: float, bond_scale: float = 1.2,
                 fp_threshold: float = 0.03, energy_threshold: float = 0.01):
        if not isinstance(fp_calc, SlabFingerprintCalculator):
            raise TypeError("SlabArchive requires a SlabFingerprintCalculator")
        for name, value in (("min_dist_ang", min_dist_ang), ("bond_scale", bond_scale),
                            ("fp_threshold", fp_threshold), ("energy_threshold", energy_threshold),
                            ("cutoff", fp_calc.cutoff)):
            if (np.ndim(value) != 0 or np.iscomplexobj(value)
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if (isinstance(fp_calc.natx, bool) or not isinstance(fp_calc.natx, Integral)
                or fp_calc.natx <= 0 or fp_calc.orbital not in ("s", "sp")):
            raise ValueError("Fingerprint natx must be a positive integer and orbital must be s or sp")
        if not composition or any(
                symbol not in atomic_numbers or atomic_numbers[symbol] == 0
                or isinstance(count, bool) or not isinstance(count, Integral) or count <= 0
                for symbol, count in composition.items()):
            raise ValueError("composition must map chemical symbols to positive integer counts")

        super().__init__(fp_calc, fp_threshold=fp_threshold, energy_threshold=energy_threshold)
        self._numbers = np.array(sorted(atomic_numbers[symbol]
                                       for symbol, count in composition.items()
                                       for _ in range(count)))
        self.min_dist_ang = min_dist_ang
        self.bond_scale = bond_scale
        self._fp_settings = self._fingerprint_settings()

    def _fingerprint_settings(self):
        return (self.fp_calc.config, self.fp_calc.cutoff,
                self.fp_calc.natx, self.fp_calc.orbital)

    def add(self, atoms: Atoms, energy_per_atom: float, *, metadata: dict | None = None) -> bool:
        """Return True for insertion, False for a duplicate, or raise on failure.

        The caller supplies energy in eV/atom from a consistent calculator;
        no calculator is evaluated and quench convergence is not inferred.
        Stored enthalpy equals energy and pressure is zero. Invalid candidates,
        energies, descriptors, or changed fingerprint settings raise before
        insertion. Matcher/backend errors propagate instead of adding an entry.
        """
        if self._fingerprint_settings() != self._fp_settings:
            raise ValueError("Slab archive geometry/fingerprint settings changed; create a new archive")
        if np.ndim(energy_per_atom) != 0 or np.iscomplexobj(energy_per_atom):
            raise ValueError("Slab archive energy_per_atom must be finite")
        energy = float(energy_per_atom)
        if not np.isfinite(energy):
            raise ValueError("Slab archive energy_per_atom must be finite")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary or None")
        if not np.array_equal(np.sort(atoms.numbers), self._numbers):
            raise ValueError("Slab composition and atom count must match the archive")
        config = self._fp_settings[0]
        # Reject invalid inputs before prepare_slab could repair PBC or height.
        validate_slab_candidate(atoms, config, min_dist_ang=self.min_dist_ang,
                                bond_scale=self.bond_scale)
        stored = prepare_slab(atoms, config)
        stored = stored[np.argsort(stored.numbers, kind="stable")]
        stored.info = deepcopy(atoms.info)
        fp = np.asarray(self.fp_calc.get_fingerprints(stored))
        if np.iscomplexobj(fp):
            raise ValueError("Slab fingerprints must be real")
        fp = np.array(fp, dtype=float, copy=True)
        dimension = self.fp_calc.natx * (4 if self.fp_calc.orbital == "sp" else 1)
        if fp.shape != (len(stored), dimension) or not np.isfinite(fp).all():
            raise ValueError("Slab fingerprints must have the configured shape and finite values")
        pooled = np.asarray(self.fp_calc.pool_with_std(fp))
        if np.iscomplexobj(pooled):
            raise ValueError("Pooled slab fingerprints must be real")
        pooled = np.array(pooled, dtype=float, copy=True)
        if pooled.shape != (2 * dimension,) or not np.isfinite(pooled).all():
            raise ValueError("Pooled slab fingerprints must have the configured shape and finite values")
        types = self.fp_calc.atoms_to_cell(stored)[2]
        for entry in self.entries:
            distance = self.fp_calc._fp_dist(fp, entry.fp, types)
            if (np.ndim(distance) != 0 or np.iscomplexobj(distance)
                    or not np.isfinite(distance) or distance < 0):
                raise ValueError("Slab fingerprint distance must be finite and nonnegative")
            if distance < self.fp_threshold and abs(energy - entry.energy) < self.energy_threshold:
                return False

        meta = deepcopy(metadata) if metadata is not None else {}
        self.entries.append(ArchiveEntry(
            atoms=stored, fp=fp, fp_pooled=pooled, energy=energy,
            enthalpy=energy, pressure=0.0, metadata=meta,
            generation=meta.get("generation", 0)))
        return True

    def _unsupported_persistence(self, *args, **kwargs):
        raise NotImplementedError("Slab archive persistence requires a slab-specific checkpoint contract")

    save = load = save_checkpoint = load_checkpoint = _unsupported_persistence
