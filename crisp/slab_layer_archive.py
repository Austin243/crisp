"""Layer-aware geometry policy using existing slab identity and persistence."""

from copy import deepcopy
import json

import numpy as np
from ase import Atoms

from .slab_archive import SlabArchive, _slab_json_default, _slab_object_hook
from .slab_screening import _real_array
from .slab_layers import LayeredSlabSpec, validate_layered_slab


class LayeredSlabArchive(SlabArchive):
    """Archive reconstructed sheets/stacks with initial identities preserved.

    Reuses slab fingerprint matching, ranking, diversity and atomic JSON I/O.
    Layer settings and a separate format tag prevent checkpoint mixing with
    the earlier single-sheet archive. Energies are supplied in eV/atom at P=0.
    """

    def __init__(self, fp_calc, spec: LayeredSlabSpec, *, fp_threshold=0.03,
                 energy_threshold=0.01):
        if fp_calc.config != spec.config:
            raise ValueError("Layered archive and fingerprint geometry settings must agree")
        super().__init__(fp_calc, spec.composition, min_dist_ang=spec.min_dist_ang,
                         bond_scale=spec.bond_scale, fp_threshold=fp_threshold,
                         energy_threshold=energy_threshold)
        self.spec = spec
        self._spec_settings = spec.to_dict()

    def _validate_candidate(self, atoms):
        self._check_layer_settings()
        validate_layered_slab(atoms, self.spec)

    def _check_layer_settings(self):
        if (self.spec.to_dict() != self._spec_settings or self.spec.config != self.fp_calc.config
                or self.min_dist_ang != self.spec.min_dist_ang or self.bond_scale != self.spec.bond_scale):
            raise ValueError("Layered archive settings changed; create a new archive")

    def _persistence_target(self):
        self._check_layer_settings()
        if self._fingerprint_settings() != self._fp_settings:
            raise ValueError("Slab archive geometry/fingerprint settings changed; create a new archive")
        target = LayeredSlabArchive(self.fp_calc, self.spec, fp_threshold=self.fp_threshold,
                                    energy_threshold=self.energy_threshold)
        settings = dict(layer_spec=self.spec.to_dict(),
                        fingerprint=dict(cutoff=self.fp_calc.cutoff, natx=int(self.fp_calc.natx),
                                         orbital=self.fp_calc.orbital),
                        fp_threshold=self.fp_threshold, energy_threshold=self.energy_threshold)
        return target, json.loads(json.dumps(settings, default=_slab_json_default, allow_nan=False))

    def _to_payload(self):
        payload = super()._to_payload()
        payload["format"] = "crisp-layered-slab-archive"
        return payload

    def _load_payload(self, payload):
        if not isinstance(payload, dict) or payload.get("format") != "crisp-layered-slab-archive":
            raise ValueError("Unsupported layered slab archive format")
        payload = deepcopy(payload)
        payload["format"] = "crisp-slab-archive"
        target, _ = self._persistence_target()
        SlabArchive._load_payload(target, payload)
        # The generic loader prepares each structure again. Repeated wrapping
        # changes floating-point coordinates slightly, which would invalidate
        # exact native-job identities when a generation's mutations are replayed.
        # Restore the already canonical serialized atoms only after all records
        # passed the inherited validation, then evaluate their descriptors again.
        for entry, record in zip(target.entries, payload["entries"]):
            data = json.loads(json.dumps(record["atoms"]), object_hook=_slab_object_hook)
            atoms = Atoms.fromdict(data)
            atoms.info = deepcopy(record["info"])
            if not np.array_equal(atoms.numbers, np.sort(atoms.numbers)):
                raise ValueError("Layered archive records must use canonical species order")
            validate_layered_slab(atoms, self.spec)
            dimension = self.fp_calc.natx * (4 if self.fp_calc.orbital == "sp" else 1)
            entry.atoms = atoms
            entry.fp = _real_array(self.fp_calc.get_fingerprints(atoms), (len(atoms), dimension))
            entry.fp_pooled = _real_array(self.fp_calc.pool_with_std(entry.fp), (2 * dimension,))
        self.entries = target.entries
