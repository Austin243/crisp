"""Opt-in, fixed-composition slab archive with species-aware identity."""

from copy import deepcopy
from dataclasses import asdict
import json
from numbers import Integral
import os
from pathlib import Path
import tempfile

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols
from ase.io.jsonio import MyEncoder as ASEJSONEncoder, object_hook as ase_object_hook

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
    Treat exposed entries and cached arrays as read-only. Save/load use a
    separate, versioned slab format; search checkpoints are not supported.
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

    def _persistence_target(self):
        """Validate current settings and create an empty compatible archive."""
        if self._fingerprint_settings() != self._fp_settings:
            raise ValueError("Slab archive geometry/fingerprint settings changed; create a new archive")
        numbers, counts = np.unique(self._numbers, return_counts=True)
        composition = {chemical_symbols[number]: int(count)
                       for number, count in zip(numbers, counts)}
        options = dict(min_dist_ang=self.min_dist_ang, bond_scale=self.bond_scale,
                       fp_threshold=self.fp_threshold, energy_threshold=self.energy_threshold)
        target = SlabArchive(self.fp_calc, composition, **options)
        settings = dict(slab=asdict(self.fp_calc.config), composition=composition,
                        fingerprint=dict(cutoff=self.fp_calc.cutoff, natx=int(self.fp_calc.natx),
                                         orbital=self.fp_calc.orbital), **options)
        # Normalize tuples and NumPy scalar settings to their JSON representation.
        settings = json.loads(json.dumps(settings, default=_slab_json_default, allow_nan=False))
        return target, settings

    def save(self, path: str | Path) -> None:
        """Atomically write a slab archive JSON file, without evaluating a calculator.

        Entries are read-only by convention; fingerprints are not saved or
        recomputed here. Keep validation/duplicate settings fixed while populated.
        Metadata and Atoms.info must contain finite JSON values with string keys;
        tuples become lists. Ordinary ASE arrays and constraints are preserved.
        """
        _write_slab_json(path, self._to_payload())

    def _to_payload(self):
        """Share the archive format with the opt-in search checkpoint."""
        _, settings = self._persistence_target()
        records = []
        for entry in self.entries:
            atoms = entry.atoms.todict()
            info = atoms.pop("info", {})
            _check_json_value(info)
            _check_json_value(entry.metadata)
            _check_json_value(entry.generation)
            if any(array.dtype.kind not in "biufcU" for array in entry.atoms.arrays.values()):
                raise TypeError("Slab archive atom arrays must be numeric, boolean or Unicode")
            records.append(dict(atoms=atoms, info=info, energy_per_atom=entry.energy,
                                metadata=entry.metadata, generation=entry.generation))
        return dict(format="crisp-slab-archive", version=1, settings=settings, entries=records)

    def load(self, path: str | Path) -> None:
        """Replace entries only after a compatible file fully validates.

        Recompute fingerprints through add(), preserving saved insertion order.
        Invalid or duplicate records and backend errors leave this archive intact.
        This restores archive contents, not calculator, GP, RNG, or search state.
        """
        self._load_payload(_read_slab_json(path))

    def _load_payload(self, payload):
        """Validate an archive payload before replacing any existing entries."""
        target, settings = self._persistence_target()
        _check_fields(payload, {"format", "version", "settings", "entries"})
        if (payload["format"] != "crisp-slab-archive"
                or type(payload["version"]) is not int or payload["version"] != 1):
            raise ValueError("Unsupported slab archive format/version")
        if payload["settings"] != settings:
            raise ValueError("Slab archive settings do not match the destination archive")
        if not isinstance(payload["entries"], list):
            raise ValueError("Slab archive entries must be a list")
        for record in payload["entries"]:
            _check_fields(record, {"atoms", "info", "energy_per_atom", "metadata", "generation"})
            if (not isinstance(record["atoms"], dict) or "info" in record["atoms"]
                    or not isinstance(record["info"], dict)
                    or not isinstance(record["metadata"], dict)):
                raise ValueError("Invalid slab archive atoms/info/metadata record")
            for name in ("info", "metadata", "generation"):
                _check_json_value(record[name])
            # Decode only structural ASE data: user metadata keys must stay literal.
            atom_data = json.loads(json.dumps(record["atoms"]), object_hook=_slab_object_hook)
            # ASE otherwise silently casts fractional species, truthy PBC or complex positions.
            for name, kinds, shape in (("numbers", "iu", (len(self._numbers),)),
                                       ("pbc", "b", (3,)),
                                       ("positions", "f", (len(self._numbers), 3)),
                                       ("cell", "f", (3, 3))):
                value = atom_data.get(name)
                if (not isinstance(value, np.ndarray) or value.dtype.kind not in kinds
                        or value.shape != shape):
                    raise ValueError(f"Invalid slab archive atom field: {name}")
            atoms = Atoms.fromdict(atom_data)
            if any(array.dtype.kind not in "biufcU" for array in atoms.arrays.values()):
                raise TypeError("Slab archive atom arrays must be numeric, boolean or Unicode")
            atoms.info = record["info"]
            if not target.add(atoms, record["energy_per_atom"], metadata=record["metadata"]):
                raise ValueError("Slab archive contains duplicate entries")
            target.entries[-1].generation = record["generation"]
        self.entries = target.entries

    def _unsupported_checkpoint(self, *args, **kwargs):
        raise NotImplementedError("Slab search checkpoints require a separate GP/RNG/search-state contract")

    save_checkpoint = load_checkpoint = _unsupported_checkpoint


def _read_slab_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=_invalid_json_constant, object_pairs_hook=_unique_json_fields)


def _write_slab_json(path, payload):
    payload = json.dumps(payload, default=_slab_json_default, allow_nan=False, indent=2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _check_json_value(value):
    """Reject lossy key coercion and unsupported objects in user metadata."""
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Slab archive metadata keys must be strings")
            _check_json_value(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_json_value(item)
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise TypeError("Slab archive metadata must contain JSON-compatible values")
    elif isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Slab archive metadata numbers must be finite")


def _check_fields(value, expected):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Invalid slab archive record fields")


def _invalid_json_constant(value):
    raise ValueError(f"Nonfinite JSON value in slab archive: {value}")


def _unique_json_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Repeated JSON key in slab archive: {key}")
        result[key] = value
    return result


def _slab_object_hook(value):
    if "__ndarray__" in value:
        shape, dtype, data = value["__ndarray__"]
        if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                or not isinstance(data, list) or np.dtype(dtype).kind not in "biufcU"):
            raise ValueError("Invalid slab archive array encoding")
        size = 1
        for n in shape:
            size *= n
        if len(data) != size * (2 if np.dtype(dtype).kind == "c" else 1):
            raise ValueError("Slab archive array data does not match its shape")
    decoded = ase_object_hook(value)
    if "__ndarray__" in value:
        encoded = json.dumps(decoded, default=_slab_json_default, allow_nan=False, sort_keys=True)
        if encoded != json.dumps(value, allow_nan=False, sort_keys=True):
            raise ValueError("Slab archive array decoding would change its data")
    return decoded


def _slab_json_default(value):
    # ASE 3.22 lacks floating-scalar support and emits undecodable Unicode dtype names.
    if isinstance(value, np.floating):
        return float(value)
    encoded = ASEJSONEncoder().default(value)
    if isinstance(value, np.ndarray):
        shape, _, data = encoded["__ndarray__"]
        encoded["__ndarray__"] = (shape, str(value.dtype), data)
    return encoded
