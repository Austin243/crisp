"""Initial placement groups and reconstruction-aware slab geometry.

Layer membership is provenance, never a constraint on the relaxed positions.
All groups share one xy cell. A group can describe one atomic plane or a
finite-thickness sheet; the final bonded components need not match the groups.
"""

from collections import Counter
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Mapping

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.neighborlist import neighbor_list

from .slab import SlabConfig, prepare_slab, validate_slab
from .slab_generation import SlabGenerationError
from .slab_validation import _check_periodic_connectivity


def _finite(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


@dataclass(frozen=True)
class InitialLayer:
    """Composition, z extent (Angstrom), and fractional xy offset at creation."""

    composition: Mapping[str, int]
    initial_thickness: float = 0.0
    lateral_offset: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        if not isinstance(self.composition, Mapping) or not self.composition:
            raise ValueError("Each initial layer needs a composition")
        for symbol, count in self.composition.items():
            if (symbol not in atomic_numbers or atomic_numbers[symbol] == 0
                    or isinstance(count, bool) or not isinstance(count, Integral) or count <= 0):
                raise ValueError("Layer composition must map chemical symbols to positive integer counts")
        composition = {symbol: int(self.composition[symbol])
                       for symbol in sorted(self.composition, key=atomic_numbers.get)}
        object.__setattr__(self, "composition", MappingProxyType(composition))
        thickness = _finite(self.initial_thickness, "initial_thickness")
        if self.natoms == 1 and thickness != 0:
            raise ValueError("A one-atom initial group must have zero thickness")
        object.__setattr__(self, "initial_thickness", thickness)
        offset = tuple(self.lateral_offset)
        if (len(offset) != 2 or any(isinstance(v, bool) or not isinstance(v, Real)
                                    or not np.isfinite(v) for v in offset)):
            raise ValueError("lateral_offset must contain two finite fractional coordinates")
        object.__setattr__(self, "lateral_offset", tuple(float(v) % 1.0 for v in offset))

    @property
    def natoms(self):
        return sum(self.composition.values())

    def to_dict(self):
        return dict(composition=dict(self.composition), initial_thickness=self.initial_thickness,
                    lateral_offset=list(self.lateral_offset))


@dataclass(frozen=True)
class LayeredSlabSpec:
    """One exact composition/cell-height search with initial-only layer inputs.

    Gaps are nearest-z-extent separations in Angstrom. Empty gaps mean zero
    between every pair of adjacent groups. ``initial_vacuum_gap`` is a minimum
    at generation, not a constraint on later reconstruction. ``config`` holds
    the overall relaxed bounds; its single-sheet ``initial_thickness`` is not
    used for layered generation. Pair keys are two chemical symbols, unordered.
    """

    layers: tuple[InitialLayer, ...]
    config: SlabConfig
    initial_layer_gaps: tuple[float, ...] = ()
    min_dist_ang: float = 1.0
    bond_scale: float = 1.2
    pair_min_distances: Mapping[tuple[str, str], float] | None = None
    initial_vacuum_gap: float | None = None

    def __post_init__(self):
        layers = tuple(self.layers)
        if not layers or not all(isinstance(layer, InitialLayer) for layer in layers):
            raise ValueError("layers must contain InitialLayer objects")
        if not isinstance(self.config, SlabConfig):
            raise TypeError("config must be SlabConfig")
        object.__setattr__(self, "layers", layers)
        gaps = tuple(self.initial_layer_gaps) or (0.0,) * (len(layers) - 1)
        if len(gaps) != len(layers) - 1:
            raise ValueError("initial_layer_gaps must have one value between adjacent groups")
        object.__setattr__(self, "initial_layer_gaps",
                           tuple(_finite(v, "initial_layer_gap") for v in gaps))
        for name in ("min_dist_ang", "bond_scale"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, positive=True))
        vacuum = self.config.min_vacuum if self.initial_vacuum_gap is None else self.initial_vacuum_gap
        object.__setattr__(self, "initial_vacuum_gap", _finite(vacuum, "initial_vacuum_gap", positive=True))
        if self.initial_stack_thickness > self.config.max_thickness + 1e-8:
            raise ValueError("Initial stack exceeds config.max_thickness")
        if self.initial_stack_thickness + self.initial_vacuum_gap > self.config.cell_height + 1e-8:
            raise ValueError("cell_height cannot accommodate the initial stack and vacuum")
        distances = {}
        if self.pair_min_distances is not None and not isinstance(self.pair_min_distances, Mapping):
            raise TypeError("pair_min_distances must be a mapping")
        for pair, value in (self.pair_min_distances or {}).items():
            if (not isinstance(pair, tuple) or len(pair) != 2
                    or any(s not in self.composition for s in pair)):
                raise ValueError("Distance pair keys must be two symbols in the composition")
            key = tuple(sorted(pair))
            distance = _finite(value, "pair minimum distance", positive=True)
            if key in distances and distances[key] != distance:
                raise ValueError("Conflicting minimum distances for the same species pair")
            distances[key] = distance
        object.__setattr__(self, "pair_min_distances", MappingProxyType(distances))

    @property
    def composition(self):
        composition = Counter()
        for layer in self.layers:
            composition.update(layer.composition)
        return dict(composition)

    @property
    def natoms(self):
        return sum(layer.natoms for layer in self.layers)

    @property
    def initial_stack_thickness(self):
        return sum(layer.initial_thickness for layer in self.layers) + sum(self.initial_layer_gaps)

    def initial_identities(self):
        """Return species numbers and placement-group IDs in original ID order."""
        numbers, groups = [], []
        for group, layer in enumerate(self.layers):
            for symbol, count in layer.composition.items():
                numbers.extend([atomic_numbers[symbol]] * count)
                groups.extend([group] * count)
        return np.array(numbers, dtype=int), np.array(groups, dtype=int)

    def to_dict(self):
        return dict(layers=[layer.to_dict() for layer in self.layers],
                    config=asdict(self.config), initial_layer_gaps=list(self.initial_layer_gaps),
                    min_dist_ang=self.min_dist_ang, bond_scale=self.bond_scale,
                    pair_min_distances=[[a, b, distance] for (a, b), distance
                                        in sorted(self.pair_min_distances.items())],
                    initial_vacuum_gap=self.initial_vacuum_gap)

    @classmethod
    def from_dict(cls, data):
        """Restore the explicit JSON-friendly representation from to_dict()."""
        data = dict(data)
        data["layers"] = tuple(InitialLayer(**layer) for layer in data["layers"])
        data["config"] = SlabConfig(**data["config"])
        pairs = data.get("pair_min_distances", [])
        if len({tuple(sorted((a, b))) for a, b, _ in pairs}) != len(pairs):
            raise ValueError("Repeated species pair in layer specification")
        data["pair_min_distances"] = {(a, b): distance for a, b, distance in pairs}
        return cls(**data)


def _check_provenance(atoms, spec):
    ids = atoms.arrays.get("initial_atom_id")
    groups = atoms.arrays.get("initial_layer")
    for name, value in (("initial_atom_id", ids), ("initial_layer", groups)):
        if value is None or value.shape != (len(atoms),) or value.dtype.kind not in "iu":
            raise ValueError(f"Missing or invalid integer provenance array: {name}")
    if not np.array_equal(np.sort(ids), np.arange(spec.natoms)):
        raise ValueError("initial_atom_id must be a permutation of original atom identities")
    numbers, initial_groups = spec.initial_identities()
    if not np.array_equal(atoms.numbers, numbers[ids]) or not np.array_equal(groups, initial_groups[ids]):
        raise ValueError("Atom species/initial_layer do not match initial_atom_id provenance")


def _check_distances(atoms, spec):
    pairs = {(atomic_numbers[a], atomic_numbers[b]): d
             for (a, b), d in spec.pair_min_distances.items()}
    pairs.update({(b, a): d for (a, b), d in tuple(pairs.items())})
    cutoff = max([spec.min_dist_ang, *pairs.values()])
    i, j, distances = neighbor_list("ijd", atoms, cutoff)
    limits = np.array([pairs.get((atoms.numbers[a], atoms.numbers[b]), spec.min_dist_ang)
                       for a, b in zip(i, j)])
    if np.any(distances < limits):
        raise ValueError("Layered slab violates the species-pair minimum distance policy")


def _check_sheet_components(atoms, spec):
    """Each quotient bond component must be a connected, fully periodic 2D net."""
    cutoffs = spec.bond_scale * covalent_radii[atoms.numbers]
    if not np.isfinite(cutoffs).all():
        raise ValueError("Scaled covalent radii must be finite")
    i, j, shifts = neighbor_list("ijS", atoms, cutoffs)
    neighbors = [[] for _ in atoms]
    for a, b in zip(i, j):
        neighbors[a].append(b)
    remaining = set(range(len(atoms)))
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        component, pending = [first], [first]
        while pending:
            for target in neighbors[pending.pop()]:
                if target in remaining:
                    remaining.remove(target)
                    pending.append(target)
                    component.append(target)
        mapping = np.full(len(atoms), -1, dtype=int)
        mapping[component] = np.arange(len(component))
        edges = mapping[i] >= 0
        _check_periodic_connectivity(len(component), mapping[i[edges]], mapping[j[edges]], shifts[edges])


def validate_layered_slab(atoms: Atoms, spec: LayeredSlabSpec, *,
                          check_provenance: bool = True, check_connectivity: bool = True) -> None:
    """Validate reconstruction without enforcing original groups' z or chemistry.

    All atoms must belong to one or more 2D sheets after relaxation; detached
    molecules/chains and unsupported interpenetrating periodic nets are rejected.
    For *unrelaxed* candidates, pass check_connectivity=False. Distances include
    periodic self-images. This is a geometric heuristic, not a stability test.
    Inputs must have contiguous, unwrapped z; no coordinate repair is performed.
    """
    validate_slab(atoms, spec.config)
    if np.any(atoms.numbers <= 0) or np.any(atoms.numbers >= len(covalent_radii)):
        raise ValueError("Layered slab requires known chemical elements")
    if Counter(atoms.get_chemical_symbols()) != spec.composition:
        raise ValueError("Layered slab total composition differs from the specification")
    if check_provenance:
        _check_provenance(atoms, spec)
    _check_distances(atoms, spec)
    if check_connectivity:
        _check_sheet_components(atoms, spec)


def generate_layered_slabs(spec: LayeredSlabSpec, n: int, *, seed=None,
                           max_attempts: int = 1000) -> list[Atoms]:
    """Seeded random placement in a shared oblique xy cell (no imposed symmetry).

    Finite groups span their requested initial thickness exactly; gaps are
    nearest-extent gaps. Proposals satisfy geometry/provenance/distance checks,
    but need not already be bonded sheets before physical relaxation.
    """
    for name, value, minimum in (("n", n, 0), ("max_attempts", max_attempts, 1)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    rng = np.random.default_rng(seed)
    structures = []
    numbers, groups = spec.initial_identities()
    last_error = None
    for _ in range(max_attempts if n else 0):
        attempt_seed = int(rng.integers(2**32))
        trial = np.random.default_rng(attempt_seed)
        area = trial.uniform(*spec.config.area_per_atom_range) * spec.natoms
        angle = trial.uniform(np.pi / 3, 2 * np.pi / 3)
        a = np.sqrt(area / np.sin(angle)) * np.exp(trial.uniform(-0.25, 0.25))
        b = area / (a * np.sin(angle))
        cell = np.array([[a, 0, 0], [b * np.cos(angle), b * np.sin(angle), 0],
                         [0, 0, spec.config.cell_height]])
        positions = []
        bottom = 0.0
        for group, layer in enumerate(spec.layers):
            xy = (trial.random((layer.natoms, 2)) + layer.lateral_offset) % 1.0
            z = trial.random(layer.natoms) if layer.initial_thickness else np.zeros(layer.natoms)
            if layer.initial_thickness:
                z = (z - z.min()) / (z.max() - z.min()) * layer.initial_thickness
            block = xy @ cell[:2]
            block[:, 2] = bottom + z
            positions.extend(block)
            bottom += layer.initial_thickness
            if group < len(spec.initial_layer_gaps):
                bottom += spec.initial_layer_gaps[group]
        atoms = prepare_slab(Atoms(numbers=numbers, positions=positions, cell=cell), spec.config)
        atoms.set_array("initial_atom_id", np.arange(spec.natoms, dtype=int))
        atoms.set_array("initial_layer", groups.copy())
        atoms.info.update(origin="layered_random", generation_seed=attempt_seed,
                          layer_policy="initial_only")
        try:
            validate_layered_slab(atoms, spec, check_connectivity=False)
        except ValueError as exc:
            last_error = exc
            continue
        structures.append(atoms)
        if len(structures) == n:
            return structures
    if n == 0:
        return []
    raise SlabGenerationError(f"Generated {len(structures)}/{n} layered slabs after {max_attempts} attempts; "
                              f"last rejection: {last_error}") from last_error
