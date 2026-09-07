# Initial layers and reconstructed sheets

The optional layered helpers describe the starting arrangement of a fixed-composition
2D search. They preserve the original bulk workflow and existing single-sheet
helpers. They do not launch VASP or implement a search driver.

```python
from crisp.slab import SlabConfig
from crisp.slab_layers import InitialLayer, LayeredSlabSpec, generate_layered_slabs

spec = LayeredSlabSpec(
    layers=(InitialLayer({"S": 2}), InitialLayer({"Mo": 2}), InitialLayer({"S": 2})),
    initial_layer_gaps=(1.5, 1.5),
    config=SlabConfig(
        area_per_atom_range=(2.5, 4.0), initial_thickness=0.0,
        max_thickness=8.0, cell_height=28.0, min_vacuum=18.0,
    ),
    initial_vacuum_gap=20.0,
    min_dist_ang=1.2,
    pair_min_distances={("Mo", "S"): 1.6},
)
candidates = generate_layered_slabs(spec, 4, seed=2026)
```

Lengths are in Angstrom, area per atom in Angstrom squared, and `lateral_offset`
is a pair of fractional xy translations. Layer and atom counts are derived from
the compositions. A finite-thickness group must contain at least two atoms.
`initial_layer_gaps` contains one nearest-extent gap between adjacent groups;
omitting it starts groups with zero gaps. All groups use the same sampled xy
cell, with no imposed symmetry. The generator uses random placement, rejects
short contacts, and has a bounded total attempt budget. Dense/impossible requests
can fail explicitly with `SlabGenerationError`; it never silently returns fewer
candidates than requested.

Each finite-thickness group spans its requested initial thickness. The older
single-sheet `config.initial_thickness` is unused by this generator. Choose one
fixed `cell_height` large enough for both the initial arrangement and the
allowed reconstructed thickness plus minimum vacuum. `initial_vacuum_gap` is a
minimum target; the actual empty gap is cell height minus current stack extent.
No cell resizing or automatic wrapping in z occurs during validation.

## Reconstruction and identity

Initial layers are placement groups, not final constraints. S–Mo–S placement
planes can become one chemically connected monolayer. Two stacked monolayers
can remain separate 2D components. Atoms may buckle, exchange spatial layers,
mix, merge or change gaps while preserving total composition.

The integer ASE arrays `initial_atom_id` and `initial_layer` record each atom's
original identity and placement group. They travel with atoms through sorting
and archive I/O. Identity fixes an atom's species and provenance, not its
coordinates or final layer membership. Never regenerate these IDs from a
relaxed geometry. External relaxation readers must restore the original atom
order and these arrays after reading species-sorted output.

`validate_layered_slab(atoms, spec)` checks geometry, exact total composition,
provenance, the species-pair minimum-distance policy and final periodic bonding.
Every final atom must belong to a connected 2D sheet; several sheets are allowed.
Detached molecules, chains and unsupported interpenetrating periodic nets are
rejected. Bonding uses scaled ASE covalent radii and is a configurable geometric
heuristic, not a physical stability or discovery test. Per-pair distance values
override the scalar fallback and include periodic copies of the same atom.

For random candidates **before** relaxation, use `check_connectivity=False`:
an initial candidate need not already have a bonded 2D topology. Final archives
always require connectivity. `check_provenance=False` permits geometry-only
inspection of external structures but does not make them eligible for insertion.

## Archive

```python
from crisp.slab_fingerprint import SlabFingerprintCalculator
from crisp.slab_layer_archive import LayeredSlabArchive

fp = SlabFingerprintCalculator(spec.config, cutoff=6.0, natx=100)
archive = LayeredSlabArchive(fp, spec)
# archive.add(relaxed_atoms, energy_per_atom, metadata={"candidate_id": "g0-c0"})
archive.save("layered-archive.json")
```

The caller supplies final, compatible zero-pressure energies in eV/atom and
separately verifies convergence. Fingerprints require an actual vacuum gap
larger than their cutoff. The layered archive reuses species-aware duplicate
matching, energy ranking, diversity selection and atomic JSON persistence.
Its separate format/settings prevent loading an earlier single-sheet archive
or a different layer specification accidentally. Fingerprints are recomputed
when loading; invalid loads leave the destination archive unchanged. Archive
files alone do not contain a search's GP, RNG, relaxation jobs or run state.
