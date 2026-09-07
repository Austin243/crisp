# 2D search foundations

These opt-in helpers cover geometry, fingerprints, generation, atomic relaxation,
structural validation, and an in-memory archive for a future 2D search mode.
They do **not** enable 2D searches in `CRISPSearch`. Existing bulk search behavior,
imports, fingerprint code, relaxation, archives, and dependencies are unchanged.

## Geometry contract

- A right-handed cell with `a` and `b` in the xy plane; oblique cells are allowed.
- A perpendicular, fixed `c` vector and physical PBC `(True, True, False)`.
- Atomic x, y, **and z** coordinates remain free. Buckled and finite-thickness
  sheets are allowed; 2D does not imply that every atom has the same z.
- Thickness is `max(z) - min(z)`. The total empty image gap is
  `cell_height - thickness`, not a vacuum amount on each face.
- Geometry settings use Angstrom and Angstrom squared per atom. Only zero
  external pressure is supported by this configuration.

Bounds are explicit because they depend on the material. `initial_thickness`
controls candidate generation; it is not imposed on an existing
structure. `max_thickness` and the minimum image gap bound prepared structures.
Area bounds apply to the in-plane area divided by the number of atoms.

```python
from ase import Atoms
from crisp.slab import SlabConfig, prepare_slab, validate_slab

config = SlabConfig(
    area_per_atom_range=(2.0, 8.0),
    initial_thickness=1.0,
    max_thickness=4.0,
    cell_height=20.0,
    min_vacuum=12.0,
)
atoms = Atoms(
    "C2", positions=[[0.0, 0.0, -0.3], [1.5, 1.5, 0.3]],
    cell=[3.0, 3.0, 0.0],
)
slab = prepare_slab(atoms, config)
validate_slab(slab, config)
```

`prepare_slab` returns a copy, installs the configured cell height without
scaling atoms, wraps only xy, and centers the z extent. It preserves relative
z displacements and does not mutate the input. As with `Atoms.copy()`, the
result has no attached calculator. The input must already be oriented in xy
with contiguous z coordinates; tilted cells and slabs split across a periodic
z boundary are not automatically repaired. A zero initial cell height is valid.

`validate_slab` checks geometry without changing it; rigid z translations are
allowed. These helpers do not establish 2D bonding/connectivity, minimum atomic
separations, calculator compatibility, or safe fingerprint image clearance.
The separate adapters below add distance, connectivity, and fingerprint clearance
checks; calculator-specific checks and search integration belong to later PRs.

Run the focused tests with `python -m unittest discover -s tests -p test_slab.py`.
They use the existing base dependencies and need no optional fingerprint backend.

## Slab fingerprints

Use the separate `SlabFingerprintCalculator` for structures satisfying the
geometry contract. It inherits the existing fingerprint mathematics and adds
validation and normalization before each value or derivative calculation:

```python
from crisp.slab_fingerprint import SlabFingerprintCalculator

fp_calc = SlabFingerprintCalculator(config, cutoff=3.2, natx=16, orbital="s")
fp = fp_calc.get_fingerprints(slab)
```

The current `torch_fplib` backend searches periodic images along all three
axes. This adapter requires the **actual** empty z gap to exceed the fingerprint
cutoff by more than `1e-6` Angstrom. With the perpendicular cell required here,
that excludes every z image from the local environment. The usual neighbor
capacity check also applies; an undersized `natx` is rejected.

Each call validates the input before wrapping xy and centering z on a copy.
Invalid PBC, changed cell height, and insufficient image clearance are rejected
instead of silently repaired. All five value/derivative entry points use the
same preparation, and inherited distance methods use the guarded fingerprints.

Both `s` and `sp` orbitals retain all Cartesian force components and all six
Voigt strain/stress components. Stress remains in ASE's volume-normalized
convention: compare `volume * stress` or in-plane `cell_height * stress` when
changing vacuum, not raw stress. The slab quench below freezes the entire cell.
This descriptor does not establish bonding/connectivity, physical
calculator compatibility, or a complete 2D search.

### Reproduce the backend checks

The slab tests were verified against unmodified `torch_fplib` commit
[`6b622cf8156fd0a2cfd178d3a114307ec39fe687`](https://github.com/Rutgers-ZRG/torch_fplib/commit/6b622cf8156fd0a2cfd178d3a114307ec39fe687).
Use a checkout of that revision on `PYTHONPATH` with CRISP's base dependencies
installed; this does not change CRISP's package requirements. From the CRISP
repository, using an unused temporary checkout path:

```sh
git clone https://github.com/Rutgers-ZRG/torch_fplib.git /tmp/crisp-torch-fplib
git -C /tmp/crisp-torch-fplib checkout --detach 6b622cf8156fd0a2cfd178d3a114307ec39fe687
PYTHONPATH=/tmp/crisp-torch-fplib python -c "import torch_fplib" && \
PYTHONPATH=/tmp/crisp-torch-fplib python -m unittest discover -s tests -p test_slab_fingerprint.py -v
```

The import preflight prevents a missing backend from yielding a skipped audit.
Tests compare `s`/`sp` fingerprints against an independent ASE mixed-PBC neighbor
list, check xyz forces and in-plane strain derivatives by finite differences,
and verify vacuum/translation invariance, input preservation, and buffer limits.
The normal lightweight suite still skips these backend audits when
`torch_fplib` is absent; the geometry/clearance guard tests run either way.

## Random slab candidates

`generate_slabs` uses PyXtal's `dim=2` layer groups (1–80), with explicit
composition, area per atom, and slab bounds. Install the existing optional
dependency with `python -m pip install -e ".[search]"`. Generation has been
verified with PyXtal 1.1.4; the existing bulk dependency requirements are unchanged.

```python
from crisp.slab_generation import generate_slabs

candidates = generate_slabs(
    {"C": 4}, config, n=2, seed=7, min_dist_ang=1.0,
    layer_groups=[1, 2, 31, 80], max_attempts=40,
)
for candidate in candidates:
    validate_slab(candidate, config)
```

The function returns exactly `n` ASE structures or raises a descriptive error.
By default it samples from all composition-compatible layer groups. Species
counts refer to the generated cell, with no primitive-cell reduction. Invalid
arguments and incompatible group requests fail before generation. Failed
generation or rejected candidates consume one attempt; exhaustion raises
`RuntimeError` with the last rejection, without returning a partial batch or
falling back to bulk generation. `max_attempts` defaults to `20*n` and bounds
PyXtal calls, each of which also has bounded internal trials; it is not a time limit.

In-plane area is sampled uniformly within the configured bounds, independently
of vacuum. The cell metric follows the layer group: square or hexagonal where
required, otherwise an aspect ratio from 0.5 to 2, with oblique angles from
60 to 120 degrees where permitted. These are simple sampling choices, not
chemistry-specific recommendations or an exhaustive cell-shape search.

`initial_thickness` is the generation scale for z coordinates, **not** a bound
on their final extent: symmetry operations can produce a larger span. Zero
requests planar candidates. Raw symmetry-site coordinates are exported without
wrapping z, then `prepare_slab` supplies vacuum and checks the actual thickness
against `max_thickness`. A final mixed-PBC neighbor check enforces `min_dist_ang`
for every species pair, including periodic copies of the same atom and contacts
created by planar flattening. No connected-sheet or stability claim is made.

An integer `seed` reproduces a batch with the same arguments and dependency
versions, without consuming NumPy's legacy global RNG. A supplied NumPy
`Generator` advances in place. Composition order and duplicate/order changes
to `layer_groups` do not change results. `Atoms.info` records `origin="random"`,
the generating `layer_group`, the per-attempt `generation_seed`, and
`pyxtal_version`. The per-attempt seed alone does not encode the sampled cell;
retain the full request and batch seed for reproduction. The group is generation
provenance, not a subsequent symmetry classification.

Run `python -c "import pyxtal"` before
`python -m unittest discover -s tests -p test_slab_generation.py -v` for a full
generation audit. Without PyXtal, the real generation tests skip and input
validation tests still run. This module is separate from the bulk generator.
Candidates can use the fixed-cell quench and slab archive below;
`CRISPSearch` integration remains later work.

## Fixed-cell atomic quench

`quench_slab` relaxes a valid slab copy with ASE's
[`LBFGS`](https://docs.ase-lib.org/ase/optimize.html#ase.optimize.LBFGS), keeping
the **entire cell** fixed. It allows every atom to move in x, y, and z, so a
planar candidate can buckle. It introduces no z constraint or cell filter and
requires only an ASE calculator's energy and forces, without requesting stress.
Pass a factory that supplies a fresh calculator supporting physical mixed PBC.

This small example uses ASE's EMT calculator to demonstrate the interface:

```python
from ase.calculators.emt import EMT
from crisp.slab_relaxation import quench_slab

quench_config = SlabConfig(
    area_per_atom_range=(4.0, 20.0), initial_thickness=1.0,
    max_thickness=4.0, cell_height=20.0, min_vacuum=12.0,
)
candidate = prepare_slab(
    Atoms("Cu2", positions=[[0.0, 0.0, 0.0], [2.8, 0.3, 0.4]],
          cell=[6.0, 6.0, 0.0]),
    quench_config,
)
relaxed = quench_slab(candidate, quench_config, EMT, fmax=0.05, max_steps=200)
energy_per_atom = relaxed.get_potential_energy() / len(relaxed)
```

The input must already satisfy `validate_slab` and have no ASE constraints;
constraints are rejected rather than removed. The input's positions, cell,
metadata, and attached calculator are preserved. The returned copy retains
the fresh calculator and its final energy/forces. Coordinates are not wrapped
or recentered during or after relaxation, so the results refer to exactly the
returned geometry. The generating layer-group tag remains provenance, not a
claim that relaxation preserves that symmetry.

Success requires every atomic force norm to be below `fmax` (eV/Angstrom).
`max_steps` bounds optimizer moves, not calculator calls or wall time. Zero
evaluates the starting structure and succeeds only if already converged.
The helper checks slab geometry before each evaluation and rejects nonfinite
energy or forces. A bounded loop around `LBFGS.step()` gives consistent limits
and validation order across ASE versions. It raises `ValueError` for invalid
geometry/options/constraints and `RuntimeError` for invalid calculator results
or an exhausted budget; calculator exceptions propagate. Failed runs never
return a structure labeled as relaxed. Successful results record
`slab_quench_steps` and `slab_quench_fmax` in `Atoms.info`.

The slab configuration requires zero external pressure: energy per atom is the
quantity available for later ranking, with no vacuum-dependent pressure-volume
term. Use the candidate validator below to check distances and connectivity after
relaxation. In-plane cell optimization, material-specific energy checks,
and search integration remain later work. A converged fixed-cell
candidate is not yet a validated 2D material or a minimum with respect to in-plane
strain.

Run `python -m unittest discover -s tests -p test_slab_relaxation.py -v`.
These tests need only the existing base dependencies, with no PyXtal or
fingerprint backend required. They were verified with ASE 3.22.1 and 3.29.0.

## Candidate distances and connectivity

`validate_slab_candidate` checks a candidate's geometry, minimum atomic
separations, and whether **all atoms and their xy repeats form one connected
2D bond network**. Call it explicitly after relaxation, or insert the candidate
through `SlabArchive`, which calls it before insertion. The generator, quench,
and bulk search do not call it automatically. For a reference sheet:

```python
from ase.build import graphene
from crisp.slab_validation import validate_slab_candidate

sheet = prepare_slab(graphene(), config)
validate_slab_candidate(sheet, config, min_dist_ang=1.0, bond_scale=1.2)
```

The function returns `None` on success and raises `ValueError` with a rejection
reason otherwise. It preserves the input, including its coordinate frame,
metadata, and calculator; it does not calculate energies or forces.

- `min_dist_ang` is required and applies to every pair of physical atoms,
  including periodic copies of the same atom. Distances strictly below it
  are rejected, independently of the bonding rule.
- A bond exists when the full xyz distance is strictly less than
  `bond_scale * (radius_i + radius_j)`, using ASE's covalent radii.
  `bond_scale` defaults to 1.2. Both settings must be finite and positive;
  choose them for the chemistry and retain them with the search settings.
- Bonds honor `(True, True, False)` PBC. No artificial z images, nearest-image
  shortcuts, or projected xy distances are used. Buckled and finite-thickness
  sheets can pass; molecules, chains/ribbons, detached atoms, and disconnected
  layers are rejected under the chosen bond cutoff.

Connectivity includes periodic image offsets. The validator assigns integer
offsets along a spanning tree, then collects the translations generated by
bond cycles. Those translations must generate the full xy integer lattice.
In two dimensions this is checked exactly by the greatest common divisor of
their pairwise determinants: it must equal one. Zero indicates a molecular or
1D network; a larger value indicates separate interpenetrating networks even
when all atom indices look connected within the cell. This avoids expanding
a fixed-size supercell or relying on a floating-point rank tolerance. See the
[periodic-graph connectivity explanation](https://delonecommons.github.io/pbcgraph/general/theory/)
for the underlying cycle-translation criterion; no graph package is added.

This is a configurable geometric bonding heuristic. In particular, detached
van der Waals layers outside the cutoff fail this single-network criterion.
Passing does not prove chemical bonding, energetic/dynamical stability, or
quench convergence. Material-specific energy sanity checks, calculator
compatibility, and a complete 2D search remain separate work.

Run `python -m unittest discover -s tests -p test_slab_validation.py -v`.
The validator and its tests use the existing base dependencies only. The focused
tests were verified with ASE 3.22.1 and 3.29.0.

## Slab archive and identity

`SlabArchive` stores candidates for one **exact composition and atom count**
under one slab geometry/fingerprint configuration. It validates candidates
before computing fingerprints and reuses the existing archive's ranking,
diversity selection, pooled features, and energy accessors. For an executable
interface example with ASE's EMT calculator:

```python
from crisp.slab_archive import SlabArchive

archive = SlabArchive(
    SlabFingerprintCalculator(config, cutoff=3.2, natx=32), {"Cu": 1},
    min_dist_ang=1.5, bond_scale=1.2,
)
candidate = prepare_slab(
    Atoms("Cu", positions=[[0.0, 0.0, 0.0]], cell=[2.5, 2.5, 0.0]), config,
)
relaxed = quench_slab(candidate, config, EMT)
added = archive.add(
    relaxed, relaxed.get_potential_energy() / len(relaxed),
    metadata={"generation": 0},
)
best = archive.get_best(1)[0]
```

The caller supplies a finite energy in eV/atom from the intended calculator.
Use the same energy model and conventions throughout an archive. Insertion
does not evaluate a calculator or infer relaxation convergence. At zero
pressure, stored enthalpy equals energy; ranking uses no vacuum-dependent
pressure-volume term. Different stoichiometries or supercell atom counts are
rejected instead of comparing incompatible targets.

Each candidate must pass the distance/connectivity validator and fingerprint
image-clearance guard. Validation happens **before** normalization, so invalid
PBC or a changed cell height is rejected. The stored copy is wrapped in xy,
centered in z, and sorted by atomic number. This aligns species blocks for the
existing Hungarian matcher; permutations within each species are matched by
the matcher. The input and its calculator are preserved, stored atoms have no
calculator, and nested metadata plus fingerprint arrays are copied.

A duplicate requires both the species-matched fingerprint distance to be
strictly below `fp_threshold` (default 0.03) and the energy difference to be
strictly below `energy_threshold` (default 0.01 eV/atom). `add` returns `True`
for insertion or `False` for a duplicate. It retains the **first** matching
entry, including when a later near-duplicate has a slightly lower energy.
Both duplicate thresholds must be finite and strictly positive.
Finite-cutoff fingerprints provide approximate identity: no primitive-cell
reduction, exact structural-equivalence proof, or supercell matching is added.

Invalid candidates, nonfinite energies/descriptors, and changed geometry or
fingerprint settings raise before insertion. Backend and matcher failures
propagate. Changing `config`, `cutoff`, `natx`, or `orbital` on the fingerprint
calculator requires a new archive, so cached descriptors cannot be mixed.
Treat exposed archive entries and cached arrays as read-only.

This step provides an in-memory archive. `save`, `load`, `save_checkpoint`, and
`load_checkpoint` raise `NotImplementedError` without writing or reading files;
the bulk persistence format does not record the required slab contract. Slab
checkpoint compatibility is the next separate step. Bulk archives and
`CRISPSearch` remain unchanged.

Run `python -m unittest discover -s tests -p test_slab_archive.py -v`.
Guard tests use the base dependencies; real descriptor tests require the
`torch_fplib` preflight and source checkout documented above.
