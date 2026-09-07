# 2D search

`SlabRandomSearch` provides an opt-in, automated 2D random-search baseline using
the slab geometry, fingerprints, generation, atomic quench, validation, and archive
components below. It samples in-plane cells and relaxes atoms with each cell fixed.
It does **not** add a 2D mode to `CRISPSearch`, GP screening, fingerprint-guided
movement, or in-plane lattice relaxation. Original bulk modules and package
requirements are unchanged.

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
checks. The local driver below connects them; physical calculator compatibility
remains the caller's responsibility.

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
quantity used for ranking, with no vacuum-dependent pressure-volume
term. Use the candidate validator below to check distances and connectivity after
relaxation. In-plane cell optimization, material-specific energy checks,
and integration with the full CRISP algorithm remain later work. A converged fixed-cell
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

Bulk archives and `CRISPSearch` remain unchanged.

Run `python -m unittest discover -s tests -p test_slab_archive.py -v`.
Guard tests use the base dependencies; real descriptor tests require the
`torch_fplib` preflight and source checkout documented above.

## Slab archive save/load

`SlabArchive.save(path)` writes a **single JSON file**. Construct a destination
archive with the same settings, then call `load(path)` to replace its entries:

```python
archive.save("slab-archive.json")
restored = SlabArchive(
    SlabFingerprintCalculator(config, cutoff=3.2, natx=32), {"Cu": 1},
    min_dist_ang=1.5, bond_scale=1.2,
)
restored.load("slab-archive.json")
best = restored.get_best(1)[0]
```

Format `crisp-slab-archive`, version 1, records the full `SlabConfig`, exact
composition, fingerprint cutoff/capacity/orbital, distance and bond cutoffs,
and both duplicate thresholds. All must match the destination archive;
loading does not replace its configuration. Missing fields, unknown versions,
bulk archive files, and incompatible settings are rejected.

Records preserve insertion order, energies in eV/atom, generation tags,
metadata, and ASE structures: cell, positions, atomic numbers, physical PBC,
ordinary per-atom arrays, constraints, cell display displacement, and `Atoms.info`.
Stored enthalpy and pressure are derived as energy and zero. Calculators and
cached fingerprints are omitted. Loading recomputes descriptors through `add`,
including geometry, connectivity, image-clearance, and composition validation.
A duplicate record is an error, so a load cannot silently discard entries.

Metadata and `Atoms.info` support finite JSON values: string-keyed dictionaries,
lists, strings, numbers, booleans, and `None`; tuples become lists. Convert
NumPy containers/scalars to ordinary Python values before saving metadata.
Unsupported objects and object/structured/time atom arrays raise rather than
being converted to strings. Numeric-looking and ASE-reserved metadata keys retain
their literal meaning. Additional numeric, boolean, and Unicode per-atom arrays
retain their dtype; ASE constraints must support ASE's dictionary serialization.

Saving serializes before touching the destination, then writes a temporary file
beside it and atomically replaces it. Serialization or write failure preserves
the previous file and removes the temporary file. Loading builds a separate
archive first: a malformed or invalid record, duplicate, or backend error leaves
the destination's existing entries intact. Empty archives also round-trip.

Keep archive entries and validation/duplicate settings unchanged while populated.
Saving trusts those read-only entries and performs no fingerprint or energy
calculations; loading revalidates them. Manually edited entries or changed
thresholds can therefore make a saved file fail validation on reload. Use the
same fingerprint backend/dependency versions for reproducible descriptors and
the same physical energy model when adding further structures. The file records
settings, not the calculator or a complete software environment.

`save_checkpoint` and `load_checkpoint` still raise `NotImplementedError`:
these archive methods restore archive contents, not search progress or GP state.
Use the separate driver's `save`/`load` methods below for random-search progress.

Run `python -m unittest discover -s tests -p 'test_slab_archive*.py' -v`.
The archive tests pass with ASE 3.22.1 and 3.29.0, including real `s`/`sp` round trips.

## Automated local random search

`SlabRandomSearch` generates one candidate per trial, quenches all atomic xyz
coordinates at fixed cell, checks distances and periodic connectivity, and inserts
distinct candidates into `SlabArchive`. It ranks the physical energy per atom.
Start with an empty archive, an explicit energy-model label, and a factory that
returns a fresh ASE calculator. The factory must support mixed PBC and the target
chemistry; energy and forces must be finite and real. No stress is requested.

```python
from crisp.slab_search import SlabRandomSearch

search_config = SlabConfig(
    area_per_atom_range=(4.5, 6.0), initial_thickness=1.0,
    max_thickness=3.0, cell_height=18.0, min_vacuum=12.0,
)
search_archive = SlabArchive(
    SlabFingerprintCalculator(search_config, cutoff=3.2, natx=32), {"Cu": 4},
    min_dist_ang=1.5,
)
search = SlabRandomSearch(
    search_archive, EMT, calculator_id="ASE-EMT-default", seed=23,
    layer_groups=[1, 2, 80], max_generation_attempts=6, fmax=0.05, max_steps=200,
)
search.run(8, checkpoint="slab-search.json")
print(search.completed_trials, search.counts)
best_entries = search.archive.get_best(5)
```

`run(n_trials)` always means **additional trials**, including rejected candidates
and duplicates. The archive may remain empty; inspect `counts` and `outcomes`
instead of interpreting an exhausted budget as a successful material discovery.
Every completed trial records a one-based trial number, its generation seed,
status, rejection reason, and (after successful quench) energy and step count:

| Status | Meaning |
| --- | --- |
| `accepted` | A converged, validated, distinct slab was inserted. |
| `duplicate` | The existing fingerprint-plus-energy identity rule matched an entry. |
| `generation_rejected` | The per-trial PyXtal attempt budget was exhausted. |
| `quench_rejected` | Relaxation crossed slab geometry bounds or exhausted its step budget. |
| `validation_rejected` | The converged candidate failed distance/connectivity checks. |

Each trial makes at most one quench request, at most `max_generation_attempts`
PyXtal calls, and at most `max_steps` optimizer moves. These are not limits on
wall time or exact calculator evaluations. Zero trials makes no candidate or
calculator calls; zero optimizer steps accepts only initially converged candidates.
Worst-case allowed thickness must leave more than cutoff + `1e-6` Angstrom of
empty z gap. Undersized fingerprint capacity remains a fatal configuration error.

Expected rejections use narrow exception types in the opt-in generation/quench
helpers; they retain the existing `ValueError`/`RuntimeError` inheritance.
Invalid options, calculator errors/nonfinite results, unexpected generation errors,
and descriptor or matcher failures propagate. They do not consume a trial or add
an outcome. Earlier completed work remains available, so the failed trial can be
retried after correcting the error. Logging uses `crisp.slab_search` at INFO level;
callers can also inspect the full `outcomes` list directly.

### Resume and output

`search.save(path)` writes one atomic `crisp-slab-search`, version 1, JSON file
containing search settings, all completed outcomes, and the validated archive
format. Passing `checkpoint=path` to `run` saves after **every completed trial**,
including rejections, so an interrupted run can resume at the next unsaved trial.
A failed save preserves the previous file and keeps completed in-memory work;
retry `save` to persist it. Optimizer steps within a trial are not checkpointed.

Construct a new search with the same empty-archive and search settings, supply
the same calculator factory, and call `load(path)` before continuing. For example,
the executable demonstration provides the same setup in a fresh process:

```sh
# With CRISP, PyXtal 1.1.4 and the documented torch_fplib checkout importable:
python examples/slab_random_search.py --trials 3 --checkpoint cu-search.json
python examples/slab_random_search.py --resume --trials 5 --checkpoint cu-search.json
```

The second command adds five trials, for eight total. The example refuses to
overwrite an existing checkpoint unless `--resume` is selected. In the Python
API, `save` and the `checkpoint` argument replace their explicitly supplied path;
they do not automatically load it.

Search and archive settings, including `calculator_id`, must match on load.
The label is provenance, not proof of identical model weights or parameters;
use the same actual model and dependency versions. Trial seeds derive from the
run seed and trial index using local NumPy `SeedSequence` with 64-bit output.
The driver does not consume global RNG state. With deterministic generation and
calculators, split/resumed runs reproduce outcomes; re-centering/wrapping while
loading can introduce roundoff in saved coordinates/descriptors. A malformed,
inconsistent, incompatible, or backend-failing load leaves both current archive
and progress intact. Bulk and plain archive files are not search checkpoints.

The checkpoint stores structures accessible through `search.archive.entries`.
For use outside Python, export an entry through ASE, for example
`ase.io.write("best-slab.extxyz", search.archive.get_best(1)[0].atoms)` after
checking that the archive is nonempty. A separate `search.archive.save(path)`
exports the plain archive without search progress.

### Verified scope and remaining PRs

With Python 3.12, ASE 3.29.0, PyXtal 1.1.4 and the fingerprint revision above,
the eight-trial Cu4/EMT example (seed 23) retained **7 distinct slabs and 1 duplicate**.
Quenches took `[0, 23, 13, 29, 40, 15, 0, 0]` steps. Trial 5 reduced energy from
1.703644 to 0.778408 eV/atom with 0.817374 Angstrom maximum atomic displacement.
All retained slabs passed validation and preserved their exact sampled cells.
Three trials plus reload/resume for five matched uninterrupted outcomes exactly;
coordinates agreed within `2e-15` Angstrom and fingerprints within `1e-15`.
This is a reproducible mechanics check, not evidence for copper-monolayer stability
or recovery of the ground state of an unknown material.

PR #8 is the first functioning **fixed-cell random search over sampled 2D cells**.
The following small PRs should build on it, preserving bulk defaults:

| PR | Scope | Acceptance milestone |
| --- | --- | --- |
| 9 | Reuse `ExactGP` to screen candidates, retain exploration, and persist training state. | Compare GP screening with PR #8 at equal physical-quench budgets across several seeds. |
| 10 | Add bounded atomic and in-plane cell mutations of archived parents. | Mutations preserve composition, c, physical PBC and bounds; random injection continues. |
| 11 | Add fingerprint-guided atomic movement and final unbiased physical quench. | Real calculator energy alone ranks candidates; forces and slab bounds remain valid. |
| 12 | Relax a/b and in-plane shear while keeping c and out-of-plane tilt fixed. | Verify in-plane derivatives/stress and results under changes in vacuum. |
| 13 | Add target-material recovery benchmarks and documented operating settings. | Recover reference sheets across seeds, converge budgets, validate the energy model and compare with trusted relaxation results. |

After #9 the driver is GP-assisted; after #11 it includes fingerprint-guided
movement; after #12 it can optimize in-plane lattice parameters. Readiness for
scientific predictions requires the material-specific evidence in #13, not only
a completed PR count. HPC execution and more advanced finishers can follow.

Run `python -m unittest discover -s tests -p 'test_slab_search*.py' -v`.
The real end-to-end test needs PyXtal and `torch_fplib`; lightweight control-flow
and harmonic-relaxation tests run without those optional backends.
