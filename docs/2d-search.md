# 2D search foundations

These are opt-in geometry and fingerprint foundations for a future 2D search
mode. They do **not** enable 2D searches in `CRISPSearch`. Existing bulk search
behavior, imports, fingerprint code, relaxation, archives, and dependencies
are unchanged.

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
is intended for later candidate generation; it is not imposed on an existing
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
Those checks and integration into the search pipeline belong to later PRs.

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
changing vacuum, not raw stress. Freezing cell strain is a later relaxation
concern. This descriptor does not establish bonding/connectivity, physical
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
