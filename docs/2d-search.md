# 2D search foundations

This is the geometry foundation for a future, opt-in 2D search mode. It does
**not** enable 2D searches in `CRISPSearch`. Existing search behavior, imports,
fingerprints, relaxation, archives, and dependencies are unchanged.

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
