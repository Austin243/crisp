# Native-VASP CRISP population search for layered structures

`SlabCRISPSearch` adds a 2D specialization of the original `CRISPSearch` without
editing the upstream bulk implementation. It uses the layered candidates from
[PR13](2d-layers.md) and [native VASP backend](2d-native-vasp.md). The earlier
`SlabRandomSearch` remains available with its original behavior and checkpoints.

**Experimental validation status:** controller tests exercise real GP, geometry,
archive and mutation code with synthetic native results. They do not establish
VASP compatibility, target-material recovery or search efficiency. Actual
supported-build native relaxations and a small unseeded reference search remain
required before using this route for scientific results.

## Workflow and supported methods

1. Generate a random population with explicit initial layer compositions,
   thicknesses and gaps. All groups share one in-plane cell.
2. Relax the bootstrap population through native VASP. VASP owns ionic and
   in-plane cell optimization; the perpendicular cell vector stays fixed.
3. Train the original `ExactGP` on **relaxed pooled fingerprints and final DFT
   energy/atom** for unique archived structures. At zero pressure this is the
   original bulk enthalpy target. Rejected results never train the model.
4. Generate more random candidates and displacement/xy-strain mutations of
   low-energy parents, with the original diverse parent-pool policy during
   stagnation. Initial layer membership is provenance; reconstruction is free.
5. Reuse the original confidence filter with coverage safeguards, or the
   original acquisition ranking. Relax selected candidates and update the GP.

The original bootstrap, generation dispatch, screening and diversity methods
are imported/reused. Only the run/checkpoint loop and geometry-dependent hooks
are specialized. Random injection remains present each generation. There is
one fixed composition and atom count per search.

`n_mutants=0` disables mutations. `screening_mode="filter"` is the bulk default;
`"rank"` enables acquisition selection. `SlabMutations` controls bounded
displacement/strain; its `random_every` field belongs to the earlier sequential
driver and is not used here, since this driver injects `n_random` candidates
every generation. The model uses a fixed RBF length scale in this first version.

Fingerprint-guided movement, FP-Jacobian/species-swap operators, finishers,
CAWR, MLIP pretreatment and pressure are not enabled in this specialization.
Unsupported constructor options fail rather than silently invoking bulk
geometry operations. The allowed maximum slab thickness must leave a vacuum
gap greater than the fingerprint cutoff throughout the accepted search space.

## Python use

After creating `spec`, `fingerprint` and `backend` as described in the linked
layer and native-backend docs:

```python
from crisp.slab_layer_archive import LayeredSlabArchive
from crisp.slab_crisp_search import SlabCRISPSearch

archive = LayeredSlabArchive(fingerprint, spec)
search = SlabCRISPSearch(
    archive, backend, seed=23, n_random=4, n_mutants=2,
    max_generations=10, budget_relax=40,
)
search.run(checkpoint="output/my-search.json")
for entry in archive.get_best(5):
    print(entry.metadata["candidate_id"], entry.energy)  # eV/atom
```

`run(n_generations=1, checkpoint=...)` runs one additional generation within
the configured total limits. The physical budget counts completed candidate
attempts, including scientific rejection, and is enforced before each native
call. It does not count individual VASP ionic/electronic iterations; the backend
has separate stage and wall-clock limits. An interrupted physical attempt may
consume compute without producing a completed controller record.

Empty or all-rejected populations are handled. `stop_reason` reports `budget`,
`generations`, `stagnation`, or `paused`; stagnation is a software stopping
criterion, not proof that the global minimum was found. Outcomes record stable
candidate IDs, generations, accepted/duplicate/rejected status and final energy
or rejection reason. Backend calculation metadata remains in the archive.

## Resume and execution failures

Construct the same empty archive/search with the same physical and search
settings, call `search.load(path)`, then `search.run(checkpoint=path)`.
Changing settings requires a new search. Earlier slab-search formats and their
pre-quench-feature training data are not accepted here.

Checkpoints are atomic and cover **complete generations**. They preserve
archive coordinates exactly, final training data, candidate records, generation
progress and deterministic seed derivation. The model is reconstructed from the
same relaxed data and fixed hyperparameters. Python state rolls back to the
last complete generation if execution or configuration fails. A replay produces
the same proposals and stable candidate IDs; the native backend reuses only
completed matching calculations whose file hashes still agree.

Typed scientific rejections consume a candidate attempt and allow the search to
continue. Configuration errors, unavailable executables, changed outputs and
unknown/interrupted native jobs propagate for inspection. The controller does
not assume such jobs are dead or resubmit them automatically. Inspect the native
stage directories before deciding how to recover an interrupted job. This is
not exact mid-ionic-run restart or distributed Slurm recovery.

## Runnable example and checks

`examples/slab_vasp_search.py` is a small six-atom MoS2 placement example with
S/Mo/S initial planes, free reconstruction, explicit potential paths, a VASP
version and an executable command. It is an API demonstration, **not a validated
MoS2 search or recommended convergence recipe**. Inspect and converge the
geometry bounds, k-mesh, cutoff, electronic settings and budgets for your work.

```sh
python examples/slab_vasp_search.py --help
python examples/slab_vasp_search.py \
  --vasp-version 6.5.1 --encut 520 \
  --mo-potcar /path/to/Mo/POTCAR --s-potcar /path/to/S/POTCAR \
  --work-dir output/mos2-pilot --command srun vasp_std
```

The second command performs real calculations when the required resources are
available. Add `--resume` with unchanged settings to continue. This example runs
serial candidate jobs locally or inside one allocation; it does not submit
individual Slurm jobs. Native VASP input limitations are listed in the backend
documentation, including unsupported per-atom magnetic moment and DFT+U arrays.

Controller regression checks:

```sh
python -m unittest discover -s tests -p 'test_slab_crisp_search.py' -v
```

Run the actual native/backend and unseeded reference acceptance checks before
calling this workflow validated. Systematic multi-seed recovery and cost
benchmarks are separate follow-up work.
