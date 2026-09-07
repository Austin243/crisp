# Native VASP relaxation for layered candidates

`crisp.slab_vasp.SlabVASPRelaxer` is an opt-in backend. VASP performs the ionic
steps and permitted cell optimization internally; no ASE optimizer drives it.
The original bulk CRISP and HPC backends are unchanged.

This experimental backend has synthetic-output/parser and local process tests.
**It has not been run against a licensed VASP executable.** Integration into
`feature/2d-search` makes it available for development and validation. The
real-build acceptance checks below remain required before relying on scientific
results. Passing unit tests does not establish that a particular VASP build
relaxes every permitted slab metric correctly.

## Supported geometry and build

Use a verified stock VASP **6.5.0 or newer**. Supply the exact patch version from
the beginning of an existing `OUTCAR` or the installation's build record. Each
result must report the same major/minor/patch version in `OUTCAR` and XML.
Lattice constraints with `IBRION=1/2` appeared in 6.4.3, but VASP documents a
convergence bug in versions older than 6.5.0. See the official
[LATTICE_CONSTRAINTS documentation](https://vasp.at/wiki/LATTICE_CONSTRAINTS).
An unverified patched older build is not accepted by this first implementation.

The final recipe requires `ISIF=3`, `IBRION=1` or `2`,
`LATTICE_CONSTRAINTS=T T F`, zero pressure, and a finite positive ionic-step
budget. The slab lies in xy with perpendicular, fixed c. Atomic xyz coordinates
are free; initial layer membership is provenance and may reconstruct. Cell
lengths and shear in xy can change. The worker checks every reported ionic cell
for fixed c and no out-of-plane tilt, then validates the final layered geometry.
It cannot check or constrain geometry before each internal VASP evaluation.

The actual empty gap is cell height minus unwrapped stack thickness. A fixed c
does not preserve the initial layer or vacuum gaps. Final structures violating
the requested area, thickness, distance, vacuum or 2D connectivity bounds are
rejected, rather than clipped or resized with their old energies retained.
Only output-printing roundoff within 2e-6 Angstrom is canonicalized to the
input's exact c and zero tilt after those quantities are checked.

## Minimal Python use

A `LayeredSlabSpec` and candidates with initial provenance come from the layered
generator described in `2d-layers.md`. Choose the physical INCAR recipe and
potential datasets for the chemistry; the following values illustrate the API,
not a converged transferable DFT prescription.

```python
from crisp.slab_vasp import NativeVASPConfig, SlabVASPRelaxer

config = NativeVASPConfig(
    command=("srun", "vasp_std"),  # argv inside an existing allocation
    version="6.5.0",              # replace with the actual installed version
    potentials={"B": "/licensed/PAW_PBE/B/POTCAR",
                "N": "/licensed/PAW_PBE/N/POTCAR"},
    incar={"ENCUT": 520, "ISMEAR": 0, "SIGMA": 0.05,
           "EDIFF": 1e-6, "NSW": 200},
    kpoints=(6, 6, 1),
    force_tolerance=0.03,          # eV/Angstrom, maximum atom-force norm
    stress_tolerance=0.01,        # eV/Angstrom², c * max(abs(xx, yy, xy))
    timeout_seconds=86400,
)
backend = SlabVASPRelaxer(spec, config, work_dir="native-runs")
result = backend.relax(candidate, candidate_id="generation-0-candidate-0")
print(result.energy_per_atom, result.metadata)
```

`command` is a nonempty argument sequence executed without a shell. It may be a
local executable, an MPI launcher, or `srun` within an existing allocation.
It is **not** an `sbatch` submission command: the process must block until VASP
finishes in the supplied working directory. Automatic scheduler submission and
recovery are separate planned work. A timeout terminates the local process
group; remote launcher cleanup still depends on the launcher's behavior.

Potential files stay local. Each supplied path must contain exactly one dataset
for the mapped element, verified through `VRHFIN` and the dataset terminator.
Atoms and datasets are sorted by chemical symbol for VASP, and output atoms,
forces and provenance are restored to the input order. Only hashes of datasets
enter the portable settings identity. Do not commit licensed potentials or job
directories. The existing repository ignores files named `POTCAR`.

The first backend requires a single explicit Gamma-centered `kx ky 1` mesh.
It disables spatial symmetry (`ISYM=0`), starts stages from new electronic states
(`ISTART=0`, `ICHARG=2`), and requests `NWRITE=2` for convergence evidence.
The defaults include `PREC=Accurate`, `LREAL=F`, `NELM=100`, `EDIFF=1e-6`,
`IBRION=2`, `NSW=200` and force-based `EDIFFG`. You must provide `ENCUT`.
Caller values cannot override the protected geometry/restart controls.

Explicit `MAGMOM`, noncollinear/SOC inputs, DFT+U species vectors, `DIPOL`,
external fields and competing optimizer/NEB controls are rejected in this
first version: these need mapped inputs or additional invariance checks.
This limitation matters for magnetic or correlated target chemistry. It does
not claim a general VASP input wrapper. Scalar physical settings such as XC,
smearing and van der Waals choices remain part of the explicit recipe identity.

Species-ordered `VDW_C6`, `VDW_C6AU`, `VDW_R0`, `VDW_R0AU`, `VDW_ALPHA`,
`ROPT` and `RWIGS` overrides are also rejected until an explicit mapping API is
available. This prevents sorting POSCAR/POTCAR from changing which species receives
a value. See the VASP definitions of [VDW_C6](https://vasp.at/wiki/VDW_C6),
[VDW_R0](https://vasp.at/wiki/VDW_R0), [ROPT](https://vasp.at/wiki/ROPT), and
[RWIGS](https://vasp.at/wiki/RWIGS). Scalar choices such as `IVDW` remain supported.

## Acceptance, energy convention and failures

Collection first requires a complete XML document, a normal OUTCAR footer,
matching input geometry and INCAR echo, and agreement between final XML and
CONTCAR. VASP must report successful electronic convergence at **every** ionic
frame. Every frame needs finite energy, forces and stress, with complete arrays.
Final acceptance additionally requires VASP's ionic convergence message and
independent force and in-plane stress thresholds. An energy-change-only ionic
criterion is not accepted. The relevant VASP definitions are
[EDIFF](https://vasp.at/wiki/EDIFF) and
[EDIFFG](https://vasp.at/wiki/EDIFFG).

Numeric INCAR echoes are compared at their printed precision, including VASP's
eight-decimal real output; rounding tolerance requires at least eight decimal
places or significant digits. Integer controls and logical values remain exact.
Input hashes still require the actual prepared files to remain unchanged.
Construct a new backend to change its configuration; replacing settings or
mutating nested options on an existing backend is rejected before execution or
cache reuse. This also protects the recorded identity of its acceptance criteria.

ASE parses native XML and converts VASP kbar stress to ASE eV/Angstrom³ with the
ASE sign convention. The acceptance magnitude uses xx, yy and xy times fixed
cell height; zz is not a relaxation target. The ranked energy is ASE's corrected
zero-smearing VASP XML energy divided by atom count, consistently across the
final recipe. It is not the finite-temperature free energy. The caller must
choose and converge electronic settings appropriate to the intended comparison.

Z is unwrapped in original-atom order from the input through every XML ionic
frame using the nearest periodic image. Each inferred displacement must be
strictly below `max_z_step` (default 2 Angstrom), itself less than c/2. This is a
continuity assumption: periodic output cannot reveal an entire unseen cell
crossing between adjacent frames. Excessive or half-cell-ambiguous steps are
rejected; the worker does not infer a new ordering of reconstructed layers.
Avoid shifting the final coordinates or changing the cell after a calculation
if that would change the physical problem, for example with an external field.

`VASPValidationError` denotes a scientifically rejected completed result.
`VASPExecutionError` denotes missing/mismatched files, execution or infrastructure
requiring attention. Invalid setup raises `ValueError` before execution.
Scientific rejections must not be used as energy observations in the search GP.

## Isolated stages and restart

Each candidate uses `work_dir/<candidate-id>/stage-00/` and further numbered
stage directories. A manifest records inputs, ordering, settings, potential
hashes and completion status. Existing completed or rejected stages are
reparsed only if their full inputs and output hashes still agree. A changed
recipe or geometry requires another calculation and a new candidate ID or work
directory. No output is silently adopted from an unmanaged directory.

An interrupted or failed stage is not automatically relaunched: first inspect
its process/job and files. After confirming it is stopped, move its stage
directory aside before intentionally rerunning with the same ID; keep the old
directory as a record. Completed earlier stages can still be reused. This is
stage-boundary persistence, not generic mid-ionic restart or scheduler recovery.

Optional coarse-to-fine stages are an ordered tuple, followed by the config's
final recipe. For example:

```python
from dataclasses import replace
from crisp.slab_vasp import NativeVASPStage

config = replace(config, preliminary_stages=(
    NativeVASPStage(incar={"ISIF": 2, "NSW": 40},
                    kpoints=(3, 3, 1), allow_unconverged=True),
))
```

A preliminary stage may use `ISIF=2` and explicitly continue after ionic budget
exhaustion. Electronic failures, invalid cell motion and invalid geometric
bounds never continue. Intermediate connectivity need not already be the final
sheet topology when continuation is permitted. The last stage always requires
full acceptance; only its final energy is returned for ranking. Electronic
restart files are not carried between recipes.

## Real-build acceptance still required

Before treating this backend as scientifically validated:

1. Relax a known flat sheet and a finite-thickness sheet with the actual
   supported executable, explicit potentials and a converged final recipe.
2. Exercise an oblique/sheared xy cell; inspect the full trajectory for fixed c
   and no out-of-plane tilt while a, b and xy shear respond.
3. Confirm final force/stress thresholds and compare the collected geometry and
   energy with an independently prepared run using the same final settings.
4. Check a reconstruction/wrapped-z case, an intentionally unconverged stage,
   and replay of a completed stage without another execution.
5. Record VASP version/build, hardware/launcher, settings and results without
   distributing executable or potential files.

The synthetic tests exercise the real ASE parser and worker orchestration but
are not evidence of these scientific acceptance results.
