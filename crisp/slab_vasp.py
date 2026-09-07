"""Native VASP slab relaxation, additive to CRISP's existing backends.

VASP performs the ionic and in-plane cell steps. This module owns file identity,
execution and result validation; it never runs an ASE optimizer.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import re
import signal
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.calculator import PropertyNotImplementedError
from ase.io import read, write

from .slab_layers import LayeredSlabSpec, validate_layered_slab


class NativeVASPError(RuntimeError):
    """Base class for a failed native relaxation."""


class VASPValidationError(NativeVASPError):
    """A completed physical calculation failed scientific acceptance."""


class VASPExecutionError(NativeVASPError):
    """Execution, file identity or infrastructure needs attention."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      default=lambda x: x.tolist() if isinstance(x, np.ndarray)
                      else x.item() if isinstance(x, np.generic) else str(x))


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(_json(value) + "\n")
    temporary.replace(path)


def _version(value):
    match = re.fullmatch(r"(?:vasp\.)?(\d+)\.(\d+)\.(\d+)(?:[._-][A-Za-z0-9.-]+)?", value.strip())
    if not match or tuple(map(int, match.groups())) < (6, 5, 0):
        raise ValueError("Native slab relaxation requires an explicit VASP version >= 6.5.0")
    return tuple(map(int, match.groups()))


def _positive(value, name):
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _incar_value(value):
    if isinstance(value, (bool, np.bool_)):
        return ".TRUE." if value else ".FALSE."
    if isinstance(value, (tuple, list)):
        return " ".join(_incar_value(item) for item in value)
    text = str(value)
    if not text or any(char in text for char in "\n\r;#!=") or not re.fullmatch(r"[A-Za-z0-9_+.* /-]+", text):
        raise ValueError("INCAR values must be finite literals without comments or extra assignments")
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError("INCAR values must be finite")
    return text


def _mesh(value):
    if len(value) != 3 or any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0 for x in value) or value[2] != 1:
        raise ValueError("kpoints must be three positive integers with kz=1")
    return tuple(map(int, value))


@dataclass(frozen=True)
class NativeVASPStage:
    """Optional preliminary native stage; the config recipe is always last.

    Overrides may set ISIF=2 for a fixed-cell preliminary stage. Budget
    exhaustion can continue only when allow_unconverged is explicitly true;
    electronic failure and invalid geometry never continue.
    """

    incar: dict = field(default_factory=dict)
    kpoints: tuple[int, int, int] | None = None
    allow_unconverged: bool = False


@dataclass(frozen=True)
class NativeVASPConfig:
    """Explicit executable, potentials and final physical recipe.

    command is argv, e.g. ("srun", "vasp_std"), not shell text. Potential paths
    map each element to exactly one licensed POTCAR dataset. force_tolerance is
    eV/Angstrom; stress_tolerance is max(abs(xx, yy, xy))*c in eV/Angstrom².
    Other physical inputs, including ENCUT, must be chosen for the chemistry.
    """

    command: tuple[str, ...]
    version: str
    potentials: dict[str, str]
    incar: dict
    kpoints: tuple[int, int, int] = (3, 3, 1)
    force_tolerance: float = 0.03
    stress_tolerance: float = 0.01
    timeout_seconds: float = 86400
    preliminary_stages: tuple[NativeVASPStage, ...] = ()
    max_z_step: float = 2.0


@dataclass
class NativeVASPResult:
    atoms: Atoms
    energy_per_atom: float
    metadata: dict


class SlabVASPRelaxer:
    """Run isolated candidate stages locally or within an existing allocation.

    Completed stages are reusable only with identical inputs and unchanged
    output hashes. Interrupted/failed stages require explicit operator review;
    this backend does not infer that an old process or scheduler job is dead.
    """

    def __init__(self, spec: LayeredSlabSpec, config: NativeVASPConfig, work_dir):
        self.spec = spec
        self.config = deepcopy(config)
        self.work_dir = Path(work_dir).expanduser().resolve()
        _version(config.version)
        if isinstance(config.command, str) or not config.command or any(not isinstance(x, str) or not x or "\x00" in x for x in config.command):
            raise ValueError("command must be nonempty argv, not a shell command string")
        for name in ("force_tolerance", "stress_tolerance", "timeout_seconds", "max_z_step"):
            _positive(getattr(config, name), name)
        if config.max_z_step >= spec.config.cell_height / 2:
            raise ValueError("max_z_step must be less than half the fixed cell height")
        if set(config.potentials) != set(spec.composition):
            raise ValueError("potentials must provide exactly the search's chemical species")
        self._potentials = {}
        for symbol, source in config.potentials.items():
            source = Path(source).expanduser().resolve()
            data = source.read_bytes()
            text = data.decode("utf-8")
            species = re.findall(r"VRHFIN\s*=\s*([A-Z][a-z]?)\s*:", text)
            if species != [symbol] or len(re.findall(r"End of Dataset", text)) != 1:
                raise ValueError(f"Potential for {symbol} must contain exactly its own POTCAR dataset")
            self._potentials[symbol] = (source, _hash(data))
        self._stages = []
        for stage in config.preliminary_stages:
            if not isinstance(stage, NativeVASPStage) or not isinstance(stage.allow_unconverged, bool):
                raise ValueError("preliminary_stages must contain NativeVASPStage values")
            self._stages.append((self._recipe(config.incar | stage.incar, final=False),
                                 _mesh(stage.kpoints or config.kpoints), stage.allow_unconverged))
        self._stages.append((self._recipe(config.incar, final=True), _mesh(config.kpoints), False))
        self._settings = {
            "backend": "native-vasp-slab-v1", "spec": spec.to_dict(),
            "version": config.version, "command": list(config.command),
            "potentials": {s: h for s, (_, h) in sorted(self._potentials.items())},
            "stages": [{"incar": r, "kpoints": list(k), "allow_unconverged": a}
                       for r, k, a in self._stages],
            "force_tolerance": config.force_tolerance,
            "stress_tolerance": config.stress_tolerance, "max_z_step": config.max_z_step,
            "energy": "ASE VASP XML zero-smearing corrected energy per atom",
        }
        self._settings = json.loads(_json(self._settings))

    @property
    def settings(self):
        """A detached serializable identity, excluding paths and wall-clock limits."""
        return deepcopy(self._settings)

    def _recipe(self, supplied, *, final):
        supplied = {str(k).upper(): v for k, v in supplied.items()}
        if any(not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) for key in supplied):
            raise ValueError("Invalid INCAR tag")
        if "ENCUT" not in supplied:
            raise ValueError("Provide an explicitly chosen ENCUT in eV")
        _positive(supplied["ENCUT"], "ENCUT")
        forbidden = {"KSPACING", "KGAMMA", "IMAGES", "SPRING", "LCLIMB", "IOPT",
                     "ICONST", "MDALGO", "LCHAIN", "EFIELD", "EFIELD_PEAD", "DIPOL",
                     "MAGMOM", "M_CONSTR", "LNONCOLLINEAR", "LSORBIT", "LAMBDA",
                     "LDAU", "LDAUTYPE", "LDAUL", "LDAUU", "LDAUJ"}
        # Per-atom/species vectors need an explicit mapping API before support.
        if forbidden.intersection(supplied):
            raise ValueError(f"Unsupported mapped or competing INCAR tags: {sorted(forbidden.intersection(supplied))}")
        recipe = {"PREC": "Accurate", "LREAL": False, "ISYM": 0, "IBRION": 2,
                  "ISIF": 3, "NSW": 200, "NELM": 100, "EDIFF": 1e-6,
                  "EDIFFG": -self.config.force_tolerance, "PSTRESS": 0.0,
                  "LATTICE_CONSTRAINTS": [True, True, False], "ISTART": 0,
                  "ICHARG": 2, "NWRITE": 2, "LWAVE": False, "LCHARG": False,
                  **supplied}
        if recipe["IBRION"] not in (1, 2) or recipe["ISIF"] not in ((3,) if final else (2, 3)):
            raise ValueError("Native relaxation requires IBRION=1/2 and final ISIF=3")
        fixed = {"PSTRESS": 0, "ISYM": 0, "ISTART": 0, "ICHARG": 2, "NWRITE": 2}
        if any(recipe[k] != v for k, v in fixed.items()):
            raise ValueError("PSTRESS=0, ISYM=0, ISTART=0, ICHARG=2 and NWRITE=2 are required")
        if not isinstance(recipe["LATTICE_CONSTRAINTS"], (tuple, list)) or list(recipe["LATTICE_CONSTRAINTS"]) != [True, True, False]:
            raise ValueError("LATTICE_CONSTRAINTS must be [True, True, False]")
        for name in ("NSW", "NELM"):
            if isinstance(recipe[name], bool) or not isinstance(recipe[name], Integral) or recipe[name] < 1:
                raise ValueError(f"{name} must be a positive integer")
        _positive(recipe["EDIFF"], "EDIFF")
        if not np.isfinite(recipe["EDIFFG"]) or recipe["EDIFFG"] >= 0:
            raise ValueError("EDIFFG must be negative for native force convergence")
        if final and abs(recipe["EDIFFG"]) > self.config.force_tolerance:
            raise ValueError("Final EDIFFG must be at least as strict as force_tolerance")
        for value in recipe.values():
            _incar_value(value)
        return json.loads(_json(recipe))

    def relax(self, atoms, candidate_id):
        """Return a validated result; preserve input order, arrays and provenance."""
        if not isinstance(candidate_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", candidate_id):
            raise ValueError("candidate_id must be a simple nonempty directory name")
        if atoms.constraints:
            raise ValueError("Native layered searches require unconstrained atomic xyz coordinates")
        validate_layered_slab(atoms, self.spec, check_connectivity=False)
        if json.loads(_json(self.spec.to_dict())) != self._settings["spec"]:
            raise ValueError("Layer specification changed after backend construction")
        current = atoms.copy()
        records = []
        for index, (recipe, mesh, allow_unconverged) in enumerate(self._stages):
            result = self._stage(current, candidate_id, index, recipe, mesh, allow_unconverged)
            records.append(result.metadata)
            current = result.atoms.copy()
        result.metadata = {"candidate_id": candidate_id,
                           "settings_hash": _hash(_json(self.settings).encode()),
                           "stages": records, "converged": True}
        return result

    def _stage(self, atoms, candidate_id, index, recipe, mesh, allow_unconverged):
        directory = self.work_dir / candidate_id / f"stage-{index:02d}"
        order = np.argsort(atoms.get_chemical_symbols(), kind="stable")
        sorted_atoms = atoms[order]
        sorted_atoms.pbc = True
        identity = {"settings": self.settings, "stage": index,
                    "atoms": atoms.todict(), "order": order.tolist()}
        identity = json.loads(_json(identity))
        manifest_path = directory / "manifest.json"
        if directory.exists():
            if not manifest_path.is_file():
                raise VASPExecutionError(f"Unmanaged existing stage directory: {directory}")
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, ValueError) as exc:
                raise VASPExecutionError("Unreadable stage manifest") from exc
            if manifest.get("identity") != identity:
                raise VASPExecutionError("Existing stage inputs/settings differ; use a new candidate ID or work directory")
            if manifest.get("status") not in ("completed", "rejected"):
                raise VASPExecutionError(f"Stage is {manifest.get('status')}; inspect its process/files before retrying: {directory}")
            self._verify_hashes(directory, manifest["input_hashes"])
            self._verify_hashes(directory, manifest["output_hashes"])
        else:
            directory.mkdir(parents=True)
            write(directory / "POSCAR", sorted_atoms, format="vasp", direct=True,
                  sort=False, vasp5=True, ignore_constraints=False)
            symbols = sorted(self.spec.composition)
            with (directory / "POTCAR").open("wb") as stream:
                for symbol in symbols:
                    source, expected = self._potentials[symbol]
                    data = source.read_bytes()
                    if _hash(data) != expected:
                        raise VASPExecutionError(f"POTCAR source changed for {symbol}")
                    stream.write(data)
                    if not data.endswith(b"\n"):
                        stream.write(b"\n")
            (directory / "INCAR").write_text("".join(f"{key} = {_incar_value(value)}\n" for key, value in sorted(recipe.items())))
            (directory / "KPOINTS").write_text("CRISP 2D\n0\nGamma\n" + " ".join(map(str, mesh)) + "\n0 0 0\n")
            manifest = {"identity": identity, "status": "running",
                        "input_hashes": self._hashes(directory, ("POSCAR", "POTCAR", "INCAR", "KPOINTS"))}
            _write_json(manifest_path, manifest)
            try:
                self._execute(directory)
                self._verify_hashes(directory, manifest["input_hashes"])
                manifest["output_hashes"] = self._hashes(directory, ("OUTCAR", "CONTCAR", "vasprun.xml"))
            except (OSError, VASPExecutionError) as exc:
                manifest["status"] = "failed"
                _write_json(manifest_path, manifest)
                raise VASPExecutionError(f"Native stage failed in {directory}: {exc}") from exc
        try:
            result = self._collect(directory, atoms, order, recipe, mesh, allow_unconverged)
        except VASPValidationError:
            manifest["status"] = "rejected"
            _write_json(manifest_path, manifest)
            raise
        manifest["status"] = "completed"
        _write_json(manifest_path, manifest)
        result.metadata.update(stage=index, directory=str(directory),
                               output_hashes=manifest["output_hashes"])
        return result

    @staticmethod
    def _hashes(directory, names):
        return {name: _hash((directory / name).read_bytes()) for name in names}

    def _verify_hashes(self, directory, expected):
        try:
            if self._hashes(directory, expected) != expected:
                raise VASPExecutionError("Stage files changed after execution")
        except OSError as exc:
            raise VASPExecutionError("Stage files are missing") from exc

    def _execute(self, directory):
        with (directory / "vasp.stdout").open("wb") as stdout, (directory / "vasp.stderr").open("wb") as stderr:
            process = subprocess.Popen(self.config.command, cwd=directory, stdout=stdout,
                                       stderr=stderr, start_new_session=True)
            try:
                returncode = process.wait(timeout=self.config.timeout_seconds)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise VASPExecutionError("Native command interrupted or exceeded timeout")
        if returncode:
            raise VASPExecutionError(f"Native command exited with status {returncode}")

    def _collect(self, directory, initial, order, recipe, mesh, allow_unconverged):
        """Strictly parse a complete XML document before asking ASE for frames."""
        try:
            root = ET.parse(directory / "vasprun.xml").getroot()
            text = (directory / "OUTCAR").read_text(errors="replace")
            if "General timing and accounting informations for this job" not in text:
                raise VASPValidationError("OUTCAR has no completed-run footer")
            banner = re.search(r"vasp\.(\d+\.\d+\.\d+)", text, re.IGNORECASE)
            xml_version = root.findtext("generator/i[@name='version']", "").strip()
            try:
                actual_version = _version(banner.group(1) if banner else "")
                xml_build = _version(xml_version)
            except ValueError as exc:
                raise VASPExecutionError("Output does not identify a supported VASP build") from exc
            if actual_version != _version(self.config.version) or xml_build != actual_version:
                raise VASPExecutionError("Output version differs from the declared supported VASP build")
            generation = root.find("kpoints/generation")
            if (generation is None or generation.attrib.get("param", "").lower() != "gamma"
                    or not self._same_value(generation.findtext("v[@name='divisions']", ""), mesh)
                    or not self._same_value(generation.findtext("v[@name='usershift']", ""), (0, 0, 0))):
                raise VASPExecutionError("Output k-point mesh differs from prepared Gamma grid")
            # VASP's explicit SCF loop-exit reason is stronger than merely
            # seeing fewer NELM steps or a small total-energy difference.
            reasons = [line for line in text.splitlines() if "aborting loop" in line]
            calculations = root.findall("calculation")
            if not calculations or len(reasons) != len(calculations) or any("because EDIFF is reached" not in line for line in reasons):
                raise VASPValidationError("Missing or unsuccessful electronic convergence for an ionic frame")
            for key, value in recipe.items():
                node = root.find(f"incar/*[@name='{key}']")
                if node is None or not self._same_value(node.text or "", value):
                    raise VASPExecutionError(f"Output INCAR echo differs from prepared {key}")
            frames = read(directory / "vasprun.xml", index=":", format="vasp-xml")
            if len(frames) != len(calculations) or len(frames) > recipe["NSW"]:
                raise VASPValidationError("Invalid or excessive ionic frame count")
            expected = initial[order]
            initial_xml = root.find("structure[@name='initialpos']")
            if initial_xml is None:
                raise VASPValidationError("Missing XML initial geometry")
            first_cell = self._vectors(initial_xml, "crystal/varray[@name='basis']/v")
            first_frac = self._vectors(initial_xml, "varray[@name='positions']/v")
            if not np.allclose(first_cell, expected.cell.array, atol=2e-6, rtol=0) or not self._same_fractional(first_frac, expected.get_scaled_positions(wrap=False)):
                raise VASPExecutionError("XML initial geometry does not match POSCAR input")
            previous_z = expected.positions[:, 2].copy()
            for frame, calculation in zip(frames, calculations):
                if frame.get_chemical_symbols() != expected.get_chemical_symbols():
                    raise VASPValidationError("Output composition/species order changed")
                cell = frame.cell.array
                if (not np.isfinite(cell).all() or not np.allclose(cell[2], expected.cell[2], atol=2e-6, rtol=0)
                        or np.any(np.abs(cell[:2, 2]) > 2e-6)):
                    raise VASPValidationError("Perpendicular cell vector or out-of-plane tilt changed")
                if recipe["ISIF"] == 2 and not np.allclose(cell, expected.cell, atol=2e-6, rtol=0):
                    raise VASPValidationError("Fixed-cell preliminary stage changed the cell")
                height = self.spec.config.cell_height
                raw = frame.positions[:, 2]
                step = (raw - previous_z + height / 2) % height - height / 2
                if not np.isfinite(step).all() or np.any(np.abs(step) >= self.config.max_z_step):
                    raise VASPValidationError("Ambiguous or excessive z displacement between native frames")
                previous_z += step
                if frame.constraints:
                    raise VASPExecutionError("Output unexpectedly contains atomic constraints")
                forces, stress, energy = frame.get_forces(apply_constraint=False), frame.get_stress(), frame.get_potential_energy()
                if forces.shape != (len(initial), 3) or stress.shape != (6,) or not np.isfinite(forces).all() or not np.isfinite(stress).all() or not np.isfinite(energy):
                    raise VASPValidationError("Nonfinite or incomplete energy, forces or stress")
                # XML vector counts are checked independently: some ASE readers
                # silently pad missing force/position/stress rows with zeros.
                for path, shape in (("varray[@name='forces']/v", (len(initial), 3)),
                                    ("varray[@name='stress']/v", (3, 3)),
                                    ("structure/varray[@name='positions']/v", (len(initial), 3))):
                    if self._vectors(calculation, path).shape != shape:
                        raise VASPValidationError("Incomplete XML array")
            contcar = read(directory / "CONTCAR", format="vasp")
            if (contcar.get_chemical_symbols() != frames[-1].get_chemical_symbols()
                    or not np.allclose(contcar.cell, frames[-1].cell, atol=2e-6, rtol=0)
                    or not self._same_fractional(contcar.get_scaled_positions(wrap=False), frames[-1].get_scaled_positions(wrap=False))):
                raise VASPValidationError("CONTCAR does not match the final evaluated geometry")
            final = initial.copy()
            final.info = deepcopy(initial.info)
            inverse = np.argsort(order)
            final.set_cell(frames[-1].cell, scale_atoms=False)
            final.positions = frames[-1].positions[inverse]
            final.positions[:, 2] = previous_z[inverse]
            final.pbc = (True, True, False)
            # Correct only XML's cell-printing roundoff; never resize vacuum.
            final.cell[2] = initial.cell[2]
            final.cell[:2, 2] = 0.0
            validate_layered_slab(final, self.spec, check_connectivity=not allow_unconverged)
            fmax = float(np.linalg.norm(forces, axis=1).max())
            stress_max = float(self.spec.config.cell_height * np.abs(stress[[0, 1, 5]]).max())
            ionic = "reached required accuracy - stopping structural energy minimisation" in text
            stress_ok = recipe["ISIF"] == 2 or stress_max < self.config.stress_tolerance
            converged = ionic and fmax < self.config.force_tolerance and stress_ok
            if not converged and not allow_unconverged:
                raise VASPValidationError(f"Native relaxation not converged: fmax={fmax:g}, in-plane stress={stress_max:g}")
            if not converged and not ionic and len(frames) < recipe["NSW"]:
                raise VASPValidationError("Intermediate stage stopped before its declared budget")
            final.calc = SinglePointCalculator(final, energy=energy, forces=forces[inverse], stress=stress)
            return NativeVASPResult(final, float(energy / len(final)),
                                    {"converged": converged, "ionic_steps": len(frames),
                                     "fmax": fmax, "stress_max": stress_max,
                                     "version": banner.group(1)})
        except (VASPExecutionError, VASPValidationError):
            raise
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, ET.ParseError, OSError, PropertyNotImplementedError) as exc:
            raise VASPValidationError(f"Malformed or invalid native result: {exc}") from exc

    @staticmethod
    def _vectors(node, path):
        return np.array([[float(x) for x in row.text.split()] for row in node.findall(path)])

    @staticmethod
    def _same_fractional(left, right):
        if left.shape != right.shape or not np.isfinite(left).all():
            return False
        delta = left - right
        return bool(np.all(np.abs(delta - np.rint(delta)) < 2e-6))

    @staticmethod
    def _same_value(text, expected):
        def tokens(value):
            return [part.upper().strip(".") for part in str(value).split()]
        left, right = tokens(text), tokens(_incar_value(expected))
        if left == right:
            return True
        logical = {"T": "TRUE", "F": "FALSE"}
        if [logical.get(x, x) for x in left] == [logical.get(x, x) for x in right]:
            return True
        try:
            return len(left) == len(right) and bool(np.allclose([float(x.replace("D", "E")) for x in left],
                                                               [float(x.replace("D", "E")) for x in right], rtol=1e-9, atol=1e-12))
        except ValueError:
            return False
