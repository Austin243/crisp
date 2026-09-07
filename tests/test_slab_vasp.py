"""Native worker contract tests using synthetic output, never licensed VASP.

The XML is deliberately small but passes ASE's real VASP XML reader. These
checks do not substitute for the documented supported-build acceptance runs.
"""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write
from ase.units import GPa

from crisp.slab import SlabConfig
from crisp.slab_layers import InitialLayer, LayeredSlabSpec
from crisp.slab_vasp import (NativeVASPConfig, NativeVASPStage, SlabVASPRelaxer,
                             VASPExecutionError, VASPValidationError)


@pytest.fixture
def native(tmp_path):
    config = SlabConfig((2., 5.), 0.2, 2., 16., 8.)
    spec = LayeredSlabSpec((InitialLayer({"B": 1, "N": 1}, initial_thickness=.1),), config)
    potentials = {}
    for symbol in spec.composition:
        path = tmp_path / f"{symbol}.POTCAR"
        path.write_text(f"Synthetic test fixture, not a potential\nVRHFIN ={symbol}: test\nEnd of Dataset\n")
        potentials[symbol] = str(path)
    settings = NativeVASPConfig(("vasp_std",), "6.5.0", potentials,
                                 {"ENCUT": 520, "NSW": 2, "ISMEAR": 0, "SIGMA": .05})
    relaxer = SlabVASPRelaxer(spec, settings, tmp_path / "runs")
    a = 2.5
    cell = [[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 16.]]
    atoms = Atoms("BN", scaled_positions=[[0, 0, .5], [2/3, 1/3, .5]], cell=cell, pbc=(True, True, False))
    atoms.new_array("initial_atom_id", np.array([0, 1]))
    atoms.new_array("initial_layer", np.array([0, 0]))
    atoms.info["source"] = {"seed": 12}
    return relaxer, atoms[[1, 0]]


def _vectors(parent, name, values):
    block = ET.SubElement(parent, "varray", name=name)
    for row in values:
        ET.SubElement(block, "v").text = " ".join(str(x) for x in row)


def _structure(parent, atoms, name=None):
    block = ET.SubElement(parent, "structure", **({} if name is None else {"name": name}))
    crystal = ET.SubElement(block, "crystal")
    _vectors(crystal, "basis", atoms.cell.array)
    _vectors(block, "positions", atoms.get_scaled_positions(wrap=False))
    return block


def _echo(parent, key, value):
    values = value if isinstance(value, (tuple, list)) else [value]
    kind = ("logical" if isinstance(values[0], bool) else "int" if isinstance(values[0], int)
            else "float" if isinstance(values[0], float) else "string")
    node = ET.SubElement(parent, "v" if isinstance(value, (tuple, list)) else "i", name=key, type=kind)
    node.text = " ".join(("T" if x else "F") if isinstance(x, bool) else str(x) for x in values)


def output(relaxer, directory, *, final=None, count=1, force=.001, stress=.001,
           ionic=True, scf=True, version="6.5.0", mutate_xml=None, contcar=None):
    """Write a complete synthetic native output using actual stage inputs."""
    initial = read(directory / "POSCAR", format="vasp")
    final = initial.copy() if final is None else final.copy()
    stage = int(directory.name.split("-")[-1])
    recipe = relaxer._stages[stage][0]
    root = ET.Element("modeling")
    generator = ET.SubElement(root, "generator")
    ET.SubElement(generator, "i", name="version", type="string").text = version
    for section in ("incar", "parameters"):
        node = ET.SubElement(root, section)
        for key, value in recipe.items():
            _echo(node, key, value)
    kpoints = ET.SubElement(root, "kpoints")
    generation = ET.SubElement(kpoints, "generation", param="Gamma")
    _echo(generation, "divisions", list(relaxer._stages[stage][1]))
    _echo(generation, "usershift", [0., 0., 0.])
    _vectors(kpoints, "kpointlist", [[0, 0, 0]])
    _vectors(kpoints, "weights", [[1]])
    atominfo = ET.SubElement(root, "atominfo")
    rows = ET.SubElement(ET.SubElement(atominfo, "array", name="atoms"), "set")
    for symbol in initial.get_chemical_symbols():
        row = ET.SubElement(rows, "rc")
        ET.SubElement(row, "c").text = symbol
        ET.SubElement(row, "c").text = "1"
    _structure(root, initial, "initialpos")
    for index in range(count):
        frame = final.copy()
        if count > 1:
            fraction = (index + 1) / count
            frame.set_cell(initial.cell.array + fraction * (final.cell.array - initial.cell.array))
            frame.positions = initial.positions + fraction * (final.positions - initial.positions)
        calc = ET.SubElement(root, "calculation")
        for _ in range(2):
            energy = ET.SubElement(ET.SubElement(calc, "scstep"), "energy")
            ET.SubElement(energy, "i", name="e_fr_energy").text = "-10.0"
            ET.SubElement(energy, "i", name="e_0_energy").text = "-9.99"
        energy = ET.SubElement(calc, "energy")
        ET.SubElement(energy, "i", name="e_fr_energy").text = "-10.0"
        ET.SubElement(energy, "i", name="e_0_energy").text = "-9.99"
        _structure(calc, frame)
        _vectors(calc, "forces", [[force, 0, 0]] * len(final))
        # XML stress is VASP kbar, opposite ASE sign, independently exercised.
        _vectors(calc, "stress", np.eye(3) * stress)
    if mutate_xml:
        mutate_xml(root)
    ET.ElementTree(root).write(directory / "vasprun.xml", encoding="unicode")
    reason = "because EDIFF is reached" if scf else "EDIFF was not reached (unconverged)"
    text = f"vasp.{version} synthetic fixture\n" + f"aborting loop {reason}\n" * count
    if ionic:
        text += "reached required accuracy - stopping structural energy minimisation\n"
    text += "General timing and accounting informations for this job:\n"
    (directory / "OUTCAR").write_text(text)
    write(directory / "CONTCAR", final if contcar is None else contcar, format="vasp", direct=True)


def test_real_ase_parser_restores_species_provenance_and_units(native):
    relaxer, atoms = native
    original = atoms.copy()
    with patch.object(relaxer, "_execute", side_effect=lambda path: output(relaxer, path)) as execute:
        result = relaxer.relax(atoms, "candidate-0")
        reused = relaxer.relax(atoms, "candidate-0")
    assert execute.call_count == 1
    np.testing.assert_equal(result.atoms.numbers, original.numbers)
    np.testing.assert_equal(result.atoms.arrays["initial_atom_id"], [1, 0])
    np.testing.assert_equal(result.atoms.arrays["initial_layer"], [0, 0])
    np.testing.assert_allclose(result.atoms.positions, original.positions)
    np.testing.assert_equal(result.atoms.pbc, [True, True, False])
    assert result.atoms.info == original.info
    assert result.energy_per_atom == pytest.approx(-4.995)
    assert reused.energy_per_atom == result.energy_per_atom
    assert result.metadata["stages"][0]["stress_max"] == pytest.approx(.001 * .1 * GPa * 16)
    np.testing.assert_allclose(result.atoms.get_stress()[:3], -.001 * .1 * GPa)
    np.testing.assert_allclose(atoms.positions, original.positions)
    assert atoms.calc is None
    stage = relaxer.work_dir / "candidate-0" / "stage-00"
    assert "LATTICE_CONSTRAINTS = .TRUE. .TRUE. .FALSE." in (stage / "INCAR").read_text()
    assert read(stage / "POSCAR").get_chemical_symbols() == ["B", "N"]
    potcar = (stage / "POTCAR").read_text()
    assert potcar.index("=B:") < potcar.index("=N:")


def test_settings_ownership_and_no_work_path_identity(native, tmp_path):
    relaxer, _ = native
    settings = relaxer.settings
    settings["stages"][0]["incar"]["ENCUT"] = 1
    assert relaxer.settings["stages"][0]["incar"]["ENCUT"] == 520
    other = SlabVASPRelaxer(relaxer.spec, replace(relaxer.config, timeout_seconds=1), tmp_path / "other")
    assert other.settings == relaxer.settings


@pytest.mark.parametrize("change", [
    {"version": "6.4.3"}, {"version": "6"}, {"command": "srun vasp_std"},
    {"command": ()}, {"kpoints": (3, 3, 2)}, {"force_tolerance": 0},
    {"stress_tolerance": float("nan")}, {"timeout_seconds": -1}, {"max_z_step": 8},
    {"incar": {}}, {"incar": {"ENCUT": 520, "ISIF": 4}},
    {"incar": {"ENCUT": 520, "IBRION": 0}}, {"incar": {"ENCUT": 520, "PSTRESS": 1}},
    {"incar": {"ENCUT": 520, "EDIFFG": .01}}, {"incar": {"ENCUT": 520, "EDIFFG": -.5}},
    {"incar": {"ENCUT": 520, "LATTICE_CONSTRAINTS": [True, True, True]}},
    {"incar": {"ENCUT": 520, "MAGMOM": [1, 2]}},
    {"incar": {"ENCUT": "520; ISIF=3"}}, {"incar": {"ENCUT": 520, "GGA": "PE\nISIF=4"}},
])
def test_invalid_configuration_fails_before_execution(native, change):
    relaxer, _ = native
    with pytest.raises((ValueError, TypeError)):
        SlabVASPRelaxer(relaxer.spec, replace(relaxer.config, **change), relaxer.work_dir)


@pytest.mark.parametrize("kwargs", [
    {"force": .03}, {"stress": 100}, {"scf": False}, {"ionic": False},
    {"count": 3}, {"force": float("nan")},
])
def test_unconverged_physical_results_are_durable_rejections(native, kwargs):
    relaxer, atoms = native
    with patch.object(relaxer, "_execute", side_effect=lambda path: output(relaxer, path, **kwargs)) as execute:
        for _ in range(2):
            with pytest.raises(VASPValidationError):
                relaxer.relax(atoms, "rejected")
    assert execute.call_count == 1


@pytest.mark.parametrize("damage", ["footer", "truncated", "force_row", "wrong_species", "wrong_initial", "echo", "version", "contcar", "kpoints", "constraints", "missing_stress"])
def test_stale_incomplete_or_mismatched_outputs_rejected(native, damage):
    relaxer, atoms = native
    def run(path):
        def mutate(root):
            if damage == "force_row":
                block = root.find("calculation/varray[@name='forces']")
                block.remove(block[-1])
            elif damage == "wrong_species":
                root.find("atominfo/array/set/rc/c").text = "C"
            elif damage == "wrong_initial":
                root.find("structure/varray/v").text = ".2 .2 .2"
            elif damage == "missing_stress":
                root.find("calculation").remove(root.find("calculation/varray[@name='stress']"))
            elif damage == "constraints":
                block = ET.SubElement(root.find("structure[@name='initialpos']"), "varray", name="selective")
                for _ in atoms:
                    ET.SubElement(block, "v").text = "F F F"
            elif damage == "kpoints":
                root.find("kpoints/generation/v[@name='divisions']").text = "9 9 1"
            elif damage == "echo":
                root.find("incar/i[@name='ENCUT']").text = "400"
        output(relaxer, path, version="6.5.1" if damage == "version" else "6.5.0", mutate_xml=mutate)
        if damage == "footer":
            (path / "OUTCAR").write_text("vasp.6.5.0\n")
        if damage == "truncated":
            xml = path / "vasprun.xml"
            xml.write_text(xml.read_text()[:-20])
        if damage == "contcar":
            changed = read(path / "CONTCAR")
            changed.positions[0, 0] += .1
            write(path / "CONTCAR", changed, format="vasp")
    with patch.object(relaxer, "_execute", side_effect=run), pytest.raises((VASPValidationError, VASPExecutionError)):
        relaxer.relax(atoms, damage)


def test_xy_cell_shear_allowed_but_c_and_tilt_rejected(native):
    relaxer, atoms = native
    def run(path):
        final = read(path / "POSCAR")
        cell = final.cell.array.copy()
        cell[1, 0] += .01
        final.set_cell(cell, scale_atoms=True)
        output(relaxer, path, final=final)
    with patch.object(relaxer, "_execute", side_effect=run):
        result = relaxer.relax(atoms, "shear")
    assert result.atoms.cell[1, 0] == pytest.approx(atoms.cell[1, 0] + .01)
    for index, cell_index in enumerate(((2, 2), (2, 0), (0, 2))):
        def forbidden(path):
            final = read(path / "POSCAR")
            final.cell[cell_index] += .01
            output(relaxer, path, final=final)
        with patch.object(relaxer, "_execute", side_effect=forbidden), pytest.raises(VASPValidationError):
            relaxer.relax(atoms, f"tilt-{index}")


def test_z_continuity_uses_input_identity_across_boundary(native):
    relaxer, atoms = native
    atoms.positions[:, 2] = 15.9
    def run(path):
        final = read(path / "POSCAR")
        final.positions[:, 2] = .1
        output(relaxer, path, final=final)
    with patch.object(relaxer, "_execute", side_effect=run):
        result = relaxer.relax(atoms, "wrapped")
    np.testing.assert_allclose(result.atoms.positions[:, 2], 16.1)


def test_ambiguous_z_motion_and_final_topology_rejected(native):
    relaxer, atoms = native
    for name, z in (("large-step", 11.), ("detached", 9.9)):
        def run(path):
            final = read(path / "POSCAR")
            final.positions[0, 2] = z
            output(relaxer, path, final=final)
        with patch.object(relaxer, "_execute", side_effect=run), pytest.raises(VASPValidationError):
            relaxer.relax(atoms, name)


def test_changed_inputs_or_outputs_never_reused(native):
    relaxer, atoms = native
    with patch.object(relaxer, "_execute", side_effect=lambda p: output(relaxer, p)):
        relaxer.relax(atoms, "once")
    moved = atoms.copy()
    moved.positions[:, 0] += .1
    with pytest.raises(VASPExecutionError, match="inputs/settings"):
        relaxer.relax(moved, "once")
    (relaxer.work_dir / "once/stage-00/OUTCAR").write_text("changed")
    with pytest.raises(VASPExecutionError, match="files changed"):
        relaxer.relax(atoms, "once")


def test_changed_potential_or_recipe_requires_new_evaluation(native):
    relaxer, atoms = native
    with patch.object(relaxer, "_execute", side_effect=lambda p: output(relaxer, p)):
        relaxer.relax(atoms, "once")
    changed = SlabVASPRelaxer(relaxer.spec, replace(relaxer.config, incar={"ENCUT": 600}), relaxer.work_dir)
    with pytest.raises(VASPExecutionError, match="inputs/settings"):
        changed.relax(atoms, "once")
    Path(relaxer.config.potentials["B"]).write_text("changed")
    with pytest.raises(VASPExecutionError, match="source changed"):
        relaxer.relax(atoms, "new")


def test_preliminary_native_stage_allows_explicit_budget_continuation(native):
    relaxer, atoms = native
    config = replace(relaxer.config, preliminary_stages=(NativeVASPStage({"ISIF": 2}, allow_unconverged=True),))
    relaxer = SlabVASPRelaxer(relaxer.spec, config, relaxer.work_dir)
    def run(path):
        coarse = path.name == "stage-00"
        output(relaxer, path, count=2 if coarse else 1, ionic=not coarse, force=.1 if coarse else .001)
    with patch.object(relaxer, "_execute", side_effect=run) as execute:
        result = relaxer.relax(atoms, "staged")
    assert execute.call_count == 2
    assert not result.metadata["stages"][0]["converged"]
    assert result.metadata["stages"][1]["converged"]
    assert result.energy_per_atom == pytest.approx(-4.995)


def test_failed_stage_not_automatically_relaunched(native):
    relaxer, atoms = native
    with patch.object(relaxer, "_execute", side_effect=VASPExecutionError("transport")) as execute:
        for _ in range(2):
            with pytest.raises(VASPExecutionError):
                relaxer.relax(atoms, "failed")
    assert execute.call_count == 1


def test_native_command_runs_as_argv_and_nonzero_exit_is_infrastructure(native, tmp_path):
    relaxer, _ = native
    config = replace(relaxer.config, command=(sys.executable, "-c", "print('native argv'); raise SystemExit(7)"))
    relaxer = SlabVASPRelaxer(relaxer.spec, config, relaxer.work_dir)
    with pytest.raises(VASPExecutionError, match="status 7"):
        relaxer._execute(tmp_path)
    assert (tmp_path / "vasp.stdout").read_text().strip() == "native argv"


def test_timeout_terminates_process_group(native, tmp_path):
    relaxer, _ = native
    config = replace(relaxer.config, command=(sys.executable, "-c", "import time; time.sleep(60)"), timeout_seconds=.03)
    relaxer = SlabVASPRelaxer(relaxer.spec, config, relaxer.work_dir)
    with pytest.raises(VASPExecutionError, match="timeout"):
        relaxer._execute(tmp_path)


def test_actual_old_build_is_configuration_failure(native):
    relaxer, atoms = native
    with patch.object(relaxer, "_execute", side_effect=lambda path: output(relaxer, path, version="6.4.3")):
        with pytest.raises(VASPExecutionError, match="supported VASP build"):
            relaxer.relax(atoms, "old-build")


def test_restored_forces_follow_original_atom_order(native):
    relaxer, atoms = native
    def run(path):
        def mutate(root):
            rows = root.findall("calculation/varray[@name='forces']/v")
            rows[0].text = "0.001 0 0"  # B in VASP order
            rows[1].text = "0.002 0 0"  # N in VASP order
        output(relaxer, path, mutate_xml=mutate)
    with patch.object(relaxer, "_execute", side_effect=run):
        result = relaxer.relax(atoms, "force-order")
    np.testing.assert_allclose(result.atoms.get_forces()[:, 0], [.002, .001])


def test_unconnected_start_may_reconstruct_into_valid_sheet(native):
    relaxer, atoms = native
    atoms.positions[0, 2] += 1.8
    def run(path):
        final = read(path / "POSCAR")
        final.positions[:, 2] = 8
        output(relaxer, path, final=final)
    with patch.object(relaxer, "_execute", side_effect=run):
        result = relaxer.relax(atoms, "reconstruction")
    assert np.ptp(result.atoms.positions[:, 2]) == 0


def test_intermediate_cell_motion_cannot_be_hidden_by_final_cell(native):
    relaxer, atoms = native
    def run(path):
        def mutate(root):
            root.find("calculation/structure/crystal/varray[@name='basis']/v").text = "2.5 0 0.01"
        output(relaxer, path, count=2, mutate_xml=mutate)
    with patch.object(relaxer, "_execute", side_effect=run), pytest.raises(VASPValidationError, match="tilt"):
        relaxer.relax(atoms, "intermediate-tilt")


def test_potential_dataset_species_and_count_must_match(native):
    relaxer, _ = native
    path = Path(relaxer.config.potentials["B"])
    for text in ("VRHFIN =C: test\nEnd of Dataset\n",
                 "VRHFIN =B: test\nEnd of Dataset\nVRHFIN =B: test\nEnd of Dataset\n"):
        path.write_text(text)
        with pytest.raises(ValueError, match="own POTCAR dataset"):
            SlabVASPRelaxer(relaxer.spec, relaxer.config, relaxer.work_dir)
