"""Physical xy lattice relaxation, including energy derivatives and ASE masks."""

from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes
from ase.calculators.emt import EMT

from crisp.slab import SlabConfig, validate_slab
from crisp.slab_cell_relaxation import SlabCellRelaxation, _SlabCellFilter, UnitCellFilter
from crisp.slab_relaxation import SlabQuenchBoundsError, SlabQuenchNotConverged, quench_slab


def _slab(height=18.0):
    return Atoms("Cu4", positions=[[0, 0, 9], [2.4, 0, 9],
                                  [1.2, 2.08, 9], [3.6, 2.08, 9]],
                 cell=[[4.8, 0, 0], [0.4, 4.16, 0], [0, 0, height]],
                 pbc=(True, True, False), info={"source": {"values": [1, 2]}})


def _config(height=18.0):
    return SlabConfig((3.0, 10.0), 1.0, 4.0, height, 12.0)


class _Calculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, evaluate):
        super().__init__()
        self.evaluate = evaluate
        self.states = []

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.states.append(self.atoms.copy())
        energy, forces, stress = self.evaluate(self.atoms, len(self.states))
        self.results = dict(energy=energy, forces=np.asarray(forces), stress=np.asarray(stress))


def _constant(*, forces=None, stress=None):
    return _Calculator(lambda a, _: (0.0, np.zeros((len(a), 3)) if forces is None else forces,
                                     np.zeros(6) if stress is None else stress))


class TestSlabCellRelaxation(unittest.TestCase):
    def test_options_are_finite_positive_real_scalars(self):
        self.assertEqual(SlabCellRelaxation(np.float64(0.02)).stress_max, 0.02)
        self.assertIs(type(SlabCellRelaxation(1).stress_max), float)
        for value in (0, -1, True, np.nan, np.inf, 1j, "0.01", [0.01], np.array(0.01)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SlabCellRelaxation(value)
        factory = Mock()
        with self.assertRaises(TypeError):
            quench_slab(_slab(), _config(), factory, cell_relaxation=True)
        factory.assert_not_called()

    def test_zero_step_canonicalizes_only_the_owned_cell(self):
        atoms = _slab()
        atoms.positions += [12.0, -7.0, 30.0]
        atoms.cell[:2, 2] = [1e-10, -1e-10]
        atoms.cell[2] = [1e-10, -1e-10, 18 + 1e-10]
        atoms.new_array("labels", np.array([1, 2, 3, 4]))
        original = Mock()
        atoms.calc = original
        before = atoms.copy()
        info = deepcopy(atoms.info)
        calculator = _constant(stress=[0, 0, 123, 456, 789, 0])
        factory = Mock(return_value=calculator)
        relaxed = quench_slab(atoms, _config(), factory, max_steps=0,
                              cell_relaxation=SlabCellRelaxation())
        np.testing.assert_array_equal(relaxed.positions, before.positions)
        np.testing.assert_array_equal(relaxed.cell[:2, :2], before.cell[:2, :2])
        np.testing.assert_array_equal(relaxed.cell[:2, 2], [0, 0])
        np.testing.assert_array_equal(relaxed.cell[2], [0, 0, 18])
        np.testing.assert_array_equal(relaxed.pbc, [True, True, False])
        np.testing.assert_array_equal(relaxed.arrays["labels"], before.arrays["labels"])
        np.testing.assert_array_equal(atoms.positions, before.positions)
        np.testing.assert_array_equal(atoms.cell, before.cell)
        self.assertEqual(atoms.info, info)
        self.assertIs(atoms.calc, original)
        self.assertIs(relaxed.calc, calculator)
        factory.assert_called_once_with()
        self.assertEqual(len(calculator.states), 1)
        self.assertEqual(relaxed.info["slab_quench_steps"], 0)
        self.assertEqual(relaxed.info["slab_quench_fmax"], 0)
        self.assertEqual(relaxed.info["slab_quench_stress_max"], 0)

    def test_strict_atomic_and_stress_thresholds_are_independent(self):
        for forces, stress, succeeds in (
                (np.zeros((4, 3)), [0, 0, 0, 0, 0, 0], True),
                (np.tile([0.04, 0.04, 0.04], (4, 1)), np.zeros(6), False),
                (np.tile([0.05, 0, 0], (4, 1)), np.zeros(6), False),
                (np.zeros((4, 3)), [0.01 / 18, 0, 0, 0, 0, 0], False),
                (np.zeros((4, 3)), [0, -0.01 / 18, 0, 0, 0, 0], False),
                (np.zeros((4, 3)), [0, 0, 0, 0, 0, -0.01 / 18], False),
                (np.zeros((4, 3)), [0.009 / 18, 0, 500, -500, 500, 0], True)):
            with self.subTest(forces=forces, stress=stress):
                calculator = _constant(forces=forces, stress=stress)
                if succeeds:
                    result = quench_slab(_slab(), _config(), lambda: calculator, max_steps=0,
                                         cell_relaxation=SlabCellRelaxation())
                    self.assertLess(result.info["slab_quench_stress_max"], 0.01)
                else:
                    with self.assertRaisesRegex(SlabQuenchNotConverged, "in-plane stress"):
                        quench_slab(_slab(), _config(), lambda: calculator, max_steps=0,
                                    cell_relaxation=SlabCellRelaxation())
                self.assertEqual(len(calculator.states), 1)

    def test_convergence_on_last_allowed_step(self):
        def evaluate(atoms, count):
            forces = np.zeros((len(atoms), 3))
            stress = np.zeros(6)
            if count == 1:
                forces[:, 0] = 0.1
                stress[5] = 0.01
            return 0.0, forces, stress

        calculator = _Calculator(evaluate)
        relaxed = quench_slab(_slab(), _config(), lambda: calculator, max_steps=1,
                              cell_relaxation=SlabCellRelaxation())
        self.assertEqual(relaxed.info["slab_quench_steps"], 1)
        self.assertEqual(len(calculator.states), 2)
        self.assertFalse(np.array_equal(relaxed.cell, _slab().cell))

    def test_real_atomic_and_cell_relaxation_preserves_plane_and_lowers_energy(self):
        atoms = _slab()
        atoms.calc = EMT()
        before = atoms.copy()
        initial_energy = atoms.get_potential_energy()
        relaxed = quench_slab(atoms, _config(), EMT, fmax=1e-3, max_steps=100,
                              cell_relaxation=SlabCellRelaxation(1e-3))
        self.assertLess(relaxed.get_potential_energy(), initial_energy)
        self.assertLess(np.linalg.norm(relaxed.get_forces(), axis=1).max(), 1e-3)
        self.assertLess(18 * np.abs(relaxed.get_stress()[[0, 1, 5]]).max(), 1e-3)
        self.assertGreater(relaxed.info["slab_quench_steps"], 1)
        self.assertFalse(np.allclose(relaxed.cell[:2, :2], before.cell[:2, :2]))
        self.assertFalse(np.allclose(relaxed.positions, before.positions))
        np.testing.assert_array_equal(relaxed.cell[2], before.cell[2])
        np.testing.assert_array_equal(relaxed.cell[:2, 2], [0, 0])
        np.testing.assert_array_equal(relaxed.pbc, before.pbc)
        np.testing.assert_array_equal(atoms.positions, before.positions)
        np.testing.assert_array_equal(atoms.cell, before.cell)
        validate_slab(relaxed, _config())

    def test_filter_projects_every_forbidden_cell_coordinate(self):
        atoms = _slab()
        cell_filter = _SlabCellFilter(atoms)
        proposed = cell_filter.get_positions()
        proposed[-3:] = len(atoms) * np.array([[1.07, 0.11, 0.8],
                                              [-0.04, 0.96, -0.4], [0.3, -0.2, 2]])
        before = proposed.copy()
        z = atoms.positions[:, 2].copy()
        cell_filter.set_positions(proposed)
        np.testing.assert_array_equal(proposed, before)
        np.testing.assert_array_equal(atoms.cell[2], [0, 0, 18])
        np.testing.assert_array_equal(atoms.cell[:2, 2], [0, 0])
        np.testing.assert_array_equal(atoms.positions[:, 2], z)
        np.testing.assert_allclose(atoms.cell[:2, :2],
                                   _slab().cell[:2, :2] @ before[-3:-1, :2].T / len(atoms))

    def test_generalized_energy_derivatives_at_nonidentity_shear(self):
        atoms = _slab()
        atoms.positions += [[0.05, -0.06, 0.02], [-0.04, 0.07, -0.03],
                            [0.03, 0.02, 0.05], [-0.07, -0.03, -0.04]]
        atoms.calc = EMT()
        cell_filter = _SlabCellFilter(atoms)
        coordinates = cell_filter.get_positions()
        coordinates[-3:] = len(atoms) * np.array([[1.07, 0.11, 0],
                                                  [-0.04, 0.96, 0], [0, 0, 1]])
        cell_filter.set_positions(coordinates)
        analytic = cell_filter.get_forces()
        numeric = np.zeros_like(analytic)
        delta = 1e-6
        for index in np.ndindex(coordinates.shape):
            plus = coordinates.copy()
            plus[index] += delta
            cell_filter.set_positions(plus)
            high = atoms.get_potential_energy()
            minus = coordinates.copy()
            minus[index] -= delta
            cell_filter.set_positions(minus)
            low = atoms.get_potential_energy()
            numeric[index] = -(high - low) / (2 * delta)
        cell_filter.set_positions(coordinates)
        np.testing.assert_allclose(analytic, numeric, rtol=0, atol=5e-7)
        np.testing.assert_array_equal(analytic[-1], [0, 0, 0])
        np.testing.assert_array_equal(analytic[-3:, 2], [0, 0, 0])

    def test_vacuum_does_not_change_forces_stress_or_relaxation(self):
        generalized, stresses, results = [], [], []
        for height in (18.0, 36.0):
            atoms = _slab(height)
            atoms.calc = EMT()
            cell_filter = _SlabCellFilter(atoms)
            coordinates = cell_filter.get_positions()
            coordinates[-3:] = len(atoms) * np.array([[1.07, 0.11, 0],
                                                      [-0.04, 0.96, 0], [0, 0, 1]])
            cell_filter.set_positions(coordinates)
            generalized.append(cell_filter.get_forces())
            stresses.append(height * atoms.get_stress()[[0, 1, 5]])
            results.append(quench_slab(_slab(height), _config(height), EMT, fmax=1e-3,
                                      max_steps=100, cell_relaxation=SlabCellRelaxation(1e-3)))
        np.testing.assert_allclose(generalized[0], generalized[1], rtol=0, atol=1e-12)
        np.testing.assert_allclose(stresses[0], stresses[1], rtol=0, atol=1e-12)
        np.testing.assert_allclose(results[0].positions, results[1].positions, rtol=0, atol=1e-9)
        np.testing.assert_allclose(results[0].cell[:2], results[1].cell[:2], rtol=0, atol=1e-9)
        self.assertEqual(results[0].info["slab_quench_steps"], results[1].info["slab_quench_steps"])

    def test_out_of_bounds_cell_is_rejected_before_calculation(self):
        atoms = _slab()
        area = np.linalg.det(atoms.cell[:2, :2]) / len(atoms)
        config = replace(_config(), area_per_atom_range=(area, area + 0.001))
        calculator = _constant(stress=[100, 100, 0, 0, 0, 0])
        with self.assertRaisesRegex(SlabQuenchBoundsError, "area"):
            quench_slab(atoms, config, lambda: calculator, max_steps=2,
                        cell_relaxation=SlabCellRelaxation())
        self.assertEqual(len(calculator.states), 1)

    def test_out_of_bounds_atoms_are_rejected_before_calculation(self):
        calculator = _constant(forces=[[0, 0, -70], [0, 0, 70], [0, 0, 0], [0, 0, 0]])
        config = replace(_config(), initial_thickness=0, max_thickness=0.1)
        with self.assertRaisesRegex(SlabQuenchBoundsError, "thickness"):
            quench_slab(_slab(), config, lambda: calculator,
                        cell_relaxation=SlabCellRelaxation())
        self.assertEqual(len(calculator.states), 1)

    def test_missing_stress_is_fatal_only_when_enabled(self):
        calculator = _constant()
        calculator.implemented_properties = ["energy", "forces"]
        with self.assertRaises(PropertyNotImplementedError):
            quench_slab(_slab(), _config(), lambda: calculator, max_steps=0,
                        cell_relaxation=SlabCellRelaxation())
        fixed = _constant()
        fixed.implemented_properties = ["energy", "forces"]
        with patch.object(fixed, "get_stress", side_effect=AssertionError("stress requested")):
            relaxed = quench_slab(_slab(), _config(), lambda: fixed, max_steps=0)
        self.assertNotIn("slab_quench_stress_max", relaxed.info)

    def test_all_stress_components_must_be_finite_and_real(self):
        values = [np.full(6, np.nan), np.full(6, np.inf), np.zeros(6, dtype=complex),
                  [0, 0, np.nan, 0, 0, 0], [0, 0, 0, np.inf, 0, 0],
                  [0, 0, 0, 0, 1j, 0], np.full((3, 3), np.nan)]
        for stress in values:
            with self.subTest(stress=stress):
                calculator = _constant(stress=stress)
                with self.assertRaisesRegex(RuntimeError, "stress"):
                    quench_slab(_slab(), _config(), lambda: calculator, max_steps=0,
                                cell_relaxation=SlabCellRelaxation())
                self.assertEqual(len(calculator.states), 1)

    def test_malformed_stress_and_calculator_errors_propagate(self):
        for stress in ([0, 0], np.zeros((2, 3)), np.zeros((6, 1))):
            with self.subTest(stress=stress):
                calculator = _constant(stress=stress)
                with self.assertRaises((AssertionError, ValueError, RuntimeError)):
                    quench_slab(_slab(), _config(), lambda: calculator,
                                cell_relaxation=SlabCellRelaxation())
        calculator = _constant()
        error = TypeError("stress calculator bug")
        with patch.object(calculator, "get_stress", side_effect=error):
            with self.assertRaises(TypeError) as caught:
                quench_slab(_slab(), _config(), lambda: calculator,
                            cell_relaxation=SlabCellRelaxation())
        self.assertIs(caught.exception, error)

    def test_later_invalid_stress_aborts_before_another_step(self):
        def evaluate(atoms, count):
            stress = np.zeros(6)
            stress[0] = 0.01 if count == 1 else np.nan
            return 0, np.zeros((len(atoms), 3)), stress
        calculator = _Calculator(evaluate)
        with self.assertRaisesRegex(RuntimeError, "stress"):
            quench_slab(_slab(), _config(), lambda: calculator,
                        cell_relaxation=SlabCellRelaxation())
        self.assertEqual(len(calculator.states), 2)

    def test_nonfinite_generalized_forces_abort_before_optimizer_move(self):
        cell_filter = _SlabCellFilter(_slab())
        with patch.object(UnitCellFilter, "get_forces", return_value=np.full((7, 3), np.inf)):
            with self.assertRaisesRegex(RuntimeError, "generalized forces"):
                cell_filter.get_forces()


if __name__ == "__main__":
    unittest.main()
