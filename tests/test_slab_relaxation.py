"""Fixed-cell slab quenching with small, energy/forces-only calculators."""

from dataclasses import replace
import unittest
from unittest.mock import Mock

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

from crisp.slab import SlabConfig, validate_slab
from crisp.slab_relaxation import quench_slab


def _config():
    return SlabConfig((4.0, 8.0), initial_thickness=1.0, max_thickness=4.0,
                      cell_height=20.0, min_vacuum=12.0)


def _slab():
    return Atoms("C2", positions=[[0.3, 0.4, 9.8], [1.6, 1.4, 10.2]],
                 cell=[[4.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 20.0]],
                 pbc=(True, True, False), info={"origin": "random"})


class _Calculator(Calculator):
    """Record evaluated geometries; deliberately provide no stress."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, evaluate):
        super().__init__()
        self.evaluate = evaluate
        self.states = []

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.states.append(self.atoms.copy())
        energy, forces = self.evaluate(self.atoms, len(self.states))
        self.results = {"energy": energy, "forces": np.asarray(forces)}


def _harmonic(target):
    def evaluate(atoms, _):
        displacement = atoms.positions - target
        return 0.5 * np.sum(displacement ** 2), -displacement
    return _Calculator(evaluate)


class TestSlabRelaxation(unittest.TestCase):
    def assert_input_unchanged(self, atoms, before, calculator):
        np.testing.assert_array_equal(atoms.positions, before.positions)
        np.testing.assert_array_equal(atoms.cell, before.cell)
        np.testing.assert_array_equal(atoms.pbc, before.pbc)
        np.testing.assert_array_equal(atoms.numbers, before.numbers)
        self.assertEqual(atoms.info, before.info)
        self.assertEqual(atoms.constraints, before.constraints)
        self.assertIs(atoms.calc, calculator)

    def test_xyz_quench_preserves_cell_and_input_without_stress(self):
        atoms = _slab()
        original = _Calculator(lambda *_: self.fail("Input calculator was used"))
        atoms.calc = original
        before = atoms.copy()
        target = atoms.positions + [[0.5, -0.6, -0.4], [-0.3, 0.4, 0.6]]
        calculator = _harmonic(target)
        factory = Mock(return_value=calculator)

        relaxed = quench_slab(atoms, _config(), factory, fmax=1e-6, max_steps=100)

        self.assertIsNot(relaxed, atoms)
        self.assertIs(relaxed.calc, calculator)
        factory.assert_called_once_with()
        np.testing.assert_allclose(relaxed.positions, target, atol=1e-6, rtol=0)
        self.assertLess(relaxed.get_potential_energy(), 1e-12)
        self.assertGreater(np.ptp(relaxed.positions[:, 2]), 1.0)
        np.testing.assert_array_equal(relaxed.cell, before.cell)
        np.testing.assert_array_equal(relaxed.pbc, [True, True, False])
        self.assertEqual(relaxed.info["origin"], "random")
        self.assertGreater(relaxed.info["slab_quench_steps"], 0)
        self.assertLessEqual(relaxed.info["slab_quench_steps"], 100)
        actual_fmax = np.linalg.norm(relaxed.get_forces(), axis=1).max()
        self.assertAlmostEqual(relaxed.info["slab_quench_fmax"], actual_fmax)
        self.assertLess(actual_fmax, 1e-6)
        for state in calculator.states:
            validate_slab(state, _config())
            np.testing.assert_array_equal(state.cell, before.cell)
        self.assert_input_unchanged(atoms, before, original)
        self.assertEqual(original.states, [])

    def test_initially_converged_zero_step_preserves_coordinate_frame(self):
        atoms = _slab()
        # These are valid unwrapped xy and translated z coordinates.
        atoms.positions += [12.0, -6.0, 31.0]
        before = atoms.copy()
        calculator = _harmonic(atoms.positions.copy())
        relaxed = quench_slab(atoms, _config(), lambda: calculator, max_steps=0)
        np.testing.assert_array_equal(relaxed.positions, before.positions)
        np.testing.assert_array_equal(relaxed.cell, before.cell)
        self.assertEqual(relaxed.info["slab_quench_steps"], 0)
        self.assertEqual(relaxed.info["slab_quench_fmax"], 0.0)
        self.assertEqual(len(calculator.states), 1)
        self.assert_input_unchanged(atoms, before, None)

    def test_force_threshold_uses_strict_per_atom_vector_norm(self):
        for vector in ([0.04, 0.04, 0.04], [0.05, 0.0, 0.0]):
            with self.subTest(vector=vector):
                calculator = _Calculator(lambda atoms, _: (0.0, np.tile(vector, (len(atoms), 1))))
                with self.assertRaises(RuntimeError):
                    quench_slab(_slab(), _config(), lambda: calculator,
                                fmax=0.05, max_steps=0)
                self.assertEqual(len(calculator.states), 1)

    def test_step_budget_failure_preserves_input(self):
        atoms = _slab()
        original = _harmonic(atoms.positions.copy())
        atoms.calc = original
        before = atoms.copy()
        calculator = _harmonic(atoms.positions + [2.0, 0.0, 0.0])
        factory = Mock(return_value=calculator)
        with self.assertRaisesRegex(RuntimeError, "1.*step"):
            quench_slab(atoms, _config(), factory, fmax=1e-8, max_steps=1)
        factory.assert_called_once_with()
        self.assertEqual(len(calculator.states), 2)
        self.assert_input_unchanged(atoms, before, original)
        self.assertEqual(original.states, [])

    def test_convergence_on_last_allowed_step_is_accepted(self):
        atoms = _slab()
        calculator = _harmonic(atoms.positions + [0.01, 0.0, 0.0])
        relaxed = quench_slab(atoms, _config(), lambda: calculator,
                             fmax=0.0099, max_steps=1)
        self.assertEqual(relaxed.info["slab_quench_steps"], 1)
        self.assertLess(relaxed.info["slab_quench_fmax"], 0.0099)
        self.assertEqual(len(calculator.states), 2)

    def test_invalid_options_fail_before_factory(self):
        cases = [{"fmax": value} for value in (0.0, -1.0, np.nan, np.inf)]
        cases += [{"max_steps": value} for value in (-1, 1.5, True)]
        for options in cases:
            with self.subTest(options=options):
                factory = Mock()
                with self.assertRaises(ValueError):
                    quench_slab(_slab(), _config(), factory, **options)
                factory.assert_not_called()

    def test_invalid_geometry_and_constraints_fail_before_factory(self):
        wrong_pbc = _slab()
        wrong_pbc.pbc = True
        wrong_height = _slab()
        wrong_height.cell[2, 2] = 21.0
        nonfinite = _slab()
        nonfinite.positions[0, 0] = np.nan
        constrained = _slab()
        constrained.set_constraint(FixAtoms(indices=[0]))
        for atoms in (wrong_pbc, wrong_height, nonfinite, constrained):
            with self.subTest(atoms=atoms):
                factory = Mock()
                with self.assertRaises(ValueError):
                    quench_slab(atoms, _config(), factory)
                factory.assert_not_called()

    def test_factory_must_return_a_calculator_distinct_from_input(self):
        atoms = _slab()
        atoms.calc = _harmonic(atoms.positions.copy())
        for calculator in (None, atoms.calc):
            with self.subTest(calculator=calculator):
                factory = Mock(return_value=calculator)
                with self.assertRaises(ValueError):
                    quench_slab(atoms, _config(), factory)
                factory.assert_called_once_with()
        self.assertEqual(atoms.calc.states, [])

    def test_nonfinite_or_malformed_results_are_rejected(self):
        atoms = _slab()
        zeros = np.zeros((len(atoms), 3))
        results = [(value, zeros) for value in (np.nan, np.inf, [0.0])]
        results += [(0.0, value) for value in
                    (np.full((2, 3), np.nan), np.full((2, 3), np.inf),
                     np.zeros((2, 2)), np.zeros(3))]
        for energy, forces in results:
            with self.subTest(energy=energy, forces=forces):
                calculator = _Calculator(lambda *_: (energy, forces))
                with self.assertRaises(RuntimeError):
                    quench_slab(atoms, _config(), lambda: calculator)
                self.assertEqual(len(calculator.states), 1)
                self.assertNotIn("slab_quench_steps", atoms.info)

    def test_later_nonfinite_results_abort_before_another_step(self):
        atoms = _slab()
        before = atoms.copy()
        for invalid in ("energy", "forces"):
            def evaluate(state, count):
                forces = np.tile([1.0, 0.0, 0.0], (len(state), 1))
                if count == 1:
                    return 0.0, forces
                return ((np.nan, forces) if invalid == "energy"
                        else (0.0, np.full_like(forces, np.nan)))

            with self.subTest(invalid=invalid):
                calculator = _Calculator(evaluate)
                with self.assertRaises(RuntimeError):
                    quench_slab(atoms, _config(), lambda: calculator)
                self.assertEqual(len(calculator.states), 2)
                self.assert_input_unchanged(atoms, before, None)

    def test_invalid_moved_geometry_is_rejected_before_evaluation(self):
        atoms = _slab()
        atoms.positions[:, 2] = 10.0
        config = replace(_config(), initial_thickness=0.0, max_thickness=0.1)
        calculator = _Calculator(lambda *_: (0.0, [[0.0, 0.0, -70.0],
                                                  [0.0, 0.0, 70.0]]))
        before = atoms.copy()
        with self.assertRaisesRegex(ValueError, "thickness"):
            quench_slab(atoms, config, lambda: calculator)
        self.assertEqual(len(calculator.states), 1)
        self.assert_input_unchanged(atoms, before, None)

    def test_calculator_exceptions_propagate(self):
        error = TypeError("calculator implementation bug")
        calculator = _Calculator(Mock(side_effect=error))
        with self.assertRaises(TypeError) as raised:
            quench_slab(_slab(), _config(), lambda: calculator)
        self.assertIs(raised.exception, error)


if __name__ == "__main__":
    unittest.main()
