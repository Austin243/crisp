"""Fixed-composition slab archive identity, ownership, and failure guards."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms

from crisp.fingerprint import FingerprintCalculator
from crisp.slab import SlabConfig, prepare_slab
from crisp.slab_archive import SlabArchive
from crisp.slab_fingerprint import SlabFingerprintCalculator

try:
    import torch_fplib
except ImportError:
    torch_fplib = None


def _config():
    return SlabConfig((1.0, 8.0), initial_thickness=1.0, max_thickness=2.0,
                      cell_height=14.0, min_vacuum=6.0)


def _sheet():
    a = 2.46
    cell = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 14]])
    positions = np.array([[0, 0, 0], [2 / 3, 1 / 3, 0]]) @ cell
    positions = np.concatenate((positions, positions + cell[0]))
    positions[:, 2] = [-0.1, 0.2, 0.05, -0.15]
    positions[2, 0] += 0.08  # Inequivalent environments within each species.
    cell[0] *= 2
    atoms = prepare_slab(Atoms("BNBN", positions=positions, cell=cell), _config())
    atoms.set_array("test_fp", np.repeat([[10.0], [100.0], [20.0], [200.0]], 8, axis=1))
    return atoms


class TestSlabArchive(unittest.TestCase):
    def setUp(self):
        self.calc = SlabFingerprintCalculator(_config(), cutoff=3.2, natx=8)
        self.archive = SlabArchive(self.calc, {"B": 2, "N": 2}, min_dist_ang=1.0)
        self.fp_patch = patch.object(self.calc, "get_fingerprints",
                                     side_effect=lambda atoms: atoms.arrays["test_fp"])
        self.fp_mock = self.fp_patch.start()
        self.addCleanup(self.fp_patch.stop)

    def test_species_permutations_use_real_hungarian_matching(self):
        atoms = _sheet()
        self.assertTrue(self.archive.add(atoms, -3.0))
        self.assertFalse(self.archive.add(atoms[[3, 0, 2, 1]], -3.0))
        self.assertEqual(len(self.archive.entries), 1)
        np.testing.assert_array_equal(self.archive.entries[0].atoms.numbers, [5, 5, 7, 7])
        # Swapping descriptor rows across species must not create a match.
        changed = atoms.copy()
        changed.arrays["test_fp"][[0, 1]] = changed.arrays["test_fp"][[1, 0]]
        self.assertTrue(self.archive.add(changed, -3.0))

    def test_duplicate_requires_both_strict_thresholds(self):
        self.archive.fp_threshold = 0.25
        self.archive.energy_threshold = 0.125
        atoms = _sheet()
        atoms.arrays["test_fp"][:] = 0
        self.assertTrue(self.archive.add(atoms, 0.0))
        atoms.arrays["test_fp"][:] = 0.25
        self.assertTrue(self.archive.add(atoms, 0.0))  # Exact FP boundary.
        atoms.arrays["test_fp"][:] = 0
        self.assertTrue(self.archive.add(atoms, 0.125))  # Exact energy boundary.
        atoms.arrays["test_fp"][:] = 0.125
        self.assertFalse(self.archive.add(atoms, 0.0625))

    def test_entry_owns_buffers_metadata_and_normalized_calculator_free_atoms(self):
        atoms = _sheet()
        atoms.positions += [17.3, -8.1, 33.0]
        atoms.info = {"nested": {"values": [1, 2]}}
        atoms.set_constraint(FixAtoms(indices=[0]))
        calculator = Calculator()
        calculator.calculate = Mock(side_effect=AssertionError("Unexpected evaluation"))
        atoms.calc = calculator
        before = atoms.copy()
        info = deepcopy(atoms.info)
        constraints = [constraint.todict() for constraint in atoms.constraints]
        metadata = {"generation": 3, "nested": {"values": [4, 5]}}
        fp_buffer = np.ones((4, 8))
        pooled_buffer = np.ones(16)
        with patch.object(self.calc, "get_fingerprints", return_value=fp_buffer), \
                patch.object(self.calc, "pool_with_std", return_value=pooled_buffer):
            self.assertTrue(self.archive.add(atoms, np.float64(-4.0), metadata=metadata))
        entry = self.archive.entries[0]
        np.testing.assert_array_equal(atoms.positions, before.positions)
        np.testing.assert_array_equal(atoms.cell, before.cell)
        np.testing.assert_array_equal(atoms.numbers, before.numbers)
        np.testing.assert_array_equal(atoms.pbc, before.pbc)
        self.assertEqual(atoms.info, info)
        self.assertEqual([c.todict() for c in atoms.constraints], constraints)
        self.assertIs(atoms.calc, calculator)
        calculator.calculate.assert_not_called()
        self.assertIsNone(entry.atoms.calc)
        self.assertAlmostEqual(np.mean([entry.atoms.positions[:, 2].min(),
                                       entry.atoms.positions[:, 2].max()]), 7.0)
        xy = entry.atoms.get_scaled_positions(wrap=False)[:, :2]
        self.assertTrue(np.all((xy >= -1e-10) & (xy < 1 + 1e-10)))
        self.assertEqual((entry.energy, entry.enthalpy, entry.pressure), (-4.0, -4.0, 0.0))
        self.assertEqual(entry.generation, 3)
        atoms.info["nested"]["values"].append(99)
        metadata["nested"]["values"].append(99)
        fp_buffer[:] = np.nan
        pooled_buffer[:] = np.nan
        self.assertEqual(entry.atoms.info, info)
        self.assertEqual(entry.metadata["nested"]["values"], [4, 5])
        self.assertTrue(np.isfinite(entry.fp).all())
        self.assertTrue(np.isfinite(entry.fp_pooled).all())

    def test_wrong_composition_and_invalid_geometry_fail_before_fingerprints(self):
        wrong_species = _sheet()
        wrong_species.numbers[0] = 6
        wrong_count = _sheet()[:2]
        wrong_pbc = _sheet()
        wrong_pbc.pbc = True
        disconnected = _sheet()
        disconnected.positions[1, 2] += 1.6
        collision = _sheet()
        collision.positions[1] = collision.positions[0]
        for atoms in (wrong_species, wrong_count, wrong_pbc, disconnected, collision):
            with self.subTest(symbols=atoms.symbols), self.assertRaises(ValueError):
                self.archive.add(atoms, -3.0)
        self.fp_mock.assert_not_called()
        self.assertEqual(self.archive.entries, [])

    def test_invalid_energy_and_metadata_fail_before_fingerprints(self):
        for energy in (np.nan, np.inf, -np.inf, [0.0], np.array([0.0]), 1 + 2j):
            with self.subTest(energy=energy), self.assertRaises((TypeError, ValueError)):
                self.archive.add(_sheet(), energy)
        for metadata in ([], "metadata", 3):
            with self.subTest(metadata=metadata), self.assertRaises(TypeError):
                self.archive.add(_sheet(), -3.0, metadata=metadata)
        self.fp_mock.assert_not_called()
        self.assertEqual(self.archive.entries, [])

    def test_invalid_descriptor_outputs_never_append(self):
        for fp in (np.zeros((3, 8)), np.zeros((4, 7)), np.zeros((4, 8, 1)),
                   np.full((4, 8), np.nan), np.full((4, 8), np.inf),
                   np.full((4, 8), 1 + 2j)):
            with self.subTest(shape=fp.shape), \
                    patch.object(self.calc, "get_fingerprints", return_value=fp), \
                    self.assertRaises(ValueError):
                self.archive.add(_sheet(), -3.0)
        for pooled in (np.zeros(8), np.zeros((2, 8)), np.full(16, np.nan), np.full(16, 1 + 2j)):
            with self.subTest(shape=pooled.shape), \
                    patch.object(self.calc, "pool_with_std", return_value=pooled), \
                    self.assertRaises(ValueError):
                self.archive.add(_sheet(), -3.0)
        self.assertEqual(self.archive.entries, [])

    def test_descriptor_and_comparison_errors_propagate_without_partial_adds(self):
        self.assertTrue(self.archive.add(_sheet(), -3.0))
        entry = self.archive.entries[0]
        for method in ("get_fingerprints", "pool_with_std", "_fp_dist"):
            with self.subTest(method=method), \
                    patch.object(self.calc, method, side_effect=RuntimeError("backend failed")), \
                    self.assertRaisesRegex(RuntimeError, "backend failed"):
                self.archive.add(_sheet(), -3.0)
        for value in (np.nan, np.inf, -1.0, 1 + 2j, np.array([0.0])):
            with self.subTest(distance=value), \
                    patch.object(self.calc, "_fp_dist", return_value=value), \
                    self.assertRaises(ValueError):
                self.archive.add(_sheet(), -3.0)
        self.assertEqual(len(self.archive.entries), 1)
        self.assertIs(self.archive.entries[0], entry)

    def test_constructor_validates_options_and_copies_target_composition(self):
        options = dict(min_dist_ang=1.0, bond_scale=1.2,
                       fp_threshold=0.03, energy_threshold=0.01)
        for name in options:
            for value in (0.0, -1.0, np.inf, np.nan, 1 + 2j):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    SlabArchive(self.calc, {"B": 2, "N": 2}, **(options | {name: value}))
        for composition in ({}, {"X": 1}, {"invalid": 1}, {"B": 0}, {"B": -1},
                            {"B": 1.5}, {"B": True}):
            with self.subTest(composition=composition), self.assertRaises(ValueError):
                SlabArchive(self.calc, composition, min_dist_ang=1.0)
        with self.assertRaises((TypeError, ValueError)):
            SlabArchive(FingerprintCalculator(), {"B": 2, "N": 2}, min_dist_ang=1.0)
        composition = {"B": 2, "N": 2}
        archive = SlabArchive(self.calc, composition, min_dist_ang=1.0)
        composition["B"] = 99
        self.assertTrue(archive.add(_sheet(), -3.0))

    def test_fingerprint_settings_must_remain_compatible(self):
        for name, value in (("config", replace(_config(), cell_height=15.0)),
                            ("cutoff", 3.0), ("natx", 16), ("orbital", "sp")):
            with self.subTest(name=name):
                original = getattr(self.calc, name)
                setattr(self.calc, name, value)
                try:
                    with self.assertRaises(ValueError):
                        self.archive.add(_sheet(), -3.0)
                finally:
                    setattr(self.calc, name, original)
        self.fp_mock.assert_not_called()
        self.assertEqual(self.archive.entries, [])
        for name, value in (("natx", 0), ("natx", 1.5), ("natx", True),
                            ("orbital", "p"), ("cutoff", np.nan)):
            with self.subTest(name=name, value=value):
                original = getattr(self.calc, name)
                setattr(self.calc, name, value)
                try:
                    with self.assertRaises(ValueError):
                        SlabArchive(self.calc, {"B": 2, "N": 2}, min_dist_ang=1.0)
                finally:
                    setattr(self.calc, name, original)

    def test_inherited_ranking_diversity_and_arrays_use_energy(self):
        for descriptor, energy in ((0.0, -3.0), (1.0, -5.0), (3.0, -4.0)):
            atoms = _sheet()
            atoms.arrays["test_fp"][:] = descriptor
            self.assertTrue(self.archive.add(atoms, energy))
        self.assertEqual([entry.energy for entry in self.archive.get_best(2)], [-5.0, -4.0])
        self.assertEqual(len(self.archive.get_diverse(2)), 2)
        np.testing.assert_array_equal(self.archive.get_all_energies(), [-3.0, -5.0, -4.0])
        np.testing.assert_array_equal(self.archive.get_all_enthalpies(),
                                      self.archive.get_all_energies())
        self.assertEqual(self.archive.get_all_pooled_fps().shape, (3, 16))
        np.testing.assert_array_equal(self.archive.get_repulsion_centers(),
                                      self.archive.get_all_pooled_fps())

    def test_persistence_is_explicitly_disabled_without_filesystem_effects(self):
        self.assertTrue(self.archive.add(_sheet(), -3.0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "must-not-exist"
            for method in ("save", "load", "save_checkpoint", "load_checkpoint"):
                with self.subTest(method=method), self.assertRaises(NotImplementedError):
                    getattr(self.archive, method)(str(path))
                self.assertFalse(path.exists())
                self.assertEqual(len(self.archive.entries), 1)


@unittest.skipUnless(torch_fplib is not None, "torch_fplib required")
class TestSlabArchiveBackend(unittest.TestCase):
    def test_real_multispecies_fingerprints_deduplicate_equivalent_sheets(self):
        for orbital in ("s", "sp"):
            with self.subTest(orbital=orbital):
                calc = SlabFingerprintCalculator(_config(), cutoff=3.2,
                                                 natx=32, orbital=orbital)
                archive = SlabArchive(calc, {"B": 2, "N": 2}, min_dist_ang=1.0,
                                      fp_threshold=1e-6)
                atoms = _sheet()
                self.assertTrue(archive.add(atoms, -3.0))
                equivalent = atoms[[3, 0, 2, 1]]
                equivalent.positions += [17.3, -8.1, 33.0]
                equivalent.positions[0] += 4 * equivalent.cell[0]
                equivalent.positions[1] -= 3 * equivalent.cell[1]
                self.assertFalse(archive.add(equivalent, -3.0))
                equivalent.wrap()
                self.assertFalse(archive.add(equivalent, -3.0))
                self.assertTrue(archive.add(equivalent, -2.0))
                buckled = atoms.copy()
                buckled.positions[:, 2] = [6.5, 7.5, 6.5, 7.5]
                self.assertTrue(archive.add(buckled, -3.0))
                dimension = 32 * (4 if orbital == "sp" else 1)
                for entry in archive.entries:
                    self.assertEqual(entry.fp.shape, (4, dimension))
                    self.assertEqual(entry.fp_pooled.shape, (2 * dimension,))
                    self.assertTrue(np.isfinite(entry.fp).all())
                    self.assertTrue(np.isfinite(entry.fp_pooled).all())


if __name__ == "__main__":
    unittest.main()
