"""Slab geometry contracts; no optional fingerprint backend or calculator."""

from dataclasses import replace
import unittest

import numpy as np
from ase import Atoms

from crisp.slab import SlabConfig, prepare_slab, validate_slab


class SlabTests(unittest.TestCase):
    def setUp(self):
        self.config = SlabConfig(
            area_per_atom_range=(2.0, 8.0), initial_thickness=1.0,
            max_thickness=4.0, cell_height=20.0, min_vacuum=12.0,
        )

    def atoms(self, z=(0.0, 0.0, 0.0)):
        return Atoms(
            "C3", positions=[[0.2, 0.3, z[0]], [1.4, 0.8, z[1]],
                             [2.1, 1.7, z[2]]],
            cell=[[4.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 8.0]],
            pbc=True,
        )

    def test_prepare_planar_sheet_returns_copy_and_preserves_input(self):
        atoms = self.atoms()
        before = atoms.copy()
        slab = prepare_slab(atoms, self.config)

        self.assertIsNot(slab, atoms)
        np.testing.assert_array_equal(atoms.positions, before.positions)
        np.testing.assert_array_equal(atoms.cell, before.cell)
        np.testing.assert_array_equal(atoms.pbc, before.pbc)
        np.testing.assert_array_equal(slab.numbers, atoms.numbers)
        np.testing.assert_allclose(slab.cell[:2], atoms.cell[:2])
        np.testing.assert_array_equal(slab.cell[2], [0.0, 0.0, 20.0])
        np.testing.assert_array_equal(slab.pbc, [True, True, False])
        np.testing.assert_allclose(slab.positions[:, 2], 10.0)
        validate_slab(slab, self.config)

    def test_buckling_and_distances_survive_padding_and_xy_wrapping(self):
        atoms = self.atoms(z=(-1.0, -0.5, 2.0))
        atoms.positions[0] += atoms.cell[0] * 2
        atoms.positions[1] -= atoms.cell[1]
        physical = atoms.copy()
        physical.pbc = (True, True, False)
        distances = physical.get_all_distances(mic=True)

        slab = prepare_slab(atoms, self.config)

        np.testing.assert_allclose(slab.get_all_distances(mic=True), distances)
        np.testing.assert_allclose(slab.positions[:, 2], [8.5, 9.0, 11.5])
        scaled = slab.get_scaled_positions(wrap=False)[:, :2]
        self.assertTrue(np.all(scaled >= 0) and np.all(scaled < 1))
        self.assertAlmostEqual(np.ptp(slab.positions[:, 2]), 3.0)

    def test_prepare_is_idempotent_and_independent_of_z_origin(self):
        atoms = self.atoms(z=(-1.0, 0.0, 1.0))
        slab = prepare_slab(atoms, self.config)
        atoms.positions[:, 2] += 50.0
        translated = prepare_slab(atoms, self.config)
        repeated = prepare_slab(slab, self.config)
        np.testing.assert_allclose(translated.positions, slab.positions)
        np.testing.assert_allclose(repeated.positions, slab.positions)
        np.testing.assert_array_equal(repeated.cell, slab.cell)

    def test_zero_input_height_is_allowed(self):
        atoms = self.atoms(z=(-1.0, 0.0, 1.0))
        atoms.cell[2] = 0.0
        slab = prepare_slab(atoms, self.config)
        self.assertEqual(slab.cell[2, 2], 20.0)
        np.testing.assert_allclose(slab.positions[:, 2], [9.0, 10.0, 11.0])

    def test_validate_allows_rigid_z_translation_without_mutation(self):
        slab = prepare_slab(self.atoms(), self.config)
        slab.positions[:, 2] += 30.0
        before = slab.positions.copy()
        validate_slab(slab, self.config)
        np.testing.assert_array_equal(slab.positions, before)

    def test_invalid_configurations_are_rejected(self):
        cases = [
            {"area_per_atom_range": (2.0,)},
            {"area_per_atom_range": (0.0, 2.0)},
            {"area_per_atom_range": (4.0, 2.0)},
            {"area_per_atom_range": (2.0, np.inf)},
            {"initial_thickness": -1.0}, {"initial_thickness": 5.0},
            {"max_thickness": np.nan}, {"cell_height": 0.0},
            {"cell_height": 15.0}, {"min_vacuum": 0.0},
            {"pressure_GPa": 1.0}, {"pressure_GPa": -1.0},
            {"pressure_GPa": np.nan},
        ]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config, **changes)

    def test_config_copies_mutable_area_bounds(self):
        bounds = [2.0, 8.0]
        config = replace(self.config, area_per_atom_range=bounds)
        bounds[0] = -1.0
        self.assertEqual(config.area_per_atom_range, (2.0, 8.0))

    def test_geometry_at_bounds_is_accepted(self):
        config = replace(self.config, area_per_atom_range=(4.0, 4.0),
                         min_vacuum=16.0)
        slab = prepare_slab(self.atoms(z=(-2.0, 0.0, 2.0)), config)
        validate_slab(slab, config)

    def test_excess_thickness_is_rejected_without_wrapping_z(self):
        # The old z cell is 8 A: these sites must not be wrapped together.
        with self.assertRaisesRegex(ValueError, "thickness"):
            prepare_slab(self.atoms(z=(0.0, 0.0, 8.1)), self.config)

    def test_area_outside_bounds_is_rejected(self):
        for factor in (0.5, 2.0):
            atoms = self.atoms()
            atoms.cell[:2] *= factor
            with self.subTest(factor=factor), self.assertRaisesRegex(ValueError, "area"):
                prepare_slab(atoms, self.config)

    def test_invalid_cells_are_rejected(self):
        for i, j, value in [(0, 2, 0.2), (1, 2, 0.2), (2, 0, 0.2),
                             (2, 1, 0.2), (0, 0, 0.0), (0, 0, -4.0),
                             (2, 2, -1.0), (1, 1, np.nan)]:
            atoms = self.atoms()
            atoms.cell[i, j] = value
            with self.subTest(cell=(i, j, value)), self.assertRaises(ValueError):
                prepare_slab(atoms, self.config)

    def test_empty_and_nonfinite_positions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            prepare_slab(Atoms(), self.config)
        for value in (np.nan, np.inf):
            atoms = self.atoms()
            atoms.positions[0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                prepare_slab(atoms, self.config)

    def test_validate_rejects_wrong_periodicity_and_changed_height(self):
        slab = prepare_slab(self.atoms(), self.config)
        slab.pbc = True
        with self.assertRaisesRegex(ValueError, "PBC"):
            validate_slab(slab, self.config)
        slab.pbc = (True, True, False)
        slab.cell[2, 2] += 1e-5
        with self.assertRaisesRegex(ValueError, "height"):
            validate_slab(slab, self.config)


if __name__ == "__main__":
    unittest.main()
