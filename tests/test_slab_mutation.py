"""Bounded slab mutations without PyXtal, fingerprints or energy evaluation."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import unittest
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.neighborlist import neighbor_list

from crisp.slab import SlabConfig, prepare_slab, validate_slab
from crisp.slab_generation import SlabGenerationError
from crisp.slab_mutation import SlabMutationError, SlabMutations, mutate_slab


def _config():
    return SlabConfig((7.0, 11.0), initial_thickness=0.6, max_thickness=2.0,
                      cell_height=20.0, min_vacuum=12.0)


def _parent():
    return Atoms("BNBN", positions=[[1.5, 1.5, 9.7], [4.5, 1.5, 10.3],
                                   [1.5, 4.5, 9.7], [4.5, 4.5, 10.3]],
                 cell=[6.0, 6.0, 20.0], pbc=(True, True, False))


class TestSlabMutation(unittest.TestCase):
    def test_configuration_validation_and_ownership(self):
        invalid = [
            {"max_displacement": value} for value in
            (-1, np.nan, np.inf, 1j, True, "0.2", [0.2])
        ] + [
            {"max_strain": value} for value in (-1, 1, np.nan, np.inf, 1j, True)
        ] + [
            {name: value} for name in ("random_every", "max_attempts")
            for value in (0, -1, 1.5, True)
        ] + [{"max_displacement": 0, "max_strain": 0}]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                SlabMutations(**options)
        options = SlabMutations(max_displacement=np.float64(0.2),
                                random_every=np.int64(2))
        self.assertIs(type(options.max_displacement), float)
        self.assertIs(type(options.random_every), int)
        with self.assertRaises(FrozenInstanceError):
            options.max_strain = 0.5

    def test_parent_and_options_fail_before_sampling(self):
        cases = []
        wrong_pbc = _parent()
        wrong_pbc.pbc = True
        cases.append(wrong_pbc)
        wrong_height = _parent()
        wrong_height.cell[2, 2] = 19.0
        cases.append(wrong_height)
        constrained = _parent()
        constrained.set_constraint(FixAtoms(indices=[0]))
        cases.append(constrained)
        overlap = _parent()
        overlap.positions[1] = overlap.positions[0]
        cases.append(overlap)
        with patch("crisp.slab_mutation.np.random.default_rng") as rng:
            for parent in cases:
                with self.subTest(parent=parent), self.assertRaises(ValueError):
                    mutate_slab(parent, _config(), SlabMutations())
            for distance in (0, -1, np.nan, np.inf, 1j, True):
                with self.subTest(distance=distance), self.assertRaises(ValueError):
                    mutate_slab(_parent(), _config(), SlabMutations(), min_dist_ang=distance)
            with self.assertRaises(TypeError):
                mutate_slab(_parent(), _config(), None)
        rng.assert_not_called()

    def test_reproducible_independent_copy_and_no_calculator_use(self):
        parent = _parent()
        parent.set_tags([1, 2, 3, 4])
        parent.info = dict(origin="random", layer_group=2, generation_seed=7,
                           pyxtal_version="test", slab_quench_steps=10,
                           slab_quench_fmax=0.05, custom={"values": [1, 2]})
        parent.calc = SinglePointCalculator(parent, energy=-1.0)
        original, info, calc = parent.copy(), deepcopy(parent.info), parent.calc
        state = np.random.get_state()
        with patch.object(calc, "get_potential_energy", side_effect=AssertionError("evaluation")):
            first = mutate_slab(parent, _config(), SlabMutations(), seed=17)
            second = mutate_slab(parent, _config(), SlabMutations(),
                                 seed=np.random.default_rng(17))
        after = np.random.get_state()
        np.testing.assert_array_equal(first.cell, second.cell)
        np.testing.assert_array_equal(first.positions, second.positions)
        self.assertEqual(first.info, second.info)
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        self.assertIsNone(first.calc)
        self.assertEqual(first.info, {"custom": {"values": [1, 2]},
                                      "origin": "mutation", "mutation_attempt": 1})
        first.info["custom"]["values"].append(3)
        first.arrays["tags"][0] = 999
        first.positions[0] += 1
        np.testing.assert_array_equal(parent.positions, original.positions)
        np.testing.assert_array_equal(parent.cell, original.cell)
        np.testing.assert_array_equal(parent.arrays["tags"], original.arrays["tags"])
        self.assertEqual(parent.info, info)
        self.assertIs(parent.calc, calc)

    def test_strain_is_symmetric_bounded_and_only_affine_in_xy(self):
        parent = _parent()
        options = SlabMutations(max_displacement=0, max_strain=0.15)
        for seed in range(12):
            with self.subTest(seed=seed):
                child = mutate_slab(parent, _config(), options, seed=seed)
                deformation = np.linalg.solve(parent.cell[:2, :2], child.cell[:2, :2])
                strain = deformation - np.eye(2)
                np.testing.assert_allclose(strain, strain.T, atol=1e-14)
                self.assertLessEqual(np.linalg.norm(strain), options.max_strain)
                self.assertGreater(np.linalg.eigvalsh(deformation).min(), 0)
                np.testing.assert_allclose(child.positions[:, :2],
                                           parent.positions[:, :2] @ deformation, atol=1e-14)
                np.testing.assert_array_equal(child.positions[:, 2], parent.positions[:, 2])
                np.testing.assert_array_equal(child.cell[2], parent.cell[2])
                np.testing.assert_array_equal(child.cell[:2, 2], 0)
                np.testing.assert_array_equal(child.pbc, (True, True, False))
                np.testing.assert_array_equal(child.numbers, parent.numbers)
                validate_slab(child, _config())
                self.assertEqual(len(neighbor_list("d", child, 1.5)), 0)

    def test_planar_displacements_stay_in_disk_and_do_not_change_cell(self):
        parent = _parent()
        parent.positions[:, 2] = 10.0
        config = replace(_config(), initial_thickness=0, max_thickness=0)
        options = SlabMutations(max_displacement=0.2, max_strain=0)
        for seed in range(12):
            child = mutate_slab(parent, config, options, seed=seed)
            distances = np.linalg.norm(child.positions - parent.positions, axis=1)
            self.assertTrue(np.all(distances <= 0.2 + 1e-14))
            self.assertTrue(np.all(distances > 0))
            np.testing.assert_array_equal(child.positions[:, 2], 10.0)
            np.testing.assert_array_equal(child.cell, parent.cell)

    def test_buckled_displacement_bound_precedes_rigid_centering(self):
        parent = _parent()
        options = SlabMutations(max_displacement=0.2, max_strain=0)
        raw = []

        def capture(atoms, config):
            raw.append(atoms.copy())
            return prepare_slab(atoms, config)

        with patch("crisp.slab_mutation.prepare_slab", side_effect=capture):
            child = mutate_slab(parent, _config(), options, seed=19)
        self.assertEqual(len(raw), 1)
        moves = raw[0].positions - parent.positions
        self.assertTrue(np.all(np.linalg.norm(moves, axis=1) <= 0.2))
        self.assertTrue(np.any(np.abs(moves[:, 2]) > 0.01))
        self.assertFalse(np.isclose(np.ptp(child.positions[:, 2]),
                                    np.ptp(parent.positions[:, 2])))
        self.assertAlmostEqual(child.positions[:, 2].min() + child.positions[:, 2].max(), 20)
        np.testing.assert_array_equal(child.cell, parent.cell)
        validate_slab(child, _config())

    def test_self_image_distance_rejection_and_bounded_retry(self):
        parent = Atoms("Cu", positions=[[1, 1, 10]], cell=[2, 2, 20], pbc=(True, True, False))
        config = SlabConfig((1, 7), 0, 0, 20, 12)
        options = SlabMutations(max_displacement=0, max_strain=0.9, max_attempts=1)
        with self.assertRaisesRegex(SlabMutationError, "1 attempts.*min_dist_ang") as rejected:
            mutate_slab(parent, config, options, seed=0, min_dist_ang=1.9)
        self.assertIsInstance(rejected.exception, SlabGenerationError)
        child = mutate_slab(parent, config, replace(options, max_attempts=20),
                            seed=0, min_dist_ang=1.9)
        self.assertEqual(child.info["mutation_attempt"], 2)
        self.assertEqual(len(neighbor_list("d", child, 1.9)), 0)

    def test_distinct_atom_distance_rejection(self):
        parent = Atoms("BN", positions=[[1, 2, 10], [2.55, 2, 10]],
                       cell=[4, 4, 20], pbc=(True, True, False))
        config = SlabConfig((6, 10), 0, 0, 20, 12)
        options = SlabMutations(max_displacement=0.5, max_strain=0, max_attempts=1)
        with self.assertRaisesRegex(SlabMutationError, "min_dist_ang"):
            mutate_slab(parent, config, options, seed=1)

    def test_area_exhaustion_is_bounded_and_preserves_parent(self):
        parent = _parent()
        original = parent.copy()
        config = replace(_config(), area_per_atom_range=(9.0, 9.0))
        options = SlabMutations(max_displacement=0, max_strain=0.1, max_attempts=3)
        with patch("crisp.slab_mutation.prepare_slab", wraps=prepare_slab) as prepare:
            with self.assertRaisesRegex(SlabMutationError, "3 attempts.*area"):
                mutate_slab(parent, config, options, seed=4)
        self.assertEqual(prepare.call_count, 3)
        np.testing.assert_array_equal(parent.positions, original.positions)
        np.testing.assert_array_equal(parent.cell, original.cell)

    def test_excess_thickness_is_rejected_without_changing_vacuum_cell(self):
        parent = _parent()
        parent.positions[:, 2] += 40
        original = parent.copy()
        config = replace(_config(), max_thickness=0.6)
        options = SlabMutations(max_displacement=0.2, max_strain=0, max_attempts=1)
        with self.assertRaisesRegex(SlabMutationError, "1 attempts.*thickness"):
            mutate_slab(parent, config, options, seed=0)
        np.testing.assert_array_equal(parent.positions, original.positions)
        np.testing.assert_array_equal(parent.cell, original.cell)

    def test_unexpected_errors_propagate_without_retry(self):
        with patch("crisp.slab_mutation.prepare_slab", side_effect=TypeError("API bug")) as prepare:
            with self.assertRaisesRegex(TypeError, "API bug"):
                mutate_slab(_parent(), _config(), SlabMutations(), seed=0)
        prepare.assert_called_once()


if __name__ == "__main__":
    unittest.main()
