"""Opt-in slab generation: contracts, real layer groups and rejection paths."""

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from ase.neighborlist import neighbor_list

from crisp.slab import SlabConfig, validate_slab
from crisp.slab_generation import _sample_cell, generate_slabs

try:
    import pyxtal
except ImportError:
    pyxtal = None


def _config():
    return SlabConfig((4.0, 8.0), initial_thickness=1.0, max_thickness=4.0,
                      cell_height=20.0, min_vacuum=12.0)


class TestSlabGenerationInputs(unittest.TestCase):
    def test_invalid_requests_fail_before_loading_pyxtal(self):
        cases = [
            {"composition": c} for c in
            ({}, {"invalid": 1}, {"X": 1}, {"C": 0}, {"C": -1},
             {"C": 1.5}, {"C": True})
        ] + [
            {"n": -1}, {"n": 1.5}, {"n": True},
            {"min_dist_ang": 0}, {"min_dist_ang": np.inf},
            {"min_dist_ang": np.nan}, {"max_attempts": 0},
            {"max_attempts": 2.5}, {"max_attempts": True},
            {"layer_groups": []}, {"layer_groups": [0]},
            {"layer_groups": [81]}, {"layer_groups": [True]},
            {"layer_groups": [1.5]},
        ]
        with patch.dict("sys.modules", {"pyxtal": None}):
            for changes in cases:
                args = dict(composition={"C": 4}, config=_config(), n=1)
                args.update(changes)
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    generate_slabs(**args)

    def test_empty_batch_needs_no_optional_backend(self):
        with patch.dict("sys.modules", {"pyxtal": None}):
            self.assertEqual(generate_slabs({"C": 4}, _config(), 0), [])

    def test_missing_backend_has_install_hint(self):
        with patch.dict("sys.modules", {"pyxtal": None}):
            with self.assertRaisesRegex(ImportError, r"\[search\] extra"):
                generate_slabs({"C": 4}, _config(), 1)


@unittest.skipUnless(pyxtal is not None, "PyXtal required")
class TestSlabGeneration(unittest.TestCase):
    def test_cell_metric_preserves_every_layer_group_point_operation(self):
        from pyxtal.symmetry import Group

        rng = np.random.default_rng(17)
        for number in range(1, 81):
            group = Group(number, dim=2)
            cell = _sample_cell(group, area=24.0, thickness=1.0, rng=rng)
            metric = cell @ cell.T
            with self.subTest(group=number):
                for operation in group[0].ops:
                    rotation = operation.rotation_matrix
                    np.testing.assert_allclose(rotation.T @ metric @ rotation,
                                               metric, atol=1e-8, rtol=0)

    def test_elemental_and_binary_batches_satisfy_contract(self):
        config = _config()
        for composition in ({"C": 4}, {"B": 2, "N": 2}):
            with self.subTest(composition=composition):
                slabs = generate_slabs(composition, config, 3, seed=7,
                                       min_dist_ang=1.0)
                self.assertEqual(len(slabs), 3)
                for atoms in slabs:
                    self.assertEqual(Counter(atoms.get_chemical_symbols()), composition)
                    validate_slab(atoms, config)
                    self.assertEqual(len(neighbor_list("d", atoms, 1.0)), 0)
                    z = atoms.positions[:, 2]
                    self.assertAlmostEqual((z.min() + z.max()) / 2, 10.0)
                    self.assertEqual(atoms.info["origin"], "random")
                    self.assertIn(atoms.info["layer_group"], range(1, 81))
                    self.assertIsInstance(atoms.info["generation_seed"], int)
                    self.assertEqual(atoms.info["pyxtal_version"], pyxtal.__version__)
                    self.assertNotIn("spacegroup", atoms.info)

    def test_seed_reproduces_batch_independent_of_input_order_and_global_rng(self):
        state = np.random.get_state()
        args = dict(config=_config(), n=2, seed=17, min_dist_ang=1.0)
        first = generate_slabs({"B": 2, "N": 2}, layer_groups=[2, 1, 2], **args)
        args["seed"] = np.random.default_rng(17)
        second = generate_slabs({"N": 2, "B": 2}, layer_groups=[1, 2], **args)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a.numbers, b.numbers)
            np.testing.assert_array_equal(a.positions, b.positions)
            np.testing.assert_array_equal(a.cell, b.cell)
            self.assertEqual(a.info, b.info)
        args["seed"] = 18
        different = generate_slabs({"B": 2, "N": 2}, layer_groups=[1, 2], **args)
        self.assertFalse(np.array_equal(first[0].cell, different[0].cell))

    def test_planar_request_and_fixed_area(self):
        config = replace(_config(), initial_thickness=0.0, max_thickness=0.0,
                         area_per_atom_range=(6.0, 6.0))
        for group in (31, 80):
            with self.subTest(group=group):
                atoms = generate_slabs({"C": 4}, config, 1, seed=7,
                                       layer_groups=[group], min_dist_ang=1.0)[0]
                validate_slab(atoms, config)
                np.testing.assert_array_equal(atoms.positions[:, 2], 10.0)
                self.assertAlmostEqual(np.linalg.det(atoms.cell[:2, :2]) / 4, 6.0)
                self.assertEqual(len(neighbor_list("d", atoms, 1.0)), 0)

    def test_buckled_inversion_group_preserves_finite_z_symmetry(self):
        # Inversion must match using xy images only, without any z wrapping.
        slabs = generate_slabs({"C": 4}, _config(), 3, seed=7,
                               layer_groups=[2], min_dist_ang=1.0)
        self.assertTrue(any(np.ptp(a.positions[:, 2]) > 0.5 for a in slabs))
        for atoms in slabs:
            centered = atoms.get_scaled_positions(wrap=False)
            centered[:, 2] -= 0.5
            delta = -centered[:, None, :] - centered[None, :, :]
            delta[:, :, :2] -= np.rint(delta[:, :, :2])
            distances = np.linalg.norm(delta @ atoms.cell.array, axis=2)
            np.testing.assert_allclose(distances.min(axis=1), 0.0, atol=1e-8)

    def test_incompatible_group_is_rejected_before_generation(self):
        with patch("pyxtal.pyxtal.from_random") as generate:
            with self.assertRaisesRegex(ValueError, "compatible"):
                generate_slabs({"C": 1}, _config(), 1, layer_groups=[31])
        generate.assert_not_called()

    def test_attempt_limit_and_unexpected_errors(self):
        with patch("pyxtal.pyxtal.from_random", side_effect=RuntimeError("no sites")) as generate:
            with self.assertRaisesRegex(RuntimeError, "0/2 slabs after 3 attempts.*no sites"):
                generate_slabs({"C": 4}, _config(), 2, layer_groups=[1], max_attempts=3)
        self.assertEqual(generate.call_count, 3)
        with patch("pyxtal.pyxtal.from_random", side_effect=TypeError("API bug")) as generate:
            with self.assertRaisesRegex(TypeError, "API bug"):
                generate_slabs({"C": 4}, _config(), 1, layer_groups=[1])
        generate.assert_called_once()

    def test_generated_geometry_and_composition_are_rechecked(self):
        # Mock only PyXtal's output to force defects its native checks may miss.
        cases = [
            ("composition", _config(), {"C": 1}, [[0, 0, 0], [0.5, 0.5, 0]]),
            ("thickness", _config(), {"C": 2}, [[0, 0, -3], [0.5, 0.5, 3]]),
            ("min_dist_ang", replace(_config(), area_per_atom_range=(0.25, 0.25)),
             {"C": 1}, [[0, 0, 0]]),  # Periodic self-image, not a distinct atom.
            ("min_dist_ang", replace(_config(), initial_thickness=0.0),
             {"C": 2}, [[0, 0, -0.25], [0, 0, 0.25]]),  # Flattening overlap.
        ]
        for message, config, composition, coords in cases:
            def output(xtal, **kwargs):
                xtal.lattice = kwargs["lattice"]
                xtal.atom_sites = [SimpleNamespace(specie="C", coords=np.array(coords))]

            with self.subTest(message=message, composition=composition):
                with patch("pyxtal.pyxtal.from_random", autospec=True, side_effect=output) as generate:
                    with self.assertRaisesRegex(RuntimeError, "2 attempts.*" + message):
                        generate_slabs(composition, config, 1, seed=7,
                                       layer_groups=[1], min_dist_ang=1.0, max_attempts=2)
                self.assertEqual(generate.call_count, 2)

    def test_exhaustion_does_not_return_a_partial_batch(self):
        with patch("pyxtal.pyxtal.from_random", autospec=True) as generate:
            # First call succeeds; the remaining call fails.
            def first_then_fail(xtal, **kwargs):
                if generate.call_count == 1:
                    xtal.lattice = kwargs["lattice"]
                    xtal.atom_sites = [SimpleNamespace(specie="C", coords=np.array([[0, 0, 0]]))]
                else:
                    raise RuntimeError("no more sites")
            generate.side_effect = first_then_fail
            with self.assertRaisesRegex(RuntimeError, "1/2 slabs after 2 attempts.*no more sites"):
                generate_slabs({"C": 1}, _config(), 2, seed=7,
                               layer_groups=[1], min_dist_ang=1.0, max_attempts=2)


if __name__ == "__main__":
    unittest.main()
