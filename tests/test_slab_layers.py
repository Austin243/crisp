"""Initial placement contracts, periodic contacts, and reconstructed stacks."""

from dataclasses import replace
import json
import unittest

import numpy as np
from ase import Atoms

from crisp.slab import SlabConfig
from crisp.slab_generation import SlabGenerationError
from crisp.slab_layers import InitialLayer, LayeredSlabSpec, generate_layered_slabs, validate_layered_slab
from crisp.slab_validation import validate_slab_candidate


def stack_spec(symbols=("C", "C"), **options):
    defaults = dict(layers=tuple(InitialLayer({symbol: 2}) for symbol in symbols),
                    config=SlabConfig((0.5, 20), initial_thickness=0, max_thickness=8,
                                      cell_height=24, min_vacuum=12),
                    initial_layer_gaps=(4,) * (len(symbols) - 1), min_dist_ang=1.0)
    return LayeredSlabSpec(**(defaults | options))


def sheet_stack(spec=None):
    spec = stack_spec() if spec is None else spec
    a = 2.46
    cell = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 24]])
    points = np.array([[0, 0, 0], [2 / 3, 1 / 3, 0]]) @ cell
    positions = np.concatenate([points + [0, 0, 8 + 4 * group] for group in range(len(spec.layers))])
    numbers, groups = spec.initial_identities()
    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=(True, True, False))
    atoms.set_array("initial_atom_id", np.arange(len(atoms)))
    atoms.set_array("initial_layer", groups)
    return atoms


class TestLayerSpec(unittest.TestCase):
    def test_counts_canonical_identities_and_json_roundtrip(self):
        composition = {"N": 2, "B": 2}
        spec = stack_spec(layers=(InitialLayer(composition, 0.5, (1.25, -0.5)), InitialLayer({"C": 2})),
                          initial_layer_gaps=(3,), pair_min_distances={("N", "B"): 1.1})
        composition["B"] = 9
        self.assertEqual(spec.composition, {"B": 2, "N": 2, "C": 2})
        self.assertEqual(spec.natoms, 6)
        self.assertEqual(spec.initial_stack_thickness, 3.5)
        self.assertEqual(spec.layers[0].lateral_offset, (0.25, 0.5))
        numbers, groups = spec.initial_identities()
        np.testing.assert_array_equal(numbers, [5, 5, 7, 7, 6, 6])
        np.testing.assert_array_equal(groups, [0, 0, 0, 0, 1, 1])
        restored = LayeredSlabSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
        self.assertEqual(restored.to_dict(), spec.to_dict())
        with self.assertRaises(TypeError):
            spec.layers[0].composition["B"] = 7

    def test_invalid_layer_inputs(self):
        for composition in ({}, {"X": 2}, {"C": True}, {"C": 1.5}, {"C": -1}):
            with self.subTest(composition=composition), self.assertRaises(ValueError):
                InitialLayer(composition)
        for value in (True, -1, float("nan"), float("inf")):
            with self.subTest(thickness=value), self.assertRaises(ValueError):
                InitialLayer({"C": 2}, value)
        with self.assertRaises(ValueError):
            InitialLayer({"C": 1}, initial_thickness=1)
        with self.assertRaises(ValueError):
            InitialLayer({"C": 2}, lateral_offset=(0, float("inf")))

    def test_invalid_spec_bounds_and_distances(self):
        changes = [dict(layers=()), dict(initial_layer_gaps=(1, 2)),
                   dict(initial_layer_gaps=(-1,)), dict(initial_layer_gaps=(9,)),
                   dict(initial_vacuum_gap=21), dict(min_dist_ang=0), dict(bond_scale=True),
                   dict(pair_min_distances={("C", "N"): 1}),
                   dict(pair_min_distances={("C", "C"): float("nan")})]
        for options in changes:
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                stack_spec(**options)
        with self.assertRaises(ValueError):
            stack_spec(symbols=("B", "N"), pair_min_distances={("B", "N"): 1, ("N", "B"): 2})


class TestLayerGeneration(unittest.TestCase):
    def test_seeded_common_cell_counts_extents_gaps_and_offsets(self):
        spec = stack_spec(layers=(InitialLayer({"B": 2, "N": 2}, 0.6), InitialLayer({"C": 3}, 0.3)),
                          initial_layer_gaps=(3,), initial_vacuum_gap=15,
                          config=SlabConfig((3, 5), 0, 8, 24, 12), min_dist_ang=0.3)
        first = generate_layered_slabs(spec, 3, seed=12)
        second = generate_layered_slabs(spec, 3, seed=12)
        for atoms, repeated in zip(first, second):
            np.testing.assert_array_equal(atoms.positions, repeated.positions)
            np.testing.assert_array_equal(atoms.cell, repeated.cell)
            validate_layered_slab(atoms, spec, check_connectivity=False)
            z0 = atoms.positions[atoms.arrays["initial_layer"] == 0, 2]
            z1 = atoms.positions[atoms.arrays["initial_layer"] == 1, 2]
            self.assertAlmostEqual(np.ptp(z0), 0.6)
            self.assertAlmostEqual(np.ptp(z1), 0.3)
            self.assertAlmostEqual(z1.min() - z0.max(), 3)
            self.assertGreaterEqual(atoms.cell[2, 2] - np.ptp(atoms.positions[:, 2]), 15)
            self.assertEqual(atoms.info["layer_policy"], "initial_only")
        shifted_spec = replace(spec, layers=(spec.layers[0], replace(spec.layers[1], lateral_offset=(0.2, 0.1))))
        original = generate_layered_slabs(spec, 1, seed=33)[0]
        shifted = generate_layered_slabs(shifted_spec, 1, seed=33)[0]
        delta = (shifted.get_scaled_positions() - original.get_scaled_positions())[:, :2]
        np.testing.assert_allclose(delta[:4], 0, atol=1e-14)
        np.testing.assert_allclose((delta[4:] - [0.2, 0.1] + 0.5) % 1 - 0.5, 0, atol=1e-14)

    def test_default_zero_gaps_flat_planes_and_bounded_failure(self):
        spec = stack_spec(initial_layer_gaps=(), min_dist_ang=0.1)
        atoms = generate_layered_slabs(spec, 1, seed=5)[0]
        self.assertEqual(np.ptp(atoms.positions[:, 2]), 0)
        self.assertEqual(generate_layered_slabs(spec, 0, seed=5), [])
        impossible = replace(spec, min_dist_ang=100)
        with self.assertRaisesRegex(SlabGenerationError, "0/1.*2 attempts"):
            generate_layered_slabs(impossible, 1, seed=3, max_attempts=2)
        for kwargs in (dict(n=True), dict(n=-1), dict(n=1, max_attempts=0)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate_layered_slabs(spec, **kwargs)


class TestLayerValidation(unittest.TestCase):
    def test_separated_sheets_accepted_original_single_sheet_policy_unchanged(self):
        spec = stack_spec()
        atoms = sheet_stack(spec)
        validate_layered_slab(atoms, spec)
        with self.assertRaisesRegex(ValueError, "disconnected"):
            validate_slab_candidate(atoms, spec.config, min_dist_ang=1)
        before = atoms.copy()
        moved = atoms[[3, 0, 2, 1]]
        moved.positions += [21, -10, 30]
        validate_layered_slab(moved, spec)
        np.testing.assert_array_equal(atoms.positions, before.positions)

    def test_reconstruction_can_change_initial_group_chemistry_and_merge_planes(self):
        spec = stack_spec(symbols=("B", "N"))
        atoms = sheet_stack(spec)
        # Swap only positions: original atom identities retain species, while
        # the two spatial sheets now both contain B and N.
        atoms.positions[[1, 2]] = atoms.positions[[2, 1]]
        validate_layered_slab(atoms, spec)
        # Three atomic placement groups need not be independent final sheets.
        merged_spec = stack_spec(layers=(InitialLayer({"C": 1}), InitialLayer({"C": 1})),
                                 initial_layer_gaps=(1,))
        merged = sheet_stack(stack_spec(symbols=("C",)))
        merged.set_array("initial_layer", np.array([0, 1]))
        validate_layered_slab(merged, merged_spec)

    def test_detached_fragments_and_chains_are_rejected_but_preflight_allows_them(self):
        for cell in ([8, 8, 24], [1.4, 8, 24]):
            spec = stack_spec(layers=(InitialLayer({"C": 1}),), initial_layer_gaps=(),
                              config=SlabConfig((0.1, 100), 0, 8, 24, 12))
            atoms = Atoms("C", positions=[[0, 0, 12]], cell=cell, pbc=(True, True, False))
            atoms.set_array("initial_atom_id", np.array([0]))
            atoms.set_array("initial_layer", np.array([0]))
            validate_layered_slab(atoms, spec, check_connectivity=False)
            with self.assertRaisesRegex(ValueError, "[01]D"):
                validate_layered_slab(atoms, spec)

    def test_detached_molecule_cannot_hide_beside_valid_sheet(self):
        spec = stack_spec()
        atoms = sheet_stack(spec)
        atoms.positions[3] = atoms.positions[2] + [0.1, 0, 1.4]
        with self.assertRaises(ValueError):
            validate_layered_slab(atoms, spec)

    def test_provenance_is_validated_after_order_changes(self):
        spec = stack_spec(symbols=("B", "N"))
        for field, replacement in (("initial_atom_id", [0, 0, 2, 3]),
                                   ("initial_layer", [0, 1, 1, 1])):
            atoms = sheet_stack(spec)
            atoms.arrays[field][:] = replacement
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "initial"):
                validate_layered_slab(atoms, spec)
        atoms = sheet_stack(spec)
        atoms.numbers[[0, 2]] = atoms.numbers[[2, 0]]
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_layered_slab(atoms, spec)
        atoms = sheet_stack(spec)
        del atoms.arrays["initial_atom_id"]
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_layered_slab(atoms, spec)
        validate_layered_slab(atoms, spec, check_provenance=False)

    def test_pair_policy_overrides_scalar_in_both_directions(self):
        spec = stack_spec(symbols=("B", "N"), pair_min_distances={("B", "B"): 1.5})
        with self.assertRaisesRegex(ValueError, "minimum distance"):
            validate_layered_slab(sheet_stack(spec), spec)
        spec = stack_spec(symbols=("B", "N"), min_dist_ang=1.5,
                          pair_min_distances={("B", "B"): 1, ("N", "N"): 1})
        validate_layered_slab(sheet_stack(spec), spec)
        mono = stack_spec(layers=(InitialLayer({"C": 1}),), initial_layer_gaps=(),
                          config=SlabConfig((0.1, 100), 0, 8, 24, 12))
        atoms = Atoms("C", positions=[[0, 0, 12]], cell=[0.8, 2, 24], pbc=(True, True, False))
        with self.assertRaisesRegex(ValueError, "minimum distance"):
            validate_layered_slab(atoms, mono, check_provenance=False, check_connectivity=False)

    def test_geometry_composition_and_vacuum_are_not_repaired(self):
        spec = stack_spec()
        for kind in ("height", "pbc", "thickness", "composition"):
            atoms = sheet_stack(spec)
            if kind == "height":
                atoms.cell[2, 2] = 25
            elif kind == "pbc":
                atoms.pbc = True
            elif kind == "thickness":
                atoms.positions[2:, 2] += 9
            else:
                atoms.numbers[0] = 7
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_layered_slab(atoms, spec)


if __name__ == "__main__":
    unittest.main()
