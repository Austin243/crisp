"""Physical minimum distances and connected, doubly periodic slab networks."""

from copy import deepcopy
import unittest
from unittest.mock import Mock

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms

from crisp.slab import SlabConfig
from crisp.slab_validation import (
    _check_periodic_connectivity,
    validate_slab_candidate,
)


def _config():
    return SlabConfig((0.1, 100.0), initial_thickness=1.0,
                      max_thickness=8.0, cell_height=20.0, min_vacuum=10.0)


def _sheet(symbols="C2", buckling=0.0):
    a = 2.46
    atoms = Atoms(symbols, scaled_positions=[[0, 0, 0.5], [2 / 3, 1 / 3, 0.5]],
                  cell=[[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 20]],
                  pbc=(True, True, False))
    atoms.positions[:, 2] += [-buckling / 2, buckling / 2]
    return atoms


def _slab(symbols, positions, xy=(8.0, 8.0)):
    return Atoms(symbols, positions=positions, cell=[*xy, 20.0],
                 pbc=(True, True, False))


class TestSlabValidation(unittest.TestCase):
    def test_graphene_and_buckled_binary_sheet_are_valid(self):
        for atoms in (_sheet(), _sheet("BN", buckling=0.5)):
            with self.subTest(symbols=atoms.get_chemical_formula()):
                self.assertIsNone(validate_slab_candidate(atoms, _config(),
                                                         min_dist_ang=1.0))

    def test_supercells_and_coordinate_representations_preserve_validity(self):
        base = _sheet("BN", buckling=0.5)
        # Construct the supercell explicitly: ASE 3.22's repeat() calls
        # np.product, which NumPy 2 removed.
        positions = np.concatenate([base.positions + x * base.cell[0] + y * base.cell[1]
                                    for x in range(3) for y in range(2)])
        repeated = Atoms("BN" * 6, positions=positions,
                         cell=base.cell.array * np.array([3, 2, 1])[:, None],
                         pbc=(True, True, False))
        translated = repeated.copy()
        translated.positions += [17.3, -8.1, 33.0]
        wrapped = translated.copy()
        wrapped.wrap()
        rebased = base.copy()
        # A unimodular change of xy basis describes exactly the same lattice.
        cell = rebased.cell.array.copy()
        cell[1] += 2 * cell[0]
        rebased.set_cell(cell, scale_atoms=False)
        rebased.wrap()
        permuted = repeated[np.arange(len(repeated))[::-1]]
        for atoms in (base, repeated, translated, wrapped, rebased, permuted):
            with self.subTest(natoms=len(atoms), cell=atoms.cell):
                self.assertIsNone(validate_slab_candidate(atoms, _config(),
                                                         min_dist_ang=1.0))

    def test_finite_clusters_including_boundary_crossing_dimers_are_rejected(self):
        cases = [
            _slab("C", [[0, 0, 10]]),
            _slab("C2", [[0, 0, 10], [1.4, 0, 10]]),
            _slab("C2", [[0.7, 0, 10], [7.3, 0, 10]]),
        ]
        for atoms in cases:
            with self.subTest(positions=atoms.positions), self.assertRaises(ValueError):
                validate_slab_candidate(atoms, _config(), min_dist_ang=1.0)

    def test_axis_and_diagonal_periodic_chains_are_rejected(self):
        cases = [
            _slab("C", [[0, 0, 10]], xy=(1.4, 8.0)),
            _slab("C4", [[k, k, 10] for k in range(4)], xy=(4.0, 4.0)),
        ]
        for atoms in cases:
            with self.subTest(cell=atoms.cell), self.assertRaises(ValueError):
                validate_slab_candidate(atoms, _config(), min_dist_ang=1.0)

    def test_disconnected_sheets_and_adsorbates_are_rejected(self):
        lower = _sheet()
        upper = _sheet()
        upper.positions[:, 2] += 4.0
        bilayer = lower + upper
        adsorbate = lower + Atoms("C", positions=[[0, 0, 14]])
        for atoms in (bilayer, adsorbate):
            with self.subTest(natoms=len(atoms)), self.assertRaises(ValueError):
                validate_slab_candidate(atoms, _config(), min_dist_ang=1.0)

    def test_interpenetrating_nets_are_rejected_despite_a_connected_cell_graph(self):
        # Opposite helical chains with y-periodic ribs form two separate nets.
        # Modulo the cell they appear connected and span both xy directions.
        positions = []
        for k in range(12):
            t = k / 12
            for sign in (-1, 1):
                positions.append([10 * t, sign * 3 * np.cos(np.pi * t),
                                  sign * 3 * np.sin(np.pi * t)])
        for z in (-3, 3):
            positions.extend([5, k * 12 / 16, z] for k in range(1, 16))
        atoms = Atoms("C54", positions=np.array(positions) * 1.4,
                      cell=[14.0, 16.8, 30.0], pbc=(True, True, False))
        config = SlabConfig((4.0, 5.0), initial_thickness=1.0,
                            max_thickness=9.0, cell_height=30.0, min_vacuum=10.0)
        with self.assertRaisesRegex(ValueError, "2 disconnected periodic nets"):
            validate_slab_candidate(atoms, config, min_dist_ang=1.0)

    def test_short_contacts_include_periodic_copies_of_the_same_atom(self):
        overlapping = _sheet()
        overlapping.positions[1] = overlapping.positions[0]
        boundary_collision = _slab("C2", [[0.1, 0, 10], [7.9, 0, 10]])
        self_collision = _slab("C", [[0, 0, 10]], xy=(0.5, 1.4))
        for atoms in (overlapping, boundary_collision, self_collision):
            with self.subTest(positions=atoms.positions), self.assertRaisesRegex(
                    ValueError, "min_dist|distance"):
                validate_slab_candidate(atoms, _config(), min_dist_ang=1.0)

    def test_distance_and_bond_thresholds_are_independent(self):
        # A minimum-distance threshold larger than the bonding cutoff must
        # still detect the short contact; a bond-only neighbor list misses it.
        with self.assertRaisesRegex(ValueError, "min_dist|distance"):
            validate_slab_candidate(_sheet(), _config(), min_dist_ang=1.5,
                                    bond_scale=0.5)
        # Lowering only the bond cutoff disconnects otherwise valid geometry.
        with self.assertRaises(ValueError):
            validate_slab_candidate(_sheet(), _config(), min_dist_ang=1.0,
                                    bond_scale=0.5)

    def test_invalid_options_and_unknown_elements_are_rejected(self):
        for name in ("min_dist_ang", "bond_scale"):
            for value in (0.0, -1.0, np.inf, np.nan):
                options = dict(min_dist_ang=1.0, bond_scale=1.2)
                options[name] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    validate_slab_candidate(_sheet(), _config(), **options)
        for number in (0, 119):
            atoms = _sheet()
            atoms.numbers[0] = number
            with self.subTest(number=number), self.assertRaises(ValueError):
                validate_slab_candidate(atoms, _config(), min_dist_ang=1.0)

    def test_slab_geometry_is_checked(self):
        wrong_pbc = _sheet()
        wrong_pbc.pbc = True
        wrong_cell = _sheet()
        wrong_cell.cell[2, 2] = 21.0
        too_thick = _sheet(buckling=9.0)
        nonfinite = _sheet()
        nonfinite.positions[0, 0] = np.nan
        for atoms in (wrong_pbc, wrong_cell, too_thick, nonfinite):
            with self.subTest(positions=atoms.positions), self.assertRaises(ValueError):
                validate_slab_candidate(atoms, _config(), min_dist_ang=1.0)

    def test_validation_preserves_input_and_never_evaluates_calculator(self):
        for valid in (True, False):
            atoms = _sheet() if valid else _slab("C", [[0, 0, 10]])
            atoms.info = {"origin": "random", "nested": {"value": [1, 2]}}
            atoms.set_constraint(FixAtoms(indices=[0]))
            calculator = Calculator()
            calculator.calculate = Mock(side_effect=AssertionError("Unexpected evaluation"))
            atoms.calc = calculator
            before = atoms.copy()
            info = deepcopy(atoms.info)
            constraints = [constraint.todict() for constraint in atoms.constraints]
            with self.subTest(valid=valid):
                if valid:
                    validate_slab_candidate(atoms, _config(), min_dist_ang=1.0)
                else:
                    with self.assertRaises(ValueError):
                        validate_slab_candidate(atoms, _config(), min_dist_ang=1.0)
                np.testing.assert_array_equal(atoms.positions, before.positions)
                np.testing.assert_array_equal(atoms.cell, before.cell)
                np.testing.assert_array_equal(atoms.pbc, before.pbc)
                np.testing.assert_array_equal(atoms.numbers, before.numbers)
                self.assertEqual(atoms.info, info)
                self.assertEqual([c.todict() for c in atoms.constraints], constraints)
                self.assertIs(atoms.calc, calculator)
                calculator.calculate.assert_not_called()


class TestPeriodicConnectivity(unittest.TestCase):
    def check_single_site(self, vectors):
        # Directed self-image edges are the minimal periodic quotient graph.
        shifts = np.array([[x, y, 0] for x, y in vectors], dtype=np.int64)
        shifts = np.concatenate((shifts, -shifts))
        endpoints = np.zeros(len(shifts), dtype=int)
        return _check_periodic_connectivity(1, endpoints, endpoints, shifts)

    def test_one_atom_net_requires_two_independent_periodic_directions(self):
        self.assertIsNone(self.check_single_site([(1, 0), (0, 1)]))
        for vectors in ([(1, 0)], [(1, 1), (2, 2)]):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                self.check_single_site(vectors)

    def test_rank_two_sublattice_can_still_describe_disconnected_nets(self):
        # Each reachable site has even x+y: the odd coset is a separate net.
        with self.assertRaises(ValueError):
            self.check_single_site([(1, 1), (1, -1)])

    def test_cycle_generators_need_not_contain_a_unimodular_pair(self):
        # Pair determinants are 6, 2, -3. Their gcd is 1, so all xy repeats
        # belong to one net despite no individual pair having determinant 1.
        self.assertIsNone(self.check_single_site([(2, 0), (0, 3), (1, 1)]))

    def test_cycle_arithmetic_preserves_exact_large_integer_offsets(self):
        large = 2 ** 53
        self.assertIsNone(self.check_single_site([(large, 1), (large + 1, 1)]))


if __name__ == "__main__":
    unittest.main()
