"""Slab fingerprints: guards plus independent mixed-PBC/derivative audits."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from crisp.fingerprint import FingerprintCalculator
from crisp.projector_peratom import project_peratom_forces_and_stress
from crisp.slab import SlabConfig, prepare_slab
from crisp.slab_fingerprint import SlabFingerprintCalculator

try:
    import torch_fplib
except ImportError:
    torch_fplib = None


def _config(height=14.0):
    return SlabConfig((1.0, 8.0), initial_thickness=1.0, max_thickness=2.0,
                      cell_height=height, min_vacuum=1.0)


def _slab(config, planar=False):
    atoms = Atoms(
        "BNC", positions=[[0.22, 0.38, -0.3], [1.35, 0.95, 0.62],
                           [2.1, 2.4, -0.1]],
        cell=[[3.8, 0.0, 0.0], [0.9, 3.6, 0.0], [0.0, 0.0, 0.0]],
    )
    if planar:
        atoms.positions[:, 2] = 0.0
    return prepare_slab(atoms, config)


_METHODS = (
    "get_fingerprints", "project_forces", "project_forces_and_stress",
    "get_fingerprints_and_jacobian", "get_fingerprints_jacobian_strain",
)


class TestSlabFingerprintGuards(unittest.TestCase):
    def test_invalid_cutoff_is_rejected(self):
        for cutoff in (0.0, -1.0, np.inf, np.nan):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                SlabFingerprintCalculator(_config(), cutoff=cutoff)

    def test_every_path_rejects_invalid_geometry_before_backend(self):
        config = _config()
        calc = SlabFingerprintCalculator(config, cutoff=3.2, natx=16)
        for invalid in ("pbc", "height", "tilt", "thickness"):
            atoms = _slab(config)
            if invalid == "pbc":
                atoms.pbc = True
            elif invalid == "height":
                atoms.cell[2, 2] += 1.0
            elif invalid == "tilt":
                atoms.cell[2, 0] = 0.1
            else:
                atoms.positions[0, 2] -= 3.0
            self._assert_rejected(calc, atoms)

    def test_every_path_rechecks_actual_gap_and_cutoff_boundary(self):
        config = _config(height=4.0)
        # Configured minimum gap is satisfied, but not the FP image clearance.
        for gap in (3.0, 3.2, 3.2 + 0.5e-6):
            atoms = _slab(config)
            atoms.positions[:, 2] = [1.0, 1.0, 1.0 + config.cell_height - gap]
            calc = SlabFingerprintCalculator(config, cutoff=3.2, natx=16)
            self._assert_rejected(calc, atoms, "image gap")

    def _assert_rejected(self, calc, atoms, message="Slab"):
        for name in _METHODS:
            args = (np.ones((len(atoms), 16)),) if name.startswith("project") else ()
            with self.subTest(method=name):
                with patch.object(FingerprintCalculator, name,
                                  side_effect=AssertionError("backend reached")):
                    with self.assertRaisesRegex(ValueError, message):
                        getattr(calc, name)(atoms, *args)


@unittest.skipUnless(torch_fplib is not None, "torch_fplib required")
class TestSlabFingerprintBackend(unittest.TestCase):
    def setUp(self):
        self.config = _config()
        self.atoms = _slab(self.config)

    def calc(self, orbital="s", config=None, natx=16):
        return SlabFingerprintCalculator(config or self.config, cutoff=3.2,
                                         natx=natx, orbital=orbital)

    @staticmethod
    def reference(atoms, calc):
        # Independent neighbor construction honors physical PBC. This backend
        # helper is a value reference only; it does not preserve autograd.
        i, j, displacement = neighbor_list("ijD", atoms, calc.cutoff)
        return torch_fplib.get_lfp_from_ase_neighbors(
            atoms.positions, atoms.numbers, i, j, displacement,
            cutoff=calc.cutoff, natx=calc.natx, orbital=calc.orbital,
        ).detach().numpy()

    @staticmethod
    def weights(calc, nat=3):
        size = calc.natx * (4 if calc.orbital == "sp" else 1)
        return np.random.default_rng(9).normal(size=(nat, size))

    def test_planar_and_buckled_values_match_mixed_pbc_reference(self):
        for planar in (False, True):
            for orbital in ("s", "sp"):
                with self.subTest(planar=planar, orbital=orbital):
                    atoms = _slab(self.config, planar=planar)
                    calc = self.calc(orbital)
                    np.testing.assert_allclose(calc.get_fingerprints(atoms),
                                               self.reference(atoms, calc),
                                               rtol=1e-9, atol=1e-11)

    def test_xyz_forces_and_full_jacobians_match_reference_finite_differences(self):
        step = 1e-5
        for orbital in ("s", "sp"):
            with self.subTest(orbital=orbital):
                calc = self.calc(orbital)
                weights = self.weights(calc)
                fp, dfp = calc.get_fingerprints_and_jacobian(self.atoms)
                forces = calc.project_forces(self.atoms, weights)
                np.testing.assert_allclose(fp, calc.get_fingerprints(self.atoms),
                                           rtol=1e-10, atol=1e-12)
                projected = -np.einsum("ijkm,im->jk", dfp, weights)
                np.testing.assert_allclose(forces, projected, rtol=1e-8, atol=1e-10)
                fd = np.zeros((len(self.atoms), 3))
                for i in range(len(self.atoms)):
                    for axis in range(3):
                        plus, minus = self.atoms.copy(), self.atoms.copy()
                        plus.positions[i, axis] += step
                        minus.positions[i, axis] -= step
                        fp_plus = self.reference(plus, calc)
                        fp_minus = self.reference(minus, calc)
                        fd[i, axis] = -np.sum(weights * (fp_plus - fp_minus)) / (2 * step)
                np.testing.assert_allclose(forces, fd, rtol=2e-4, atol=2e-7)
                self.assertGreater(np.linalg.norm(forces[:, 2]), 1e-4)

    def test_stress_and_strain_jacobian_match_inplane_finite_differences(self):
        step = 1e-5
        for orbital in ("s", "sp"):
            with self.subTest(orbital=orbital):
                calc = self.calc(orbital)
                weights = self.weights(calc)
                fp, dfp, dfpe = calc.get_fingerprints_jacobian_strain(self.atoms)
                forces, stress = calc.project_forces_and_stress(self.atoms, weights)
                f_full, s_full = project_peratom_forces_and_stress(
                    dfp, dfpe, weights, self.atoms.get_volume())
                np.testing.assert_allclose(fp, self.reference(self.atoms, calc),
                                           rtol=1e-9, atol=1e-11)
                np.testing.assert_allclose(forces, f_full, rtol=1e-8, atol=1e-10)
                np.testing.assert_allclose(stress, s_full, rtol=1e-8, atol=1e-10)
                for v, a, b in ((0, 0, 0), (1, 1, 1), (5, 0, 1)):
                    strain = np.zeros((3, 3))
                    strain[a, b] = step if a == b else step / 2
                    strain[b, a] = strain[a, b]
                    plus, minus = self.atoms.copy(), self.atoms.copy()
                    plus.set_cell(plus.cell @ (np.eye(3) + strain), scale_atoms=True)
                    minus.set_cell(minus.cell @ (np.eye(3) - strain), scale_atoms=True)
                    fd = np.sum(weights * (self.reference(plus, calc)
                                           - self.reference(minus, calc))) / (2 * step)
                    np.testing.assert_allclose(stress[v] * self.atoms.get_volume(),
                                               fd, rtol=2e-4, atol=2e-7)

    def test_all_paths_are_vacuum_and_translation_invariant_without_mutation(self):
        taller_config = replace(self.config, cell_height=20.0)
        translated = prepare_slab(self.atoms, taller_config)
        translated.positions[0] += 5 * translated.cell[0]
        translated.positions[1] -= 4 * translated.cell[1]
        translated.positions[:, 2] += 37.0
        before = translated.copy()
        for orbital in ("s", "sp"):
            calc, taller_calc = self.calc(orbital), self.calc(orbital, taller_config)
            weights = self.weights(calc)
            for name in _METHODS:
                with self.subTest(orbital=orbital, method=name):
                    args = (weights,) if name.startswith("project") else ()
                    expected = getattr(calc, name)(self.atoms, *args)
                    actual = getattr(taller_calc, name)(translated, *args)
                    if name == "project_forces_and_stress":
                        expected = (expected[0], expected[1] * self.atoms.get_volume())
                        actual = (actual[0], actual[1] * translated.get_volume())
                    if not isinstance(expected, tuple):
                        expected, actual = (expected,), (actual,)
                    for left, right in zip(expected, actual):
                        np.testing.assert_allclose(left, right, rtol=1e-8, atol=1e-9)
        np.testing.assert_array_equal(translated.positions, before.positions)
        np.testing.assert_array_equal(translated.cell, before.cell)
        np.testing.assert_array_equal(translated.pbc, before.pbc)

    def test_neighbor_capacity_includes_self_and_periodic_images(self):
        i = neighbor_list("i", self.atoms, 3.2)
        capacity = int(np.bincount(i, minlength=len(self.atoms)).max()) + 1
        calc = self.calc(natx=capacity)
        np.testing.assert_allclose(calc.get_fingerprints(self.atoms),
                                   self.reference(self.atoms, calc), atol=1e-11)
        with self.assertRaisesRegex(ValueError, "Too many neighbors"):
            self.calc(natx=capacity - 1).get_fingerprints(self.atoms)

    def test_inherited_distances_handle_inplane_images(self):
        shifted = self.atoms.copy()
        shifted.positions[0] += 5 * shifted.cell[0]
        calc = self.calc()
        self.assertAlmostEqual(calc.get_distance(self.atoms, shifted), 0.0, places=10)
        distance, assignment = calc.get_distance_and_assignment(self.atoms, shifted)
        self.assertAlmostEqual(distance, 0.0, places=10)
        np.testing.assert_array_equal(assignment, np.arange(len(self.atoms)))


if __name__ == "__main__":
    unittest.main()
