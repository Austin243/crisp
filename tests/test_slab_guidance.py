"""Atomic acquisition movement, its bounds, and the real pooled-FP force chain."""

from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms

from crisp.bias import BiasPotential
from crisp.projector import ForceProjector
from crisp.slab import SlabConfig, prepare_slab
from crisp.slab_fingerprint import SlabFingerprintCalculator
from crisp.slab_guidance import SlabGuidance, guide_slab
from crisp.slab_screening import candidate_features
from crisp.surrogate import ExactGP

try:
    import torch_fplib
except ImportError:
    torch_fplib = None


def _config(**kwargs):
    return SlabConfig((1, 60), initial_thickness=0, max_thickness=2,
                      cell_height=12, min_vacuum=4, **kwargs)


def _atoms():
    return Atoms("Cu2", positions=[[2, 2, 5], [4, 2, 5]],
                 cell=[10, 10, 12], pbc=[True, True, False])


class CoordinateFP(SlabFingerprintCalculator):
    """Analytical descriptor for movement/control-flow tests without torch."""

    def __init__(self, config=None, axis=0):
        super().__init__(config or _config(), cutoff=2, natx=1)
        self.axis = axis
        self.seen = []

    def get_fingerprints(self, atoms):
        self.seen.append(atoms.copy())
        return atoms.positions[:, self.axis:self.axis + 1].copy()

    def project_forces(self, atoms, weights):
        forces = np.zeros((len(atoms), 3))
        forces[:, self.axis] = -weights[:, 0]
        return forces


def _model(dimension=2):
    model = ExactGP()
    model.train(np.zeros((2, dimension)), np.array([0., 1.]))
    return model


def _linear(feature):
    return float(feature[0]), np.array([1., 0.])


class TestSlabGuidance(unittest.TestCase):
    def guide(self, atoms=None, fp=None, guidance=None, **kwargs):
        return guide_slab(atoms or _atoms(), fp or CoordinateFP(),
                          kwargs.pop("model", _model()), guidance or SlabGuidance(),
                          kappa=kwargs.pop("kappa", 1.), min_dist_ang=kwargs.pop("min_dist_ang", 1.),
                          **kwargs)

    def test_config_rejects_invalid_options(self):
        for options in ({"max_steps": 0}, {"max_steps": True}, {"max_steps": 1.5},
                        {"max_backtracks": -1}, {"max_backtracks": False},
                        {"max_step": 0}, {"max_step": True}, {"max_step": np.nan},
                        {"max_step": 1j}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SlabGuidance(**options)
        self.assertEqual(SlabGuidance(max_steps=np.int64(2), max_backtracks=0).max_steps, 2)

    def test_downhill_steps_are_bounded_owned_and_calculator_free(self):
        atoms = _atoms()
        atoms.calc = Calculator()
        atoms.info = {"nested": {"values": [1]}, "slab_quench_steps": 4, "layer_group": 1}
        atoms.set_array("custom", np.array([2, 3]))
        before = atoms.copy()
        original_info = deepcopy(atoms.info)
        with patch.object(BiasPotential, "evaluate_with_grad", side_effect=_linear):
            guided, details = self.guide(atoms)
        self.assertEqual({k: v for k, v in details.items() if k != "final_score"},
                         dict(steps=5, attempts=5, initial_score=3., stop="budget"))
        self.assertAlmostEqual(details["final_score"], 2.75)
        np.testing.assert_allclose(guided.positions[:, 0], atoms.positions[:, 0] - .25)
        np.testing.assert_array_equal(guided.positions[:, 1:], atoms.positions[:, 1:])
        for name in ("cell", "pbc", "numbers"):
            np.testing.assert_array_equal(getattr(guided, name), getattr(atoms, name))
        self.assertIsNone(guided.calc)
        self.assertNotIn("layer_group", guided.info)
        self.assertNotIn("slab_quench_steps", guided.info)
        guided.info["nested"]["values"].append(2)
        guided.arrays["custom"][0] = 9
        self.assertEqual(atoms.info, original_info)
        np.testing.assert_array_equal(atoms.positions, before.positions)
        np.testing.assert_array_equal(atoms.arrays["custom"], before.arrays["custom"])
        self.assertEqual(atoms.calc.results, {})

    def test_backtracking_accepts_only_lower_acquisition(self):
        atoms = _atoms()
        atoms.positions[:, 0] += .1
        def quadratic(feature):
            return (feature[0] - 3.) ** 2, np.array([2 * (feature[0] - 3.), 0.])
        with patch.object(BiasPotential, "evaluate_with_grad", side_effect=quadratic):
            guided, details = self.guide(atoms, guidance=SlabGuidance(1, 1., 4))
        self.assertEqual((details["steps"], details["attempts"], details["stop"]), (1, 4, "budget"))
        self.assertLess(details["final_score"], details["initial_score"])
        np.testing.assert_allclose(guided.positions[:, 0], atoms.positions[:, 0] - .125)

    def test_blocked_geometry_never_reaches_fingerprint_backend(self):
        config = replace(_config(), max_thickness=1)
        fp = CoordinateFP(config, axis=2)
        atoms = _atoms()
        atoms.positions[:, 2] = [5., 6.]
        def expand_thickness(feature):
            return -float(feature[1]), np.array([0., -1.])
        with patch.object(BiasPotential, "evaluate_with_grad", side_effect=expand_thickness):
            guided, details = self.guide(atoms, fp)
        self.assertEqual((details["steps"], details["attempts"], details["stop"]), (0, 7, "blocked"))
        np.testing.assert_array_equal(guided.positions, atoms.positions)
        self.assertEqual(len(fp.seen), 2)  # Initial value and force projection only.
        self.assertTrue(all(np.ptp(a.positions[:, 2]) <= 1 for a in fp.seen))

    def test_gap_and_periodic_self_distance_checked_before_initial_backend(self):
        atoms = _atoms()
        cases = []
        gap_fp = CoordinateFP(replace(_config(), max_thickness=10, min_vacuum=1))
        gap_atoms = atoms.copy()
        gap_atoms.positions[:, 2] = [1, 11]
        cases.append((gap_atoms, gap_fp))
        short_atoms = atoms.copy()
        short_atoms.cell[0, 0] = .5
        cases.append((short_atoms, CoordinateFP()))
        for invalid, fp in cases:
            with self.subTest(cell=invalid.cell), self.assertRaises(ValueError):
                self.guide(invalid, fp)
            self.assertEqual(fp.seen, [])

    def test_proposed_gap_and_distances_are_rejected_before_backend(self):
        for boundary in ("gap", "distance"):
            atoms = _atoms()
            if boundary == "gap":
                fp = CoordinateFP(replace(_config(), max_thickness=10, min_vacuum=1), axis=2)
                atoms.positions[:, 2] = [1., 10.99999]
                gradient = np.array([0., -1.])
            else:
                fp = CoordinateFP()
                atoms.positions[:, 0] = [2., 3.000001]
                gradient = np.array([0., 1.])
            with patch.object(BiasPotential, "evaluate_with_grad",
                              side_effect=lambda f: (float(f @ gradient), gradient)):
                guided, details = self.guide(atoms, fp)
            self.assertEqual(details["stop"], "blocked")
            self.assertEqual(len(fp.seen), 2)
            np.testing.assert_array_equal(guided.positions, atoms.positions)

    def test_flat_gradient_stops_without_trials(self):
        with patch.object(BiasPotential, "evaluate_with_grad", return_value=(1., np.zeros(2))):
            guided, details = self.guide()
        self.assertEqual(details, dict(steps=0, attempts=0, initial_score=1., final_score=1., stop="flat"))
        np.testing.assert_array_equal(guided.positions, _atoms().positions)

    def test_invalid_models_constraints_and_options_fail_before_fingerprints(self):
        models = [ExactGP(), _model(3), _model()]
        models[-1].alpha[0] = np.nan
        for model in models:
            fp = CoordinateFP()
            with self.assertRaises(ValueError):
                self.guide(fp=fp, model=model)
            self.assertEqual(fp.seen, [])
        atoms = _atoms()
        atoms.set_constraint(FixAtoms(indices=[0]))
        with self.assertRaises(ValueError):
            self.guide(atoms)
        for kwargs in ({"kappa": -1}, {"kappa": True}, {"min_dist_ang": 0}, {"min_dist_ang": np.inf}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.guide(**kwargs)

    def test_nonfinite_scores_gradients_and_forces_and_backend_errors_propagate(self):
        for value in ((np.nan, np.zeros(2)), (1j, np.zeros(2)), (1., [np.inf, 0]),
                      (1., [1j, 0]), (1., np.zeros(3))):
            with patch.object(BiasPotential, "evaluate_with_grad", return_value=value), self.assertRaises(ValueError):
                self.guide()
        for forces in (np.full((2, 3), np.nan), np.ones((2, 3), complex), np.zeros((2, 2)),
                       np.full((2, 3), 1e200)):
            with np.errstate(over="ignore"), \
                    patch.object(ForceProjector, "compute_forces", return_value=forces), self.assertRaises(ValueError):
                self.guide()
        with patch.object(CoordinateFP, "get_fingerprints", side_effect=RuntimeError("backend failed")):
            with self.assertRaisesRegex(RuntimeError, "backend failed"):
                self.guide()


@unittest.skipUnless(torch_fplib is not None, "torch_fplib required")
class TestSlabGuidanceBackend(unittest.TestCase):
    def test_acquisition_xyz_force_matches_finite_differences_for_s_and_sp(self):
        config = SlabConfig((1., 8.), initial_thickness=1., max_thickness=2.,
                            cell_height=14., min_vacuum=1.)
        atoms = prepare_slab(Atoms(
            "BNC", positions=[[.22, .38, -.3], [1.35, .95, .62], [2.1, 2.4, -.1]],
            cell=[[3.8, 0, 0], [.9, 3.6, 0], [0, 0, 14.]]), config)
        for orbital in ("s", "sp"):
            with self.subTest(orbital=orbital):
                fp = SlabFingerprintCalculator(config, cutoff=3.2, natx=16, orbital=orbital)
                features = []
                for scale in (-.2, -.1, .1, .2):
                    candidate = atoms.copy()
                    candidate.positions += np.random.default_rng(13).normal(size=(3, 3)) * scale
                    features.append(candidate_features(candidate, fp))
                model = ExactGP(length_scale=.2)
                model.train(np.array(features), np.array([-.5, -.9, -.7, -.3]))
                bias = BiasPotential(model, kappa=.5, beta=0, gamma=0)
                _, gradient = bias.evaluate_with_grad(candidate_features(atoms, fp))
                forces = len(atoms) * ForceProjector(fp).compute_forces(atoms, gradient)
                finite_difference = np.zeros_like(forces)
                eps = 1e-5
                for atom in range(len(atoms)):
                    for axis in range(3):
                        plus, minus = atoms.copy(), atoms.copy()
                        plus.positions[atom, axis] += eps
                        minus.positions[atom, axis] -= eps
                        values = [len(atoms) * bias.evaluate(candidate_features(a, fp)) for a in (plus, minus)]
                        finite_difference[atom, axis] = -(values[0] - values[1]) / (2 * eps)
                np.testing.assert_allclose(forces, finite_difference, rtol=2e-5, atol=2e-7)
                self.assertGreater(np.linalg.norm(forces[:, 2]), .01)
                guided, details = guide_slab(atoms, fp, model, SlabGuidance(), kappa=.5, min_dist_ang=1.)
                self.assertGreater(details["steps"], 0)
                self.assertLess(details["final_score"], details["initial_score"])
                np.testing.assert_array_equal(guided.cell, atoms.cell)
                np.testing.assert_array_equal(guided.pbc, atoms.pbc)
                self.assertLessEqual(np.linalg.norm(guided.positions - atoms.positions, axis=1).max(), .25 + 1e-12)


if __name__ == "__main__":
    unittest.main()
