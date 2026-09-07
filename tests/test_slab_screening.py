"""GP ranking learns candidate outcomes and guards descriptor/model failures."""

from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import Mock, patch

import numpy as np
from ase import Atoms

from crisp.slab_fingerprint import SlabFingerprintCalculator
from crisp.slab_screening import SlabGPScreening, candidate_features, select_by_gp
from test_slab_archive import _config


def _training():
    return [dict(features=[-1.0], energy_per_atom=-1.0),
            dict(features=[1.0], energy_per_atom=1.0)]


def _trained_model():
    model = Mock()
    model.X_train = np.array([[-1.0], [1.0]])
    model.y_train = np.array([-1.0, 1.0])
    model.alpha = np.array([-1.0, 1.0])
    model.K_inv = np.eye(2)
    model._y_mean, model._y_std = 0.0, 1.0
    model.length_scale, model.noise = 1.0, 1e-3
    model.predict.return_value = (0.0, 0.1)
    return model


class TestSlabGPScreening(unittest.TestCase):
    def test_config_is_frozen_and_rejects_invalid_budgets_and_kernel_options(self):
        config = SlabGPScreening(pool_size=np.int64(3), kappa=np.float64(0.0))
        self.assertIs(type(config.pool_size), int)
        self.assertIs(type(config.kappa), float)
        with self.assertRaises(FrozenInstanceError):
            config.pool_size = 4
        for field, minimum in (("pool_size", 2), ("min_training_points", 2),
                               ("max_training_points", 2), ("explore_every", 1)):
            for value in (minimum - 1, True, 2.5):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    SlabGPScreening(**{field: value})
        with self.assertRaises(ValueError):
            SlabGPScreening(min_training_points=5, max_training_points=4)
        for field in ("kappa", "length_scale", "noise"):
            values = [-1.0, np.inf, np.nan, 1j, [1.0]]
            if field != "kappa":
                values.append(0.0)
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    SlabGPScreening(**{field: value})
        for value in (0, 1, "yes", np.bool_(True)):
            with self.subTest(auto_tune=value), self.assertRaises(ValueError):
                SlabGPScreening(auto_tune=value)

    def test_real_gp_learns_lower_energy_and_breaks_equal_scores_by_pool_order(self):
        features = [[1.0], [-1.0], [-1.0]]
        for auto_tune in (False, True):
            with self.subTest(auto_tune=auto_tune):
                config = SlabGPScreening(min_training_points=2, kappa=0.0,
                                         noise=1e-6, auto_tune=auto_tune)
                selected, predictions = select_by_gp(features, _training(), config)
                self.assertEqual(selected, 1)
                self.assertGreater(predictions[0]["mean"], 0.99)
                self.assertLess(predictions[1]["mean"], -0.99)
                self.assertEqual(predictions[1], predictions[2])
                self.assertEqual(predictions[selected]["score"],
                                 min(row["score"] for row in predictions))
                for prediction in predictions:
                    self.assertEqual(prediction["score"], prediction["mean"])
                    self.assertGreaterEqual(prediction["std"], 0.0)

    def test_uncertainty_weight_can_select_an_unseen_candidate_over_known_low_energy(self):
        features = [[-1.0], [5.0]]
        exploit = SlabGPScreening(min_training_points=2, kappa=0.0, auto_tune=False)
        explore = SlabGPScreening(min_training_points=2, kappa=2.0, auto_tune=False)
        low_mean, first = select_by_gp(features, _training(), exploit)
        uncertain, second = select_by_gp(features, _training(), explore)
        self.assertEqual(low_mean, 0)
        self.assertEqual(uncertain, 1)
        self.assertGreater(second[1]["std"], 10 * second[0]["std"])
        for before, after in zip(first, second):
            self.assertEqual(before["mean"], after["mean"])
            self.assertEqual(before["std"], after["std"])
            self.assertAlmostEqual(after["score"], after["mean"] - 2 * after["std"])

    def test_candidate_features_pool_s_and_sp_and_own_backend_buffers(self):
        atoms = Atoms("BN")
        calculator = Mock()
        calculator.get_potential_energy.side_effect = AssertionError("physical evaluation")
        atoms.calc = calculator
        for orbital, dimension in (("s", 3), ("sp", 12)):
            with self.subTest(orbital=orbital):
                fp_calc = SlabFingerprintCalculator(_config(), natx=3, orbital=orbital)
                fp = np.arange(2 * dimension, dtype=float).reshape(2, dimension)
                expected = np.concatenate((fp.mean(axis=0), fp.std(axis=0)))
                with patch.object(fp_calc, "get_fingerprints", return_value=fp):
                    actual = candidate_features(atoms, fp_calc)
                np.testing.assert_allclose(actual, expected)
                fp[:] = np.nan
                np.testing.assert_allclose(actual, expected)
                pooled = np.arange(2 * dimension, dtype=float)
                with patch.object(fp_calc, "get_fingerprints", return_value=np.zeros_like(fp)), \
                        patch.object(fp_calc, "pool_with_std", return_value=pooled):
                    copied = candidate_features(atoms, fp_calc)
                pooled[:] = np.nan
                np.testing.assert_array_equal(copied, np.arange(2 * dimension))
        calculator.get_potential_energy.assert_not_called()

    def test_invalid_descriptor_shape_or_values_fail_before_pool_or_model(self):
        atoms = Atoms("BN")
        fp_calc = SlabFingerprintCalculator(_config(), natx=3)
        bad_fp = (np.zeros((1, 3)), np.zeros((2, 2)), np.zeros((2, 3, 1)),
                  np.full((2, 3), np.nan), np.full((2, 3), np.inf),
                  np.full((2, 3), "1"), np.full((2, 3), 1, dtype=object),
                  np.zeros((2, 3), dtype=complex))
        for value in bad_fp:
            with self.subTest(shape=value.shape, dtype=value.dtype), \
                    patch.object(fp_calc, "get_fingerprints", return_value=value), \
                    patch.object(fp_calc, "pool_with_std") as pool, \
                    self.assertRaises(ValueError):
                candidate_features(atoms, fp_calc)
            pool.assert_not_called()
        for value in (np.zeros(3), np.zeros((2, 3)), np.full(6, np.nan),
                      np.full(6, np.inf), np.zeros(6, dtype=complex)):
            with self.subTest(shape=value.shape, dtype=value.dtype), \
                    patch.object(fp_calc, "get_fingerprints", return_value=np.zeros((2, 3))), \
                    patch.object(fp_calc, "pool_with_std", return_value=value), \
                    self.assertRaises(ValueError):
                candidate_features(atoms, fp_calc)

    def test_descriptor_training_and_prediction_errors_propagate(self):
        fp_calc = SlabFingerprintCalculator(_config(), natx=3)
        for method in ("get_fingerprints", "pool_with_std"):
            with self.subTest(method=method), \
                    patch.object(fp_calc, "get_fingerprints", return_value=np.zeros((2, 3))), \
                    patch.object(fp_calc, method, side_effect=RuntimeError("descriptor failed")), \
                    self.assertRaisesRegex(RuntimeError, "descriptor failed"):
                candidate_features(Atoms("BN"), fp_calc)
        for method in ("train", "predict"):
            model = _trained_model()
            getattr(model, method).side_effect = RuntimeError("GP failed")
            with self.subTest(method=method), \
                    patch("crisp.slab_screening.ExactGP", return_value=model), \
                    self.assertRaisesRegex(RuntimeError, "GP failed"):
                select_by_gp([[0.0]], _training(), SlabGPScreening())

    def test_invalid_trained_state_is_fatal_before_any_prediction(self):
        for field in ("X_train", "y_train", "alpha", "K_inv", "_y_mean", "_y_std",
                      "length_scale", "noise"):
            for value in (None, np.nan, np.inf, 1j):
                model = _trained_model()
                setattr(model, field, value)
                with self.subTest(field=field, value=value), \
                        patch("crisp.slab_screening.ExactGP", return_value=model), \
                        self.assertRaises(ValueError):
                    select_by_gp([[0.0]], _training(), SlabGPScreening())
                model.predict.assert_not_called()

    def test_invalid_predictions_and_overflowing_acquisition_are_fatal(self):
        invalid = ((np.nan, 0.1), (np.inf, 0.1), (1j, 0.1), ([0.0], 0.1),
                   (0.0, np.nan), (0.0, np.inf), (0.0, 1j), (0.0, [0.1]), (0.0, -0.1))
        for prediction in invalid:
            model = _trained_model()
            model.predict.return_value = prediction
            with self.subTest(prediction=prediction), \
                    patch("crisp.slab_screening.ExactGP", return_value=model), \
                    self.assertRaises(ValueError):
                select_by_gp([[0.0]], _training(), SlabGPScreening())
        model = _trained_model()
        model.predict.return_value = (-1e308, 1e308)
        with patch("crisp.slab_screening.ExactGP", return_value=model), \
                self.assertRaises(ValueError):
            select_by_gp([[0.0]], _training(), SlabGPScreening(kappa=2.0))


if __name__ == "__main__":
    unittest.main()
