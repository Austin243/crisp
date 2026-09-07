"""Opt-in GP ranking of slab candidates before physical relaxation."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .surrogate import ExactGP


@dataclass(frozen=True)
class SlabGPScreening:
    """Bounded candidate pools and recent training data for ExactGP screening.

    Targets pair candidate (unrelaxed) descriptors with valid post-quench energy.
    Every explore_every-th trial uses the first random candidate regardless of GP.
    Kernel tuning uses only training data; it is optional and deterministic.
    """

    pool_size: int = 4
    min_training_points: int = 4
    explore_every: int = 5
    max_training_points: int = 128
    kappa: float = 1.0
    length_scale: float = 1.0
    noise: float = 1e-3
    auto_tune: bool = True

    def __post_init__(self):
        for name, minimum in (("pool_size", 2), ("min_training_points", 2),
                              ("explore_every", 1), ("max_training_points", 2)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer at least {minimum}")
            object.__setattr__(self, name, int(value))
        if self.max_training_points < self.min_training_points:
            raise ValueError("max_training_points must accommodate min_training_points")
        for name in ("kappa", "length_scale", "noise"):
            value = getattr(self, name)
            if np.ndim(value) != 0 or np.iscomplexobj(value):
                raise ValueError(f"{name} must be a finite real scalar")
            value = float(value)
            if not np.isfinite(value) or value < 0 or (name != "kappa" and value == 0):
                raise ValueError(f"{name} must be finite and {'nonnegative' if name == 'kappa' else 'positive'}")
            object.__setattr__(self, name, value)
        if type(self.auto_tune) is not bool:
            raise ValueError("auto_tune must be boolean")


def candidate_features(atoms, fp_calc):
    """Copy finite pooled candidate descriptors, without a physical calculation."""
    dimension = fp_calc.natx * (4 if fp_calc.orbital == "sp" else 1)
    fp = _real_array(fp_calc.get_fingerprints(atoms), (len(atoms), dimension))
    return _real_array(fp_calc.pool_with_std(fp), (2 * dimension,))


def fit_slab_gp(training, config):
    """Fit one deterministic model for screening and optional atomic guidance."""
    model = ExactGP(length_scale=config.length_scale, noise=config.noise,
                    auto_tune=config.auto_tune, auto_tune_min_points=config.min_training_points)
    model.train(np.array([row["features"] for row in training]),
                np.array([row["energy_per_atom"] for row in training]))
    for value in (model.X_train, model.y_train, model.alpha, model.K_inv,
                  model._y_mean, model._y_std, model.length_scale, model.noise):
        if value is None or np.iscomplexobj(value) or not np.isfinite(value).all():
            raise ValueError("Slab GP training produced nonfinite or complex state")
    return model


def select_by_gp(features, training, config, *, model=None):
    """Rank by mean - kappa * std; optionally reuse an already fitted model."""
    if model is None:
        model = fit_slab_gp(training, config)
    predictions = []
    for feature in features:
        mean, std = model.predict(feature)
        for value in (mean, std):
            if np.ndim(value) != 0 or np.iscomplexobj(value) or not np.isfinite(value):
                raise ValueError("Slab GP predictions must be finite real scalars")
        score = float(mean - config.kappa * std)
        if std < 0 or not np.isfinite(score):
            raise ValueError("Slab GP uncertainty/selection score is invalid")
        predictions.append(dict(mean=float(mean), std=float(std), score=score))
    selected = min(range(len(predictions)), key=lambda index: predictions[index]["score"])
    return selected, predictions


def _real_array(value, shape):
    value = np.asarray(value)
    if value.dtype.kind not in "biuf":
        raise ValueError("Slab GP descriptors must be real numeric arrays")
    value = np.array(value, dtype=float, copy=True)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError("Slab GP descriptors must have the configured shape and finite values")
    return value
