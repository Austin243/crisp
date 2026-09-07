"""Bounded atomic proposals following a fingerprint-space acquisition gradient."""

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from ase.neighborlist import neighbor_list

from .bias import BiasPotential
from .projector import ForceProjector
from .slab import validate_slab
from .slab_fingerprint import SlabFingerprintCalculator, _IMAGE_MARGIN
from .slab_screening import _real_array, candidate_features


@dataclass(frozen=True)
class SlabGuidance:
    """Bounds for atomic acquisition descent before an unbiased physical quench.

    max_step bounds each atom's displacement in Angstrom per accepted move.
    Each move tries the full step followed by at most max_backtracks halvings.
    The acquisition is a basin-energy heuristic, not a physical potential.
    """

    max_steps: int = 5
    max_step: float = 0.05
    max_backtracks: int = 6

    def __post_init__(self):
        for name, minimum in (("max_steps", 1), ("max_backtracks", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer at least {minimum}")
            object.__setattr__(self, name, int(value))
        if (isinstance(self.max_step, bool) or not isinstance(self.max_step, Real)
                or not np.isfinite(self.max_step) or self.max_step <= 0):
            raise ValueError("max_step must be finite and positive")
        object.__setattr__(self, "max_step", float(self.max_step))


def guide_slab(atoms, fp_calc, model, guidance: SlabGuidance, *,
               kappa: float, min_dist_ang: float):
    """Return a calculator-free copy and a bounded descent summary.

    Reuse a trained, frozen GP and CRISP's pooled-fingerprint chain rule to
    descend mu - kappa * std. Projected directions are the negative gradient
    of N times that acquisition; they are not physical forces. Every accepted
    move strictly decreases the per-atom acquisition. A maximum projected
    per-atom norm <= 1e-12 stops as flat; a blocked line search keeps the last
    valid point. These stops are proposal outcomes, not physical convergence.

    Cell, PBC and atom order stay exactly fixed; all xyz coordinates may move.
    No wrapping/centering, physical calculation, stress or connectivity check
    is performed. Geometry, actual fingerprint image clearance and distances
    (including periodic self-images) are checked before each FP evaluation.
    Invalid inputs and unexpected numerical/backend errors propagate.
    """
    if not isinstance(guidance, SlabGuidance):
        raise TypeError("guidance must be SlabGuidance")
    if not isinstance(fp_calc, SlabFingerprintCalculator):
        raise TypeError("fp_calc must be SlabFingerprintCalculator")
    for name, value, allow_zero in (("kappa", kappa, True),
                                    ("min_dist_ang", min_dist_ang, False)):
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0 or (not allow_zero and value == 0)):
            raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    if atoms.constraints:
        raise ValueError("Slab guidance requires unconstrained atoms")
    dimension = 2 * fp_calc.natx * (4 if fp_calc.orbital == "sp" else 1)
    if getattr(model, "X_train", None) is None:
        raise ValueError("Slab guidance requires a trained GP")
    rows = len(model.X_train)
    _real_array(model.X_train, (rows, dimension))
    if rows == 0:
        raise ValueError("Slab guidance requires a trained GP")
    for value in (model.y_train, model.alpha, model.K_inv, model._y_mean,
                  model._y_std, model.length_scale, model.noise):
        if value is None or np.iscomplexobj(value) or not np.isfinite(value).all():
            raise ValueError("Slab guidance requires finite real GP state")

    def valid_geometry(candidate):
        validate_slab(candidate, fp_calc.config)
        if candidate.cell[2, 2] - np.ptp(candidate.positions[:, 2]) <= fp_calc.cutoff + _IMAGE_MARGIN:
            raise ValueError("Slab image gap must exceed fingerprint cutoff + 1e-6 Angstrom")
        if len(neighbor_list("d", candidate, min_dist_ang)):
            raise ValueError("Slab guidance violates min_dist_ang")

    valid_geometry(atoms)
    guided = atoms.copy()
    guided.info = deepcopy(atoms.info)
    for key in list(guided.info):
        if isinstance(key, str) and key.startswith("slab_quench_"):
            del guided.info[key]
    bias = BiasPotential(model, kappa=float(kappa), beta=0, gamma=0)
    projector = ForceProjector(fp_calc)

    def evaluate(candidate):
        score, gradient = bias.evaluate_with_grad(candidate_features(candidate, fp_calc))
        if np.ndim(score) != 0 or np.iscomplexobj(score) or not np.isfinite(score):
            raise ValueError("Slab guidance acquisition must be a finite real scalar")
        return float(score), _real_array(gradient, (dimension,))

    score, gradient = evaluate(guided)
    details = dict(steps=0, attempts=0, initial_score=score, final_score=score, stop="budget")
    for _ in range(guidance.max_steps):
        direction = _real_array(projector.compute_forces(guided, gradient), (len(guided), 3))
        direction *= len(guided)
        norm = float(np.linalg.norm(direction, axis=1).max())
        if not np.isfinite(norm):
            raise ValueError("Slab guidance projected force norm must be finite")
        if norm <= 1e-12:
            details["stop"] = "flat"
            break
        direction *= guidance.max_step / norm
        for backtrack in range(guidance.max_backtracks + 1):
            details["attempts"] += 1
            trial = guided.copy()
            trial.positions += direction * (0.5 ** backtrack)
            try:
                valid_geometry(trial)
            except ValueError:
                continue
            trial_score, trial_gradient = evaluate(trial)
            if trial_score < score:
                guided = trial
                score, gradient = trial_score, trial_gradient
                details.update(steps=details["steps"] + 1, final_score=score)
                break
        else:
            details["stop"] = "blocked"
            break
    if details["steps"]:
        guided.info.pop("layer_group", None)
    return guided, details
