"""Bounded local random search over slab cells, separate from CRISPSearch."""

from collections import Counter
from copy import deepcopy
import logging
from numbers import Integral

import numpy as np

from .slab import _ATOL
from .slab_archive import SlabArchive, _check_fields, _read_slab_json, _write_slab_json
from .slab_fingerprint import _IMAGE_MARGIN
from .slab_generation import SlabGenerationError, generate_slabs
from .slab_relaxation import SlabQuenchBoundsError, SlabQuenchNotConverged, quench_slab
from .slab_validation import validate_slab_candidate


logger = logging.getLogger(__name__)
_RELAXED = {"accepted", "duplicate", "validation_rejected"}
_REJECTED = {"generation_rejected", "quench_rejected", "validation_rejected"}


class SlabRandomSearch:
    """Generate, quench, validate and archive a fixed-composition 2D search.

    Start with an empty SlabArchive and a fresh-calculator factory supporting
    mixed PBC. The entire sampled cell stays fixed during each atomic quench.
    run(n_trials) requests that many additional trials, including rejections;
    each trial allows at most one quench. GP/bias/cell relaxation are not used.

    calculator_id is a caller-supplied energy-model label, checked on resume;
    the caller must also supply the same actual model and dependency versions.
    Treat archive, outcomes and settings as read-only during a search.
    """

    def __init__(self, archive: SlabArchive, calc_factory, *, calculator_id: str,
                 seed: int = 0, layer_groups=None, max_generation_attempts: int = 20,
                 fmax: float = 0.05, max_steps: int = 200):
        if not isinstance(archive, SlabArchive) or archive.entries:
            raise ValueError("Start SlabRandomSearch with an empty SlabArchive; use load to resume")
        if not callable(calc_factory):
            raise TypeError("calc_factory must be callable")
        if not isinstance(calculator_id, str) or not calculator_id.strip():
            raise ValueError("calculator_id must identify the physical energy model")
        for name, value, minimum in (("seed", seed, 0), ("max_steps", max_steps, 0),
                                     ("max_generation_attempts", max_generation_attempts, 1)):
            _integer(name, value, minimum)
        if (np.ndim(fmax) != 0 or np.iscomplexobj(fmax)
                or not np.isfinite(fmax) or fmax <= 0):
            raise ValueError("fmax must be finite and positive")
        _, archive_settings = archive._persistence_target()
        config = archive.fp_calc.config
        if config.cell_height - config.max_thickness - _ATOL <= archive.fp_calc.cutoff + _IMAGE_MARGIN:
            raise ValueError("All allowed slab thicknesses must leave fingerprint image clearance")
        groups = tuple(range(1, 81) if layer_groups is None else layer_groups)
        # Reuse generation's argument validation without importing PyXtal or consuming RNG.
        generate_slabs(archive_settings["composition"], config, 0, layer_groups=groups,
                       min_dist_ang=archive.min_dist_ang,
                       max_attempts=max_generation_attempts)
        self.archive = archive
        self.calc_factory = calc_factory
        self.outcomes = []
        self._archive_settings = archive_settings
        self._settings = dict(calculator_id=calculator_id, seed=int(seed),
                              layer_groups=sorted(set(int(g) for g in groups)),
                              max_generation_attempts=int(max_generation_attempts),
                              fmax=float(fmax), max_steps=int(max_steps))

    @property
    def completed_trials(self):
        return len(self.outcomes)

    @property
    def counts(self):
        """Counts of completed trial outcomes; an empty archive is a valid result."""
        return dict(Counter(outcome["status"] for outcome in self.outcomes))

    def _trial_seed(self, index):
        # Independent streams let resumed runs reproduce trials without RNG snapshots.
        sequence = np.random.SeedSequence([self._settings["seed"], index])
        return int(sequence.generate_state(1, dtype=np.uint64)[0])

    def _check_archive(self):
        target, settings = self.archive._persistence_target()
        if settings != self._archive_settings:
            raise ValueError("Slab search archive settings changed")
        if len(self.archive.entries) != self.counts.get("accepted", 0):
            raise ValueError("Slab search archive entries changed outside the search")
        return target

    def run(self, n_trials: int, *, checkpoint=None) -> SlabArchive:
        """Run additional bounded trials, saving after each completed outcome.

        Expected generation, quench-bound/nonconvergence and structural rejections
        consume a trial and are recorded. Calculator, fingerprint, matcher and
        configuration errors propagate; the last completed checkpoint is retained.
        Budgets count trials, generation attempts and optimizer moves, not elapsed
        time or exact calculator evaluations. A failed trial can be retried after
        fixing its fatal error. A failed save leaves completed in-memory work intact.
        """
        _integer("n_trials", n_trials, 0)
        self._check_archive()
        for _ in range(n_trials):
            outcome = self._run_trial(self.completed_trials)
            self.outcomes.append(outcome)
            logger.info("Slab trial %d: %s%s", outcome["trial"], outcome["status"],
                        f" ({outcome['reason']})" if outcome["reason"] else "")
            if checkpoint is not None:
                self.save(checkpoint)
        if n_trials == 0 and checkpoint is not None:
            self.save(checkpoint)
        return self.archive

    def _run_trial(self, index):
        options = self._settings
        config = self.archive.fp_calc.config
        outcome = dict(trial=index + 1, seed=self._trial_seed(index), status=None,
                       reason=None, energy_per_atom=None, quench_steps=None)
        try:
            candidate = generate_slabs(
                self._archive_settings["composition"], config, 1, seed=outcome["seed"],
                layer_groups=options["layer_groups"], min_dist_ang=self.archive.min_dist_ang,
                max_attempts=options["max_generation_attempts"])[0]
        except SlabGenerationError as exc:
            return outcome | dict(status="generation_rejected", reason=str(exc))
        try:
            relaxed = quench_slab(candidate, config, self.calc_factory,
                                  fmax=options["fmax"], max_steps=options["max_steps"])
        except (SlabQuenchBoundsError, SlabQuenchNotConverged) as exc:
            return outcome | dict(status="quench_rejected", reason=str(exc))
        total_energy = relaxed.get_potential_energy()
        if (np.ndim(total_energy) != 0 or np.iscomplexobj(total_energy)
                or not np.isfinite(total_energy)):
            raise RuntimeError("Slab calculator must return a finite real scalar energy")
        energy = float(total_energy / len(relaxed))
        if not np.isfinite(energy):
            raise RuntimeError("Slab energy per atom must be finite")
        outcome.update(energy_per_atom=energy, quench_steps=relaxed.info["slab_quench_steps"])
        try:
            validate_slab_candidate(relaxed, config, min_dist_ang=self.archive.min_dist_ang,
                                    bond_scale=self.archive.bond_scale)
        except ValueError as exc:
            return outcome | dict(status="validation_rejected", reason=str(exc))
        added = self.archive.add(relaxed, energy, metadata=self._entry_metadata(outcome))
        return outcome | dict(status="accepted" if added else "duplicate")

    def _entry_metadata(self, outcome):
        return dict(search_trial=outcome["trial"], trial_seed=outcome["seed"],
                    calculator_id=self._settings["calculator_id"])

    def save(self, path):
        """Atomically save the archive and all completed trials in one file."""
        self._check_archive()
        _write_slab_json(path, dict(format="crisp-slab-search", version=1,
                                   settings=self._settings, outcomes=self.outcomes,
                                   archive=self.archive._to_payload()))

    def load(self, path):
        """Restore a compatible checkpoint without partial archive/progress updates."""
        target = self._check_archive()
        payload = _read_slab_json(path)
        _check_fields(payload, {"format", "version", "settings", "outcomes", "archive"})
        if (payload["format"] != "crisp-slab-search"
                or type(payload["version"]) is not int or payload["version"] != 1):
            raise ValueError("Unsupported slab search checkpoint format/version")
        if payload["settings"] != self._settings:
            raise ValueError("Slab search settings/calculator_id do not match")
        outcomes = payload["outcomes"]
        if not isinstance(outcomes, list):
            raise ValueError("Slab search outcomes must be a list")
        for index, outcome in enumerate(outcomes):
            _check_fields(outcome, {"trial", "seed", "status", "reason", "energy_per_atom", "quench_steps"})
            if (type(outcome["trial"]) is not int or outcome["trial"] != index + 1
                    or type(outcome["seed"]) is not int or outcome["seed"] != self._trial_seed(index)
                    or not isinstance(outcome["status"], str)
                    or outcome["status"] not in _RELAXED | _REJECTED):
                raise ValueError("Invalid slab search trial sequence/status")
            reason = outcome["reason"]
            rejected = outcome["status"] in _REJECTED
            if (rejected and not isinstance(reason, str)) or (not rejected and reason is not None):
                raise ValueError("Invalid slab search rejection reason")
            if outcome["status"] in _RELAXED:
                energy = outcome["energy_per_atom"]
                if type(energy) not in (int, float) or not np.isfinite(energy):
                    raise ValueError("Invalid slab search energy")
                _integer("quench_steps", outcome["quench_steps"], 0)
                if outcome["quench_steps"] > self._settings["max_steps"]:
                    raise ValueError("Slab search quench exceeds its step budget")
            elif outcome["energy_per_atom"] is not None or outcome["quench_steps"] is not None:
                raise ValueError("Unrelaxed trial cannot contain quench results")
        target._load_payload(payload["archive"])
        accepted = [outcome for outcome in outcomes if outcome["status"] == "accepted"]
        if len(target.entries) != len(accepted) or any(
                entry.metadata != self._entry_metadata(outcome) or entry.energy != outcome["energy_per_atom"]
                for entry, outcome in zip(target.entries, accepted)):
            raise ValueError("Slab search archive does not match its accepted trials")
        self.archive.entries = target.entries
        self.outcomes = deepcopy(outcomes)


def _integer(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
