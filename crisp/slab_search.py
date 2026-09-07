"""Bounded local slab search with optional GP screening and parent mutations."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import logging
from numbers import Integral

import numpy as np

from .slab import _ATOL
from .slab_archive import SlabArchive, _check_fields, _read_slab_json, _write_slab_json
from .slab_fingerprint import _IMAGE_MARGIN
from .slab_generation import SlabGenerationError, generate_slabs
from .slab_mutation import SlabMutations, mutate_slab
from .slab_relaxation import SlabQuenchBoundsError, SlabQuenchNotConverged, quench_slab
from .slab_screening import SlabGPScreening, _real_array, candidate_features, select_by_gp
from .slab_validation import validate_slab_candidate


logger = logging.getLogger(__name__)
_RELAXED = {"accepted", "duplicate", "validation_rejected"}
_REJECTED = {"generation_rejected", "quench_rejected", "validation_rejected"}


class SlabRandomSearch:
    """Generate, quench, validate and archive a fixed-composition 2D search.

    Start with an empty SlabArchive and a fresh-calculator factory supporting
    mixed PBC. The entire sampled cell stays fixed during each atomic quench.
    run(n_trials) requests that many additional trials, including rejections;
    each trial allows at most one quench. Optional screening ranks candidate
    pools with ExactGP. Optional mutations perturb archived parents between
    random trials; atomic bias and cell relaxation are not used.

    calculator_id is a caller-supplied energy-model label, checked on resume;
    the caller must also supply the same actual model and dependency versions.
    Treat archive, outcomes, training_rows and settings as read-only during a search.
    """

    def __init__(self, archive: SlabArchive, calc_factory, *, calculator_id: str,
                 seed: int = 0, layer_groups=None, max_generation_attempts: int = 20,
                 fmax: float = 0.05, max_steps: int = 200,
                 screening: SlabGPScreening | None = None,
                 mutations: SlabMutations | None = None):
        if screening is not None and not isinstance(screening, SlabGPScreening):
            raise TypeError("screening must be SlabGPScreening or None")
        if mutations is not None and not isinstance(mutations, SlabMutations):
            raise TypeError("mutations must be SlabMutations or None")
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
        self.training_rows = []
        self._screening = screening
        self._mutations = mutations
        self._archive_settings = archive_settings
        self._settings = dict(calculator_id=calculator_id, seed=int(seed),
                              layer_groups=sorted(set(int(g) for g in groups)),
                              max_generation_attempts=int(max_generation_attempts),
                              fmax=float(fmax), max_steps=int(max_steps))
        if screening is not None:
            self._settings["screening"] = asdict(screening)
        if mutations is not None:
            self._settings["mutations"] = asdict(mutations)

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
        if self._mutations is not None:
            outcome["proposal"] = None
        feature = None
        if self._screening is None:
            try:
                candidate, proposal = self._candidate(index)
            except SlabGenerationError as exc:
                return outcome | dict(status="generation_rejected", reason=str(exc))
        else:
            candidate, feature, details, proposal, error = self._screen_candidate(index)
            outcome["screening"] = details
            if candidate is None:
                return outcome | dict(status="generation_rejected", reason=error)
        if self._mutations is not None:
            outcome["proposal"] = proposal
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
        if feature is not None:
            row = dict(trial=index + 1, features=feature.tolist(), energy_per_atom=energy)
            training = (self.training_rows + [row])[-self._screening.max_training_points:]
        added = self.archive.add(relaxed, energy, metadata=self._entry_metadata(outcome))
        if feature is not None:
            self.training_rows = training
        return outcome | dict(status="accepted" if added else "duplicate")

    def _generate(self, seed):
        return generate_slabs(
            self._archive_settings["composition"], self.archive.fp_calc.config, 1, seed=seed,
            layer_groups=self._settings["layer_groups"], min_dist_ang=self.archive.min_dist_ang,
            max_attempts=self._settings["max_generation_attempts"])[0]

    def _pool_seed(self, index, member):
        if member == 0:
            return self._trial_seed(index)
        sequence = np.random.SeedSequence([self._settings["seed"], index, member, 9])
        return int(sequence.generate_state(1, dtype=np.uint64)[0])

    def _proposal(self, index, member, accepted_trials, mode=None):
        """Derive source and parent from completed history, without mutable RNG state."""
        if (not accepted_trials or (index + 1) % self._mutations.random_every == 0
                or (self._screening is not None and mode != "gp")):
            return dict(source="random", parent_trial=None)
        rng = np.random.default_rng(np.random.SeedSequence([self._pool_seed(index, member), 10]))
        return dict(source="mutation", parent_trial=accepted_trials[int(rng.integers(len(accepted_trials)))])

    def _candidate(self, index, member=0, mode=None):
        seed = self._pool_seed(index, member)
        if self._mutations is None:
            return self._generate(seed), None
        accepted_trials = [entry.metadata["search_trial"] for entry in self.archive.entries]
        proposal = self._proposal(index, member, accepted_trials, mode)
        if proposal["source"] == "random":
            return self._generate(seed), proposal
        parent = self.archive.entries[accepted_trials.index(proposal["parent_trial"])].atoms
        candidate = mutate_slab(parent, self.archive.fp_calc.config, self._mutations,
                                seed=seed, min_dist_ang=self.archive.min_dist_ang)
        return candidate, proposal

    def _screening_mode(self, index, n_training):
        if n_training < self._screening.min_training_points:
            return "bootstrap"
        return "explore" if (index + 1) % self._screening.explore_every == 0 else "gp"

    def _screen_candidate(self, index):
        mode = self._screening_mode(index, len(self.training_rows))
        requested = self._screening.pool_size if mode == "gp" else 1
        details = dict(mode=mode, requested=requested, generated=0, selected_member=None,
                       candidate_seed=None, mean=None, std=None, score=None)
        candidates, features, members, proposals = [], [], [], []
        error = None
        for member in range(requested):
            try:
                candidate, proposal = self._candidate(index, member, mode)
            except SlabGenerationError as exc:
                error = str(exc)
                continue
            features.append(candidate_features(candidate, self.archive.fp_calc))
            candidates.append(candidate)
            members.append(member)
            proposals.append(proposal)
        details["generated"] = len(candidates)
        if not candidates:
            return None, None, details, None, error
        selected = 0
        if mode == "gp":
            selected, predictions = select_by_gp(features, self.training_rows, self._screening)
            details.update(predictions[selected])
        details.update(selected_member=members[selected],
                       candidate_seed=self._pool_seed(index, members[selected]))
        return candidates[selected], features[selected], details, proposals[selected], None

    def _entry_metadata(self, outcome):
        metadata = dict(search_trial=outcome["trial"], trial_seed=outcome["seed"],
                        calculator_id=self._settings["calculator_id"])
        if self._screening is not None:
            metadata["candidate_seed"] = outcome["screening"]["candidate_seed"]
        if self._mutations is not None:
            metadata["proposal"] = deepcopy(outcome["proposal"])
        return metadata

    def save(self, path):
        """Atomically save the archive and all completed trials in one file."""
        self._check_archive()
        payload = dict(format="crisp-slab-search", version=1,
                       settings=self._settings, outcomes=self.outcomes,
                       archive=self.archive._to_payload())
        if self._screening is not None:
            payload.update(version=2, training=self.training_rows)
        if self._mutations is not None:
            payload["version"] = 3
        _write_slab_json(path, payload)

    def load(self, path):
        """Restore a compatible checkpoint without partial archive/progress updates."""
        target = self._check_archive()
        payload = _read_slab_json(path)
        fields = {"format", "version", "settings", "outcomes", "archive"}
        if self._screening is not None:
            fields.add("training")
        _check_fields(payload, fields)
        version = 1 if self._screening is None else 2
        if self._mutations is not None:
            version = 3
        if (payload["format"] != "crisp-slab-search"
                or type(payload["version"]) is not int or payload["version"] != version):
            raise ValueError("Unsupported slab search checkpoint format/version")
        if payload["settings"] != self._settings:
            raise ValueError("Slab search settings/calculator_id do not match")
        outcomes = payload["outcomes"]
        if not isinstance(outcomes, list):
            raise ValueError("Slab search outcomes must be a list")
        n_training = 0
        accepted_trials = []
        for index, outcome in enumerate(outcomes):
            fields = {"trial", "seed", "status", "reason", "energy_per_atom", "quench_steps"}
            if self._screening is not None:
                fields.add("screening")
            if self._mutations is not None:
                fields.add("proposal")
            _check_fields(outcome, fields)
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
            if self._screening is not None:
                self._validate_screening(outcome, index, n_training)
                n_training += outcome["status"] in ("accepted", "duplicate")
            if self._mutations is not None:
                self._validate_proposal(outcome, index, accepted_trials)
            if outcome["status"] == "accepted":
                accepted_trials.append(outcome["trial"])
        training = self._validated_training(payload["training"], outcomes) if self._screening else []
        target._load_payload(payload["archive"])
        accepted = [outcome for outcome in outcomes if outcome["status"] == "accepted"]
        if len(target.entries) != len(accepted) or any(
                entry.metadata != self._entry_metadata(outcome) or entry.energy != outcome["energy_per_atom"]
                for entry, outcome in zip(target.entries, accepted)):
            raise ValueError("Slab search archive does not match its accepted trials")
        self.archive.entries = target.entries
        self.outcomes = deepcopy(outcomes)
        self.training_rows = training

    def _validate_proposal(self, outcome, index, accepted_trials):
        proposal = outcome["proposal"]
        if outcome["status"] == "generation_rejected":
            if proposal is not None:
                raise ValueError("Failed generation cannot contain a selected proposal")
            return
        _check_fields(proposal, {"source", "parent_trial"})
        if proposal["parent_trial"] is not None:
            _integer("parent_trial", proposal["parent_trial"], 1)
        selection = outcome.get("screening", {})
        expected = self._proposal(index, selection.get("selected_member", 0), accepted_trials,
                                  selection.get("mode"))
        if proposal != expected:
            raise ValueError("Slab proposal source/parent does not match completed history")

    def _validate_screening(self, outcome, index, n_training):
        details = outcome["screening"]
        _check_fields(details, {"mode", "requested", "generated", "selected_member",
                                "candidate_seed", "mean", "std", "score"})
        mode = self._screening_mode(index, n_training)
        requested = self._screening.pool_size if mode == "gp" else 1
        if details["mode"] != mode or type(details["requested"]) is not int or details["requested"] != requested:
            raise ValueError("Invalid slab GP selection mode/pool size")
        _integer("generated", details["generated"], 0)
        if details["generated"] > requested:
            raise ValueError("Slab GP generation exceeded its pool budget")
        if outcome["status"] == "generation_rejected":
            if details["generated"] != 0 or any(details[key] is not None for key in
                    ("selected_member", "candidate_seed", "mean", "std", "score")):
                raise ValueError("Failed generation cannot contain a GP selection")
            return
        member = details["selected_member"]
        _integer("selected_member", member, 0)
        if (details["generated"] == 0 or member >= requested
                or type(details["candidate_seed"]) is not int
                or details["candidate_seed"] != self._pool_seed(index, member)):
            raise ValueError("Invalid slab GP selected candidate")
        if mode == "gp":
            for name in ("mean", "std", "score"):
                value = details[name]
                if type(value) not in (int, float) or not np.isfinite(value):
                    raise ValueError("Invalid slab GP prediction")
            if details["std"] < 0 or not np.isclose(
                    details["score"], details["mean"] - self._screening.kappa * details["std"],
                    atol=1e-12, rtol=1e-12):
                raise ValueError("Invalid slab GP acquisition score")
        elif any(details[key] is not None for key in ("mean", "std", "score")):
            raise ValueError("Random selections cannot contain GP predictions")

    def _validated_training(self, rows, outcomes):
        eligible = [outcome for outcome in outcomes if outcome["status"] in ("accepted", "duplicate")]
        eligible = eligible[-self._screening.max_training_points:]
        if not isinstance(rows, list) or len(rows) != len(eligible):
            raise ValueError("Slab GP training rows do not match completed valid quenches")
        dimension = 2 * self.archive.fp_calc.natx * (4 if self.archive.fp_calc.orbital == "sp" else 1)
        restored = []
        for row, outcome in zip(rows, eligible):
            _check_fields(row, {"trial", "features", "energy_per_atom"})
            if (type(row["trial"]) is not int or row["trial"] != outcome["trial"]
                    or type(row["energy_per_atom"]) not in (int, float)
                    or row["energy_per_atom"] != outcome["energy_per_atom"]):
                raise ValueError("Slab GP training target does not match its completed trial")
            feature = _real_array(row["features"], (dimension,))
            restored.append(dict(trial=row["trial"], features=feature.tolist(),
                                 energy_per_atom=float(row["energy_per_atom"])))
        return restored


def _integer(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
