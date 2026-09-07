"""Additive bulk-style population search with native VASP slab relaxation."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from .bias import BiasPotential
from .projector import ForceProjector
from .search import CRISPSearch
from .slab_archive import _read_slab_json, _write_slab_json, _slab_json_default
from .slab_generation import SlabGenerationError
from .slab_fingerprint import _IMAGE_MARGIN
from .slab_layer_archive import LayeredSlabArchive
from .slab_layers import generate_layered_slabs, validate_layered_slab
from .slab_mutation import SlabMutations, mutate_slab
from .slab_screening import candidate_features
from .slab_vasp import VASPValidationError
from .surrogate import ExactGP


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _real(name, value, minimum=0.0, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not np.isfinite(value) or value < minimum
            or (positive and value == minimum)):
        raise ValueError(f"Invalid finite scalar {name}")
    return float(value)


def _json(value):
    return json.loads(json.dumps(value, default=_slab_json_default, allow_nan=False))


class SlabCRISPSearch(CRISPSearch):
    """Reuse CRISP populations, screening and GP training for layered slabs.

    The backend must implement ``spec``, JSON-compatible physical ``settings``
    and ``relax(atoms, candidate_id)`` returning atoms, energy_per_atom and
    metadata. SlabVASPRelaxer supplies this contract. Physical optimization is
    entirely native; no MLIP or ASE optimizer is called by this driver.

    Checkpoints commit complete generations. An interrupted generation is
    deterministically replayed from its previous archive; the native backend
    reuses matching completed calculations. This is not mid-ionic-step restart.
    Training pairs *relaxed* fingerprints with final energy/atom, as bulk CRISP
    does at zero pressure. SlabRandomSearch has a separate, unchanged contract.
    """

    def __init__(self, archive, relaxer, *, seed=0, n_random=4, n_mutants=2,
                 n_select=4, max_generations=20, budget_relax=None,
                 mutations=SlabMutations(), screening_mode="filter",
                 convergence_gens=5, max_generation_attempts=1000,
                 gp_length_scale=1.0, kappa=1.0, beta=0.1, gamma=0.5,
                 gp_energy_margin=0.2, gp_confidence_frac=0.15,
                 max_skip_frac=0.6, min_relax_per_gen=1):
        if not isinstance(archive, LayeredSlabArchive):
            raise TypeError("archive must be a LayeredSlabArchive")
        if archive.entries:
            raise ValueError("Start with an empty archive; use load() to resume")
        geometry = archive.spec.config
        if geometry.cell_height - geometry.max_thickness <= archive.fp_calc.cutoff + _IMAGE_MARGIN:
            raise ValueError("Allowed stack thickness must leave a vacuum gap larger than the fingerprint cutoff")
        if not isinstance(mutations, SlabMutations):
            raise TypeError("mutations must be SlabMutations; use n_mutants=0 to disable")
        if not callable(getattr(relaxer, "relax", None)):
            raise TypeError("relaxer must implement the native relaxation contract")
        if relaxer.spec.to_dict() != archive.spec.to_dict():
            raise ValueError("Relaxer and archive layer specifications differ")
        if screening_mode not in ("filter", "rank"):
            raise ValueError("screening_mode must be filter or rank")
        integers = dict(n_random=(n_random, 1), n_mutants=(n_mutants, 0),
                        n_select=(n_select, 1), max_generations=(max_generations, 1),
                        convergence_gens=(convergence_gens, 1),
                        min_relax_per_gen=(min_relax_per_gen, 0))
        validated = {name: _integer(name, value, minimum)
                     for name, (value, minimum) in integers.items()}
        if budget_relax is not None:
            budget_relax = _integer("budget_relax", budget_relax, 1)
        scalars = dict(kappa=kappa, beta=beta, gamma=gamma,
                       gp_energy_margin=gp_energy_margin,
                       gp_confidence_frac=gp_confidence_frac, max_skip_frac=max_skip_frac)
        scalars = {name: _real(name, value) for name, value in scalars.items()}
        if scalars["gp_confidence_frac"] > 1 or scalars["max_skip_frac"] > 1:
            raise ValueError("Confidence and skip fractions must be <= 1")
        gp_length_scale = _real("gp_length_scale", gp_length_scale, positive=True)
        super().__init__(
            hpc_relaxer=relaxer, fp_calc=archive.fp_calc,
            composition=archive.spec.composition, pressure_GPa=0.0,
            min_dist_ang=archive.spec.min_dist_ang,
            screening_mode=screening_mode, budget_relax=budget_relax,
            gp_length_scale=gp_length_scale, use_mutations=validated["n_mutants"] > 0,
            enable_flow=False, enable_fp_finisher=False, enable_fpj_mutations=False,
            enable_gp_guided=False, enable_swap_mutations=False,
            enable_cawr_pretreat=False, local_relax_mode="plain", gp_auto_tune=False,
            **validated, **scalars)
        self.archive = archive
        self.relaxer = relaxer
        self.seed = _integer("seed", seed)
        self.mutations = mutations
        self.max_generation_attempts = _integer(
            "max_generation_attempts", max_generation_attempts, 1)
        self.next_generation = 0
        self.outcomes = []
        self.history = []
        self.stop_reason = None
        self._generation = 0
        self._random_call = 0
        self._mutation_count = 0
        self._stagnation = 0
        self._no_new = 0
        self._best_energy = None
        self._saved_settings = self._settings()
        self._rebuild_model()

    def _settings(self):
        names = ("seed", "n_random", "n_mutants", "n_select", "max_generations",
                 "budget_relax", "screening_mode", "convergence_gens",
                 "max_generation_attempts", "gp_length_scale", "kappa", "beta",
                 "gamma", "gp_energy_margin", "gp_confidence_frac", "max_skip_frac",
                 "min_relax_per_gen")
        _, archive_settings = self.archive._persistence_target()
        return _json(dict(search={name: getattr(self, name) for name in names},
                          mutations=asdict(self.mutations), archive=archive_settings,
                          backend=self.relaxer.settings,
                          training_target="relaxed-fingerprint/final-energy-per-atom"))

    def _check_settings(self):
        if self.fp_calc is not self.archive.fp_calc:
            raise ValueError("Search and archive must share the same fingerprint calculator")
        flags = (self.enable_flow, self.enable_fp_finisher, self.enable_fpj_mutations,
                 self.enable_gp_guided, self.enable_swap_mutations, self.gp_auto_tune)
        if (any(flags) or self._finisher is not None or self._cawr_config is not None
                or self.mlip_calc_factory is not None or self.pressure_GPa != 0
                or not self._use_hpc or self.use_mutations != (self.n_mutants > 0)):
            raise ValueError("Unsupported bulk option in the native 2D search")
        if self._settings() != self._saved_settings:
            raise ValueError("Search, fingerprint or physical settings changed; start a new search")

    def _model_for(self, archive):
        gp = ExactGP(length_scale=self.gp_length_scale, auto_tune=False)
        bias = BiasPotential(gp, kappa=self.kappa, beta=self.beta, gamma=self.gamma)
        projector = ForceProjector(self.fp_calc)
        if len(archive.entries) >= 2:
            gp.train(archive.get_all_pooled_fps(), archive.get_all_enthalpies())
            anchors = archive.get_diverse(n=3, pool="best", pool_size=10)
            bias.set_anchors([entry.fp_pooled for entry in anchors])
            bias.set_repulsion_centers([entry.fp_pooled for entry in archive.entries])
        return gp, bias, projector

    def _rebuild_model(self):
        self.gp, self._bias, self._projector = self._model_for(self.archive)

    def _seed(self, kind, index):
        return int(np.random.SeedSequence(
            [self.seed, self._generation, kind, index]).generate_state(1)[0])

    def _generate_random(self, n):
        candidates = []
        for _ in range(n):
            seed = self._seed(0, self._random_call)
            self._random_call += 1
            try:
                atoms = generate_layered_slabs(
                    self.archive.spec, 1, seed=seed,
                    max_attempts=self.max_generation_attempts)[0]
            except SlabGenerationError:
                continue
            # Fail clearly on invalid/unavailable descriptor configuration before
            # the inherited screening helpers could swallow backend exceptions.
            candidate_features(atoms, self.fp_calc)
            atoms.info.update(origin="random", generation_seed=seed)
            candidates.append(atoms)
        return candidates

    def _generate_mutants(self, archive, stagnation_count=0):
        if not archive.entries or not self.n_mutants:
            return []
        entries = sorted(archive.entries, key=lambda entry: entry.enthalpy)
        stagnant = stagnation_count >= 5
        pool = (self._diverse_parent_pool(entries, n=20)
                if stagnant and len(entries) > 10 else entries[:10])
        rng = np.random.default_rng(self._seed(1, 0))
        restarts = max(1, self.n_mutants // 3) if stagnant else 0
        options = self.mutations
        if stagnant:
            options = replace(options, max_displacement=2 * options.max_displacement,
                              max_strain=min(2 * options.max_strain, 0.99))
        candidates = []
        # The existing mutation primitive has one scalar contact threshold.
        # Use the least restrictive bound here, then apply the full pair policy.
        min_distance = min([archive.spec.min_dist_ang,
                            *archive.spec.pair_min_distances.values()])
        for slot in range(self.n_mutants - restarts):
            parent = pool[int(rng.integers(len(pool)))]
            seed = self._seed(2, slot)
            try:
                atoms = mutate_slab(parent.atoms, archive.spec.config, options,
                                    seed=seed, min_dist_ang=min_distance)
                validate_layered_slab(atoms, archive.spec, check_connectivity=False)
            except (SlabGenerationError, ValueError):
                continue
            candidate_features(atoms, self.fp_calc)
            atoms.info.update(origin="mutant", mutation_seed=seed,
                              parent_candidate_id=parent.metadata["candidate_id"],
                              parent_enthalpy=float(parent.enthalpy))
            candidates.append(atoms)
        if restarts:
            candidates.extend(self._generate_random(restarts))
        self._mutation_count = len(candidates)
        return candidates

    def _relax_batch_hpc(self, candidates, archive, generation):
        new_count = 0
        for index, atoms in enumerate(candidates):
            if self.budget_relax is not None and self.n_relaxed >= self.budget_relax:
                break
            candidate_id = f"g{generation:06d}-c{index:04d}"
            record = dict(candidate_id=candidate_id, generation=generation,
                          origin=atoms.info.get("origin", "random"), status="rejected",
                          energy_per_atom=None, error=None)
            try:
                result = self.relaxer.relax(atoms, candidate_id)
            except VASPValidationError as exc:
                record["error"] = str(exc)
            else:
                try:
                    validate_layered_slab(result.atoms, archive.spec)
                except ValueError as exc:
                    record["error"] = str(exc)
                else:
                    candidate_features(result.atoms, self.fp_calc)
                    metadata = deepcopy(result.metadata)
                    metadata.update(candidate_id=candidate_id, generation=generation,
                                    origin=record["origin"])
                    if "parent_candidate_id" in atoms.info:
                        metadata["parent_candidate_id"] = atoms.info["parent_candidate_id"]
                    added = archive.add(result.atoms, result.energy_per_atom, metadata=metadata)
                    record.update(status="accepted" if added else "duplicate",
                                  energy_per_atom=float(result.energy_per_atom))
                    new_count += int(added)
            self.outcomes.append(record)
            self.n_relaxed += 1
        return new_count

    def run(self, n_generations=None, *, checkpoint=None):
        """Run additional complete generations, respecting the total search budget.

        Execution/configuration failures propagate and restore the last complete
        generation in memory. Native candidate files are retained for replay.
        """
        self._check_settings()
        if n_generations is None:
            n_generations = self.max_generations - self.next_generation
        n_generations = _integer("n_generations", n_generations)
        if checkpoint is not None and Path(checkpoint).exists() and not self.next_generation:
            if _read_slab_json(checkpoint) != _json(self._payload()):
                raise ValueError("Checkpoint already exists; load it before continuing")
        if checkpoint is not None:
            self.save(checkpoint)
        for _ in range(n_generations):
            if self.next_generation >= self.max_generations:
                self.stop_reason = "generations"
                break
            if self.budget_relax is not None and self.n_relaxed >= self.budget_relax:
                self.stop_reason = "budget"
                break
            if self._no_new >= self.convergence_gens:
                self.stop_reason = "stagnation"
                break
            before = _json(self._payload())
            self._generation = self.next_generation
            self._random_call = self._mutation_count = 0
            try:
                candidates = self._generate_random(self.n_random)
                if self._generation == 0:
                    new_count = self._run_bootstrap(candidates, self.archive, self.gp, self._bias)
                else:
                    new_count = self._run_generation(
                        candidates, self.archive, self.gp, self._bias, self._projector,
                        self._generation, stagnation_count=self._stagnation)
                best = (min(entry.energy for entry in self.archive.entries)
                        if self.archive.entries else None)
                improved = best is not None and (
                    self._best_energy is None or best < self._best_energy - 0.001)
                self._stagnation = 0 if improved else self._stagnation + 1
                self._no_new = 0 if new_count else self._no_new + 1
                self._best_energy = best
                self.history.append(dict(generation=self._generation, new=new_count,
                                         attempted=self.n_relaxed - before["n_relaxed"],
                                         generated=len(candidates),
                                         mutations=self._mutation_count, best_energy=best))
                self.next_generation += 1
                self.stop_reason = ("budget" if self.budget_relax is not None
                                    and self.n_relaxed >= self.budget_relax else
                                    "generations" if self.next_generation >= self.max_generations else
                                    "stagnation" if self._no_new >= self.convergence_gens else "paused")
                if checkpoint is not None:
                    self.save(checkpoint)
            except Exception:
                self._load_payload(before)
                raise
        if checkpoint is not None:
            self.save(checkpoint)
        return self.archive

    def _payload(self):
        return dict(format="crisp-layered-population-search", version=1,
                    settings=self._saved_settings, archive=self.archive._to_payload(),
                    next_generation=self.next_generation, n_relaxed=self.n_relaxed,
                    outcomes=deepcopy(self.outcomes), history=deepcopy(self.history),
                    stagnation=self._stagnation, no_new=self._no_new,
                    best_energy=self._best_energy, stop_reason=self.stop_reason)

    def save(self, path):
        """Atomically save a complete-generation checkpoint."""
        self._check_settings()
        _write_slab_json(path, self._payload())

    def load(self, path):
        """Load only a compatible, validated checkpoint; failures leave state intact."""
        self._check_settings()
        self._load_payload(_read_slab_json(path))

    def _load_payload(self, payload):
        expected = set(self._payload())
        if (not isinstance(payload, dict) or set(payload) != expected
                or payload["format"] != "crisp-layered-population-search"
                or type(payload["version"]) is not int or payload["version"] != 1):
            raise ValueError("Unsupported native slab search checkpoint")
        if payload["settings"] != self._saved_settings:
            raise ValueError("Checkpoint search/physical settings do not match")
        generation = _integer("next_generation", payload["next_generation"])
        relaxed = _integer("n_relaxed", payload["n_relaxed"])
        if generation > self.max_generations or (self.budget_relax is not None
                                                 and relaxed > self.budget_relax):
            raise ValueError("Checkpoint exceeds search budget")
        outcomes, history = payload["outcomes"], payload["history"]
        if not isinstance(outcomes, list) or len(outcomes) != relaxed:
            raise ValueError("Invalid checkpoint outcome count")
        if not isinstance(history, list) or len(history) != generation:
            raise ValueError("Invalid checkpoint generation history")
        target, _ = self.archive._persistence_target()
        target._load_payload(payload["archive"])
        accepted = {}
        seen = set()
        for record in outcomes:
            if (not isinstance(record, dict) or set(record) !=
                    {"candidate_id", "generation", "origin", "status", "energy_per_atom", "error"}):
                raise ValueError("Invalid checkpoint outcome")
            gen = _integer("outcome generation", record["generation"])
            ident = record["candidate_id"]
            if (gen >= generation or not isinstance(ident, str) or ident in seen
                    or not ident.startswith(f"g{gen:06d}-c")):
                raise ValueError("Invalid or duplicate checkpoint candidate identity")
            seen.add(ident)
            if record["status"] in ("accepted", "duplicate"):
                if not isinstance(record["energy_per_atom"], Real) or not np.isfinite(record["energy_per_atom"]):
                    raise ValueError("Invalid checkpoint energy")
                if record["error"] is not None:
                    raise ValueError("Successful result has an error")
                if record["status"] == "accepted":
                    accepted[ident] = record
            elif (record["status"] != "rejected" or record["energy_per_atom"] is not None
                  or not isinstance(record["error"], str)):
                raise ValueError("Invalid checkpoint rejection")
        if len(accepted) != len(target.entries):
            raise ValueError("Checkpoint archive/outcomes disagree")
        for entry in target.entries:
            record = accepted.get(entry.metadata.get("candidate_id"))
            if (record is None or record["generation"] != entry.generation
                    or record["energy_per_atom"] != entry.energy):
                raise ValueError("Checkpoint archive result identity/energy mismatch")
        no_new = stagnation = 0
        best = None
        for gen, row in enumerate(history):
            if not isinstance(row, dict) or set(row) != {
                    "generation", "new", "attempted", "generated", "mutations", "best_energy"}:
                raise ValueError("Invalid checkpoint history row")
            records = [record for record in outcomes if record["generation"] == gen]
            new = sum(record["status"] == "accepted" for record in records)
            for name in ("generation", "new", "attempted", "generated", "mutations"):
                _integer(name, row[name])
            if row["generation"] != gen or row["new"] != new or row["attempted"] != len(records):
                raise ValueError("Checkpoint history/outcome counts disagree")
            energies = [record["energy_per_atom"] for record in outcomes
                        if record["generation"] <= gen and record["status"] == "accepted"]
            current_best = min(energies) if energies else None
            if row["best_energy"] != current_best:
                raise ValueError("Checkpoint best energy disagrees with accepted results")
            improved = current_best is not None and (best is None or current_best < best - 0.001)
            stagnation = 0 if improved else stagnation + 1
            no_new = 0 if new else no_new + 1
            best = current_best
        if (payload["no_new"] != no_new or payload["stagnation"] != stagnation
                or payload["best_energy"] != best or payload["stop_reason"] not in
                (None, "paused", "budget", "generations", "stagnation")):
            raise ValueError("Invalid checkpoint stopping state")
        # Build the model before committing any restored public state.
        gp, bias, projector = self._model_for(target)
        self.archive.entries = target.entries
        self.next_generation, self.n_relaxed = generation, relaxed
        self.outcomes, self.history = deepcopy(outcomes), deepcopy(history)
        self._stagnation, self._no_new, self._best_energy = stagnation, no_new, best
        self.stop_reason = payload["stop_reason"]
        self.gp, self._bias, self._projector = gp, bias, projector
