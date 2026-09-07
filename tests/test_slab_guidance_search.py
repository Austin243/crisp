"""Guided proposals retain physical targets, bounded work, and restart state."""

from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

from crisp.slab_archive import SlabArchive
from crisp.slab_fingerprint import SlabFingerprintCalculator
from crisp.slab_generation import SlabGenerationError
from crisp.slab_guidance import SlabGuidance
from crisp.slab_mutation import SlabMutations
from crisp.slab_screening import SlabGPScreening, candidate_features, select_by_gp
from crisp.slab_search import SlabRandomSearch
from crisp.surrogate import ExactGP
from test_slab_archive import _config
from test_slab_gp_search import _candidate, _quench, _seeded_generation
from test_slab_mutation_search import _seeded_mutation


def _guided(atoms, fp_calc, model, guidance, *, kappa, min_dist_ang):
    mean, std = model.predict(candidate_features(atoms, fp_calc))
    score = float(mean - kappa * std)
    child = atoms.copy()
    child.arrays["test_fp"] += 0.5
    child.positions[0, 0] += 0.01
    return child, dict(steps=guidance.max_steps, attempts=guidance.max_steps,
                       initial_score=score, final_score=score - 0.25, stop="budget")


class TestSlabGuidanceSearch(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "guided-search.json"
        self.fp_calc = SlabFingerprintCalculator(_config(), cutoff=3.2, natx=8)
        fingerprints = patch.object(self.fp_calc, "get_fingerprints",
                                    side_effect=lambda atoms: atoms.arrays["test_fp"])
        fingerprints.start()
        self.addCleanup(fingerprints.stop)

    def make_search(self, **options):
        screening = SlabGPScreening(pool_size=3, min_training_points=2, explore_every=4,
                                    max_training_points=3, kappa=0, length_scale=10,
                                    auto_tune=False)
        archive = SlabArchive(self.fp_calc, {"B": 2, "N": 2}, min_dist_ang=1.0)
        return SlabRandomSearch(archive, Mock(), **(dict(
            calculator_id="test-model-v1", seed=73, layer_groups=[1],
            max_generation_attempts=6, max_steps=9, screening=screening,
            guidance=SlabGuidance(max_steps=2, max_step=0.03, max_backtracks=3)) | options))

    def run_seeded(self, search, trials, **options):
        with patch("crisp.slab_search.generate_slabs", side_effect=_seeded_generation), \
                patch("crisp.slab_search.mutate_slab", side_effect=_seeded_mutation), \
                patch("crisp.slab_search.guide_slab", side_effect=_guided), \
                patch("crisp.slab_search.quench_slab", side_effect=_quench):
            search.run(trials, **options)

    def assert_load_failure_is_atomic(self, search):
        entries, outcomes, training = search.archive.entries, search.outcomes, search.training_rows
        previous_outcomes, previous_training = deepcopy(outcomes), deepcopy(training)
        with self.assertRaises((ValueError, TypeError)):
            search.load(self.path)
        self.assertIs(search.archive.entries, entries)
        self.assertIs(search.outcomes, outcomes)
        self.assertIs(search.training_rows, training)
        self.assertEqual(outcomes, previous_outcomes)
        self.assertEqual(training, previous_training)

    def test_only_selected_gp_proposals_are_guided_and_random_injection_bypasses(self):
        search = self.make_search(mutations=SlabMutations(random_every=5))
        events, fitted = [], []
        original_train = ExactGP.train

        def train(model, *args, **kwargs):
            fitted.append(model)
            return original_train(model, *args, **kwargs)

        def guide(*args, **kwargs):
            self.assertIsNone(args[0].calc)
            self.assertEqual(search.calc_factory.call_count, events.count("physical"))
            events.append("guide")
            return _guided(*args, **kwargs)

        def quench(*args, **kwargs):
            events.append("physical")
            args[2]()
            return _quench(*args, **kwargs)

        with patch("crisp.slab_search.generate_slabs", side_effect=_seeded_generation), \
                patch("crisp.slab_search.mutate_slab", side_effect=_seeded_mutation), \
                patch("crisp.surrogate.ExactGP.train", new=train), \
                patch("crisp.slab_search.select_by_gp", wraps=select_by_gp) as select, \
                patch("crisp.slab_search.guide_slab", side_effect=guide) as move, \
                patch("crisp.slab_search.quench_slab", side_effect=quench) as relax:
            search.run(6)
        self.assertEqual([row["guidance"]["mode"] for row in search.outcomes], [
            "bootstrap", "bootstrap", "guided", "explore", "random_injection", "guided"])
        self.assertEqual((move.call_count, relax.call_count, len(fitted)), (2, 6, 3))
        self.assertEqual(events, ["physical", "physical", "guide", "physical",
                                  "physical", "physical", "guide", "physical"])
        for call, model in zip(select.call_args_list, fitted):
            self.assertIs(call.kwargs["model"], model)
        for call, model in zip(move.call_args_list, (fitted[0], fitted[2])):
            self.assertIs(call.args[1], self.fp_calc)
            self.assertIs(call.args[2], model)
            self.assertEqual(call.kwargs, dict(kappa=0.0, min_dist_ang=1.0))
        for row in search.outcomes:
            details = row["guidance"]
            if details["mode"] != "guided":
                self.assertEqual(details, dict(mode=details["mode"], steps=0, attempts=0,
                                               initial_score=None, final_score=None, stop=None))

    def test_training_uses_guided_input_features_and_unbiased_physical_energy(self):
        search = self.make_search()
        self.run_seeded(search, 2)
        selected = _candidate(7.0, energy=-12.0)
        with patch("crisp.slab_search.generate_slabs", side_effect=lambda *a, **kw: [selected.copy()]), \
                patch("crisp.slab_search.guide_slab", side_effect=_guided) as guide, \
                patch("crisp.slab_search.quench_slab", side_effect=_quench) as quench:
            search.run(1)
        guide.assert_called_once()
        quench.assert_called_once()
        self.assertIsNot(quench.call_args.args[0], guide.call_args.args[0])
        self.assertEqual(search.training_rows[-1]["energy_per_atom"], -12.0)
        np.testing.assert_array_equal(search.training_rows[-1]["features"], [7.5] * 8 + [0.0] * 8)
        np.testing.assert_array_equal(search.archive.entries[-1].fp_pooled, [100.0] * 8 + [0.0] * 8)
        self.assertEqual(search.archive.get_best(1)[0].energy, -12.0)
        details = search.outcomes[-1]["guidance"]
        self.assertEqual(search.archive.entries[-1].metadata["guidance"], details)
        self.assertIsNot(search.archive.entries[-1].metadata["guidance"], details)
        np.testing.assert_array_equal(selected.arrays["test_fp"], 7.0)

    def test_guided_duplicates_train_but_failed_generation_and_rejections_do_not(self):
        screening = SlabGPScreening(pool_size=2, min_training_points=2, explore_every=100,
                                    max_training_points=8, auto_tune=False)
        search = self.make_search(screening=screening)
        with patch("crisp.slab_search.generate_slabs", side_effect=[
                [_candidate(1, energy=-3)], [_candidate(2, energy=-3)]]), \
                patch("crisp.slab_search.quench_slab", side_effect=_quench):
            search.run(2)
        for kind in (0, 2, 3):
            with patch("crisp.slab_search.generate_slabs", side_effect=lambda *a, **kw: [
                    _candidate(4, energy=-3, kind=kind)]), \
                    patch("crisp.slab_search.guide_slab", side_effect=_guided), \
                    patch("crisp.slab_search.quench_slab", side_effect=_quench):
                search.run(1)
        with patch("crisp.slab_search.generate_slabs", side_effect=SlabGenerationError("exhausted")), \
                patch("crisp.slab_search.guide_slab") as guide, \
                patch("crisp.slab_search.quench_slab") as quench:
            search.run(1, checkpoint=self.path)
        guide.assert_not_called()
        quench.assert_not_called()
        self.assertEqual([row["status"] for row in search.outcomes], [
            "accepted", "duplicate", "duplicate", "quench_rejected", "validation_rejected", "generation_rejected"])
        self.assertEqual([row["trial"] for row in search.training_rows], [1, 2, 3])
        self.assertIsNone(search.outcomes[-1]["guidance"])
        resumed = self.make_search(screening=screening)
        resumed.load(self.path)
        self.assertEqual(resumed.outcomes, search.outcomes)

    def test_guidance_and_invalid_post_guidance_features_fail_before_physical_quench(self):
        for failure in (ValueError("invalid gradient"), RuntimeError("backend failed"), "features"):
            with self.subTest(failure=failure):
                search = self.make_search()
                self.run_seeded(search, 2, checkpoint=self.path)
                previous, outcomes = self.path.read_bytes(), deepcopy(search.outcomes)
                training, entries = deepcopy(search.training_rows), search.archive.entries[:]

                def bad_guide(*args, **kwargs):
                    if failure != "features":
                        raise failure
                    child, details = _guided(*args, **kwargs)
                    child.arrays["test_fp"][:] = np.nan
                    return child, details

                with patch("crisp.slab_search.generate_slabs", side_effect=_seeded_generation), \
                        patch("crisp.slab_search.guide_slab", side_effect=bad_guide), \
                        patch("crisp.slab_search.quench_slab") as quench, \
                        self.assertRaises((ValueError, RuntimeError)):
                    search.run(1, checkpoint=self.path)
                quench.assert_not_called()
                self.assertEqual(search.outcomes, outcomes)
                self.assertEqual(search.training_rows, training)
                self.assertEqual(search.archive.entries, entries)
                self.assertEqual(self.path.read_bytes(), previous)

    def test_flat_and_blocked_guidance_still_receive_the_physical_quench(self):
        for stop, attempts in (("flat", 0), ("blocked", 4)):
            with self.subTest(stop=stop):
                search = self.make_search()
                self.run_seeded(search, 2)

                def no_move(atoms, *args, **kwargs):
                    return atoms.copy(), dict(steps=0, attempts=attempts, stop=stop,
                                               initial_score=-1.0, final_score=-1.0)

                with patch("crisp.slab_search.generate_slabs", side_effect=lambda *a, **kw: [_candidate(7)]), \
                        patch("crisp.slab_search.guide_slab", side_effect=no_move), \
                        patch("crisp.slab_search.quench_slab", side_effect=_quench) as quench:
                    search.run(1, checkpoint=self.path)
                quench.assert_called_once()
                self.assertEqual(search.outcomes[-1]["guidance"]["stop"], stop)
                np.testing.assert_array_equal(search.training_rows[-1]["features"], [7.0] * 8 + [0.0] * 8)
                resumed = self.make_search()
                resumed.load(self.path)
                self.assertEqual(resumed.outcomes, search.outcomes)

    def test_nonphysical_energy_never_enters_guided_archive_or_training(self):
        search = self.make_search()
        self.run_seeded(search, 2, checkpoint=self.path)
        previous, outcomes, training = self.path.read_bytes(), deepcopy(search.outcomes), deepcopy(search.training_rows)

        def bad_quench(*args, **kwargs):
            atoms = _quench(*args, **kwargs)
            atoms.calc = SinglePointCalculator(atoms, energy=1j, forces=np.zeros((len(atoms), 3)))
            return atoms

        with patch("crisp.slab_search.generate_slabs", side_effect=_seeded_generation), \
                patch("crisp.slab_search.guide_slab", side_effect=_guided), \
                patch("crisp.slab_search.quench_slab", side_effect=bad_quench), \
                self.assertRaises(RuntimeError):
            search.run(1, checkpoint=self.path)
        self.assertEqual(search.outcomes, outcomes)
        self.assertEqual(search.training_rows, training)
        self.assertEqual(self.path.read_bytes(), previous)

    def test_version_four_resume_preserves_guidance_lineage_training_and_rng(self):
        for mutations in (None, SlabMutations(random_every=5)):
            with self.subTest(mutations=mutations):
                full, first, resumed = (self.make_search(mutations=mutations) for _ in range(3))
                numpy_state, python_state = np.random.get_state(), random.getstate()
                self.run_seeded(full, 9)
                self.run_seeded(first, 4, checkpoint=self.path)
                self.assertEqual(json.loads(self.path.read_text())["version"], 4)
                resumed.load(self.path)
                self.run_seeded(resumed, 5)
                self.assertEqual(resumed.outcomes, full.outcomes)
                self.assertEqual(resumed.training_rows, full.training_rows)
                self.assertEqual(len(resumed.archive.entries), len(full.archive.entries))
                for expected, actual in zip(full.archive.entries, resumed.archive.entries):
                    self.assertEqual(actual.energy, expected.energy)
                    self.assertEqual(actual.metadata, expected.metadata)
                    np.testing.assert_allclose(actual.atoms.positions, expected.atoms.positions, atol=1e-12)
                current = np.random.get_state()
                self.assertEqual(numpy_state[0], current[0])
                np.testing.assert_array_equal(numpy_state[1], current[1])
                self.assertEqual(numpy_state[2:], current[2:])
                self.assertEqual(python_state, random.getstate())

    def test_corrupt_guidance_settings_modes_budgets_and_scores_cannot_partially_load(self):
        source, destination = self.make_search(), self.make_search()
        self.run_seeded(source, 6, checkpoint=self.path)
        self.run_seeded(destination, 1)
        original = json.loads(self.path.read_text())
        documents = []
        for field, value in (("mode", "bootstrap"), ("steps", 3), ("steps", True),
                             ("attempts", 9), ("attempts", -1), ("initial_score", None),
                             ("final_score", np.inf), ("stop", "unknown")):
            document = deepcopy(original)
            document["outcomes"][2]["guidance"][field] = value
            documents.append(document)
        document = deepcopy(original)
        details = document["outcomes"][2]["guidance"]
        details["final_score"] = details["initial_score"] + 1
        documents.append(document)
        for steps, attempts, stop in ((0, 1, "flat"), (1, 1, "blocked"), (1, 5, "flat")):
            document = deepcopy(original)
            details = document["outcomes"][2]["guidance"]
            details.update(steps=steps, attempts=attempts, stop=stop)
            if steps == 0:
                details["final_score"] = details["initial_score"]
            # Keep archive provenance consistent so the movement budget check
            # itself must reject these impossible traces.
            for entry in document["archive"]["entries"]:
                if entry["metadata"]["search_trial"] == 3:
                    entry["metadata"]["guidance"] = deepcopy(details)
            documents.append(document)
        document = deepcopy(original)
        document["outcomes"][0]["guidance"]["steps"] = 1
        documents.append(document)
        document = deepcopy(original)
        document["settings"]["guidance"]["max_step"] *= 2
        documents.append(document)
        document = deepcopy(original)
        document["archive"]["entries"][0]["metadata"]["guidance"]["mode"] = "guided"
        documents.append(document)
        for document in documents:
            self.path.write_text(json.dumps(document))
            self.assert_load_failure_is_atomic(destination)

    def test_guidance_requires_screening_and_disabled_mode_preserves_old_schemas(self):
        with self.assertRaises((ValueError, TypeError)):
            self.make_search(screening=None)
        with self.assertRaises(TypeError):
            self.make_search(guidance={"max_steps": 2})
        for screening, mutations, version in ((None, None, 1), (SlabGPScreening(), None, 2),
                                               (None, SlabMutations(), 3)):
            search = self.make_search(screening=screening, mutations=mutations, guidance=None)
            with patch("crisp.slab_search.generate_slabs", side_effect=_seeded_generation), \
                    patch("crisp.slab_search.guide_slab") as guide, \
                    patch("crisp.slab_search.quench_slab", side_effect=_quench):
                search.run(1, checkpoint=self.path)
            guide.assert_not_called()
            payload = json.loads(self.path.read_text())
            self.assertEqual(payload["version"], version)
            self.assertNotIn("guidance", payload["settings"])
            self.assertNotIn("guidance", payload["outcomes"][0])
            self.assertNotIn("guidance", payload["archive"]["entries"][0]["metadata"])


if __name__ == "__main__":
    unittest.main()
