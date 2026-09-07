"""Candidate screening, physical outcomes, and resumable GP training state."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

from crisp.slab_archive import SlabArchive
from crisp.slab_fingerprint import SlabFingerprintCalculator
from crisp.slab_generation import SlabGenerationError
from crisp.slab_screening import SlabGPScreening
from crisp.slab_search import SlabRandomSearch
from test_slab_archive import _config, _sheet
from test_slab_search import _relax


def _candidate(value, *, energy=None, kind=0):
    atoms = _sheet()
    atoms.arrays["test_fp"][:] = value
    atoms.info.update(test_kind=kind, energy=value if energy is None else energy)
    return atoms


def _quench(atoms, config, calc_factory, **options):
    relaxed = _relax(atoms, config, calc_factory, **options)
    # A different post-quench descriptor prevents accidental relaxed-feature labels.
    relaxed.arrays["test_fp"][:] = 100.0
    relaxed.calc = SinglePointCalculator(
        relaxed, energy=atoms.info["energy"] * len(relaxed),
        forces=np.zeros((len(relaxed), 3)))
    return relaxed


def _seeded_generation(*args, seed, **kwargs):
    return [_candidate(float(np.random.default_rng(seed).uniform(-4, 4)))]


class TestSlabGPSearch(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "gp-search.json"
        self.fp_calc = SlabFingerprintCalculator(_config(), cutoff=3.2, natx=8)
        descriptor = patch.object(self.fp_calc, "get_fingerprints",
                                  side_effect=lambda atoms: atoms.arrays["test_fp"])
        self.fingerprints = descriptor.start()
        self.addCleanup(descriptor.stop)

    def make_search(self, **options):
        screening = SlabGPScreening(pool_size=3, min_training_points=2, explore_every=4,
                                    max_training_points=3, kappa=0.0, length_scale=10.0,
                                    auto_tune=False)
        archive = SlabArchive(self.fp_calc, {"B": 2, "N": 2}, min_dist_ang=1.0)
        return SlabRandomSearch(archive, Mock(), **(dict(
            calculator_id="test-model-v1", seed=73, layer_groups=[1],
            max_generation_attempts=6, max_steps=9, screening=screening) | options))

    def run_seeded(self, search, trials, **options):
        with patch("crisp.slab_search.generate_slabs", side_effect=_seeded_generation), \
                patch("crisp.slab_search.quench_slab", side_effect=_quench):
            search.run(trials, **options)

    def assert_load_failure_is_atomic(self, search):
        entries, outcomes, training = search.archive.entries, search.outcomes, search.training_rows
        original_outcomes, original_training = deepcopy(outcomes), deepcopy(training)
        with self.assertRaises((ValueError, TypeError)):
            search.load(self.path)
        self.assertIs(search.archive.entries, entries)
        self.assertIs(search.outcomes, outcomes)
        self.assertIs(search.training_rows, training)
        self.assertEqual(outcomes, original_outcomes)
        self.assertEqual(training, original_training)

    def test_training_pairs_raw_features_with_physical_energy_including_duplicates(self):
        search = self.make_search()
        candidates = [_candidate(1.0, energy=-3.0), _candidate(7.0, energy=-3.0)]
        with patch("crisp.slab_search.generate_slabs", side_effect=[[a] for a in candidates]), \
                patch("crisp.slab_search.quench_slab", side_effect=_quench) as quench:
            search.run(2)
        self.assertEqual(search.counts, {"accepted": 1, "duplicate": 1})
        self.assertEqual(quench.call_count, 2)
        self.assertEqual([row["energy_per_atom"] for row in search.training_rows], [-3.0, -3.0])
        self.assertEqual([row["trial"] for row in search.training_rows], [1, 2])
        for row, value in zip(search.training_rows, (1.0, 7.0)):
            np.testing.assert_array_equal(row["features"], [value] * 8 + [0.0] * 8)
        np.testing.assert_array_equal(search.archive.entries[0].fp_pooled, [100.0] * 8 + [0.0] * 8)
        candidates[0].arrays["test_fp"][:] = np.nan
        self.assertTrue(np.isfinite(search.training_rows[0]["features"]).all())

    def test_bootstrap_partial_gp_pool_and_periodic_exploration_keep_quench_budget(self):
        search = self.make_search()
        low, high, random = _candidate(1.0), _candidate(5.0), _candidate(8.0)
        responses = [[_candidate(1.0)], [_candidate(5.0)], SlabGenerationError("attempt limit"),
                     [low], [high], [random]]
        with patch("crisp.slab_search.generate_slabs", side_effect=responses) as generate, \
                patch("crisp.slab_search.quench_slab", side_effect=_quench) as quench:
            search.run(4)
        self.assertEqual(quench.call_count, 4)
        self.assertEqual(generate.call_count, 6)
        details = [outcome["screening"] for outcome in search.outcomes]
        self.assertEqual([row["mode"] for row in details], ["bootstrap", "bootstrap", "gp", "explore"])
        self.assertEqual([row["requested"] for row in details], [1, 1, 3, 1])
        self.assertEqual([row["generated"] for row in details], [1, 1, 2, 1])
        self.assertIs(quench.call_args_list[2].args[0], low)
        self.assertIs(quench.call_args_list[3].args[0], random)
        self.assertEqual(details[2]["selected_member"], 1)
        self.assertEqual(details[2]["candidate_seed"], generate.call_args_list[3].kwargs["seed"])
        self.assertTrue(np.isfinite(details[2]["mean"]))
        self.assertIsNone(details[3]["mean"])
        for call in generate.call_args_list:
            self.assertEqual(call.args[2], 1)
            self.assertEqual(call.kwargs["max_attempts"], 6)
        self.assertEqual([row["trial"] for row in search.training_rows], [2, 3, 4])

    def test_rejections_never_train_and_exhausted_gp_pool_never_quenches(self):
        search = self.make_search()
        responses = [SlabGenerationError("generation failed"), [_candidate(1.0, kind=2)],
                     [_candidate(1.0, kind=3)], [_candidate(1.0)], [_candidate(5.0)]]
        responses += [SlabGenerationError("generation failed")] * 3
        with patch("crisp.slab_search.generate_slabs", side_effect=responses) as generate, \
                patch("crisp.slab_search.quench_slab", side_effect=_quench) as quench:
            search.run(6, checkpoint=self.path)
        self.assertEqual([row["status"] for row in search.outcomes], [
            "generation_rejected", "quench_rejected", "validation_rejected",
            "accepted", "accepted", "generation_rejected"])
        self.assertEqual([row["trial"] for row in search.training_rows], [4, 5])
        self.assertEqual(generate.call_count, 8)
        self.assertEqual(quench.call_count, 4)
        last = search.outcomes[-1]["screening"]
        self.assertEqual((last["mode"], last["requested"], last["generated"]), ("gp", 3, 0))
        self.assertIsNone(last["selected_member"])
        resumed = self.make_search()
        resumed.load(self.path)
        self.assertEqual(resumed.outcomes, search.outcomes)
        self.assertEqual(resumed.training_rows, search.training_rows)

    def test_version_two_resume_reproduces_real_gp_choices_with_bounded_training(self):
        full, first, resumed = (self.make_search() for _ in range(3))
        self.run_seeded(full, 9)
        self.run_seeded(first, 4, checkpoint=self.path)
        payload = json.loads(self.path.read_text())
        self.assertEqual(payload["version"], 2)
        self.assertEqual(payload["training"], first.training_rows)
        resumed.load(self.path)
        self.run_seeded(resumed, 5, checkpoint=self.path)
        self.assertEqual(resumed.outcomes, full.outcomes)
        self.assertEqual(resumed.training_rows, full.training_rows)
        self.assertEqual([row["trial"] for row in resumed.training_rows], [7, 8, 9])
        self.assertIn("gp", [row["screening"]["mode"] for row in resumed.outcomes])
        self.assertEqual(resumed.archive.get_all_energies().tolist(), full.archive.get_all_energies().tolist())
        for actual, expected in zip(resumed.archive.entries, full.archive.entries):
            self.assertEqual(actual.metadata, expected.metadata)
            np.testing.assert_allclose(actual.atoms.positions, expected.atoms.positions, atol=1e-12)
        plain = self.make_search(screening=None)
        self.assert_load_failure_is_atomic(plain)
        plain.save(self.path)
        self.assertEqual(json.loads(self.path.read_text())["version"], 1)
        self.assert_load_failure_is_atomic(resumed)

    def test_incompatible_and_corrupt_training_or_selection_cannot_partially_load(self):
        source, destination = self.make_search(), self.make_search()
        self.run_seeded(source, 6, checkpoint=self.path)
        self.run_seeded(destination, 1)
        original = json.loads(self.path.read_text())
        changed = self.make_search(screening=SlabGPScreening(pool_size=2))
        self.assert_load_failure_is_atomic(changed)
        documents = [original | {"training": {}}, original | {"training": []},
                     original | {"version": 1}]
        for field, value in (("trial", 1), ("energy_per_atom", 12345.0), ("energy_per_atom", True),
                             ("features", [1.0]), ("features", [np.inf] * 16)):
            document = deepcopy(original)
            document["training"][0][field] = value
            documents.append(document)
        for field, value in (("mode", "bootstrap"), ("requested", 4), ("generated", 0),
                             ("selected_member", 3), ("candidate_seed", 0),
                             ("mean", None), ("std", -1.0), ("score", 12345.0)):
            document = deepcopy(original)
            document["outcomes"][2]["screening"][field] = value
            documents.append(document)
        document = deepcopy(original)
        document["archive"]["entries"][0]["metadata"]["candidate_seed"] = 0
        documents.append(document)
        for document in documents:
            self.path.write_text(json.dumps(document))
            with self.subTest(document=str(document)[:120]):
                self.assert_load_failure_is_atomic(destination)

    def test_fatal_descriptors_and_gp_errors_prevent_quench_and_archive_errors_preserve_training(self):
        for origin in ("descriptor", "train", "predict", "archive"):
            with self.subTest(origin=origin):
                search = self.make_search()
                self.run_seeded(search, 2, checkpoint=self.path)
                training, outcomes = search.training_rows, deepcopy(search.outcomes)
                original_training = deepcopy(training)
                entries, previous = search.archive.entries[:], self.path.read_bytes()
                if origin == "descriptor":
                    failure = patch.object(self.fp_calc, "get_fingerprints", side_effect=RuntimeError("fatal"))
                elif origin == "archive":
                    failure = patch.object(self.fp_calc, "_fp_dist", side_effect=RuntimeError("fatal"))
                else:
                    failure = patch("crisp.slab_screening.ExactGP." + origin, side_effect=RuntimeError("fatal"))
                with patch("crisp.slab_search.generate_slabs", side_effect=_seeded_generation), \
                        patch("crisp.slab_search.quench_slab", side_effect=_quench) as quench, \
                        failure, self.assertRaisesRegex(RuntimeError, "fatal"):
                    search.run(1, checkpoint=self.path)
                self.assertEqual(quench.call_count, 1 if origin == "archive" else 0)
                self.assertIs(search.training_rows, training)
                self.assertEqual(search.training_rows, original_training)
                self.assertEqual(search.outcomes, outcomes)
                self.assertEqual(search.archive.entries, entries)
                self.assertEqual(self.path.read_bytes(), previous)
                self.run_seeded(search, 1, checkpoint=self.path)
                self.assertEqual(search.completed_trials, 3)
                self.assertEqual(search.training_rows[-1]["trial"], 3)

    def test_failed_save_keeps_completed_gp_work_and_previous_checkpoint(self):
        search = self.make_search()
        self.run_seeded(search, 2, checkpoint=self.path)
        previous = self.path.read_bytes()
        with patch("crisp.slab_archive.os.fsync", side_effect=OSError("disk error")), \
                self.assertRaises(OSError):
            self.run_seeded(search, 1, checkpoint=self.path)
        self.assertEqual(search.completed_trials, 3)
        self.assertEqual(search.training_rows[-1]["trial"], 3)
        self.assertEqual(self.path.read_bytes(), previous)
        previous_run = self.make_search()
        previous_run.load(self.path)
        self.assertEqual(previous_run.completed_trials, 2)
        search.save(self.path)
        previous_run.load(self.path)
        self.assertEqual(previous_run.training_rows, search.training_rows)


if __name__ == "__main__":
    unittest.main()
