"""Mutation scheduling, archived-parent lineage, and checkpoint continuity."""

from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from crisp.slab_archive import SlabArchive
from crisp.slab_fingerprint import SlabFingerprintCalculator
from crisp.slab_generation import SlabGenerationError
from crisp.slab_mutation import SlabMutationError, SlabMutations, mutate_slab
from crisp.slab_screening import SlabGPScreening
from crisp.slab_search import SlabRandomSearch
from test_slab_archive import _config
from test_slab_gp_search import _candidate, _quench, _seeded_generation


def _seeded_mutation(parent, config, mutations, *, seed, min_dist_ang):
    child = parent.copy()
    value = float(np.random.default_rng(seed).uniform(-4, 4))
    child.arrays["test_fp"][:] = value
    child.info.update(energy=value, test_kind=0)
    return child


class TestSlabMutationSearch(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "mutation-search.json"
        self.fp_calc = SlabFingerprintCalculator(_config(), cutoff=3.2, natx=8)
        fingerprints = patch.object(self.fp_calc, "get_fingerprints",
                                    side_effect=lambda atoms: atoms.arrays["test_fp"])
        fingerprints.start()
        self.addCleanup(fingerprints.stop)

    def make_search(self, **options):
        archive = SlabArchive(self.fp_calc, {"B": 2, "N": 2}, min_dist_ang=1.0)
        return SlabRandomSearch(archive, Mock(), **(dict(
            calculator_id="test-model-v1", seed=73, layer_groups=[1],
            max_generation_attempts=6, max_steps=9,
            mutations=SlabMutations(random_every=5, max_attempts=7)) | options))

    def run_seeded(self, search, trials, **options):
        with patch("crisp.slab_search.generate_slabs", side_effect=_seeded_generation), \
                patch("crisp.slab_search.mutate_slab", side_effect=_seeded_mutation), \
                patch("crisp.slab_search.quench_slab", side_effect=_quench):
            search.run(trials, **options)

    def assert_load_failure_is_atomic(self, search):
        entries, outcomes, training = search.archive.entries, search.outcomes, search.training_rows
        original_outcomes, original_training = deepcopy(outcomes), deepcopy(training)
        with self.assertRaises((TypeError, ValueError)):
            search.load(self.path)
        self.assertIs(search.archive.entries, entries)
        self.assertIs(search.outcomes, outcomes)
        self.assertIs(search.training_rows, training)
        self.assertEqual(outcomes, original_outcomes)
        self.assertEqual(training, original_training)

    def test_random_injection_counts_failed_trials_and_only_accepted_parents_are_used(self):
        mutations = SlabMutations(random_every=3, max_attempts=7)
        search = self.make_search(mutations=mutations)
        random_candidates = [[_candidate(1.0, energy=-3.0)], SlabGenerationError("exhausted"),
                             [_candidate(5.0, energy=-5.0)]]
        children = [_candidate(2.0, energy=-3.0), _candidate(3.0, kind=3),
                    _candidate(4.0, energy=-4.0)]
        with patch("crisp.slab_search.generate_slabs", side_effect=random_candidates) as generate, \
                patch("crisp.slab_search.mutate_slab", side_effect=children) as mutate, \
                patch("crisp.slab_search.quench_slab", side_effect=_quench) as quench:
            search.run(6)
        self.assertEqual([row["status"] for row in search.outcomes], [
            "accepted", "duplicate", "generation_rejected", "validation_rejected", "accepted", "accepted"])
        self.assertEqual([row["proposal"] for row in search.outcomes], [
            dict(source="random", parent_trial=None), dict(source="mutation", parent_trial=1), None,
            dict(source="mutation", parent_trial=1), dict(source="mutation", parent_trial=1),
            dict(source="random", parent_trial=None)])
        self.assertEqual((generate.call_count, mutate.call_count, quench.call_count), (3, 3, 5))
        for call, trial in zip(mutate.call_args_list, (2, 4, 5)):
            self.assertEqual(call.args[1:], (_config(), mutations))
            self.assertEqual(call.kwargs, dict(seed=search.outcomes[trial - 1]["seed"], min_dist_ang=1.0))
            np.testing.assert_array_equal(call.args[0].positions, search.archive.entries[0].atoms.positions)
        for entry in search.archive.entries:
            self.assertEqual(entry.metadata["proposal"], search.outcomes[entry.metadata["search_trial"] - 1]["proposal"])

    def test_empty_archive_keeps_generating_random_candidates(self):
        search = self.make_search()
        with patch("crisp.slab_search.generate_slabs", side_effect=[
                SlabGenerationError("exhausted"), [_candidate(1.0, kind=2)], [_candidate(2.0)]]) as generate, \
                patch("crisp.slab_search.mutate_slab") as mutate, \
                patch("crisp.slab_search.quench_slab", side_effect=_quench):
            search.run(3)
        self.assertEqual(generate.call_count, 3)
        mutate.assert_not_called()
        self.assertEqual(search.outcomes[2]["proposal"], dict(source="random", parent_trial=None))
        self.assertEqual([entry.metadata["search_trial"] for entry in search.archive.entries], [3])

    def test_mutation_exhaustion_has_no_random_fallback_and_fatal_errors_do_not_consume_trials(self):
        search = self.make_search()
        self.run_seeded(search, 1, checkpoint=self.path)
        with patch("crisp.slab_search.mutate_slab", side_effect=SlabMutationError("attempt limit")), \
                patch("crisp.slab_search.generate_slabs") as generate, \
                patch("crisp.slab_search.quench_slab") as quench:
            search.run(1, checkpoint=self.path)
        generate.assert_not_called()
        quench.assert_not_called()
        self.assertEqual(search.outcomes[-1]["status"], "generation_rejected")
        self.assertIsNone(search.outcomes[-1]["proposal"])
        previous = self.path.read_bytes()
        outcomes, training, entries = deepcopy(search.outcomes), deepcopy(search.training_rows), search.archive.entries[:]
        for error in (ValueError("fatal geometry/configuration"), RuntimeError("unexpected mutation error")):
            with self.subTest(error=error), \
                    patch("crisp.slab_search.mutate_slab", side_effect=error), \
                    patch("crisp.slab_search.generate_slabs") as generate, \
                    patch("crisp.slab_search.quench_slab") as quench, self.assertRaises(type(error)):
                search.run(1, checkpoint=self.path)
            generate.assert_not_called()
            quench.assert_not_called()
            self.assertEqual(search.outcomes, outcomes)
            self.assertEqual(search.training_rows, training)
            self.assertEqual(search.archive.entries, entries)
            self.assertEqual(self.path.read_bytes(), previous)
        self.run_seeded(search, 1)
        self.assertEqual(search.completed_trials, 3)

    def test_gp_bootstrap_exploration_and_random_interval_apply_to_whole_pools(self):
        screening = SlabGPScreening(pool_size=3, min_training_points=2, explore_every=4,
                                    max_training_points=3, auto_tune=False)
        search = self.make_search(screening=screening)
        with patch("crisp.slab_search.generate_slabs", side_effect=_seeded_generation) as generate, \
                patch("crisp.slab_search.mutate_slab", side_effect=_seeded_mutation) as mutate, \
                patch("crisp.slab_search.quench_slab", side_effect=_quench) as quench:
            search.run(6)
        self.assertEqual([row["screening"]["mode"] for row in search.outcomes], [
            "bootstrap", "bootstrap", "gp", "explore", "gp", "gp"])
        self.assertEqual([row["proposal"]["source"] for row in search.outcomes], [
            "random", "random", "mutation", "random", "random", "mutation"])
        self.assertEqual((generate.call_count, mutate.call_count, quench.call_count), (6, 6, 6))
        for row in search.outcomes:
            if row["proposal"]["source"] == "mutation":
                parent = row["proposal"]["parent_trial"]
                self.assertLess(parent, row["trial"])
                self.assertEqual(search.outcomes[parent - 1]["status"], "accepted")

    def test_partial_gp_mutation_pool_records_selected_parent_and_never_archives_unselected_members(self):
        screening = SlabGPScreening(pool_size=3, min_training_points=2, explore_every=9, auto_tune=False)
        search = self.make_search(screening=screening)
        self.run_seeded(search, 2)
        previous_entries = search.archive.entries[:]
        parents = []
        def mutate(parent, config, mutations, **options):
            self.assertEqual(len(search.archive.entries), len(previous_entries))
            parents.append(parent)
            if len(parents) == 1:
                raise SlabMutationError("attempt limit")
            return _candidate(float(len(parents)), energy=-10.0 - len(parents))
        predictions = [dict(mean=1.0, std=0.0, score=1.0), dict(mean=0.0, std=0.0, score=0.0)]
        with patch("crisp.slab_search.mutate_slab", side_effect=mutate), \
                patch("crisp.slab_search.select_by_gp", return_value=(1, predictions)), \
                patch("crisp.slab_search.quench_slab", side_effect=_quench) as quench:
            search.run(1)
        outcome = search.outcomes[-1]
        self.assertEqual((outcome["screening"]["generated"], outcome["screening"]["selected_member"]), (2, 2))
        self.assertEqual(quench.call_count, 1)
        self.assertEqual(len(search.archive.entries), len(previous_entries) + 1)
        parent_trial = outcome["proposal"]["parent_trial"]
        parent_entry = next(entry for entry in previous_entries if entry.metadata["search_trial"] == parent_trial)
        self.assertEqual(parents[2].info, parent_entry.atoms.info)
        self.assertEqual(search.archive.entries[-1].metadata["proposal"], outcome["proposal"])
        self.assertEqual(search.training_rows[-1]["energy_per_atom"], -13.0)

    def test_real_mutation_preserves_archived_parent_buffers_and_metadata(self):
        search = self.make_search(mutations=SlabMutations(max_displacement=0.02, max_strain=0.01))
        self.run_seeded(search, 1)
        parent = search.archive.entries[0]
        before, info, metadata = parent.atoms.copy(), deepcopy(parent.atoms.info), deepcopy(parent.metadata)
        fingerprint = parent.fp.copy()
        with patch("crisp.slab_search.mutate_slab", wraps=mutate_slab) as mutate, \
                patch("crisp.slab_search.quench_slab", side_effect=_quench):
            search.run(1)
        mutate.assert_called_once()
        self.assertEqual(search.outcomes[-1]["proposal"], dict(source="mutation", parent_trial=1))
        np.testing.assert_array_equal(parent.atoms.positions, before.positions)
        np.testing.assert_array_equal(parent.atoms.cell, before.cell)
        np.testing.assert_array_equal(parent.atoms.arrays["test_fp"], before.arrays["test_fp"])
        np.testing.assert_array_equal(parent.fp, fingerprint)
        self.assertEqual(parent.atoms.info, info)
        self.assertEqual(parent.metadata, metadata)
        self.assertIsNone(parent.atoms.calc)

    def test_mutation_resume_reproduces_lineage_training_and_global_rng_state(self):
        screening = SlabGPScreening(pool_size=3, min_training_points=2, explore_every=4,
                                    max_training_points=3, auto_tune=False)
        for gp in (None, screening):
            with self.subTest(screening=gp):
                full, split, resumed = (self.make_search(screening=gp) for _ in range(3))
                numpy_state, python_state = np.random.get_state(), random.getstate()
                self.run_seeded(full, 12)
                self.run_seeded(split, 5, checkpoint=self.path)
                payload = json.loads(self.path.read_text())
                self.assertEqual(payload["version"], 3)
                self.assertEqual("training" in payload, gp is not None)
                resumed.load(self.path)
                self.run_seeded(resumed, 7)
                self.assertEqual(resumed.outcomes, full.outcomes)
                self.assertEqual(resumed.training_rows, full.training_rows)
                self.assertGreater(len(full.archive.entries), 2)
                for actual, expected in zip(resumed.archive.entries, full.archive.entries):
                    self.assertEqual(actual.metadata, expected.metadata)
                    self.assertEqual(actual.energy, expected.energy)
                    np.testing.assert_allclose(actual.atoms.positions, expected.atoms.positions, atol=1e-12)
                current = np.random.get_state()
                self.assertEqual(numpy_state[0], current[0])
                np.testing.assert_array_equal(numpy_state[1], current[1])
                self.assertEqual(numpy_state[2:], current[2:])
                self.assertEqual(python_state, random.getstate())

    def test_corrupt_lineage_settings_or_archive_metadata_cannot_partially_load(self):
        screening = SlabGPScreening(pool_size=3, min_training_points=2, explore_every=4,
                                    max_training_points=3, auto_tune=False)
        for gp in (None, screening):
            with self.subTest(screening=gp):
                source, destination = (self.make_search(screening=gp) for _ in range(2))
                self.run_seeded(source, 9, checkpoint=self.path)
                self.run_seeded(destination, 3)
                original = json.loads(self.path.read_text())
                index = next(i for i, row in enumerate(original["outcomes"])
                             if row["proposal"]["source"] == "mutation")
                documents = []
                for replacement in (None, {}, dict(source="random", parent_trial=None),
                                    dict(source="mutation", parent_trial=True),
                                    dict(source="mutation", parent_trial=9)):
                    document = deepcopy(original)
                    document["outcomes"][index]["proposal"] = replacement
                    documents.append(document)
                document = deepcopy(original)
                document["settings"]["mutations"]["random_every"] = 99
                documents.append(document)
                document = deepcopy(original)
                document["archive"]["entries"][0]["metadata"]["proposal"]["parent_trial"] = 1
                documents.append(document)
                for document in documents:
                    self.path.write_text(json.dumps(document))
                    self.assert_load_failure_is_atomic(destination)

    def test_disabled_mutations_preserve_checkpoint_schemas_and_empty_v3_resumes(self):
        for gp, version in ((None, 1), (SlabGPScreening(), 2)):
            with self.subTest(version=version):
                search = self.make_search(mutations=None, screening=gp)
                with patch("crisp.slab_search.mutate_slab") as mutate, \
                        patch("crisp.slab_search.generate_slabs", side_effect=_seeded_generation), \
                        patch("crisp.slab_search.quench_slab", side_effect=_quench):
                    search.run(1, checkpoint=self.path)
                mutate.assert_not_called()
                payload = json.loads(self.path.read_text())
                self.assertEqual(payload["version"], version)
                self.assertNotIn("mutations", payload["settings"])
                self.assertNotIn("proposal", payload["outcomes"][0])
                self.assertNotIn("proposal", payload["archive"]["entries"][0]["metadata"])
        empty = self.make_search()
        empty.run(0, checkpoint=self.path)
        resumed = self.make_search()
        resumed.load(self.path)
        self.assertEqual(resumed.completed_trials, 0)
        self.assertEqual(resumed.archive.entries, [])
        with self.assertRaises(TypeError):
            self.make_search(mutations={"random_every": 5})


if __name__ == "__main__":
    unittest.main()
