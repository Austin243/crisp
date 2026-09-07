"""Optional physical cell relaxation, energy targets, and restart provenance."""

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
from crisp.slab_cell_relaxation import SlabCellRelaxation
from crisp.slab_fingerprint import SlabFingerprintCalculator
from crisp.slab_generation import SlabGenerationError
from crisp.slab_guidance import SlabGuidance
from crisp.slab_mutation import SlabMutations
from crisp.slab_screening import SlabGPScreening, candidate_features
from crisp.slab_search import SlabRandomSearch
from test_slab_archive import _config
from test_slab_gp_search import _candidate, _quench, _seeded_generation
from test_slab_guidance_search import _guided
from test_slab_mutation_search import _seeded_mutation


def _cell_quench(atoms, config, calc_factory, **options):
    relaxed = _quench(atoms, config, calc_factory, **options)
    cell = relaxed.cell.array.copy()
    cell[:2, :2] *= 1.002
    relaxed.set_cell(cell, scale_atoms=True)
    relaxed.info["slab_quench_stress_max"] = 0.002
    relaxed.calc = SinglePointCalculator(
        relaxed, energy=atoms.info["energy"] * len(relaxed),
        forces=np.zeros((len(relaxed), 3)))
    return relaxed


class TestSlabCellSearch(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "cell-search.json"
        self.fp_calc = SlabFingerprintCalculator(_config(), cutoff=3.2, natx=8)
        fingerprints = patch.object(self.fp_calc, "get_fingerprints",
                                    side_effect=lambda atoms: atoms.arrays["test_fp"])
        fingerprints.start()
        self.addCleanup(fingerprints.stop)

    def make_search(self, **options):
        archive = SlabArchive(self.fp_calc, {"B": 2, "N": 2}, min_dist_ang=1.0)
        return SlabRandomSearch(archive, Mock(), **(dict(
            calculator_id="test-model-v1", seed=73, layer_groups=[1],
            max_generation_attempts=6, fmax=.01, max_steps=9,
            cell_relaxation=SlabCellRelaxation(stress_max=.01)) | options))

    def gp_options(self):
        return dict(screening=SlabGPScreening(
            pool_size=3, min_training_points=2, explore_every=4,
            max_training_points=3, kappa=0, length_scale=10, auto_tune=False))

    def run_seeded(self, search, trials, **options):
        with patch("crisp.slab_search.generate_slabs", side_effect=_seeded_generation), \
                patch("crisp.slab_search.mutate_slab", side_effect=_seeded_mutation), \
                patch("crisp.slab_search.guide_slab", side_effect=_guided), \
                patch("crisp.slab_search.quench_slab", side_effect=_cell_quench):
            search.run(trials, **options)

    def assert_load_failure_is_atomic(self, search):
        entries, outcomes, training = search.archive.entries, search.outcomes, search.training_rows
        old_outcomes, old_training = deepcopy(outcomes), deepcopy(training)
        with self.assertRaises((ValueError, TypeError)):
            search.load(self.path)
        self.assertIs(search.archive.entries, entries)
        self.assertIs(search.outcomes, outcomes)
        self.assertIs(search.training_rows, training)
        self.assertEqual(outcomes, old_outcomes)
        self.assertEqual(training, old_training)

    def test_independent_option_forwards_physical_budgets_and_records_stress_for_all_statuses(self):
        settings = SlabCellRelaxation(stress_max=.03)
        search = self.make_search(cell_relaxation=settings)
        candidates = [[_candidate(1, energy=-3)], SlabGenerationError("attempt limit"),
                      [_candidate(2, kind=2)], [_candidate(3, kind=3)],
                      [_candidate(4, energy=-3)], [_candidate(5, kind=5)],
                      [_candidate(6, energy=-4)]]
        with patch("crisp.slab_search.generate_slabs", side_effect=candidates), \
                patch("crisp.slab_search.quench_slab", side_effect=_cell_quench) as quench:
            search.run(7, checkpoint=self.path)
        self.assertEqual([row["status"] for row in search.outcomes], [
            "accepted", "generation_rejected", "quench_rejected", "validation_rejected",
            "duplicate", "quench_rejected", "accepted"])
        self.assertEqual(quench.call_count, 6)
        for call in quench.call_args_list:
            self.assertIs(call.args[2], search.calc_factory)
            self.assertEqual(call.kwargs, dict(fmax=.01, max_steps=9, cell_relaxation=settings))
        for row in search.outcomes:
            self.assertEqual(row["stress_max"], None if row["quench_steps"] is None else .002)
        self.assertEqual([entry.energy for entry in search.archive.get_best()], [-4., -3.])
        for entry in search.archive.entries:
            self.assertEqual(entry.metadata["stress_max"], .002)
            self.assertIsNone(entry.atoms.calc)
        restored = self.make_search(cell_relaxation=settings)
        restored.load(self.path)
        self.assertEqual(restored.outcomes, search.outcomes)
        payload = json.loads(self.path.read_text())
        self.assertEqual(payload["version"], 5)
        self.assertNotIn("training", payload)

    def test_gp_and_guidance_train_on_physical_input_and_rank_only_final_energy(self):
        options = self.gp_options() | dict(guidance=SlabGuidance(max_steps=2),
                                         mutations=SlabMutations(random_every=5))
        search = self.make_search(**options)
        inputs = {}

        def observed_quench(candidate, *args, **kwargs):
            inputs[search.completed_trials + 1] = candidate.copy()
            return _cell_quench(candidate, *args, **kwargs)

        with patch("crisp.slab_search.generate_slabs", side_effect=_seeded_generation), \
                patch("crisp.slab_search.mutate_slab", side_effect=_seeded_mutation), \
                patch("crisp.slab_search.guide_slab", side_effect=_guided) as guide, \
                patch("crisp.slab_search.quench_slab", side_effect=observed_quench):
            search.run(6, checkpoint=self.path)
        self.assertGreater(guide.call_count, 0)
        self.assertTrue(any(row["proposal"]["source"] == "mutation" for row in search.outcomes))
        for row in search.training_rows:
            candidate = inputs[row["trial"]]
            expected = candidate_features(candidate, self.fp_calc)
            np.testing.assert_array_equal(row["features"], expected)
            self.assertEqual(row["energy_per_atom"], candidate.info["energy"])
        for entry in search.archive.entries:
            candidate = inputs[entry.metadata["search_trial"]]
            self.assertEqual(entry.energy, candidate.info["energy"])
            self.assertEqual(entry.enthalpy, entry.energy)
            self.assertEqual(entry.pressure, 0)
            np.testing.assert_allclose(entry.atoms.cell[:2, :2], candidate.cell[:2, :2] * 1.002)
            np.testing.assert_array_equal(entry.atoms.cell[2], candidate.cell[2])
            np.testing.assert_array_equal(entry.fp_pooled, [100.] * 8 + [0.] * 8)

    def test_version_five_resume_preserves_all_modes_and_global_rng(self):
        modes = ({}, dict(mutations=SlabMutations(random_every=5)), self.gp_options(),
                 self.gp_options() | dict(mutations=SlabMutations(random_every=5),
                                          guidance=SlabGuidance(max_steps=2)))
        for options in modes:
            with self.subTest(options=options):
                full, first, resumed = (self.make_search(**options) for _ in range(3))
                numpy_state, python_state = np.random.get_state(), random.getstate()
                self.run_seeded(full, 9)
                self.run_seeded(first, 4, checkpoint=self.path)
                payload = json.loads(self.path.read_text())
                self.assertEqual(payload["version"], 5)
                self.assertEqual("training" in payload, "screening" in options)
                resumed.load(self.path)
                self.run_seeded(resumed, 5)
                self.assertEqual(resumed.outcomes, full.outcomes)
                self.assertEqual(resumed.training_rows, full.training_rows)
                self.assertEqual(len(resumed.archive.entries), len(full.archive.entries))
                for expected, actual in zip(full.archive.entries, resumed.archive.entries):
                    self.assertEqual(actual.energy, expected.energy)
                    self.assertEqual(actual.metadata, expected.metadata)
                    np.testing.assert_allclose(actual.atoms.positions, expected.atoms.positions, atol=1e-12)
                    np.testing.assert_array_equal(actual.atoms.cell, expected.atoms.cell)
                current = np.random.get_state()
                self.assertEqual(numpy_state[0], current[0])
                np.testing.assert_array_equal(numpy_state[1], current[1])
                self.assertEqual(numpy_state[2:], current[2:])
                self.assertEqual(python_state, random.getstate())

    def test_corrupt_stress_settings_and_metadata_do_not_partially_load(self):
        options = self.gp_options()
        source, destination = self.make_search(**options), self.make_search(**options)
        self.run_seeded(source, 3, checkpoint=self.path)
        self.run_seeded(destination, 1)
        original = json.loads(self.path.read_text())
        documents = []
        for value in (None, True, -.001, .01, .02, np.inf, [], "0.002"):
            document = deepcopy(original)
            document["outcomes"][0]["stress_max"] = value
            # Matching provenance must not bypass convergence validation.
            document["archive"]["entries"][0]["metadata"]["stress_max"] = value
            documents.append(document)
        for key, value in (("settings", .02), ("metadata", .003), ("version", 4)):
            document = deepcopy(original)
            if key == "settings":
                document[key]["cell_relaxation"]["stress_max"] = value
            elif key == "metadata":
                document["archive"]["entries"][0][key]["stress_max"] = value
            else:
                document[key] = value
            documents.append(document)
        document = deepcopy(original)
        del document["outcomes"][0]["stress_max"]
        documents.append(document)
        document = deepcopy(original)
        document["archive"]["entries"][0]["info"]["slab_quench_stress_max"] = .003
        documents.append(document)
        for document in documents:
            self.path.write_text(json.dumps(document))
            self.assert_load_failure_is_atomic(destination)
        failed = self.make_search(**options)
        with patch("crisp.slab_search.generate_slabs", side_effect=SlabGenerationError("exhausted")):
            failed.run(1, checkpoint=self.path)
        document = json.loads(self.path.read_text())
        document["outcomes"][0]["stress_max"] = .002
        self.path.write_text(json.dumps(document))
        self.assert_load_failure_is_atomic(destination)

    def test_invalid_final_stress_or_calculator_failure_is_fatal_before_archive(self):
        for invalid in (None, True, -.01, .01, .02, np.nan, 1j, np.array([.002]), "missing"):
            with self.subTest(invalid=invalid):
                search = self.make_search(**self.gp_options())
                self.run_seeded(search, 1, checkpoint=self.path)
                previous = self.path.read_bytes()
                outcomes, training, entries = deepcopy(search.outcomes), deepcopy(search.training_rows), search.archive.entries[:]

                def bad_quench(*args, **kwargs):
                    relaxed = _cell_quench(*args, **kwargs)
                    if isinstance(invalid, str):
                        del relaxed.info["slab_quench_stress_max"]
                    else:
                        relaxed.info["slab_quench_stress_max"] = invalid
                    return relaxed

                with patch("crisp.slab_search.generate_slabs", side_effect=_seeded_generation), \
                        patch("crisp.slab_search.quench_slab", side_effect=bad_quench), \
                        patch("crisp.slab_search.validate_slab_candidate") as validate, \
                        self.assertRaises((ValueError, TypeError, RuntimeError, KeyError)):
                    search.run(1, checkpoint=self.path)
                validate.assert_not_called()
                self.assertEqual(search.outcomes, outcomes)
                self.assertEqual(search.training_rows, training)
                self.assertEqual(search.archive.entries, entries)
                self.assertEqual(self.path.read_bytes(), previous)
        search = self.make_search()
        with patch("crisp.slab_search.generate_slabs", side_effect=_seeded_generation), \
                patch("crisp.slab_search.quench_slab", side_effect=RuntimeError("stress unavailable")), \
                self.assertRaisesRegex(RuntimeError, "stress unavailable"):
            search.run(1)
        self.assertEqual(search.completed_trials, 0)

    def test_disabled_cell_mode_keeps_prior_schemas_and_quench_signature(self):
        modes = (({}, 1), (self.gp_options(), 2), (dict(mutations=SlabMutations()), 3),
                 (self.gp_options() | dict(mutations=SlabMutations()), 3),
                 (self.gp_options() | dict(guidance=SlabGuidance()), 4))
        for options, version in modes:
            with self.subTest(version=version, options=options):
                search = self.make_search(cell_relaxation=None, **options)
                with patch("crisp.slab_search.generate_slabs", side_effect=_seeded_generation), \
                        patch("crisp.slab_search.quench_slab", side_effect=_quench) as quench:
                    search.run(1, checkpoint=self.path)
                self.assertEqual(quench.call_args.kwargs, dict(fmax=.01, max_steps=9))
                payload = json.loads(self.path.read_text())
                self.assertEqual(payload["version"], version)
                self.assertNotIn("cell_relaxation", payload["settings"])
                self.assertNotIn("stress_max", payload["outcomes"][0])
                self.assertNotIn("stress_max", payload["archive"]["entries"][0]["metadata"])
                resumed = self.make_search(cell_relaxation=None, **options)
                resumed.load(self.path)
                self.assertEqual(resumed.outcomes, search.outcomes)
                with self.assertRaises(ValueError):
                    self.make_search(**options).load(self.path)
        with self.assertRaises(TypeError):
            self.make_search(cell_relaxation={"stress_max": .01})


if __name__ == "__main__":
    unittest.main()
