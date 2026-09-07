"""Real PyXtal/EMT/fingerprint search and checkpoint continuation smoke test."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from ase.calculators.emt import EMT

from crisp.slab import SlabConfig
from crisp.slab_archive import SlabArchive
from crisp.slab_fingerprint import SlabFingerprintCalculator
from crisp.slab_mutation import SlabMutations
from crisp.slab_screening import SlabGPScreening
from crisp.slab_search import SlabRandomSearch
from crisp.slab_validation import validate_slab_candidate

try:
    import pyxtal
except ImportError:
    pyxtal = None

try:
    import torch_fplib
except ImportError:
    torch_fplib = None


def _search(screening=None):
    config = SlabConfig((5.5, 6.5), initial_thickness=0, max_thickness=1,
                        cell_height=18, min_vacuum=12)
    archive = SlabArchive(
        SlabFingerprintCalculator(config, cutoff=3.2, natx=32),
        {"Cu": 1}, min_dist_ang=1.5,
    )
    return SlabRandomSearch(
        archive, EMT, calculator_id="ASE-EMT-default", seed=19,
        layer_groups=[80], max_generation_attempts=6, max_steps=0,
        screening=screening,
    )


def _mutation_search(screening=None):
    config = SlabConfig((4.5, 6.0), initial_thickness=1, max_thickness=3,
                        cell_height=18, min_vacuum=12)
    archive = SlabArchive(
        SlabFingerprintCalculator(config, cutoff=3.2, natx=64),
        {"Cu": 4}, min_dist_ang=1.5,
    )
    return SlabRandomSearch(
        archive, EMT, calculator_id="ASE-EMT-default", seed=23,
        layer_groups=[1, 2, 80], max_generation_attempts=6, max_steps=200,
        mutations=SlabMutations(random_every=4), screening=screening,
    )


@unittest.skipUnless(pyxtal is not None and torch_fplib is not None,
                     "PyXtal and torch_fplib required")
class TestSlabSearchBackend(unittest.TestCase):
    def test_real_seeded_search_matches_split_checkpoint_resume(self):
        self._check_split_checkpoint_resume()

    def test_real_gp_search_matches_split_checkpoint_resume(self):
        self._check_split_checkpoint_resume(SlabGPScreening(
            pool_size=3, min_training_points=2, explore_every=4,
            max_training_points=4,
        ))

    def test_real_mutations_and_gp_preserve_geometry_and_resume(self):
        # Cu4 requires actual optimizer moves, unlike the symmetric Cu1 smoke test.
        for screening in (None, SlabGPScreening(
                pool_size=3, min_training_points=2, explore_every=4,
                max_training_points=4)):
            with self.subTest(screening=screening):
                continuous = _mutation_search(screening)
                continuous.run(8)
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory) / "mutations.json"
                    first = _mutation_search(screening)
                    first.run(3, checkpoint=checkpoint)
                    resumed = _mutation_search(screening)
                    resumed.load(checkpoint)
                    resumed.run(5, checkpoint=checkpoint)

                self.assertEqual(continuous.counts, resumed.counts)
                for expected, actual in zip(continuous.outcomes, resumed.outcomes):
                    for key, value in expected.items():
                        if key == "energy_per_atom" and value is not None:
                            np.testing.assert_allclose(actual[key], value, atol=1e-10, rtol=0)
                        elif key == "screening":
                            for detail, target in value.items():
                                if detail in ("mean", "std", "score") and target is not None:
                                    np.testing.assert_allclose(actual[key][detail], target,
                                                               atol=1e-9, rtol=0)
                                else:
                                    self.assertEqual(actual[key][detail], target)
                        else:
                            self.assertEqual(actual[key], value)
                self.assertEqual(len(continuous.training_rows), len(resumed.training_rows))
                for expected, actual in zip(continuous.training_rows, resumed.training_rows):
                    self.assertEqual(actual["trial"], expected["trial"])
                    np.testing.assert_allclose(actual["features"], expected["features"],
                                               atol=1e-10, rtol=0)
                    np.testing.assert_allclose(actual["energy_per_atom"], expected["energy_per_atom"],
                                               atol=1e-10, rtol=0)
                entries = continuous.archive.entries
                by_trial = {entry.metadata["search_trial"]: entry for entry in entries}
                mutated = [entry for entry in entries
                           if entry.metadata["proposal"]["source"] == "mutation"]
                self.assertTrue(mutated)
                self.assertTrue(any(entry.atoms.info["slab_quench_steps"] > 0 for entry in mutated))
                for entry in mutated:
                    parent = by_trial[entry.metadata["proposal"]["parent_trial"]]
                    self.assertGreater(np.linalg.norm(entry.atoms.cell - parent.atoms.cell), 1e-5)
                for trial in (4, 8):
                    self.assertEqual(continuous.outcomes[trial - 1]["proposal"]["source"], "random")
                for expected, actual in zip(entries, resumed.archive.entries):
                    self.assertEqual(actual.metadata, expected.metadata)
                    np.testing.assert_array_equal(actual.atoms.numbers, [29] * 4)
                    np.testing.assert_array_equal(actual.atoms.pbc, [True, True, False])
                    np.testing.assert_array_equal(actual.atoms.cell[2], [0, 0, 18])
                    np.testing.assert_allclose(actual.atoms.positions, expected.atoms.positions,
                                               atol=1e-9, rtol=0)
                    np.testing.assert_allclose(actual.atoms.cell, expected.atoms.cell,
                                               atol=1e-12, rtol=0)
                    np.testing.assert_allclose(actual.fp, expected.fp, atol=1e-10, rtol=0)
                    validate_slab_candidate(actual.atoms, resumed.archive.fp_calc.config,
                                            min_dist_ang=1.5)

    def _check_split_checkpoint_resume(self, screening=None):
        # A one-atom triangular sheet has zero forces by symmetry. This tests
        # the complete pipeline and restart, not structural stability/recovery.
        state = np.random.get_state()
        continuous = _search(screening)
        continuous.run(6)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "search.json"
            first = _search(screening)
            first.run(2, checkpoint=checkpoint)
            resumed = _search(screening)
            resumed.load(checkpoint)
            self.assertEqual(resumed.completed_trials, 2)
            resumed.run(4, checkpoint=checkpoint)

        self.assertEqual(continuous.completed_trials, 6)
        self.assertEqual(resumed.completed_trials, 6)
        self.assertGreater(continuous.counts.get("accepted", 0), 0)
        self.assertEqual(continuous.outcomes, resumed.outcomes)
        self.assertEqual(continuous.counts, resumed.counts)
        self.assertTrue(all(outcome["status"] in ("accepted", "duplicate")
                            for outcome in continuous.outcomes))
        self.assertEqual(continuous.training_rows, resumed.training_rows)
        if screening is not None:
            self.assertEqual([outcome["screening"]["mode"] for outcome in continuous.outcomes],
                             ["bootstrap", "bootstrap", "gp", "explore", "gp", "gp"])
            self.assertEqual(len(continuous.training_rows), 4)
            self.assertEqual([row["trial"] for row in continuous.training_rows], [3, 4, 5, 6])
        self.assertEqual(len(continuous.archive.entries), len(resumed.archive.entries))
        for expected, actual in zip(continuous.archive.entries, resumed.archive.entries):
            self.assertEqual(expected.energy, actual.energy)
            self.assertEqual(expected.metadata, actual.metadata)
            self.assertEqual(expected.atoms.info, actual.atoms.info)
            np.testing.assert_array_equal(actual.atoms.numbers, expected.atoms.numbers)
            np.testing.assert_array_equal(actual.atoms.pbc, [True, True, False])
            np.testing.assert_allclose(actual.atoms.positions, expected.atoms.positions,
                                       atol=1e-12, rtol=0)
            np.testing.assert_allclose(actual.atoms.cell, expected.atoms.cell,
                                       atol=1e-12, rtol=0)
            np.testing.assert_allclose(actual.fp, expected.fp, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(actual.fp_pooled, expected.fp_pooled,
                                       atol=1e-12, rtol=1e-12)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])


if __name__ == "__main__":
    unittest.main()
