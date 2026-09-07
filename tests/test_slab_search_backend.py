"""Real PyXtal/EMT/fingerprint search and checkpoint continuation smoke test."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from ase.calculators.emt import EMT

from crisp.slab import SlabConfig
from crisp.slab_archive import SlabArchive
from crisp.slab_fingerprint import SlabFingerprintCalculator
from crisp.slab_screening import SlabGPScreening
from crisp.slab_search import SlabRandomSearch

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
