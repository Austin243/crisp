"""Real fingerprint guidance followed by physical EMT relaxation and restart."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ase.calculators.emt import EMT
from ase.neighborlist import neighbor_list

from crisp.slab import SlabConfig, validate_slab
from crisp.slab_archive import SlabArchive
from crisp.slab_fingerprint import SlabFingerprintCalculator
from crisp.slab_guidance import SlabGuidance
from crisp.slab_mutation import SlabMutations
from crisp.slab_relaxation import quench_slab
from crisp.slab_screening import SlabGPScreening, candidate_features
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


def _search(mutations):
    config = SlabConfig((4.5, 6.0), initial_thickness=1, max_thickness=3,
                        cell_height=18, min_vacuum=12)
    archive = SlabArchive(
        SlabFingerprintCalculator(config, cutoff=3.2, natx=64),
        {"Cu": 4}, min_dist_ang=1.5,
    )
    return SlabRandomSearch(
        archive, EMT, calculator_id="ASE-EMT-default", seed=23,
        layer_groups=[1, 2, 80], max_generation_attempts=6, max_steps=200,
        screening=SlabGPScreening(pool_size=3, min_training_points=2,
                                 explore_every=4, max_training_points=4),
        mutations=mutations, guidance=SlabGuidance(),
    )


@unittest.skipUnless(pyxtal is not None and torch_fplib is not None,
                     "PyXtal and torch_fplib required")
class TestSlabGuidanceBackend(unittest.TestCase):
    def assert_same_records(self, actual, expected):
        # Reloading archive fingerprints may introduce roundoff, while all
        # choices, stop reasons, trial counts, and lineage must agree exactly.
        if isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_same_records(actual[key], expected[key])
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))
            for got, wanted in zip(actual, expected):
                self.assert_same_records(got, wanted)
        elif isinstance(expected, float):
            np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
        else:
            self.assertEqual(actual, expected)

    def test_real_guidance_then_unbiased_quench_matches_checkpoint_resume(self):
        for mutations in (None, SlabMutations(random_every=5)):
            with self.subTest(mutations=mutations):
                continuous = _search(mutations)
                inputs, outputs = {}, {}

                def observed_quench(candidate, *args, **kwargs):
                    trial = continuous.completed_trials + 1
                    self.assertNotIn(trial, inputs)
                    self.assertIsNone(candidate.calc)
                    inputs[trial] = candidate.copy()
                    result = quench_slab(candidate, *args, **kwargs)
                    self.assertIsInstance(result.calc, EMT)
                    outputs[trial] = result.copy()
                    return result

                with patch("crisp.slab_search.quench_slab", side_effect=observed_quench):
                    continuous.run(8)
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory) / "guided-search.json"
                    first = _search(mutations)
                    first.run(3, checkpoint=checkpoint)
                    resumed = _search(mutations)
                    resumed.load(checkpoint)
                    resumed.run(5, checkpoint=checkpoint)

                self.assert_same_records(resumed.outcomes, continuous.outcomes)
                self.assert_same_records(resumed.training_rows, continuous.training_rows)
                guided = [row for row in continuous.outcomes
                          if row["guidance"] is not None and row["guidance"]["steps"] > 0]
                self.assertTrue(guided, "Exercise actual Cartesian guidance, not only bypasses")
                for row in guided:
                    self.assertLess(row["guidance"]["final_score"], row["guidance"]["initial_score"])
                relaxed_guided = [row for row in guided if row["trial"] in outputs]
                self.assertTrue(relaxed_guided)
                self.assertTrue(any(np.linalg.norm(outputs[row["trial"]].positions -
                                                   inputs[row["trial"]].positions) > 1e-5
                                    for row in relaxed_guided))
                # Repeating the existing physical quench alone gives the same
                # geometry: the GP guidance force is absent from relaxation.
                trial = relaxed_guided[0]["trial"]
                physical = quench_slab(inputs[trial], continuous.archive.fp_calc.config,
                                       EMT, fmax=.05, max_steps=200)
                np.testing.assert_allclose(physical.positions, outputs[trial].positions,
                                           atol=1e-12, rtol=0)
                for trial, candidate in inputs.items():
                    validate_slab(candidate, continuous.archive.fp_calc.config)
                    self.assertEqual(len(neighbor_list("d", candidate, 1.5)), 0)
                    np.testing.assert_array_equal(candidate.numbers, [29] * 4)
                    np.testing.assert_array_equal(candidate.pbc, [True, True, False])
                    np.testing.assert_array_equal(candidate.cell[2], [0, 0, 18])
                    if trial in outputs:
                        np.testing.assert_array_equal(outputs[trial].cell, candidate.cell)
                for row in continuous.training_rows:
                    # Training describes the structure actually handed to EMT,
                    # after guidance, rather than the earlier screened proposal.
                    expected = candidate_features(inputs[row["trial"]], continuous.archive.fp_calc)
                    np.testing.assert_allclose(row["features"], expected, atol=1e-12, rtol=0)
                self.assertEqual(len(continuous.archive.entries), len(resumed.archive.entries))
                for expected, actual in zip(continuous.archive.entries, resumed.archive.entries):
                    self.assert_same_records(actual.metadata, expected.metadata)
                    np.testing.assert_allclose(actual.atoms.positions, expected.atoms.positions,
                                               atol=1e-8, rtol=0)
                    np.testing.assert_allclose(actual.atoms.cell, expected.atoms.cell,
                                               atol=1e-12, rtol=0)
                    np.testing.assert_allclose(actual.fp, expected.fp, atol=1e-9, rtol=0)
                    validate_slab_candidate(actual.atoms, resumed.archive.fp_calc.config,
                                            min_dist_ang=1.5)


if __name__ == "__main__":
    unittest.main()
