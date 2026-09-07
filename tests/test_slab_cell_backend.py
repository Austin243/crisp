"""Physical EMT cell relaxation, vacuum independence and search restart."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT

from crisp.slab import SlabConfig
from crisp.slab_archive import SlabArchive
from crisp.slab_cell_relaxation import SlabCellRelaxation
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


def _search(combined):
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
                                 explore_every=4, max_training_points=4) if combined else None,
        mutations=SlabMutations(random_every=5) if combined else None,
        guidance=SlabGuidance() if combined else None,
        cell_relaxation=SlabCellRelaxation(),
    )


class TestSlabCellBackend(unittest.TestCase):
    def test_real_emt_relaxation_is_independent_of_vacuum_height(self):
        a = np.sqrt(2 * 5.1 / np.sqrt(3))
        b = np.sqrt(3) * a / 2
        parent = Atoms("Cu4", positions=[[0, 0, 9], [a / 2, b, 9],
                                         [a, 0, 9], [1.5 * a, b, 9]],
                       cell=[[2 * a, 0, 0], [a, 2 * b, 0], [0, 0, 18]],
                       pbc=[True, True, False])
        strain = np.array([[1.04, .04], [.04, .96]])
        parent.cell[:2, :2] = parent.cell[:2, :2] @ strain
        parent.positions[:, :2] = parent.positions[:, :2] @ strain
        parent.positions[:, :2] += [[.04, -.03], [-.01, .02], [-.02, -.01], [-.01, .02]]
        outputs = []
        for height in (18, 36):
            candidate = parent.copy()
            candidate.cell[2, 2] = height
            config = SlabConfig((4.5, 6), 0, 1, height, 12)
            result = quench_slab(candidate, config, EMT, fmax=.005, max_steps=200,
                                 cell_relaxation=SlabCellRelaxation(stress_max=.002))
            self.assertGreater(result.info["slab_quench_steps"], 0)
            self.assertGreater(np.linalg.norm(result.cell[:2] - candidate.cell[:2]), 1e-3)
            np.testing.assert_array_equal(result.cell[2], [0, 0, height])
            np.testing.assert_array_equal(result.cell[:2, 2], [0, 0])
            self.assertLess(np.linalg.norm(result.get_forces(), axis=1).max(), .005)
            self.assertLess(height * np.abs(result.get_stress()[[0, 1, 5]]).max(), .002)
            candidate.calc = EMT()
            self.assertLess(result.get_potential_energy(), candidate.get_potential_energy())
            outputs.append(result)

        self.assertEqual(outputs[0].info["slab_quench_steps"], outputs[1].info["slab_quench_steps"])
        np.testing.assert_allclose(outputs[0].cell[:2], outputs[1].cell[:2], atol=1e-10, rtol=0)
        np.testing.assert_allclose(outputs[0].positions, outputs[1].positions, atol=1e-10, rtol=0)
        np.testing.assert_allclose(outputs[0].get_potential_energy(),
                                   outputs[1].get_potential_energy(), atol=1e-10, rtol=0)
        np.testing.assert_allclose(18 * outputs[0].get_stress()[[0, 1, 5]],
                                   36 * outputs[1].get_stress()[[0, 1, 5]], atol=1e-10, rtol=0)

    def assert_same_records(self, actual, expected):
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

    @unittest.skipUnless(pyxtal is not None and torch_fplib is not None,
                         "PyXtal and torch_fplib required")
    def test_real_cell_search_and_combined_guidance_match_checkpoint_resume(self):
        for combined in (False, True):
            with self.subTest(combined=combined):
                continuous = _search(combined)
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
                    checkpoint = Path(directory) / "cell-search.json"
                    first = _search(combined)
                    first.run(3, checkpoint=checkpoint)
                    resumed = _search(combined)
                    resumed.load(checkpoint)
                    resumed.run(5, checkpoint=checkpoint)

                self.assertGreater(continuous.counts.get("accepted", 0), 0)
                self.assert_same_records(resumed.outcomes, continuous.outcomes)
                self.assert_same_records(resumed.training_rows, continuous.training_rows)
                nonaffine_motion = []
                for trial, result in outputs.items():
                    candidate = inputs[trial]
                    result.calc = EMT()  # Independently reevaluate the final physical state.
                    row = continuous.outcomes[trial - 1]
                    self.assertLess(np.linalg.norm(result.get_forces(), axis=1).max(), .05)
                    stress = 18 * np.abs(result.get_stress()[[0, 1, 5]]).max()
                    self.assertLess(stress, .01)
                    self.assertAlmostEqual(row["stress_max"], stress, places=12)
                    self.assertAlmostEqual(row["energy_per_atom"],
                                           result.get_potential_energy() / len(result), places=12)
                    np.testing.assert_array_equal(result.pbc, [True, True, False])
                    np.testing.assert_array_equal(result.cell[2], [0, 0, 18])
                    np.testing.assert_array_equal(result.cell[:2, 2], [0, 0])
                    self.assertGreater(np.linalg.norm(result.cell - candidate.cell), 1e-5)
                    affine = candidate.positions.copy()
                    affine[:, :2] = affine[:, :2] @ np.linalg.solve(candidate.cell[:2, :2],
                                                                    result.cell[:2, :2])
                    nonaffine_motion.append(np.linalg.norm(result.positions - affine))
                    validate_slab_candidate(result, continuous.archive.fp_calc.config,
                                            min_dist_ang=1.5)
                self.assertGreater(max(nonaffine_motion), 1e-3)

                if combined:
                    guided = [row for row in continuous.outcomes
                              if row["trial"] in outputs and row["guidance"]["steps"] > 0]
                    self.assertTrue(guided, "Exercise guidance and physical cell relaxation together")
                    self.assertTrue(any(row["proposal"]["source"] == "mutation" for row in guided))
                    trial = guided[0]["trial"]
                    repeated = quench_slab(inputs[trial], continuous.archive.fp_calc.config, EMT,
                                           fmax=.05, max_steps=200,
                                           cell_relaxation=SlabCellRelaxation())
                    np.testing.assert_allclose(repeated.positions, outputs[trial].positions,
                                               atol=1e-12, rtol=0)
                    np.testing.assert_allclose(repeated.cell, outputs[trial].cell, atol=1e-12, rtol=0)
                    for row in continuous.training_rows:
                        expected = candidate_features(inputs[row["trial"]], continuous.archive.fp_calc)
                        np.testing.assert_allclose(row["features"], expected, atol=1e-12, rtol=0)
                self.assertEqual(len(continuous.archive.entries), len(resumed.archive.entries))
                for expected, actual in zip(continuous.archive.entries, resumed.archive.entries):
                    self.assert_same_records(actual.metadata, expected.metadata)
                    np.testing.assert_allclose(actual.atoms.positions, expected.atoms.positions,
                                               atol=1e-8, rtol=0)
                    np.testing.assert_allclose(actual.atoms.cell, expected.atoms.cell, atol=1e-8, rtol=0)
                    np.testing.assert_allclose(actual.fp, expected.fp, atol=1e-9, rtol=0)


if __name__ == "__main__":
    unittest.main()
