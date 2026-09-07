"""Real CRISP/native-file integration with synthetic VASP process output.

Only process execution is mocked. This tests the parser and search contracts,
not VASP physics, material stability, or supported-build acceptance.
"""

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ase import Atoms

from crisp.fingerprint import _HAS_TORCH_FPLIB
from crisp.slab import SlabConfig
from crisp.slab_crisp_search import SlabCRISPSearch
from crisp.slab_fingerprint import SlabFingerprintCalculator
from crisp.slab_layer_archive import LayeredSlabArchive
from crisp.slab_layers import InitialLayer, LayeredSlabSpec, validate_layered_slab
from crisp.slab_mutation import SlabMutations
from crisp.slab_vasp import NativeVASPConfig, SlabVASPRelaxer
from test_slab_vasp import output


@unittest.skipUnless(_HAS_TORCH_FPLIB, "torch_fplib required for real fingerprints")
class TestSlabCRISPSearchBackend(unittest.TestCase):
    def test_real_fingerprints_native_manifests_and_population_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            potential = directory / "Cu.POTCAR"
            potential.write_text("Synthetic fixture, not a potential\nVRHFIN =Cu: test\nEnd of Dataset\n")
            geometry = SlabConfig((5, 6), initial_thickness=0, max_thickness=2,
                                  cell_height=18, min_vacuum=12)
            spec = LayeredSlabSpec((InitialLayer({"Cu": 1}),), geometry,
                                   min_dist_ang=1.0, bond_scale=1.5)
            config = NativeVASPConfig(
                ("synthetic-test-only-vasp",), "6.5.0", {"Cu": str(potential)},
                {"ENCUT": 520, "NSW": 2, "ISMEAR": 0, "SIGMA": .05})

            def make_search(work_directory):
                backend = SlabVASPRelaxer(spec, config, work_directory)
                archive = LayeredSlabArchive(
                    SlabFingerprintCalculator(geometry, cutoff=3.2, natx=32), spec,
                    fp_threshold=1e-8, energy_threshold=.001)
                search = SlabCRISPSearch(
                    archive, backend, seed=41, n_random=2, n_mutants=2,
                    max_generations=3, convergence_gens=5, max_skip_frac=0,
                    mutations=SlabMutations(max_displacement=.03, max_strain=.02))
                return search, backend

            def run(search, backend, count, checkpoint=None):
                with patch.object(backend, "_execute", side_effect=lambda path: output(
                        backend, path, force=0, stress=0)) as execute, redirect_stdout(io.StringIO()):
                    search.run(count, checkpoint=checkpoint)
                return execute.call_count

            full, full_backend = make_search(directory / "full")
            full_launches = run(full, full_backend, 3)
            checkpoint = directory / "search.json"
            split, split_backend = make_search(directory / "split")
            split_launches = run(split, split_backend, 1, checkpoint)
            resumed, resumed_backend = make_search(split_backend.work_dir)
            resumed.load(checkpoint)
            resumed_launches = run(resumed, resumed_backend, 2, checkpoint)

            self.assertEqual(full.outcomes, resumed.outcomes)
            self.assertEqual(full.next_generation, 3)
            self.assertEqual(resumed.next_generation, 3)
            self.assertEqual(full.stop_reason, resumed.stop_reason)
            self.assertEqual(full_launches, full.n_relaxed)
            self.assertEqual(split_launches + resumed_launches, full_launches)
            self.assertTrue(any(row["origin"] == "mutant" for row in resumed.outcomes))
            self.assertGreaterEqual(len(resumed.archive.entries), 2)
            areas = [np.linalg.det(entry.atoms.cell[:2, :2]) for entry in resumed.archive.entries]
            self.assertGreater(np.ptp(areas), .01)
            np.testing.assert_array_equal(full.gp.X_train, resumed.gp.X_train)
            np.testing.assert_array_equal(resumed.gp.X_train, resumed.archive.get_all_pooled_fps())
            np.testing.assert_allclose(resumed.gp.y_train * resumed.gp._y_std + resumed.gp._y_mean,
                                       resumed.archive.get_all_enthalpies(), atol=1e-12)

            for before, after in zip(full.archive.entries, resumed.archive.entries):
                expected_metadata, actual_metadata = deepcopy(before.metadata), deepcopy(after.metadata)
                for metadata, work_dir in ((expected_metadata, full_backend.work_dir),
                                           (actual_metadata, resumed_backend.work_dir)):
                    for stage in metadata["stages"]:
                        stage["directory"] = str(Path(stage["directory"]).relative_to(work_dir))
                self.assertEqual(expected_metadata, actual_metadata)
                self.assertAlmostEqual(after.energy, -9.99, places=12)
                np.testing.assert_array_equal(before.fp, after.fp)
                np.testing.assert_array_equal(before.atoms.positions, after.atoms.positions)
                np.testing.assert_array_equal(before.atoms.cell, after.atoms.cell)
                np.testing.assert_array_equal(after.atoms.pbc, [True, True, False])
                np.testing.assert_array_equal(after.atoms.cell[2], [0, 0, 18])
                np.testing.assert_array_equal(after.atoms.arrays["initial_atom_id"], [0])
                np.testing.assert_array_equal(after.atoms.arrays["initial_layer"], [0])
                validate_layered_slab(after.atoms, spec)

            for row in resumed.outcomes:
                relative = Path(row["candidate_id"]) / "stage-00" / "manifest.json"
                expected = json.loads((full_backend.work_dir / relative).read_text())
                actual = json.loads((resumed_backend.work_dir / relative).read_text())
                self.assertEqual(actual["identity"], expected["identity"])
                self.assertEqual(actual["status"], "completed")

            # Recollect an actual completed mutant through the native manifest
            # path. Exact identity must reuse the parsed result without execution.
            mutant = next(row for row in resumed.outcomes if row["origin"] == "mutant")
            path = resumed_backend.work_dir / mutant["candidate_id"] / "stage-00" / "manifest.json"
            saved = deepcopy(json.loads(path.read_text())["identity"]["atoms"])
            for key in ("numbers", "positions", "cell", "pbc", "initial_atom_id", "initial_layer"):
                saved[key] = np.asarray(saved[key])
            candidate = Atoms.fromdict(saved)
            with patch.object(resumed_backend, "_execute", side_effect=AssertionError("cached job relaunched")):
                result = resumed_backend.relax(candidate, mutant["candidate_id"])
            self.assertEqual(result.energy_per_atom, mutant["energy_per_atom"])
            validate_layered_slab(result.atoms, spec)


if __name__ == "__main__":
    unittest.main()
