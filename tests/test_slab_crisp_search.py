"""Population controller contracts using synthetic fingerprints and native results.

The GP, archive and layered geometry checks are real. The backend below is a
test double: these tests neither launch VASP nor establish DFT correctness.
"""

from copy import deepcopy
from dataclasses import replace
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ase import Atoms

from crisp.slab import SlabConfig
from crisp.slab_archive import _write_slab_json
from crisp.slab_crisp_search import SlabCRISPSearch
from crisp.slab_fingerprint import SlabFingerprintCalculator
from crisp.slab_generation import SlabGenerationError
from crisp.slab_layer_archive import LayeredSlabArchive
from crisp.slab_layers import InitialLayer, LayeredSlabSpec
from crisp.slab_mutation import SlabMutations
from crisp.slab_vasp import NativeVASPResult, VASPExecutionError, VASPValidationError


def _sheet(value, kind="accepted"):
    atoms = Atoms("Cu4", positions=[[0, 0, 9], [2.1, 0, 9],
                                   [0, 2.1, 9], [2.1, 2.1, 9]],
                  cell=[4.2, 4.2, 18], pbc=[True, True, False])
    atoms.set_array("initial_atom_id", np.arange(4, dtype=int))
    atoms.set_array("initial_layer", np.zeros(4, dtype=int))
    atoms.set_array("test_fp", np.full((4, 8), float(value)))
    atoms.info.update(test_kind=kind, origin="layered_random")
    return atoms


def _generate(spec, n, *, seed=None, max_attempts=1000):
    rng = np.random.default_rng(seed)
    return [_sheet(value) for value in rng.uniform(0, 5, n)]


class _NativeResults:
    """Cache immutable fake physical results by deterministic candidate ID."""

    def __init__(self, spec, *, fail_at=None):
        self.spec = spec
        self.settings = {"backend": "test-double", "final_recipe": "toy-v1"}
        self.calls = []
        self.cache = {}
        self.inputs = {}
        self.fail_at = fail_at

    def relax(self, atoms, candidate_id):
        self.calls.append(candidate_id)
        if candidate_id in self.inputs:
            previous = self.inputs[candidate_id]
            np.testing.assert_array_equal(atoms.positions, previous.positions)
            np.testing.assert_array_equal(atoms.cell, previous.cell)
            np.testing.assert_array_equal(atoms.arrays["test_fp"], previous.arrays["test_fp"])
        else:
            self.inputs[candidate_id] = atoms.copy()
        if candidate_id in self.cache:
            cached = self.cache[candidate_id]
            if isinstance(cached, Exception):
                raise type(cached)(str(cached))
            return deepcopy(cached)
        if self.fail_at == len(self.calls):
            self.fail_at = None
            raise VASPExecutionError("simulated worker interruption")
        if atoms.info.get("test_kind") == "native_rejected":
            error = VASPValidationError("simulated unconverged native result")
            self.cache[candidate_id] = error
            raise error
        relaxed = atoms.copy()
        # Distinguish actual relaxed-state descriptors from proposal descriptors.
        relaxed.arrays["test_fp"] += 100
        relaxed.positions[:, 2] += [-.05, .05, .05, -.05]
        if atoms.info.get("test_kind") == "geometry_rejected":
            relaxed.positions[1] = relaxed.positions[0]
        result = NativeVASPResult(
            relaxed, -10.0 - float(atoms.arrays["test_fp"].mean()),
            {"candidate_id": candidate_id, "backend": "test-double", "final_recipe": "toy-v1"})
        self.cache[candidate_id] = deepcopy(result)
        return result


class TestSlabCRISPSearch(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "population.json"
        config = SlabConfig((3, 7), initial_thickness=0, max_thickness=2,
                            cell_height=18, min_vacuum=12)
        self.spec = LayeredSlabSpec((InitialLayer({"Cu": 4}),), config,
                                   min_dist_ang=1.0)
        self.fp_calc = SlabFingerprintCalculator(config, cutoff=3.2, natx=8)
        fingerprints = patch.object(self.fp_calc, "get_fingerprints",
                                    side_effect=lambda atoms: atoms.arrays["test_fp"])
        fingerprints.start()
        self.addCleanup(fingerprints.stop)

    def make_search(self, relaxer=None, **options):
        archive = LayeredSlabArchive(self.fp_calc, self.spec,
                                     fp_threshold=.001, energy_threshold=.001)
        relaxer = relaxer or _NativeResults(self.spec)
        defaults = dict(seed=71, n_random=4, n_mutants=1, n_select=3,
                        max_generations=4, convergence_gens=10,
                        mutations=SlabMutations(max_displacement=.02, max_strain=.001),
                        gp_length_scale=100, max_skip_frac=0, min_relax_per_gen=1)
        return SlabCRISPSearch(archive, relaxer, **(defaults | options))

    def run_seeded(self, search, n_generations=None, **options):
        with patch("crisp.slab_crisp_search.generate_layered_slabs", side_effect=_generate), \
                redirect_stdout(io.StringIO()):
            return search.run(n_generations, **options)

    def assert_same_search(self, left, right):
        self.assertEqual(left.outcomes, right.outcomes)
        self.assertEqual(left.next_generation, right.next_generation)
        self.assertEqual(left.stop_reason, right.stop_reason)
        self.assertEqual(left.n_relaxed, right.n_relaxed)
        self.assertEqual(len(left.archive.entries), len(right.archive.entries))
        for before, after in zip(left.archive.entries, right.archive.entries):
            self.assertEqual(before.energy, after.energy)
            self.assertEqual(before.metadata, after.metadata)
            np.testing.assert_allclose(before.atoms.positions, after.atoms.positions, atol=1e-12)
            np.testing.assert_allclose(before.atoms.cell, after.atoms.cell, atol=1e-12)
            np.testing.assert_array_equal(before.fp, after.fp)
        if left.gp.X_train is None:
            self.assertIsNone(right.gp.X_train)
        else:
            np.testing.assert_array_equal(left.gp.X_train, right.gp.X_train)
            np.testing.assert_allclose(left.gp.y_train, right.gp.y_train, atol=1e-12)

    def test_gp_learns_only_unique_final_relaxed_states_and_physical_energies(self):
        search = self.make_search(n_random=5, n_mutants=0)
        proposals = iter([_sheet(1), _sheet(1), _sheet(2),
                          _sheet(3, "native_rejected"), _sheet(4, "geometry_rejected")])
        with patch("crisp.slab_crisp_search.generate_layered_slabs",
                   side_effect=lambda spec, n, **kw: [next(proposals) for _ in range(n)]), \
                redirect_stdout(io.StringIO()):
            search.run(1)
        self.assertEqual(search.n_relaxed, 5)
        self.assertEqual(len(search.archive.entries), 2)
        np.testing.assert_array_equal(search.gp.X_train, search.archive.get_all_pooled_fps())
        np.testing.assert_allclose(search.gp.y_train * search.gp._y_std + search.gp._y_mean,
                                   [-11, -12])
        # The first half is the mean descriptor; the second half is its std.
        np.testing.assert_array_equal(search.gp.X_train[:, :8], [[101] * 8, [102] * 8])
        self.assertEqual(len(search.outcomes), 5)
        self.assertEqual(sum(row["status"] == "accepted" for row in search.outcomes), 2)
        self.assertEqual(sum(row["status"] == "duplicate" for row in search.outcomes), 1)

    def test_budget_is_strict_even_during_bootstrap_and_stops_further_launches(self):
        backend = _NativeResults(self.spec)
        search = self.make_search(backend, n_random=5, budget_relax=3)
        self.run_seeded(search)
        self.assertEqual(search.n_relaxed, 3)
        self.assertEqual(len(backend.calls), 3)
        self.assertIn("budget", search.stop_reason)
        self.run_seeded(search, 1)
        self.assertEqual(len(backend.calls), 3)

    def test_empty_and_all_rejected_generations_are_resumable_without_gp(self):
        for rejection in ("generation", "native_rejected", "geometry_rejected"):
            with self.subTest(rejection=rejection):
                self.path.unlink(missing_ok=True)
                search = self.make_search(n_mutants=0, max_generations=3)
                def generate(spec, n, **options):
                    if rejection == "generation":
                        raise SlabGenerationError("simulated exhausted placement budget")
                    return [_sheet(i, rejection) for i in range(n)]
                with patch("crisp.slab_crisp_search.generate_layered_slabs", side_effect=generate), \
                        redirect_stdout(io.StringIO()):
                    search.run(2, checkpoint=self.path)
                    resumed = self.make_search(n_mutants=0, max_generations=3)
                    resumed.load(self.path)
                    self.assert_same_search(search, resumed)
                    resumed.run(1)
                self.assertEqual(resumed.next_generation, 3)
                self.assertEqual(resumed.archive.entries, [])
                self.assertIsNone(resumed.gp.X_train)
                self.assertEqual(resumed.n_relaxed, 0 if rejection == "generation" else 12)

    def test_native_execution_error_rolls_back_generation_and_replays_cached_work(self):
        for previous_generations in (0, 1):
            with self.subTest(previous_generations=previous_generations):
                self.path.unlink(missing_ok=True)
                backend = _NativeResults(self.spec, fail_at=previous_generations * 4 + 2)
                search = self.make_search(backend, n_mutants=0)
                self.run_seeded(search, previous_generations, checkpoint=self.path)
                previous = self.make_search(n_mutants=0)
                previous.load(self.path)
                first_call = len(backend.calls)
                with self.assertRaisesRegex(VASPExecutionError, "worker interruption"):
                    self.run_seeded(search, 1, checkpoint=self.path)
                self.assert_same_search(previous, search)
                first_id = backend.calls[first_call]
                self.run_seeded(search, 1, checkpoint=self.path)
                self.assertEqual(backend.calls.count(first_id), 2)
                self.assertEqual(len(backend.cache), 4 * (previous_generations + 1))
                uninterrupted = self.make_search(n_mutants=0)
                self.run_seeded(uninterrupted, previous_generations + 1)
                self.assert_same_search(uninterrupted, search)

    def test_backend_configuration_error_is_fatal_not_a_physical_rejection(self):
        backend = _NativeResults(self.spec)
        search = self.make_search(backend)
        with patch.object(backend, "relax", side_effect=ValueError("changed physical recipe")), \
                self.assertRaisesRegex(ValueError, "changed physical recipe"):
            self.run_seeded(search, 1)
        self.assertEqual(search.next_generation, 0)
        self.assertEqual(search.n_relaxed, 0)
        self.assertEqual(search.outcomes, [])
        self.assertEqual(search.archive.entries, [])

    def test_cached_mutant_inputs_replay_exactly_after_generation_failure(self):
        backend = _NativeResults(self.spec)
        search = self.make_search(backend, n_random=2, n_mutants=4)
        self.run_seeded(search, 1, checkpoint=self.path)
        backend.fail_at = len(backend.calls) + 5
        with self.assertRaisesRegex(VASPExecutionError, "worker interruption"):
            self.run_seeded(search, 1, checkpoint=self.path)
        self.assertEqual(search.next_generation, 1)
        cached_mutants = [candidate_id for candidate_id in backend.cache
                          if backend.inputs[candidate_id].info["origin"] == "mutant"]
        self.assertEqual(len(cached_mutants), 2)
        # The backend requires exact input equality before cache reuse. Even
        # last-bit coordinate drift from archive normalization must fail here.
        self.run_seeded(search, 1, checkpoint=self.path)
        self.assertEqual(search.n_relaxed, 8)
        self.assertEqual(len(backend.cache), 8)
        for candidate_id in cached_mutants:
            self.assertEqual(backend.calls.count(candidate_id), 2)
        full_backend = _NativeResults(self.spec)
        full = self.make_search(full_backend, n_random=2, n_mutants=4)
        self.run_seeded(full, 2)
        self.assert_same_search(full, search)
        for candidate_id, replayed in backend.inputs.items():
            expected = full_backend.inputs[candidate_id]
            np.testing.assert_array_equal(replayed.positions, expected.positions)
            np.testing.assert_array_equal(replayed.cell, expected.cell)

    def test_failed_checkpoint_write_rolls_back_and_reuses_completed_mutants(self):
        backend = _NativeResults(self.spec)
        search = self.make_search(backend, n_random=2, n_mutants=4)
        self.run_seeded(search, 1, checkpoint=self.path)
        previous = self.make_search(n_random=2, n_mutants=4)
        previous.load(self.path)
        saved = self.path.read_bytes()

        def fail_new_generation(path, payload):
            if payload["next_generation"] == 2:
                raise OSError("simulated checkpoint write failure")
            return _write_slab_json(path, payload)

        with patch("crisp.slab_crisp_search._write_slab_json", side_effect=fail_new_generation), \
                self.assertRaisesRegex(OSError, "checkpoint write failure"):
            self.run_seeded(search, 1, checkpoint=self.path)
        self.assert_same_search(previous, search)
        self.assertEqual(self.path.read_bytes(), saved)
        self.assertEqual(len(backend.cache), 8)
        self.run_seeded(search, 1, checkpoint=self.path)
        self.assertEqual(len(backend.cache), 8)
        self.assertEqual(search.n_relaxed, 8)
        uninterrupted = self.make_search(n_random=2, n_mutants=4)
        self.run_seeded(uninterrupted, 2)
        self.assert_same_search(uninterrupted, search)

    def test_split_population_search_matches_full_run_in_both_screening_modes(self):
        for mode in ("filter", "rank"):
            with self.subTest(mode=mode):
                self.path.unlink(missing_ok=True)
                full_backend = _NativeResults(self.spec)
                resumed_backend = _NativeResults(self.spec)
                full = self.make_search(full_backend, screening_mode=mode)
                split = self.make_search(screening_mode=mode)
                resumed = self.make_search(resumed_backend, screening_mode=mode)
                self.run_seeded(full, 4)
                self.run_seeded(split, 2, checkpoint=self.path)
                resumed.load(self.path)
                self.run_seeded(resumed, 2, checkpoint=self.path)
                self.assert_same_search(full, resumed)
                self.assertGreater(full.n_relaxed, 4)
                self.assertGreater(len(full.archive.entries), 4)
                for candidate_id, actual in resumed_backend.inputs.items():
                    expected = full_backend.inputs[candidate_id]
                    np.testing.assert_array_equal(actual.positions, expected.positions)
                    np.testing.assert_array_equal(actual.cell, expected.cell)
                    np.testing.assert_array_equal(actual.arrays["test_fp"], expected.arrays["test_fp"])

    def test_confidence_filter_retains_uncertain_candidate_and_coverage_floor(self):
        search = self.make_search(max_skip_frac=.6)
        self.run_seeded(search, 1)
        # Four confidently poor random proposals plus one uncertain mutant:
        # the latter survives, and the coverage cap rescues one random proposal.
        with patch.object(search.gp, "predict", side_effect=[(-5, .001)] * 4 + [(-5, 1)]):
            self.run_seeded(search, 1)
        self.assertEqual(search.n_relaxed, 6)
        self.assertEqual([row["origin"] for row in search.outcomes[4:]], ["random", "mutant"])

    def test_mutations_honor_pair_distance_override_below_scalar_fallback(self):
        self.spec = replace(self.spec, min_dist_ang=2.5,
                            pair_min_distances={("Cu", "Cu"): 1.0})
        search = self.make_search()
        # Cu neighbors are 2.1 Angstrom apart: valid under the pair override,
        # and impossible under the scalar fallback even after a small mutation.
        self.run_seeded(search, 2)
        self.assertEqual(search.n_relaxed, 9)
        mutants = [row for row in search.outcomes if row["origin"] == "mutant"]
        self.assertEqual(len(mutants), 1)
        self.assertEqual(mutants[0]["status"], "accepted")

    def test_incompatible_checkpoint_settings_fail_before_replacing_live_state(self):
        source = self.make_search()
        self.run_seeded(source, 1, checkpoint=self.path)
        for changed in ({"seed": 72}, {"screening_mode": "rank"}, {"n_random": 3}):
            with self.subTest(changed=changed):
                target = self.make_search(**changed)
                self.run_seeded(target, 1)
                entries, outcomes, gp = target.archive.entries, target.outcomes, target.gp
                before = deepcopy(outcomes)
                with self.assertRaises(ValueError):
                    target.load(self.path)
                self.assertIs(target.archive.entries, entries)
                self.assertIs(target.outcomes, outcomes)
                self.assertIs(target.gp, gp)
                self.assertEqual(target.outcomes, before)
        backend = _NativeResults(self.spec)
        backend.settings["final_recipe"] = "different-toy-energy-model"
        with self.assertRaises(ValueError):
            self.make_search(backend).load(self.path)

    def test_checkpoint_archive_candidate_ids_must_map_one_to_one_to_accepted_results(self):
        backend = _NativeResults(self.spec)
        native_relax = backend.relax

        def equal_energy_result(atoms, candidate_id):
            result = native_relax(atoms, candidate_id)
            result.energy_per_atom = -10.0
            return result

        source = self.make_search(backend, n_random=2, n_mutants=0)
        with patch.object(backend, "relax", side_effect=equal_energy_result):
            self.run_seeded(source, 1, checkpoint=self.path)
        payload = json.loads(self.path.read_text())
        entries = payload["archive"]["entries"]
        self.assertEqual(len(entries), 2)
        self.assertNotEqual(entries[0]["metadata"]["candidate_id"],
                            entries[1]["metadata"]["candidate_id"])
        # Preserve both valid structures and their equal energies; corrupt only
        # the mapping from the second structure to its native calculation ID.
        entries[1]["metadata"]["candidate_id"] = entries[0]["metadata"]["candidate_id"]
        self.path.write_text(json.dumps(payload))
        target = self.make_search(n_random=2, n_mutants=0)
        self.run_seeded(target, 1)
        archive_entries, outcomes, gp = target.archive.entries, target.outcomes, target.gp
        previous_outcomes = deepcopy(outcomes)
        with self.assertRaises(ValueError):
            target.load(self.path)
        self.assertIs(target.archive.entries, archive_entries)
        self.assertIs(target.outcomes, outcomes)
        self.assertIs(target.gp, gp)
        self.assertEqual(target.outcomes, previous_outcomes)

    def test_unsupported_bulk_operators_are_not_silently_enabled(self):
        for options in ({"enable_flow": True}, {"enable_fp_finisher": True},
                        {"enable_cawr_pretreat": True}, {"enable_swap_mutations": True},
                        {"pressure_GPa": 1}, {"local_relax_mode": "legacy"}):
            with self.subTest(options=options), self.assertRaises(TypeError):
                self.make_search(**options)

    def test_running_and_resuming_leave_process_global_random_streams_unchanged(self):
        numpy_state, python_state = np.random.get_state(), random.getstate()
        search = self.make_search()
        self.run_seeded(search, 2, checkpoint=self.path)
        resumed = self.make_search()
        resumed.load(self.path)
        self.run_seeded(resumed, 2)
        current = np.random.get_state()
        self.assertEqual(numpy_state[0], current[0])
        np.testing.assert_array_equal(numpy_state[1], current[1])
        self.assertEqual(numpy_state[2:], current[2:])
        self.assertEqual(python_state, random.getstate())


if __name__ == "__main__":
    unittest.main()
