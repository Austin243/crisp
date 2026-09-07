"""Bounded slab searches, independent trial streams, and resumable checkpoints."""

from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

from crisp.slab import prepare_slab
from crisp.slab_archive import SlabArchive
from crisp.slab_fingerprint import SlabFingerprintCalculator
from crisp.slab_generation import SlabGenerationError
from crisp.slab_relaxation import SlabQuenchBoundsError, SlabQuenchNotConverged
from crisp.slab_search import SlabRandomSearch
from test_slab_archive import _config, _sheet
from test_slab_relaxation import _Calculator, _harmonic


def _candidate(kind=0):
    if kind == 1:
        raise SlabGenerationError("generation attempt limit")
    atoms = _sheet()
    atoms.info["test_kind"] = kind
    return [atoms]


def _relax(atoms, config, calc_factory, **options):
    kind = atoms.info.get("test_kind", 0)
    if kind == 2:
        raise SlabQuenchBoundsError("thickness limit")
    if kind == 5:
        raise SlabQuenchNotConverged("optimizer step limit")
    relaxed = atoms.copy()
    if kind == 3:
        relaxed.positions[1] = relaxed.positions[0]  # Actual distance validation rejects this.
    relaxed.info.update(slab_quench_steps=1, slab_quench_fmax=0.0)
    energy = -4.0 if kind == 6 else -3.0
    relaxed.calc = SinglePointCalculator(relaxed, energy=energy * len(relaxed),
                                        forces=np.zeros((len(relaxed), 3)))
    return relaxed


class TestSlabSearch(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "search.json"
        self.calc = SlabFingerprintCalculator(_config(), cutoff=3.2, natx=8)
        fp_patch = patch.object(self.calc, "get_fingerprints",
                                side_effect=lambda atoms: atoms.arrays["test_fp"])
        self.fingerprints = fp_patch.start()
        self.addCleanup(fp_patch.stop)

    def make_search(self, **options):
        archive = SlabArchive(self.calc, {"B": 2, "N": 2}, min_dist_ang=1.0)
        return SlabRandomSearch(archive, Mock(), **(dict(
            calculator_id="test-model-v1", seed=73, layer_groups=[2, 1, 2],
            max_generation_attempts=7, fmax=0.01, max_steps=9) | options))

    def run_kinds(self, search, kinds, **options):
        candidates = iter(kinds)
        with patch("crisp.slab_search.generate_slabs",
                   side_effect=lambda *args, **kwargs: _candidate(next(candidates))), \
                patch("crisp.slab_search.quench_slab", side_effect=_relax):
            return search.run(len(kinds), **options)

    def assert_load_failure_is_atomic(self, search, error=ValueError):
        entries, outcomes = search.archive.entries, search.outcomes
        original_entries, original_outcomes = list(entries), deepcopy(outcomes)
        with self.assertRaises(error):
            search.load(self.path)
        self.assertIs(search.archive.entries, entries)
        self.assertIs(search.outcomes, outcomes)
        self.assertEqual(outcomes, original_outcomes)
        self.assertEqual(len(entries), len(original_entries))
        for before, after in zip(original_entries, entries):
            self.assertIs(before, after)

    def test_all_outcomes_consume_exact_additional_trial_and_forward_budgets(self):
        search = self.make_search()
        kinds = iter(range(7))
        with patch("crisp.slab_search.generate_slabs",
                   side_effect=lambda *args, **kwargs: _candidate(next(kinds))) as generate, \
                patch("crisp.slab_search.quench_slab", side_effect=_relax) as quench:
            self.assertIs(search.run(3), search.archive)
            self.assertEqual(search.completed_trials, 3)
            search.run(4)
            search.run(0)
        self.assertEqual([o["status"] for o in search.outcomes], [
            "accepted", "generation_rejected", "quench_rejected", "validation_rejected",
            "duplicate", "quench_rejected", "accepted"])
        self.assertEqual(search.counts, dict(accepted=2, generation_rejected=1,
                                            quench_rejected=2, validation_rejected=1, duplicate=1))
        self.assertEqual(generate.call_count, 7)
        self.assertEqual(quench.call_count, 6)
        for index, call in enumerate(generate.call_args_list):
            self.assertEqual(call.args, ({"B": 2, "N": 2}, _config(), 1))
            self.assertEqual(call.kwargs, dict(seed=search.outcomes[index]["seed"],
                                              layer_groups=[1, 2], min_dist_ang=1.0, max_attempts=7))
        for call in quench.call_args_list:
            self.assertIs(call.args[2], search.calc_factory)
            self.assertEqual(call.kwargs, dict(fmax=0.01, max_steps=9))
        self.assertEqual([e.energy for e in search.archive.get_best()], [-4.0, -3.0])
        for entry, index in zip(search.archive.entries, (0, 6)):
            outcome = search.outcomes[index]
            self.assertEqual(entry.metadata, dict(search_trial=index + 1,
                                                  trial_seed=outcome["seed"],
                                                  calculator_id="test-model-v1"))
            self.assertEqual(entry.energy, outcome["energy_per_atom"])
            self.assertIsNone(entry.atoms.calc)
        for outcome in search.outcomes:
            if outcome["status"] in ("generation_rejected", "quench_rejected"):
                self.assertIsNone(outcome["energy_per_atom"])
                self.assertIsNone(outcome["quench_steps"])
            else:
                self.assertEqual(outcome["quench_steps"], 1)

    def test_uninterrupted_and_resumed_trials_match_without_global_rng_changes(self):
        full, split, resumed = (self.make_search() for _ in range(3))
        numpy_state, python_state = np.random.get_state(), random.getstate()
        with patch("crisp.slab_search.generate_slabs",
                   side_effect=lambda *args, seed, **kwargs: _candidate(seed % 7)), \
                patch("crisp.slab_search.quench_slab", side_effect=_relax):
            full.run(12)
            split.run(5, checkpoint=self.path)
            resumed.load(self.path)
            resumed.run(7, checkpoint=self.path)
        self.assertEqual(full.outcomes, resumed.outcomes)
        self.assertEqual(full.counts, resumed.counts)
        self.assertIn("generation_rejected", full.counts)
        self.assertIn("quench_rejected", full.counts)
        self.assertGreater(len(full.archive.entries), 0)
        for before, after in zip(full.archive.entries, resumed.archive.entries):
            self.assertEqual(before.energy, after.energy)
            self.assertEqual(before.metadata, after.metadata)
            np.testing.assert_allclose(before.atoms.positions, after.atoms.positions)
            np.testing.assert_allclose(before.fp, after.fp)
        current_numpy = np.random.get_state()
        self.assertEqual(numpy_state[0], current_numpy[0])
        np.testing.assert_array_equal(numpy_state[1], current_numpy[1])
        self.assertEqual(numpy_state[2:], current_numpy[2:])
        self.assertEqual(python_state, random.getstate())

    def test_empty_and_all_rejected_searches_save_and_resume(self):
        search = self.make_search()
        with patch("crisp.slab_search.generate_slabs") as generate:
            search.run(0, checkpoint=self.path)
        generate.assert_not_called()
        restored = self.make_search()
        restored.load(self.path)
        self.assertEqual(restored.counts, {})
        self.run_kinds(search, [1, 2, 3, 5], checkpoint=self.path)
        restored.load(self.path)
        self.assertEqual(restored.completed_trials, 4)
        self.assertEqual(restored.outcomes, search.outcomes)
        self.assertEqual(restored.archive.get_best(), [])
        self.run_kinds(restored, [0])
        self.assertEqual(restored.archive.entries[0].metadata["search_trial"], 5)

    def test_real_quench_moves_atoms_with_fresh_calculators_and_fixed_cell(self):
        search = self.make_search(fmax=1e-6, max_steps=100)
        candidate = _sheet()
        before = candidate.copy()
        target = candidate.positions + [[0.08, -0.04, 0.07], [-0.03, 0.02, -0.04],
                                        [0.01, 0.03, 0.02], [-0.02, -0.02, -0.05]]
        calculators = []

        def factory():
            calculator = _harmonic(target)
            calculators.append(calculator)
            return calculator

        search.calc_factory = Mock(side_effect=factory)
        with patch("crisp.slab_search.generate_slabs", return_value=[candidate]):
            search.run(2)
        self.assertEqual(search.counts, {"accepted": 1, "duplicate": 1})
        self.assertEqual(search.calc_factory.call_count, 2)
        self.assertIsNot(calculators[0], calculators[1])
        self.assertTrue(all(outcome["quench_steps"] > 0 for outcome in search.outcomes))
        self.assertLess(search.outcomes[0]["energy_per_atom"], 1e-12)
        expected = candidate.copy()
        expected.positions = target
        expected = prepare_slab(expected, _config())
        expected = expected[np.argsort(expected.numbers, kind="stable")]
        np.testing.assert_allclose(search.archive.entries[0].atoms.positions,
                                   expected.positions, atol=1e-6, rtol=0)
        for calculator in calculators:
            np.testing.assert_allclose(calculator.states[-1].positions, target, atol=1e-6, rtol=0)
            for state in calculator.states:
                np.testing.assert_array_equal(state.cell, before.cell)
                np.testing.assert_array_equal(state.pbc, [True, True, False])
        np.testing.assert_array_equal(candidate.positions, before.positions)
        self.assertIsNone(candidate.calc)

    def test_fatal_errors_propagate_without_consuming_trial_or_replacing_checkpoint(self):
        for origin in ("generator", "calculator_runtime", "calculator_value", "fingerprint", "matcher"):
            with self.subTest(origin=origin):
                search = self.make_search()
                self.run_kinds(search, [0], checkpoint=self.path)
                previous = self.path.read_bytes()
                entries, outcomes = search.archive.entries[:], deepcopy(search.outcomes)
                error = ValueError("fatal calculator") if origin == "calculator_value" else RuntimeError("fatal")
                generate = patch("crisp.slab_search.generate_slabs", return_value=_candidate())
                quench = patch("crisp.slab_search.quench_slab", side_effect=_relax)
                if origin == "generator":
                    failure = patch("crisp.slab_search.generate_slabs", side_effect=error)
                elif origin.startswith("calculator"):
                    search.calc_factory = Mock(side_effect=error)
                    # Exercise the real quench so generic calculator exceptions cannot be swallowed.
                    from crisp.slab_relaxation import quench_slab
                    failure = patch("crisp.slab_search.quench_slab", side_effect=quench_slab)
                elif origin == "fingerprint":
                    failure = patch.object(self.calc, "get_fingerprints", side_effect=error)
                else:
                    failure = patch.object(self.calc, "_fp_dist", side_effect=error)
                with generate, quench, failure, self.assertRaises(type(error)):
                    search.run(1, checkpoint=self.path)
                self.assertEqual(search.outcomes, outcomes)
                self.assertEqual(search.archive.entries, entries)
                self.assertEqual(self.path.read_bytes(), previous)
                self.run_kinds(search, [6], checkpoint=self.path)
                self.assertEqual(search.completed_trials, 2)
                self.assertEqual(search.outcomes[-1]["trial"], 2)

    def test_invalid_options_or_external_archive_changes_fail_before_generation(self):
        options = [{"seed": value} for value in (-1, True, 1.5)]
        options += [{"max_steps": value} for value in (-1, True, 1.5)]
        options += [{"max_generation_attempts": value} for value in (0, True, 1.5)]
        options += [{"fmax": value} for value in (0, np.inf, np.nan, 1j)]
        options += [{"calculator_id": value} for value in ("", " ", None)]
        options += [{"layer_groups": value} for value in ([], [0], [81], [True])]
        for option in options:
            with self.subTest(option=option), self.assertRaises((TypeError, ValueError)):
                self.make_search(**option)
        # Include the geometry validator's allowed thickness tolerance in clearance.
        config = _config()
        calc = SlabFingerprintCalculator(
            config, cutoff=config.cell_height - config.max_thickness - 1.005e-6, natx=8)
        archive = SlabArchive(calc, {"B": 2, "N": 2}, min_dist_ang=1.0)
        with self.assertRaisesRegex(ValueError, "clearance"):
            SlabRandomSearch(archive, Mock(), calculator_id="model")
        search = self.make_search()
        with patch("crisp.slab_search.generate_slabs") as generate:
            for n in (-1, True, 0.5):
                with self.assertRaises(ValueError):
                    search.run(n)
            search.archive.bond_scale = 1.3
            with self.assertRaisesRegex(ValueError, "settings changed"):
                search.run(1)
        generate.assert_not_called()
        populated = self.make_search()
        self.run_kinds(populated, [0])
        with self.assertRaises(ValueError):
            SlabRandomSearch(populated.archive, Mock(), calculator_id="model")
        populated.archive.entries.clear()
        with self.assertRaisesRegex(ValueError, "entries changed"):
            populated.run(1)

    def test_invalid_final_energy_is_fatal_even_for_a_structurally_invalid_candidate(self):
        for invalid_energy in (np.nan, np.inf, 1j, np.array([3.0])):
            for invalid_structure in (False, True):
                with self.subTest(energy=invalid_energy, invalid_structure=invalid_structure):
                    search = self.make_search()
                    self.run_kinds(search, [0], checkpoint=self.path)
                    previous = self.path.read_bytes()
                    outcomes = deepcopy(search.outcomes)
                    candidate = _sheet()
                    if invalid_structure:
                        candidate.positions[1] = candidate.positions[0]
                    calculator = _Calculator(lambda atoms, _: (-12.0, np.zeros((len(atoms), 3))))
                    # Quench sees finite energy and zero forces; the driver's final getter fails.
                    calculator.get_potential_energy = Mock(side_effect=[-12.0, invalid_energy])
                    search.calc_factory = Mock(return_value=calculator)
                    self.fingerprints.reset_mock()
                    with patch("crisp.slab_search.generate_slabs", return_value=[candidate]), \
                            patch("crisp.slab_search.validate_slab_candidate") as validate, \
                            self.assertRaises(RuntimeError):
                        search.run(1, checkpoint=self.path)
                    self.assertEqual(calculator.get_potential_energy.call_count, 2)
                    validate.assert_not_called()
                    self.fingerprints.assert_not_called()
                    self.assertEqual(search.outcomes, outcomes)
                    self.assertEqual(search.counts, {"accepted": 1})
                    self.assertEqual(self.path.read_bytes(), previous)

    def test_complex_calculator_energy_or_forces_are_fatal_during_real_quench(self):
        zeros = np.zeros((4, 3))
        for energy, forces in ((-12.0 + 0j, zeros), (-12.0 + 1j, zeros),
                               (-12.0, zeros.astype(complex)), (-12.0, zeros + 1j)):
            with self.subTest(energy=energy, force_dtype=forces.dtype):
                search = self.make_search(max_steps=0)
                calculator = _Calculator(lambda *_: (energy, forces))
                search.calc_factory = Mock(return_value=calculator)
                self.fingerprints.reset_mock()
                with patch("crisp.slab_search.generate_slabs", return_value=_candidate()), \
                        self.assertRaises(RuntimeError):
                    search.run(1)
                self.assertEqual(search.completed_trials, 0)
                self.assertEqual(search.counts, {})
                self.assertEqual(search.archive.entries, [])
                self.fingerprints.assert_not_called()

    def test_search_settings_and_energy_model_mismatch_reject_before_backend(self):
        source = self.make_search()
        self.run_kinds(source, [0], checkpoint=self.path)
        for options in ({"calculator_id": "other-model"}, {"seed": 74}, {"layer_groups": [1]},
                        {"max_generation_attempts": 8}, {"fmax": 0.02}, {"max_steps": 10}):
            destination = self.make_search(**options)
            self.fingerprints.reset_mock()
            with self.subTest(options=options):
                self.assert_load_failure_is_atomic(destination)
                self.fingerprints.assert_not_called()

    def test_corrupt_progress_and_accepted_entry_mismatches_leave_destination_intact(self):
        source, destination = self.make_search(), self.make_search()
        self.run_kinds(source, list(range(7)), checkpoint=self.path)
        original = json.loads(self.path.read_text())
        self.run_kinds(destination, [6])
        documents = ["{", "[]", json.dumps(original | {"version": True}),
                     json.dumps(original | {"version": 2}),
                     json.dumps(original | {"format": "bulk"}),
                     json.dumps(original | {"outcomes": {}})]
        for index, field, value in ((0, "trial", 2), (0, "seed", 0), (0, "status", "unknown"),
                                     (0, "reason", "unexpected"), (1, "reason", None),
                                     (1, "energy_per_atom", 0.0), (1, "quench_steps", 0),
                                     (0, "energy_per_atom", True), (0, "energy_per_atom", np.inf),
                                     (0, "quench_steps", 10), (0, "quench_steps", True),
                                     (0, "status", "duplicate")):
            data = deepcopy(original)
            data["outcomes"][index][field] = value
            documents.append(json.dumps(data))
        for field, value in (("metadata", {}), ("energy_per_atom", -20.0)):
            data = deepcopy(original)
            data["archive"]["entries"][0][field] = value
            documents.append(json.dumps(data))
        missing_entry = deepcopy(original)
        missing_entry["archive"]["entries"].pop()
        documents.append(json.dumps(missing_entry))
        for document in documents:
            self.path.write_text(document)
            with self.subTest(document=document[:100]):
                self.assert_load_failure_is_atomic(destination)
        self.path.write_text(json.dumps(original))
        with patch.object(self.calc, "get_fingerprints", side_effect=RuntimeError("backend")):
            self.assert_load_failure_is_atomic(destination, RuntimeError)

    def test_failed_checkpoint_save_retains_completed_work_and_previous_file(self):
        search = self.make_search()
        self.run_kinds(search, [0], checkpoint=self.path)
        original = self.path.read_bytes()
        with patch("crisp.slab_archive.os.fsync", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                self.run_kinds(search, [6], checkpoint=self.path)
        self.assertEqual(search.completed_trials, 2)
        self.assertEqual(search.counts, {"accepted": 2})
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        search.save(self.path)
        restored = self.make_search()
        restored.load(self.path)
        self.assertEqual(restored.outcomes, search.outcomes)
        self.assertEqual(restored.archive.get_all_energies().tolist(), [-3.0, -4.0])


if __name__ == "__main__":
    unittest.main()
