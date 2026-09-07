"""Slab archive round trips, compatibility, and failure recovery."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ase.constraints import FixAtoms

from crisp.slab_archive import SlabArchive
from crisp.slab_fingerprint import SlabFingerprintCalculator
from test_slab_archive import _config, _sheet, torch_fplib


class TestSlabArchivePersistence(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "archive.json"
        self.calc = SlabFingerprintCalculator(_config(), cutoff=3.2, natx=8)
        self.archive = self.make_archive()
        fp_patch = patch.object(self.calc, "get_fingerprints",
                                side_effect=lambda atoms: atoms.arrays["test_fp"])
        self.fingerprints = fp_patch.start()
        self.addCleanup(fp_patch.stop)

    def make_archive(self, **options):
        return SlabArchive(self.calc, {"B": 2, "N": 2},
                           **(dict(min_dist_ang=1.0) | options))

    def populate(self):
        for energy in (-3.0, -5.0, -4.0):
            self.assertTrue(self.archive.add(_sheet(), energy,
                                             metadata={"generation": int(-energy)}))
        self.archive.save(self.path)

    def assert_load_failure_preserves_entries(self, destination, error=ValueError):
        entries = destination.entries
        originals = list(entries)
        with self.assertRaises(error):
            destination.load(self.path)
        self.assertIs(destination.entries, entries)
        self.assertEqual(len(entries), len(originals))
        for current, original in zip(entries, originals):
            self.assertIs(current, original)

    def test_roundtrip_preserves_atoms_metadata_order_and_future_identity(self):
        atoms = _sheet()
        atoms.set_array("labels", np.array(["north", "south", "east", "west"]))
        atoms.set_array("flags", np.array([True, False, False, True]))
        atoms.set_array("counts", np.arange(4, dtype=np.int16))
        atoms.set_array("weights", np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
        atoms.set_array("complex_values", np.array([1 + 2j] * 4, dtype=np.complex64))
        atoms.set_constraint(FixAtoms(indices=[0, 3]))
        atoms.set_celldisp([[0.1], [0.2], [0.3]])
        literal = {"01": {"__ndarray__": "literal", "__ase_objtype__": "literal"},
                   "nested": [None, True, 1, 2.5, "text"], "__complex__": [3, 4]}
        atoms.info = deepcopy(literal)
        for energy in (-3.0, -5.0, -4.0):
            self.archive.add(atoms, energy, metadata=literal | {"generation": 4})
        self.archive.entries[0].generation = 9
        self.archive.save(self.path)
        restored = self.make_archive()
        restored.add(_sheet(), 10.0)  # Successful load replaces old contents.
        restored.load(self.path)
        self.assertEqual(restored.get_all_energies().tolist(), [-3.0, -5.0, -4.0])
        self.assertEqual([e.energy for e in restored.get_best(2)], [-5.0, -4.0])
        for before, after in zip(self.archive.entries, restored.entries):
            self.assertEqual((after.energy, after.enthalpy, after.pressure),
                             (before.energy, before.energy, 0.0))
            self.assertEqual(after.metadata, before.metadata)
            self.assertEqual(after.generation, before.generation)
            self.assertEqual(after.atoms.info, before.atoms.info)
            self.assertIsNone(after.atoms.calc)
            np.testing.assert_allclose(after.atoms.cell, before.atoms.cell)
            np.testing.assert_array_equal(after.atoms.pbc, before.atoms.pbc)
            np.testing.assert_array_equal(after.atoms.get_celldisp(), before.atoms.get_celldisp())
            np.testing.assert_array_equal(after.atoms.constraints[0].get_indices(),
                                          before.atoms.constraints[0].get_indices())
            self.assertEqual(set(after.atoms.arrays), set(before.atoms.arrays))
            for name in before.atoms.arrays:
                if name == "positions":
                    np.testing.assert_allclose(after.atoms.arrays[name], before.atoms.arrays[name])
                else:
                    np.testing.assert_array_equal(after.atoms.arrays[name], before.atoms.arrays[name])
                self.assertEqual(after.atoms.arrays[name].dtype, before.atoms.arrays[name].dtype)
            np.testing.assert_allclose(after.fp, before.fp)
            np.testing.assert_allclose(after.fp_pooled, before.fp_pooled)
        self.assertFalse(restored.add(atoms[[3, 0, 2, 1]], -3.0))
        self.assertTrue(restored.add(atoms, -7.0))
        restored.entries[0].metadata["nested"].append("changed")
        self.assertEqual(self.archive.entries[0].metadata["nested"], literal["nested"])

    def test_empty_archive_roundtrip_clears_destination(self):
        self.archive.save(self.path)
        destination = self.make_archive()
        destination.add(_sheet(), -3.0)
        self.fingerprints.reset_mock()
        destination.load(self.path)
        self.assertEqual(destination.entries, [])
        self.fingerprints.assert_not_called()

    def test_save_uses_no_backend_and_load_recomputes_every_descriptor(self):
        self.populate()
        self.fingerprints.reset_mock()
        with patch.object(self.calc, "pool_with_std", side_effect=AssertionError("no pooling")):
            self.archive.save(self.path)
        self.fingerprints.assert_not_called()
        restored = self.make_archive()
        with patch.object(self.calc, "pool_with_std", wraps=self.calc.pool_with_std) as pooled:
            restored.load(self.path)
        self.assertEqual(self.fingerprints.call_count, 3)
        self.assertEqual(pooled.call_count, 3)

    def test_every_setting_must_match_before_backend_calls(self):
        self.populate()
        original = json.loads(self.path.read_text())
        changes = [(("composition",), {"B": 1, "N": 3}),
                   (("min_dist_ang",), 1.01), (("bond_scale",), 1.3),
                   (("fp_threshold",), 0.04), (("energy_threshold",), 0.02),
                   (("fingerprint", "cutoff"), 3.3),
                   (("fingerprint", "natx"), 16), (("fingerprint", "orbital"), "sp")]
        changes += [(("slab", name), value) for name, value in (
            ("area_per_atom_range", [1.0, 9.0]), ("initial_thickness", 0.5),
            ("max_thickness", 3.0), ("cell_height", 15.0),
            ("min_vacuum", 7.0), ("pressure_GPa", 1.0))]
        for keys, value in changes:
            document = deepcopy(original)
            settings = document["settings"]
            for key in keys[:-1]:
                settings = settings[key]
            settings[keys[-1]] = value
            self.path.write_text(json.dumps(document))
            self.fingerprints.reset_mock()
            with self.subTest(setting=keys):
                self.assert_load_failure_preserves_entries(self.archive)
                self.fingerprints.assert_not_called()

    def test_missing_truncated_invalid_version_and_corrupt_records_are_atomic(self):
        self.populate()
        original = json.loads(self.path.read_text())
        self.path.unlink()
        self.assert_load_failure_preserves_entries(self.archive, FileNotFoundError)
        invalid_documents = ["{", "[]", "null", '{"format": "bulk"}']
        for field, value in (("version", 2), ("version", True), ("format", "bulk"),
                             ("entries", {})):
            invalid_documents.append(json.dumps(original | {field: value}))
        bad_energy = deepcopy(original)
        bad_energy["entries"][1]["energy_per_atom"] = "not-an-energy"
        duplicate = deepcopy(original)
        duplicate["entries"][1] = deepcopy(duplicate["entries"][0])
        missing = deepcopy(original)
        del missing["entries"][1]["atoms"]
        bad_pbc = deepcopy(original)
        bad_pbc["entries"][1]["atoms"]["pbc"] = [True, True, True]
        invalid_documents += [json.dumps(data) for data in
                              (bad_energy, duplicate, missing, bad_pbc)]
        invalid_documents.append(json.dumps(original).replace('"energy_per_atom": -5.0',
                                                              '"energy_per_atom": NaN'))
        for document in invalid_documents:
            with self.subTest(document=document[:60]):
                self.path.write_text(document)
                self.assert_load_failure_preserves_entries(self.archive)

    def test_backend_failure_on_later_entry_keeps_existing_entries(self):
        self.populate()
        with patch.object(self.calc, "get_fingerprints", side_effect=[
                self.archive.entries[0].fp, RuntimeError("backend failed")]):
            self.assert_load_failure_preserves_entries(self.archive, RuntimeError)

    def test_overflow_numbers_and_duplicate_json_keys_are_rejected(self):
        self.populate()
        original = json.loads(self.path.read_text())
        for field in ("info", "metadata", "generation"):
            data = deepcopy(original)
            data["entries"][1][field] = ("OVERFLOW" if field == "generation"
                                          else {"nested": ["OVERFLOW"]})
            self.path.write_text(json.dumps(data).replace('"OVERFLOW"', "1e999"))
            with self.subTest(field=field):
                self.assert_load_failure_preserves_entries(self.archive)
        self.path.write_text(json.dumps(original).replace('"version": 1',
                                                         '"version": 2, "version": 1'))
        self.assert_load_failure_preserves_entries(self.archive)

    def test_invalid_array_encodings_cannot_be_silently_coerced_or_broadcast(self):
        self.populate()
        original = json.loads(self.path.read_text())
        encodings = [
            ("numbers", [[4], "float64", [5.5, 5.5, 7.0, 7.0]]),
            ("numbers", [[4], "int64", [5.5, 5.5, 7, 7]]),
            ("pbc", [[3], "int64", [1, 1, 0]]),
            ("pbc", [[3], "bool", [1, 1, 0]]),
            ("labels", [[4], "<U1", ["truncated"] * 4]),
            ("positions", [[4, 3], "complex128", [0.0, 0.0] * 12]),
            ("cell", [[3, 3], "complex128", [0.0, 0.0] * 9]),
            ("positions", [[4, 3], "float64", [0.0]]),
            ("objects", [[4], "object", [None] * 4]),
        ]
        for name, encoding in encodings:
            data = deepcopy(original)
            data["entries"][1]["atoms"][name] = {"__ndarray__": encoding}
            self.path.write_text(json.dumps(data))
            with self.subTest(name=name, dtype=encoding[1]):
                self.assert_load_failure_preserves_entries(self.archive)

    def test_failed_save_preserves_existing_file_and_removes_temporary_file(self):
        self.populate()
        original = self.path.read_bytes()
        self.archive.add(_sheet(), -8.0)
        with patch("crisp.slab_archive.os.fsync", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                self.archive.save(self.path)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        with patch.object(Path, "replace", side_effect=OSError("replace failed")):
            with self.assertRaises(OSError):
                self.archive.save(self.path)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_unsupported_metadata_and_object_arrays_do_not_replace_file(self):
        self.populate()
        original = self.path.read_bytes()
        entry = self.archive.entries[0]
        for attribute, value in (("metadata", {1: "ambiguous key"}),
                                 ("metadata", {"bad": object()}),
                                 ("metadata", {"bad": np.array([1])}),
                                 ("metadata", {"bad": np.inf}),
                                 ("generation", np.nan)):
            previous = getattr(entry, attribute)
            setattr(entry, attribute, value)
            try:
                with self.subTest(attribute=attribute, value=value), \
                        self.assertRaises((TypeError, ValueError)):
                    self.archive.save(self.path)
            finally:
                setattr(entry, attribute, previous)
            self.assertEqual(self.path.read_bytes(), original)
        entry.atoms.info = {"bad": {1, 2}}
        with self.assertRaises(TypeError):
            self.archive.save(self.path)
        entry.atoms.info = {}
        entry.atoms.set_array("objects", np.array([{}, {}, {}, {}], dtype=object))
        with self.assertRaises(TypeError):
            self.archive.save(self.path)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_changed_fingerprint_settings_reject_save_and_load(self):
        self.populate()
        original = self.path.read_bytes()
        self.calc.config = replace(_config(), cell_height=15.0)
        with self.assertRaises(ValueError):
            self.archive.save(self.path)
        self.assertEqual(self.path.read_bytes(), original)
        self.assert_load_failure_preserves_entries(self.archive)


@unittest.skipUnless(torch_fplib is not None, "torch_fplib required")
class TestSlabArchivePersistenceBackend(unittest.TestCase):
    def test_real_s_and_sp_roundtrips_preserve_fingerprints_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            for orbital in ("s", "sp"):
                with self.subTest(orbital=orbital):
                    calc = SlabFingerprintCalculator(_config(), cutoff=3.2,
                                                     natx=32, orbital=orbital)
                    archive = SlabArchive(calc, {"B": 2, "N": 2}, min_dist_ang=1.0)
                    archive.add(_sheet(), -3.0)
                    path = Path(directory) / f"{orbital}.json"
                    archive.save(path)
                    restored = SlabArchive(calc, {"B": 2, "N": 2}, min_dist_ang=1.0)
                    restored.load(path)
                    np.testing.assert_allclose(restored.entries[0].fp, archive.entries[0].fp)
                    np.testing.assert_allclose(restored.entries[0].fp_pooled,
                                               archive.entries[0].fp_pooled)
                    equivalent = _sheet()[[3, 0, 2, 1]]
                    equivalent.positions += [17.3, -8.1, 33.0]
                    self.assertFalse(restored.add(equivalent, -3.0))


if __name__ == "__main__":
    unittest.main()
