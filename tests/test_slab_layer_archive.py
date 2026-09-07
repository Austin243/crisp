"""Layered archive reuse, strict reconstruction checks, and provenance I/O."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from crisp.slab_archive import SlabArchive, _slab_json_default
from crisp.slab_fingerprint import SlabFingerprintCalculator
from crisp.slab_layer_archive import LayeredSlabArchive
from test_slab_layers import sheet_stack, stack_spec


class TestLayeredSlabArchive(unittest.TestCase):
    def setUp(self):
        self.spec = stack_spec(symbols=("N", "B"))
        self.calc = SlabFingerprintCalculator(self.spec.config, cutoff=3.2, natx=8)
        self.archive = LayeredSlabArchive(self.calc, self.spec)
        fp_patch = patch.object(self.calc, "get_fingerprints", side_effect=lambda atoms:
                                np.repeat(atoms.numbers[:, None], 8, axis=1).astype(float))
        fp_patch.start()
        self.addCleanup(fp_patch.stop)
        distance_patch = patch.object(self.calc, "_fp_dist", side_effect=lambda a, b, types:
                                      float(np.linalg.norm(a - b)))
        distance_patch.start()
        self.addCleanup(distance_patch.stop)

    def test_reuses_sorted_species_matching_ranking_and_diversity(self):
        atoms = sheet_stack(self.spec)
        self.assertTrue(self.archive.add(atoms, -3, metadata={"generation": 2}))
        self.assertFalse(self.archive.add(atoms[[3, 0, 2, 1]], -3))
        self.assertTrue(self.archive.add(atoms, -4))
        self.assertEqual(self.archive.get_all_enthalpies().tolist(), [-3, -4])
        self.assertEqual(self.archive.get_best(1)[0].energy, -4)
        self.assertEqual(len(self.archive.get_diverse(2)), 2)
        stored = self.archive.entries[0].atoms
        np.testing.assert_array_equal(stored.numbers, [5, 5, 7, 7])
        np.testing.assert_array_equal(stored.arrays["initial_atom_id"], [2, 3, 0, 1])
        np.testing.assert_array_equal(stored.arrays["initial_layer"], [1, 1, 0, 0])
        self.assertEqual(self.archive.entries[0].generation, 2)

    def test_roundtrip_preserves_provenance_and_settings(self):
        atoms = sheet_stack(self.spec)
        atoms.positions[[1, 2]] = atoms.positions[[2, 1]]
        atoms.info["layer_policy"] = "initial_only"
        self.archive.add(atoms, -4, metadata={"candidate_id": "g0-c1"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "layers.json"
            self.archive.save(path)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["format"], "crisp-layered-slab-archive")
            restored = LayeredSlabArchive(self.calc, self.spec)
            restored.load(path)
            for name in ("initial_atom_id", "initial_layer"):
                np.testing.assert_array_equal(restored.entries[0].atoms.arrays[name],
                                              self.archive.entries[0].atoms.arrays[name])
            np.testing.assert_array_equal(restored.entries[0].atoms.positions,
                                          self.archive.entries[0].atoms.positions)
            self.assertEqual(restored.entries[0].metadata, {"candidate_id": "g0-c1"})
            self.assertEqual(restored.entries[0].atoms.info, atoms.info)
            self.assertFalse(restored.add(atoms[[3, 2, 1, 0]], -4))

    def test_wrong_format_or_spec_and_corrupt_identity_load_atomically(self):
        self.archive.add(sheet_stack(self.spec), -4)
        original = self.archive._to_payload()
        invalid = []
        invalid.append(original | {"format": "crisp-slab-archive"})
        changed = deepcopy(original)
        changed["settings"]["layer_spec"]["initial_layer_gaps"] = [3.0]
        invalid.append(changed)
        changed = deepcopy(original)
        # _to_payload contains ordinary ndarray values until JSON serialization.
        changed["entries"][0]["atoms"]["initial_atom_id"][:] = 0
        invalid.append(json.loads(json.dumps(changed, default=_slab_json_default)))
        for payload in invalid:
            entries = self.archive.entries
            with self.subTest(format=payload["format"]), self.assertRaises((ValueError, TypeError)):
                self.archive._load_payload(payload)
            self.assertIs(self.archive.entries, entries)
        old = SlabArchive(self.calc, self.spec.composition, min_dist_ang=1)
        with self.assertRaises(ValueError):
            old._load_payload(original)

    def test_geometry_and_provenance_rejected_before_backend(self):
        for kind in ("fragment", "identity", "nonfinite_energy"):
            atoms = sheet_stack(self.spec)
            energy = -4
            if kind == "fragment":
                atoms.positions[3] = atoms.positions[2] + [0, 0, 1.4]
            elif kind == "identity":
                atoms.arrays["initial_atom_id"][0] = 3
            else:
                energy = float("nan")
            with patch.object(self.calc, "get_fingerprints", side_effect=AssertionError("unexpected")):
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    self.archive.add(atoms, energy)
        self.assertEqual(self.archive.entries, [])

    def test_exact_restore_is_atomic_if_descriptor_recomputation_fails(self):
        self.archive.add(sheet_stack(self.spec), -4)
        payload = json.loads(json.dumps(self.archive._to_payload(), default=_slab_json_default))
        original_entries = self.archive.entries
        # First evaluation belongs to inherited validation, second to exact restore.
        valid = self.archive.entries[0].fp.copy()
        with patch.object(self.calc, "get_fingerprints", side_effect=[valid, np.full_like(valid, np.nan)]):
            with self.assertRaisesRegex(ValueError, "configured shape and finite"):
                self.archive._load_payload(payload)
        self.assertIs(self.archive.entries, original_entries)

    def test_noncanonical_record_order_is_rejected_without_replacing_archive(self):
        self.archive.add(sheet_stack(self.spec), -4)
        payload = self.archive._to_payload()
        reordered = self.archive.entries[0].atoms[[3, 0, 2, 1]].todict()
        reordered.pop("info", None)
        payload["entries"][0]["atoms"] = reordered
        payload = json.loads(json.dumps(payload, default=_slab_json_default))
        original_entries = self.archive.entries
        with self.assertRaisesRegex(ValueError, "canonical species order"):
            self.archive._load_payload(payload)
        self.assertIs(self.archive.entries, original_entries)

    def test_settings_cannot_drift_on_existing_archive(self):
        atoms = sheet_stack(self.spec)
        self.archive.spec = replace(self.spec, initial_layer_gaps=(3,))
        with self.assertRaisesRegex(ValueError, "settings changed"):
            self.archive.add(atoms, -4)
        self.archive.spec = self.spec
        self.archive.min_dist_ang = 0.9
        with self.assertRaisesRegex(ValueError, "settings changed"):
            self.archive._to_payload()


if __name__ == "__main__":
    unittest.main()
