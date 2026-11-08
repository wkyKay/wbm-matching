"""Protocol leakage, provenance and retrieval-denominator regression checks."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.patch = mock.patch.object(common, "ROOT", self.root)
        self.patch.start()
        labels = np.zeros((17, 8), dtype=np.uint8)
        labels[:6, 0] = 1
        labels[6:12, 1] = 1
        maps = np.ones((17, 7, 7), dtype=np.uint8)
        maps[:12, 2:5, 2:5] = 2
        self.data = self.root / "data.npz"
        np.savez(self.data, arr_0=maps, arr_1=labels)
        self.split = self.root / "split.csv"
        self.rows = []
        for start in (0, 6):
            for offset in range(6):
                i = start + offset
                row = {"sample_id": i, "original_index": i, "valid_index": i,
                       "split": ("train", "valid", "test")[offset // 2]}
                row.update({f"label_{k}": int(labels[i, k]) for k in range(8)})
                self.rows.append(row)
        common.save_csv(self.split, self.rows, list(self.rows[0]))
        self.config = {"method": "wu", "seed": 2026, "data_file": str(self.data),
                       "split_file": str(self.split), "output_dir": str(self.root / "runs/wu"),
                       "expected_classes": 2, "normal_split": {"train": 2, "valid": 1, "holdout": 2},
                       "subset": {}, "valid_pool": {"queries": 4, "candidates": 3, "seed": 2026},
                       "ks": [1, 2]}

    def tearDown(self):
        self.patch.stop()
        self.temporary.cleanup()

    def test_frozen_partition_and_normal_sidecar(self):
        before = self.split.read_bytes()
        common.ensure_run(self.config, [])
        bundle = common.prepare_data(self.config)
        self.assertEqual(bundle.train_ids.tolist(), [0, 1, 6, 7])
        self.assertEqual(bundle.valid_ids.tolist(), [2, 3, 8, 9])
        self.assertEqual(bundle.reserved_ids.tolist(), [4, 5, 10, 11])
        self.assertEqual((len(bundle.normal_train_ids), len(bundle.normal_valid_ids),
                          len(bundle.normal_reserved_ids)), (2, 1, 2))
        selected = set(np.concatenate([bundle.train_ids, bundle.valid_ids, bundle.normal_train_ids,
                                      bundle.normal_valid_ids]).tolist())
        self.assertFalse(selected & set(bundle.reserved_ids))
        self.assertFalse(selected & set(bundle.normal_reserved_ids))
        self.assertEqual(before, self.split.read_bytes())
        repeated = common.prepare_data(self.config)
        self.assertEqual(bundle.signature, repeated.signature)
        np.testing.assert_array_equal(bundle.normal_train_ids, repeated.normal_train_ids)
        self.assertIsInstance(bundle.maps, np.memmap)

    def test_same_pool_across_methods_and_no_self(self):
        common.ensure_run(self.config, [])
        bundle = common.prepare_data(self.config)
        pool = common.make_valid_pool(bundle, self.config)
        alternative = {**self.config, "method": "yu", "output_dir": str(self.root / "runs/yu")}
        common.ensure_run(alternative, [])
        second = common.prepare_data(alternative)
        self.assertEqual(pool, common.make_valid_pool(second, alternative))
        for q, candidates in pool.items():
            self.assertNotIn(q, candidates)
            self.assertEqual(set(candidates), set(bundle.valid_ids) - {q})
        with self.assertRaises(ValueError):
            common.make_valid_pool(bundle, {**self.config, "valid_pool": {"queries": 5, "candidates": 3}})

    def test_run_signature_rejects_scientific_changes(self):
        first = common.ensure_run(self.config, [])
        self.assertEqual(first, common.ensure_run({**self.config, "device": "cuda", "num_workers": 4}, []))
        with self.assertRaises(ValueError):
            common.ensure_run({**self.config, "seed": 1}, [])
        other = self.root / "unowned"
        other.mkdir()
        (other / "user.txt").write_text("preserve")
        with self.assertRaises(FileExistsError):
            common.ensure_run({**self.config, "output_dir": str(other)}, [])
        self.assertEqual((other / "user.txt").read_text(), "preserve")

    def test_manifest_label_mismatch_is_rejected(self):
        self.rows[0]["label_0"] = 0
        common.save_csv(self.split, self.rows, list(self.rows[0]))
        with self.assertRaises(ValueError):
            common.prepare_data(self.config)

    def test_small_subset_preserves_classes_and_split(self):
        config = {**self.config, "subset": {"train_per_class": 1, "valid_per_class": 1,
                                             "normal_train": 1, "normal_valid": 1}}
        bundle = common.prepare_data(config)
        self.assertEqual(len(bundle.train_ids), 2)
        self.assertEqual(set(bundle.class_ids[bundle.train_ids]), {0, 1})
        self.assertTrue(set(bundle.train_ids) <= {0, 1, 6, 7})
        self.assertTrue(set(bundle.valid_ids) <= {2, 3, 8, 9})
        with self.assertRaises(ValueError):
            common.prepare_data({**self.config, "subset": {"train_per_class": 3}})

    def test_truncated_ap_uses_full_candidate_denominator(self):
        # Independent four-item example: two relevant candidates, one irrelevant.
        bundle = SimpleNamespace(valid_ids=np.arange(4), class_ids=np.array([0, 0, 1, 0]))
        q, positive, second_positive, negative = 0, 1, 3, 2
        pool = {q: [positive, negative, second_positive]}
        result = common.evaluate_rankings(bundle, pool, {q: [positive, negative]}, [1, 2], self.root / "metrics")
        self.assertEqual(result["metrics"]["Precision@2"], 0.5)
        self.assertEqual(result["metrics"]["Recall@2"], 0.5)
        self.assertEqual(result["metrics"]["mAP@2"], 0.5)
        empty = common.evaluate_rankings(bundle, pool, {q: []}, [1, 2], self.root / "empty")
        self.assertEqual(empty["metrics"]["mAP@2"], 0.)
        self.assertEqual(empty["metrics"]["Precision@2"], 0.)

    def test_zero_relevant_and_invalid_rankings(self):
        bundle = common.prepare_data(self.config)
        result = common.evaluate_rankings(bundle, {2: [8, 9]}, {2: [8, 9]}, [1, 2], self.root / "zero")
        self.assertEqual(result["queries_without_relevant_candidates"], 1)
        self.assertIsNone(result["metrics"]["mAP@2"])
        self.assertEqual(result["metrics"]["Precision@2"], 0.)
        for ranks in ({2: [8, 8]}, {2: [2]}, {2: [12]}):
            with self.assertRaises(ValueError):
                common.evaluate_rankings(bundle, {2: [8, 9]}, ranks, [1], self.root / "bad")
        with self.assertRaises(ValueError):
            common.evaluate_rankings(bundle, {4: [8, 9]}, {4: [8, 9]}, [1], self.root / "test_forbidden")

    def test_relative_config_paths_independent_of_cwd(self):
        path = self.root / "config.json"
        config = {**self.config, "data_file": "data.npz", "split_file": "split.csv",
                  "output_dir": "runs/wu"}
        path.write_text(json.dumps(config))
        loaded = common.load_config(path)
        self.assertEqual(loaded["data_file"], str(self.data))
        self.assertEqual(loaded["output_dir"], str(self.root / "runs/wu"))


if __name__ == "__main__":
    unittest.main()
