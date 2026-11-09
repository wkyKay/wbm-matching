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
from shared import common
from shared.retrieval_protocol import build_retrieval_protocol, FullGalleryPool


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
        # Every source image is distinct; duplicate fixtures belong in the
        # dedicated regression tests below, not in unrelated protocol tests.
        for sample_id in range(len(maps)):
            maps[sample_id, 0, :5] = [(sample_id >> bit) & 1 for bit in range(5)]
        self.maps, self.labels = maps, labels
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
        self.protocol_revision = 0
        self.config = {"method": "wu", "seed": 2026, "data_file": str(self.data),
                       "split_file": str(self.split), "output_dir": str(self.root / "runs/wu"),
                       "expected_classes": 2, "normal_split": {"train": 2, "valid": 1, "holdout": 2},
                       "subset": {},
                       "ks": [1, 2]}
        self.refresh_retrieval_protocol()

    def refresh_retrieval_protocol(self):
        self.protocol_revision += 1
        directory = self.root / f"retrieval_{self.protocol_revision}"
        build_retrieval_protocol(self.data, self.split, directory, valid_queries=4, min_per_class=1)
        self.config["retrieval_protocol_dir"] = str(directory)

    def tearDown(self):
        self.patch.stop()
        self.temporary.cleanup()

    def test_feature_only_config_does_not_require_metric_settings(self):
        config = {k: v for k, v in self.config.items() if k != "ks"}
        path = self.root / "feature_only.json"
        common.save_json(path, config)
        loaded = common.load_config(path, require_ks=False)
        self.assertNotIn("ks", loaded)
        self.assertEqual(loaded["split_file"], str(self.split))
        with self.assertRaises(ValueError):
            common.load_config(path)  # Existing metric-producing methods stay strict.
        for invalid in ([], [0], [True], "1"):
            with self.subTest(ks=invalid):
                common.save_json(path, {**config, "ks": invalid})
                with self.assertRaises(ValueError):
                    common.load_config(path, require_ks=False)

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
        other_pool = common.make_valid_pool(second, alternative)
        np.testing.assert_array_equal(pool.queries, other_pool.queries)
        np.testing.assert_array_equal(pool.gallery, other_pool.gallery)
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

    def test_run_signature_tracks_deduplication_implementation(self):
        common.ensure_run(self.config, [])
        real_hash = common.hash_file

        def changed_protocol_hash(path):
            return "0" * 64 if Path(path).name == "data_protocol.py" else real_hash(path)

        with mock.patch.object(common, "hash_file", side_effect=changed_protocol_hash):
            with self.assertRaises(ValueError):
                common.ensure_run(self.config, [])

    def test_manifest_label_mismatch_is_rejected(self):
        self.rows[0]["label_0"] = 0
        common.save_csv(self.split, self.rows, list(self.rows[0]))
        with self.assertRaises(ValueError):
            common.prepare_data(self.config)

    def test_duplicate_source_keeps_only_canonical_ids_in_splits_and_pool(self):
        # Duplicate a train image, a valid image and a Normal image at new IDs.
        # The manifest still contains only the lowest original IDs.
        maps = np.concatenate([self.maps, self.maps[[0, 2, 12]]])
        labels = np.concatenate([self.labels, self.labels[[0, 2, 12]]])
        np.savez(self.data, arr_0=maps, arr_1=labels)
        self.refresh_retrieval_protocol()
        source_before, split_before = self.data.read_bytes(), self.split.read_bytes()
        bundle = common.prepare_data(self.config)
        all_ids = np.concatenate([bundle.train_ids, bundle.valid_ids, bundle.reserved_ids,
                                  bundle.normal_train_ids, bundle.normal_valid_ids,
                                  bundle.normal_reserved_ids]).tolist()
        self.assertEqual(sorted(all_ids), list(range(17)))
        self.assertEqual(len({maps[i].tobytes() for i in all_ids}), len(all_ids))
        pool = common.make_valid_pool(bundle, self.config)
        for query, candidates in pool.items():
            self.assertTrue(all(not np.array_equal(maps[query], maps[c]) for c in candidates))
            self.assertEqual(len({maps[c].tobytes() for c in candidates}), len(candidates))
        self.assertEqual(source_before, self.data.read_bytes())
        self.assertEqual(split_before, self.split.read_bytes())

    def test_legacy_manifest_with_cross_split_duplicate_is_rejected(self):
        maps = self.maps.copy()
        maps[2] = maps[0]  # Existing rows assign these IDs to valid and train.
        np.savez(self.data, arr_0=maps, arr_1=self.labels)
        with self.assertRaises(ValueError):
            common.prepare_data(self.config)

    def test_manifest_cannot_omit_a_unique_image_or_choose_later_duplicate(self):
        maps = self.maps.copy()
        maps[2] = maps[0]
        np.savez(self.data, arr_0=maps, arr_1=self.labels)
        # Keeping ID 2 instead of the canonical ID 0 must also be rejected.
        later_only = [row for row in self.rows if row["sample_id"] != 0]
        common.save_csv(self.split, later_only, list(self.rows[0]))
        with self.assertRaises(ValueError):
            common.prepare_data(self.config)
        # Even after removing the true duplicate, dropping another unique
        # sample must not silently shrink the benchmark.
        missing_unique = [row for row in self.rows if row["sample_id"] not in (2, 3)]
        common.save_csv(self.split, missing_unique, list(self.rows[0]))
        with self.assertRaises(ValueError):
            common.prepare_data(self.config)

    def test_conflicting_labels_for_identical_images_are_rejected(self):
        maps = self.maps.copy()
        maps[6] = maps[0]  # Different combination classes cannot share a map.
        np.savez(self.data, arr_0=maps, arr_1=self.labels)
        with self.assertRaises(ValueError):
            common.prepare_data(self.config)

    def test_normal_counts_apply_to_unique_normal_images(self):
        maps = self.maps.copy()
        maps[13] = maps[12]
        np.savez(self.data, arr_0=maps, arr_1=self.labels)
        with self.assertRaises(ValueError):
            common.prepare_data(self.config)  # Old 2+1+2 count now exceeds four.
        self.refresh_retrieval_protocol()
        config = {**self.config, "normal_split": {"train": 2, "valid": 1, "holdout": 1}}
        bundle = common.prepare_data(config)
        ids = np.concatenate([bundle.normal_train_ids, bundle.normal_valid_ids,
                              bundle.normal_reserved_ids]).tolist()
        self.assertEqual(sorted(ids), [12, 14, 15, 16])

    def test_negative_valid_index_is_rejected(self):
        # Python's negative indexing must not allow a malformed source index.
        self.rows[-1]["valid_index"] = -1
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

    def test_frozen_queries_do_not_change_with_training_seed(self):
        first = common.prepare_data(self.config)
        one = common.make_valid_pool(first, self.config)
        alternative = {**self.config, "seed": 7, "output_dir": str(self.root / "different_seed")}
        second = common.prepare_data(alternative)
        two = common.make_valid_pool(second, alternative)
        np.testing.assert_array_equal(one.queries, two.queries)
        np.testing.assert_array_equal(one.gallery, two.gallery)

    def test_full_test_pool_and_streaming_evaluation(self):
        bundle = common.prepare_data(self.config)
        pool = common.make_retrieval_pool(bundle, self.config, "test")
        self.assertIsInstance(pool, FullGalleryPool)
        np.testing.assert_array_equal(pool.queries, bundle.reserved_ids)
        np.testing.assert_array_equal(pool.gallery, bundle.reserved_ids)
        for q, candidates in pool.items():
            self.assertEqual(set(candidates), set(bundle.reserved_ids) - {q})
            self.assertFalse(set(candidates) & set(bundle.train_ids))
            self.assertFalse(set(candidates) & set(bundle.valid_ids))
        def rankings():
            for q, candidates in pool.items():
                ordered = sorted(candidates, key=lambda c: (bundle.class_ids[c] != bundle.class_ids[q], c))
                yield q, ordered[:2]
        result = common.evaluate_rankings(bundle, pool, rankings(), [1, 2], self.root / "test_metrics", "test")
        self.assertTrue(result["test_evaluated"])
        self.assertEqual(result["metrics"]["mAP@2"], 1.)
        self.assertEqual(result["metrics"]["Precision@2"], .5)
        with self.assertRaises(ValueError):
            common.evaluate_rankings(bundle, pool, rankings(), [1], self.root / "wrong_split")

    def test_streaming_rankings_reject_missing_duplicate_and_wrong_split(self):
        bundle = common.prepare_data(self.config)
        pool = common.make_valid_pool(bundle, self.config)
        records = [(q, cs[:1].tolist()) for q, cs in pool.items()]
        for invalid in (records[:-1], records + records[:1], [(4, [5])] + records):
            with self.assertRaises(ValueError):
                common.evaluate_rankings(bundle, pool, iter(invalid), [1], self.root / "invalid_stream")

    def test_smoke_intersects_frozen_queries_and_cannot_evaluate_test(self):
        config = {**self.config, "smoke": True, "subset": {"valid_per_class": 1}}
        bundle = common.prepare_data(config)
        pool = common.make_valid_pool(bundle, config)
        np.testing.assert_array_equal(pool.queries, bundle.valid_ids)
        np.testing.assert_array_equal(pool.gallery, bundle.valid_ids)
        with self.assertRaises(ValueError):
            common.make_retrieval_pool(bundle, config, "test")
        with self.assertRaises(ValueError):
            common.make_valid_pool(bundle, {**config, "smoke": False})

    def test_frozen_files_and_implementation_are_signed(self):
        common.ensure_run(self.config, [])
        real_hash = common.hash_file
        with mock.patch.object(common, "hash_file", side_effect=lambda p: "a" * 64
                               if Path(p).name == "retrieval_protocol.py" else real_hash(p)):
            with self.assertRaises(ValueError):
                common.ensure_run(self.config, [])
        bundle = common.prepare_data(self.config)
        path = Path(self.config["retrieval_protocol_dir"]) / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["extra_note"] = "changed after preparation"
        path.write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            common.ensure_run(self.config, [])
        with self.assertRaises(ValueError):
            common.make_valid_pool(bundle, self.config)


if __name__ == "__main__":
    unittest.main()
