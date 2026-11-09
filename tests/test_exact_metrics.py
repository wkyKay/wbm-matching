"""Hand-calculated exact retrieval metrics and label-independent tie ordering."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import exact_metrics as metrics
from shared.retrieval_protocol import FullGalleryPool


class ExactMetricTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        # Class 0/1 are single-label, class 2 is mixed, class 3 is a singleton.
        class_ids = np.array([0, 0, 0, 1, 1, 2, 2, 3])
        class_labels = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
        self.bundle = SimpleNamespace(
            valid_ids=np.arange(8), reserved_ids=np.arange(8),
            class_ids=class_ids, labels=class_labels[class_ids],
        )
        self.pool = {0: [1, 2, 3, 4, 5, 6, 7]}

    def tearDown(self):
        self.temporary.cleanup()

    def evaluate(self, rankings, ks=(1, 2, 3), pool=None, name="result", **kwargs):
        return metrics.evaluate_rankings(
            self.bundle, self.pool if pool is None else pool, rankings, ks,
            self.root / name, **kwargs,
        )

    def test_hand_calculated_hit_precision_recall_and_truncated_ap(self):
        result = self.evaluate(iter([(0, [1, 3, 2])]))
        expected = {
            "Hit@1": 1., "Precision@1": 1., "Recall@1": .5, "mAP@1": .5,
            "Hit@2": 1., "Precision@2": .5, "Recall@2": .5, "mAP@2": .5,
            "Hit@3": 1., "Precision@3": 2 / 3, "Recall@3": 1., "mAP@3": 5 / 6,
        }
        for name, value in expected.items():
            self.assertAlmostEqual(result["metrics"][name], value)
        self.assertEqual(result["metric_version"], "exact_v1")
        self.assertNotIn("mAP", result["metrics"])
        self.assertIn("truncated", result["ap_definition"])
        self.assertEqual(json.loads((self.root / "result/metrics.json").read_text()), result)
        with (self.root / "result/per_query.csv").open(newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["relevant_in_pool"], "2")
        self.assertAlmostEqual(float(row["AP@3"]), 5 / 6)

    def test_gating_does_not_reduce_full_gallery_denominator(self):
        result = self.evaluate({0: [1]}, (1, 3))
        self.assertEqual(result["metrics"]["mAP@3"], .5)
        self.assertEqual(result["metrics"]["Recall@3"], .5)
        self.assertEqual(result["metrics"]["Precision@3"], 1 / 3)
        self.assertEqual(result["metrics"]["Hit@3"], 1.)

    def test_empty_returns_are_misses(self):
        result = self.evaluate({0: []})
        self.assertTrue(all(value == 0. for value in result["metrics"].values()))

    def test_hit_and_precision_count_multiple_hits_differently(self):
        one = self.evaluate({0: [1, 3]}, (2,), name="one")
        two = self.evaluate({0: [1, 2]}, (2,), name="two")
        self.assertEqual(one["metrics"]["Hit@2"], two["metrics"]["Hit@2"])
        self.assertEqual(one["metrics"]["Precision@2"], .5)
        self.assertEqual(two["metrics"]["Precision@2"], 1.)

    def test_zero_relevant_policy_and_class_macro_counts(self):
        pool = {0: [1, 2, 3], 7: [0, 1, 3]}
        result = self.evaluate({0: [1], 7: [0]}, (1,), pool=pool)
        self.assertEqual(result["queries_without_relevant_candidates"], 1)
        self.assertEqual(result["metrics"]["Hit@1"], .5)
        self.assertEqual(result["metrics"]["Precision@1"], .5)
        self.assertEqual(result["metrics"]["mAP@1"], .5)
        self.assertEqual(result["metric_query_counts"]["mAP@1"], 1)
        self.assertEqual(result["class_macro_metric_class_counts"]["mAP@1"], 1)
        empty = self.evaluate({7: []}, (1,), pool={7: [0, 1]}, name="zero")
        self.assertIsNone(empty["metrics"]["mAP@1"])
        self.assertIsNone(empty["class_macro_metrics"]["Recall@1"])
        self.assertEqual(empty["metrics"]["Hit@1"], 0.)

    def test_query_means_class_macro_and_cardinality_groups(self):
        pool = {0: [1, 2, 3], 1: [0, 2, 3], 5: [0, 3, 6]}
        result = self.evaluate(iter([(0, [1]), (1, [0]), (5, [3])]), (1,), pool=pool)
        self.assertEqual(result["num_queries"], 3)
        self.assertEqual(result["num_classes"], 2)
        self.assertAlmostEqual(result["metrics"]["Hit@1"], 2 / 3)
        self.assertEqual(result["class_macro_metrics"]["Hit@1"], .5)
        self.assertAlmostEqual(result["metrics"]["mAP@1"], 1 / 3)
        self.assertEqual(result["class_macro_metrics"]["mAP@1"], .25)
        self.assertEqual(result["groups"]["single_label"]["num_queries"], 2)
        self.assertEqual(result["groups"]["mixed_label"]["metrics"]["Hit@1"], 0.)
        self.assertEqual(set(result["groups"]["by_label_count"]), {"1", "2"})
        with (self.root / "result/per_class.csv").open(newline="") as handle:
            classes = list(csv.DictReader(handle))
        self.assertEqual([row["num_queries"] for row in classes], ["2", "1"])
        self.assertEqual(classes[1]["label_signature"], "0|1")

    def test_exact_combination_does_not_treat_partial_overlap_as_relevant(self):
        result = self.evaluate({5: [0, 6]}, (1, 2), pool={5: [0, 3, 6]})
        self.assertEqual(result["metrics"]["Hit@1"], 0.)
        self.assertEqual(result["metrics"]["Recall@2"], 1.)
        self.assertEqual(result["metrics"]["mAP@2"], .5)

    def test_full_gallery_test_stream_and_metadata(self):
        pool = FullGalleryPool([0, 5], np.arange(8), "test", "frozen-signature")
        def once():
            yield np.int64(0), np.array([1, 2])
            yield np.int64(5), np.array([6, 0])
        result = self.evaluate(once(), (1, 2), pool=pool, split_name="test", metadata={"variant": "global"})
        self.assertTrue(result["test_evaluated"])
        self.assertEqual(result["retrieval_protocol_signature"], "frozen-signature")
        self.assertEqual(result["metadata"], {"variant": "global"})
        with self.assertRaises(ValueError):
            self.evaluate(once(), (1,), pool=pool, name="wrong")

    def test_query_stream_requires_unique_complete_frozen_coverage(self):
        pool = {0: [1, 2, 3], 5: [0, 3, 6]}
        bad_streams = [[], [(0, [1])], [(0, [1]), (0, [2]), (5, [6])],
                       [(0, [1]), (6, [5])], [(False, [1]), (5, [6])],
                       [(0., [1]), (5, [6])], [("0", [1]), (5, [6])], [(0,)]]
        for stream in bad_streams:
            with self.subTest(stream=stream), self.assertRaises(ValueError):
                self.evaluate(iter(stream), (1,), pool=pool)
        self.assertFalse((self.root / "result").exists())

    def test_invalid_candidate_ids_ranked_ids_and_cutoffs(self):
        for ranking in ([1, 1], [0], [8], [-1], [1.0], [True], ["1"], [[1]]):
            with self.subTest(ranking=ranking), self.assertRaises(ValueError):
                self.evaluate({0: ranking}, (1,))
        for candidates in ([1, 1], [0, 1], [8], [-1], [], [1.0], [True]):
            with self.subTest(pool=candidates), self.assertRaises(ValueError):
                self.evaluate({0: []}, (1,), pool={0: candidates})
        for ks in ([], [0], [True], [1.0], [1, 1], [8], "1", None):
            with self.subTest(ks=ks), self.assertRaises(ValueError):
                self.evaluate({0: []}, ks)

    def test_wrong_split_and_invalid_label_mapping_are_rejected(self):
        self.bundle.valid_ids = np.array([0, 1, 2])
        with self.assertRaises(ValueError):
            self.evaluate({0: [1]}, (1,), pool={0: [1, 3]})
        self.bundle.labels[1] = [0, 1, 0]
        with self.assertRaises(ValueError):
            self.evaluate({0: [1]}, (1,), pool={0: [1, 2]})
        self.bundle.labels[1] = [0, 0, 0]
        with self.assertRaises(ValueError):
            self.evaluate({0: [1]}, (1,), pool={0: [1, 2]})

    def test_existing_reports_are_preserved(self):
        self.evaluate({0: [1]})
        original = (self.root / "result/metrics.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.evaluate({0: [2]})
        self.assertEqual((self.root / "result/metrics.json").read_bytes(), original)


class ExactOrderingTests(unittest.TestCase):
    def test_sha256_specification_and_input_order_independence(self):
        ids = np.array([10, 4, 200, 1])
        keys = metrics.tie_keys(9, ids)
        self.assertEqual(keys.dtype, np.dtype("uint64"))
        expected = [int.from_bytes(hashlib.sha256(f"exact_v1:tie:2026:9:{i}".encode("ascii")).digest()[:8], "big") for i in ids]
        np.testing.assert_array_equal(keys, np.asarray(expected, dtype=np.uint64))
        scores = np.array([1., 1., 1., 1.])
        first = ids[metrics.order_scores(ids, scores, 9)]
        permutation = np.array([2, 0, 3, 1])
        second = ids[permutation][metrics.order_scores(ids[permutation], scores[permutation], 9)]
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(keys, metrics.tie_keys(9, ids, seed=2027)))
        self.assertFalse(np.array_equal(keys, metrics.tie_keys(8, ids)))

    def test_score_and_secondary_precede_hash(self):
        ids = np.array([10, 11, 12, 13, 14])
        scores = np.array([.1, .8, .8, -np.inf, .8])
        secondary = np.array([0., 2., 1., 0., 3.])
        np.testing.assert_array_equal(metrics.order_scores(ids, scores, 0, secondary=secondary), [2, 1, 4, 0, 3])

    def test_negative_infinity_and_empty_results_are_supported(self):
        ids = np.array([3, 7, 1])
        order = metrics.order_scores(ids, [0., -np.inf, -np.inf], 0)
        self.assertEqual(order[0], 0)
        self.assertEqual(set(order[1:]), {1, 2})
        self.assertEqual(metrics.order_scores([], [], 0).shape, (0,))

    def test_nonfinite_misaligned_and_noninteger_inputs_are_rejected(self):
        for scores in ([np.nan, 0.], [np.inf, 0.], [0.], [[0., 1.]], [True, False], ["0", "1"]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                metrics.order_scores([1, 2], scores, 0)
        for ids in ([1., 2.], [True, 2], [-1, 2], [1, 1], [[1], [2]], [2**64]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                metrics.tie_keys(0, ids)
        for secondary in ([np.nan, 1.], [np.inf, 1.], [-np.inf, 1.], [1.]):
            with self.subTest(secondary=secondary), self.assertRaises(ValueError):
                metrics.order_scores([1, 2], [0., 0.], 0, secondary=secondary)
        for query in (True, -1, 0., "0"):
            with self.subTest(query=query), self.assertRaises(ValueError):
                metrics.tie_keys(query, [1, 2])


if __name__ == "__main__":
    unittest.main()
