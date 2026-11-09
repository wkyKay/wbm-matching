"""Second-task directional category-coverage metric tests."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

import numpy as np

from shared import coverage_metrics


class CoverageMetricTests(unittest.TestCase):
    def test_direction_is_query_to_candidate_and_extra_labels_are_free(self):
        ab = np.array([[1, 1, 0, 0]], dtype=np.uint8)
        abc = np.array([[1, 1, 1, 0]], dtype=np.uint8)
        self.assertEqual(coverage_metrics.coverage_values(ab, abc)[0], 1.0)
        self.assertAlmostEqual(coverage_metrics.coverage_values(abc, ab)[0], 2 / 3)

    def test_graded_ranking_and_full_coverage_use_complete_gallery(self):
        query = np.array([[1, 1, 0, 0]], dtype=np.uint8)
        gallery = np.array([
            [1, 1, 1, 0],  # full coverage by superset
            [1, 0, 0, 0],  # partial
            [0, 0, 1, 0],  # disjoint
        ], dtype=np.uint8)
        ranked = gallery[[1, 0, 2]]
        result = coverage_metrics.query_metrics(query, gallery, ranked, (1, 2, 3))
        self.assertEqual(result["full_coverage_in_pool"], 1)
        self.assertAlmostEqual(result["MeanCoverage@1"], 0.5)
        self.assertEqual(result["FullCoverageHit@1"], 0.0)
        self.assertEqual(result["FullCoverageHit@2"], 1.0)
        self.assertEqual(result["FullCoverageRecall@2"], 1.0)
        self.assertAlmostEqual(result["FullCoverageAP@2"], 0.5)
        self.assertLess(result["Coverage-nDCG@2"], 1.0)

    def test_zero_coverage_gallery_has_undefined_ndcg_and_full_recall(self):
        query = np.array([[1, 1, 0, 0]], dtype=np.uint8)
        gallery = np.array([[0, 0, 1, 0]], dtype=np.uint8)
        result = coverage_metrics.query_metrics(query, gallery, gallery, (1,))
        self.assertIsNone(result["Coverage-nDCG@1"])
        self.assertIsNone(result["FullCoverageRecall@1"])
        self.assertIsNone(result["FullCoverageAP@1"])
        self.assertEqual(result["FullCoveragePrecision@1"], 0.0)

    def test_ranking_csv_writes_separate_coverage_reports(self):
        labels = np.array([
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ], dtype=np.uint8)
        bundle = SimpleNamespace(labels=labels, class_ids=np.arange(4),
                                 valid_ids=np.arange(4), reserved_ids=np.arange(4))
        pool = {0: [1, 2, 3]}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            ranking_path = output / "rankings.csv"
            ranking_path.write_text(
                "query_id,rank,candidate_id\n0,1,2\n0,2,1\n0,3,3\n", encoding="utf-8")
            result = coverage_metrics.evaluate_rankings_file(
                bundle, pool, ranking_path, (1, 2, 3), output, split_name="valid")
            self.assertEqual(result["metric_version"], "coverage_v1")
            coverage = json.loads((output / "coverage_metrics.json").read_text())
            self.assertEqual(coverage["metric_version"], "coverage_v1")
            self.assertEqual(coverage["full_coverage_condition"], "Lq subset Lc; coverage == 1")
            self.assertEqual(coverage["metrics"]["FullCoverageHit@2"], 1.0)
            for name in ("coverage_per_query.csv", "coverage_per_class.csv"):
                self.assertTrue((output / name).is_file())


if __name__ == "__main__":
    unittest.main()
