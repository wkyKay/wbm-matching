"""Serial/process equivalence and bounded query submission, including ties."""
from concurrent.futures import Future
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import evaluation
from shared.evaluation_adapters import PreparedMethod
from methods.wbbs.wbbs import WaferFeatures, parameters as wbbs_parameters
from methods.wmhd.wmhd import WMHDFeatures, parameters as wmhd_parameters
from methods.wu.wu import parameters as wu_parameters


class ParallelRankingTests(unittest.TestCase):
    def test_serial_and_spawn_match_all_methods_and_full_candidates(self):
        ids = np.array([37, 2, 19, 50], dtype=np.int64)
        rng = np.random.default_rng(2026)
        points = [np.array([[0., 0.], [(i + 1) / 10, 0.]]) for i in range(4)]
        full = rng.normal(size=(4, 8)).astype(np.float32)
        features = np.zeros((4, 116), dtype=np.float32)  # coarse-boundary ties
        cases = [
            ("wu", (features, full, full), wu_parameters({"coarse_candidates": 2})),
            ("wbbs", [WaferFeatures(p, np.ones((2, 3)), 2) for p in points], wbbs_parameters()),
            ("wmhd", [WMHDFeatures(p if i else np.empty((0, 2)),
                         np.ones((2 if i else 0, 9)), 1., 1., 2, 25)
                       for i, p in enumerate(points)], wmhd_parameters()),
            ("yu", {"codes": np.zeros((4, 256), dtype=np.uint8)}, {}),
            ("nakazawa", {"codes": np.zeros((4, 256), dtype=np.uint8)}, {}),
        ]
        for method, data, settings in cases:
            with self.subTest(method=method):
                ranker = PreparedMethod(method, ids, data, settings, {}, tie_seed=2026)
                items = [(int(q), ids[ids != q][::-1]) for q in ids]
                serial = list(evaluation.iter_ranked_queries(ranker, iter(items), 1))
                parallel = list(evaluation.iter_ranked_queries(ranker, iter(items), 2))
                for a, b in zip(serial, parallel):
                    self.assertEqual(a[0], b[0])
                    np.testing.assert_array_equal(a[1], b[1])
                    for field in ("ids", "scores", "coarse_distances"):
                        np.testing.assert_array_equal(getattr(a[2], field), getattr(b[2], field))
                    for field in ("scored_count", "eligible_count", "coarse_scored_count"):
                        self.assertEqual(getattr(a[2], field), getattr(b[2], field))

    def test_submission_is_bounded_and_close_releases_executor(self):
        consumed, shutdown = [], []

        class Executor:
            def __init__(self, **kwargs):
                pass

            def submit(self, function, item):
                future = Future()
                future.set_result((None, 0.))
                return future

            def shutdown(self, **kwargs):
                shutdown.append(kwargs)

        def items():
            for q in range(100):
                consumed.append(q)
                yield q, np.array([q + 100])

        with patch.object(evaluation, "ProcessPoolExecutor", Executor):
            stream = evaluation.iter_ranked_queries(None, items(), 3)
            self.assertEqual(next(stream)[0], 0)
            self.assertEqual(len(consumed), 6)
            self.assertEqual(next(stream)[0], 1)
            self.assertEqual(len(consumed), 7)
            stream.close()
        self.assertEqual(shutdown, [{"wait": True, "cancel_futures": True}])

    def test_worker_failure_is_propagated(self):
        ranker = PreparedMethod("yu", np.array([1, 2]),
            {"codes": np.zeros((2, 256), dtype=np.uint8)}, {}, {}, tie_seed=2026)
        with self.assertRaisesRegex(ValueError, "exclude"):
            list(evaluation.iter_ranked_queries(ranker, [(1, np.array([1, 2]))], 2))


if __name__ == "__main__":
    unittest.main()
