"""Equation and protocol checks for WMHD; no retrieval metric definitions."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods.wmhd import wmhd
from shared.retrieval_protocol import FullGalleryPool


def feature(points, values=1.0):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    neighborhoods = np.full((len(points), 9), values, dtype=float)
    return wmhd.WMHDFeatures(points, neighborhoods, 1.0, 1.0, len(points), 100)


class WMHDTests(unittest.TestCase):
    def test_mountain_uses_all_die_denominator_and_exponential_distance(self):
        support = np.ones((1, 3), dtype=bool)
        defects = np.array([[True, False, True]])
        field, beta = wmhd.mountain_field(defects, support, {"mountain_m": 1.0})
        self.assertEqual(beta, 3 / 2)
        np.testing.assert_allclose(field, [[1 + np.exp(-3), 2 * np.exp(-1.5), 1 + np.exp(-3)]])

    def test_esrn_enhances_and_removes_without_mutating_input(self):
        support = np.ones((9, 9), dtype=bool)
        defects = np.zeros((9, 9), dtype=bool)
        defects[2:5, 2:5] = True
        defects[3, 3] = False
        defects[7, 7] = True
        original = defects.copy()
        cleaned = wmhd.esrn(defects, support)
        self.assertTrue(cleaned[3, 3])
        self.assertFalse(cleaned[7, 7])
        np.testing.assert_array_equal(defects, original)
        np.testing.assert_array_equal(wmhd.esrn(defects, support, {"esrn_iterations": 0}), defects)

    def test_background_not_a_die_or_a_good_neighbor(self):
        wafer = np.zeros((5, 5), dtype=np.uint8)
        wafer[0, 0] = 3
        wafer[2, 2] = 2
        wafer[2, 3] = 2
        result = wmhd.extract_features(wafer)
        self.assertEqual(result.support_dies, 2)
        self.assertEqual(len(result.points), 2)
        self.assertEqual(result.raw_defects, 2)

    def test_coordinates_use_support_not_defect_extent(self):
        support = np.ones((5, 5), dtype=bool)
        coords, center, radius = wmhd.wafer_coordinates(support)
        np.testing.assert_array_equal(center, [2, 2])
        self.assertAlmostEqual(radius, np.sqrt(8))
        self.assertAlmostEqual(np.linalg.norm(coords, axis=1).max(), 1)
        a = np.ones((5, 5), dtype=np.uint8)
        b = a.copy()
        a[2, 2] = 2
        b[0, 0] = 2
        fa = wmhd.extract_features(a, {"esrn_iterations": 0})
        fb = wmhd.extract_features(b, {"esrn_iterations": 0})
        self.assertEqual(fa.radius, fb.radius)

    def test_weight_two_is_sum_including_center_and_zero_ratio_is_zero(self):
        query, candidate = np.full((1, 9), 2.0), np.ones((1, 9))
        self.assertEqual(wmhd.weight_matrix(query, candidate, "W0")[0, 0], 1)
        self.assertEqual(wmhd.weight_matrix(query, candidate, "W1")[0, 0], 0.5)
        self.assertEqual(wmhd.weight_matrix(query, candidate, "W2")[0, 0], 4.5)
        zeros = np.zeros((1, 9))
        self.assertEqual(wmhd.weight_matrix(zeros, zeros, "W2")[0, 0], 0)

    def test_outlier_penalty_is_not_averaged_and_denominator_is_original_na(self):
        query, candidate = feature([[0, 0], [2, 0]]), feature([[0.1, 0]])
        p = {"matching_tolerance": 1.0, "weight_design": "W0"}
        self.assertAlmostEqual(wmhd.distance(query, candidate, p), 0.1 / 2 + 1.0)

    def test_matching_minimizes_weighted_distance_not_spatial_nearest_only(self):
        query = feature([[0, 0]], 2)
        candidate = feature([[0.1, 0], [0.2, 0]], 1)
        candidate.neighborhoods[1] = 2
        self.assertEqual(wmhd.distance(query, candidate, {"matching_tolerance": 0.5}), 0)

    def test_far_zero_weight_point_cannot_match_under_o1(self):
        query = feature([[0, 0]], 2)
        candidate = feature([[0.2, 0], [3, 0]], 1)
        candidate.neighborhoods[1] = 2
        p = {"weight_design": "W1", "matching_tolerance": 0.5}
        self.assertAlmostEqual(wmhd.distance(query, candidate, p), 0.1)
        self.assertEqual(wmhd.distance(query, candidate, {**p, "outlier_penalty": False}), 0)

    def test_directionality_and_strict_tolerance(self):
        a, b = feature([[0, 0]]), feature([[0, 0], [1, 0]])
        p = {"weight_design": "W0", "matching_tolerance": 0.5}
        self.assertEqual(wmhd.distance(a, b, p), 0)
        self.assertEqual(wmhd.distance(b, a, p), 0.5)
        # At exactly Sout the pair is excluded; W1 would otherwise make cost 0.
        self.assertEqual(wmhd.distance(a, feature([[0.5, 0]]), {"matching_tolerance": 0.5}), 0.5)

    def test_empty_and_single_center_cases_are_defined(self):
        empty, nonempty = feature([]), feature([[0, 0]])
        self.assertEqual(wmhd.distance(empty, empty), 0)
        self.assertTrue(np.isinf(wmhd.distance(empty, nonempty)))
        self.assertEqual(wmhd.distance(nonempty, empty), 0.1)
        self.assertTrue(np.isinf(wmhd.distance(nonempty, empty, {"outlier_penalty": False})))
        wafer = np.ones((5, 5), dtype=np.uint8)
        wafer[2, 2] = 2
        encoded = wmhd.extract_features(wafer, {"esrn_iterations": 0})
        self.assertTrue(np.isfinite(encoded.neighborhoods).all())
        self.assertEqual(encoded.neighborhoods[0, 4], 1)

    def test_chunked_distance_matches_scalar_equation_reference(self):
        rng = np.random.default_rng(2026)
        q = feature(rng.uniform(-1, 1, (9, 2)))
        c = feature(rng.uniform(-1, 1, (13, 2)))
        q.neighborhoods[:] = rng.uniform(0.1, 2, (9, 9))
        c.neighborhoods[:] = rng.uniform(0.1, 2, (13, 9))
        tolerance = 0.7
        summed, outliers = 0.0, 0
        for a, pa in zip(q.points, q.neighborhoods):
            possible = []
            for b, pb in zip(c.points, c.neighborhoods):
                spatial = abs(a[0] - b[0]) + abs(a[1] - b[1])
                if spatial < tolerance:
                    weight = sum(abs(float(x) - float(y)) / max(float(x), float(y)) for x, y in zip(pa, pb))
                    possible.append(spatial * weight)
            if possible:
                summed += min(possible)
            else:
                outliers += 1
        expected = summed / len(q.points) + outliers * tolerance
        for chunks in [1, 3, 256]:
            self.assertAlmostEqual(wmhd.distance(q, c, {"point_chunk_size": chunks,
                                                       "matching_tolerance": tolerance}), expected)

    def test_ranking_consumes_all_candidates_and_uses_id_ties(self):
        q = feature([[0, 0]])
        candidates = [feature([[0, 0]]), feature([[0.4, 0]]), feature([[0, 0]])]
        ids, scores = wmhd.rank_candidates(q, iter(candidates), [9, 5, 2])
        np.testing.assert_array_equal(ids, [2, 9, 5])
        np.testing.assert_allclose(scores, [0, 0, -0.1])

    def test_ranking_optional_self_exclusion_and_id_validation(self):
        q = feature([[0, 0]])
        ids, scores = wmhd.rank_candidates(q, iter([q, q]), [7, 2], query_id=7)
        np.testing.assert_array_equal(ids, [2])
        np.testing.assert_array_equal(scores, [0])
        ids, scores = wmhd.rank_candidates(q, [], [])
        self.assertEqual(len(ids), 0)
        self.assertEqual(len(scores), 0)
        with self.assertRaises(ValueError):
            wmhd.rank_candidates(q, [q], [2.5])
        with self.assertRaises(ValueError):
            wmhd.rank_candidates(q, [q], [2, 3])

    def test_invalid_parameters_and_maps_fail(self):
        for p in [{"mountain_m": 0}, {"matching_tolerance": np.nan}, {"weight_design": "W3"},
                  {"esrn_good_threshold": 0.5}, {"esrn_iterations": -1}, {"point_chunk_size": 0},
                  {"unknown": 1}, {"outlier_penalty": 1}]:
            with self.assertRaises(ValueError):
                wmhd.parameters(p)
        with self.assertRaises(ValueError):
            wmhd.extract_features(np.full((3, 3), 4))

    def test_prepare_reads_only_frozen_valid_gallery_and_hash_checks_cache(self):
        pool = FullGalleryPool([4], [3, 4, 8], "valid", "frozen")
        accesses = []

        class GuardedMaps:
            def __getitem__(self, index):
                index = int(index)
                if index not in (3, 4, 8):
                    raise AssertionError("Read outside fixed valid gallery")
                accesses.append(index)
                wafer = np.ones((5, 5), dtype=np.uint8)
                wafer[1:4, 1:4] = 2
                return wafer

        bundle = SimpleNamespace(signature="bundle", maps=GuardedMaps())
        with tempfile.TemporaryDirectory() as directory:
            config = {"output_dir": directory}
            with patch("shared.common.make_retrieval_pool", return_value=pool) as make_pool:
                record = wmhd.prepare(bundle, config)
                make_pool.assert_called_once_with(bundle, config, split_name="valid")
            self.assertEqual(accesses, [3, 4, 8])
            self.assertEqual(record["rows"], 3)
            self.assertFalse(record["metrics_computed"])
            self.assertFalse(record["test_accessed"])
            loaded = wmhd.load_features(directory)
            self.assertEqual(set(loaded), {3, 4, 8})
            self.assertEqual(wmhd.distance(loaded[3], loaded[4]), 0)
            with patch("shared.common.make_retrieval_pool", return_value=pool):
                self.assertEqual(wmhd.prepare(bundle, config), record)
            self.assertEqual(accesses, [3, 4, 8])
            with (Path(directory) / "features.npz").open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                wmhd.load_features(directory)


if __name__ == "__main__":
    unittest.main()
