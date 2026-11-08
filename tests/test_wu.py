"""Behavioral checks for the Wu WMSR reproduction; synthetic inputs only."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compared_methods.wu import wu


def circle_wafer():
    row, col = np.ogrid[:52, :52]
    return (((row - 25.5) ** 2 + (col - 25.5) ** 2) <= 25**2).astype(np.uint8)


class GuardedMaps:
    """Any train, Normal or held-out image access fails the integration test."""
    def __init__(self, arrays, allowed):
        self.arrays, self.allowed, self.accessed = arrays, set(allowed), []

    def __getitem__(self, index):
        if int(index) not in self.allowed:
            raise AssertionError(f"Forbidden image access: {index}")
        self.accessed.append(int(index))
        return self.arrays[index]


class WuFeatureTests(unittest.TestCase):
    def test_separate_area_and_perimeter_regions_make_14_plus_3_plus_1(self):
        support = np.ones((64, 64), dtype=bool)
        defects = np.zeros_like(support)
        defects[8:14, 8:14] = True  # area 36, perimeter 20
        defects[40, 10:35] = True  # area 25, perimeter 23
        regional = wu.regional_features(defects, support)
        self.assertEqual(regional.shape, (14,))
        self.assertAlmostEqual(regional[0], 36.0 / (np.pi * 32**2))
        self.assertAlmostEqual(regional[1], 23.0 / 32)
        self.assertGreater(regional[8], 0.0)  # Sa has a nonzero minor axis.
        self.assertEqual(regional[9], 0.0)  # Sp is a line.
        self.assertLess(regional[12], regional[13])  # square vs line eccentricity
        self.assertEqual(wu.statistical_features(defects, support).shape, (3,))
        self.assertEqual(wu.geometry_features(defects, support).shape, (18,))

    def test_outer_rings_have_separate_denominators(self):
        support = np.ones((7, 7), dtype=bool)
        defects = np.zeros_like(support)
        defects[0, 0] = defects[1, 1] = defects[3, 3] = True
        np.testing.assert_allclose(wu.statistical_features(defects, support),
                                   [3 / 49, 1 / 24, 1 / 16])

    def test_116_contains_original_and_median_versions(self):
        wafer = circle_wafer()
        wafer[26, 10:42] = 2
        feature = wu.extract_features(wafer)
        self.assertEqual(feature.shape, (116,))
        self.assertTrue(np.isfinite(feature).all())
        self.assertGreater(np.count_nonzero(feature[:58]), 0)
        # A one-pixel scratch is removed by the fixed 3x3 median filter.
        np.testing.assert_array_equal(feature[58:], np.zeros(58))

    def test_empty_normal_ignored_and_point_are_finite(self):
        point = circle_wafer()
        point[25, 25] = 2
        for wafer in (np.zeros((52, 52), dtype=np.uint8), circle_wafer(),
                      np.full((52, 52), 3, dtype=np.uint8), point):
            feature, whole, region = wu.encode_wafer(wafer)
            self.assertEqual(feature.shape, (116,))
            self.assertTrue(all(np.isfinite(x).all() for x in (feature, whole, region)))
        defects, support = wu.wafer_masks(np.array([[0, 1], [2, 3]], dtype=np.uint8))
        np.testing.assert_array_equal(support, [[False, True], [True, False]])
        np.testing.assert_array_equal(defects, [[False, False], [True, False]])

    def test_largest_region_preserves_canvas_and_stable_tie(self):
        defects = np.zeros((20, 20), dtype=bool)
        defects[2:4, 3:5] = True
        defects[12:14, 13:15] = True
        salient = wu.largest_region(defects)
        self.assertEqual(salient.shape, defects.shape)
        self.assertEqual(int(salient.sum()), 4)
        self.assertTrue(salient[2:4, 3:5].all())
        self.assertFalse(salient[12:14, 13:15].any())

    def test_deterministic_hough_single_line_and_separated_segments(self):
        support = circle_wafer().astype(bool)
        mask = np.zeros_like(support)
        mask[26, 10:42] = True
        self.assertEqual(wu.hough_line_count(mask, support), 1)
        self.assertEqual(wu.hough_line_count(mask, support), 1)
        mask[:, :] = False
        mask[26, 10:20] = mask[26, 30:40] = True
        self.assertEqual(wu.hough_line_count(mask, support), 2)


class WuRankingTests(unittest.TestCase):
    def test_ncc_zero_variance_and_alignment(self):
        image = np.zeros((12, 12)); image[2:5, 2:5] = 1
        shifted = np.zeros_like(image); shifted[7:10, 7:10] = 1
        self.assertEqual(wu.ncc_score(np.zeros((3, 3)), np.zeros((3, 3))), 0)
        self.assertEqual(wu.ncc_score(np.ones((3, 3)), np.ones((3, 3))), 0)
        self.assertAlmostEqual(wu.ncc_score(image, image), 1, places=6)
        self.assertAlmostEqual(wu.ncc_score(image, 1 - image), -1, places=6)
        self.assertLess(wu.ncc_score(image, shifted), 0)

    def test_ncc_reranks_but_cannot_recover_a_screened_candidate(self):
        query_feature = np.zeros(116)
        features = np.zeros((3, 116)); features[:, 0] = [0, 1, 2]
        query_ncc = np.array([1, -1], dtype=np.float32) / np.sqrt(2)
        vectors = np.stack([-query_ncc, query_ncc, query_ncc])
        ids, scores, distances = wu.rank_candidates(
            query_feature, features, np.array([10, 20, 30]),
            query_ncc, vectors, query_ncc, vectors, {"coarse_candidates": 2})
        self.assertEqual(ids.tolist(), [20, 10])
        np.testing.assert_allclose(scores, [1, -1], atol=1e-6)
        np.testing.assert_allclose(distances, [1, 0])

    def test_ties_use_distance_then_id_and_prefilter_uses_id(self):
        features = np.zeros((3, 116)); features[0, 0] = 1
        zero_vectors = np.zeros((3, 4))
        ids, _, _ = wu.rank_candidates(np.zeros(116), features, np.array([5, 9, 2]),
                                       np.zeros(4), zero_vectors, np.zeros(4), zero_vectors)
        self.assertEqual(ids.tolist(), [2, 9, 5])
        ids, _, _ = wu.rank_candidates(np.zeros(116), features, np.array([5, 9, 2]),
                                       np.zeros(4), zero_vectors, np.zeros(4), zero_vectors,
                                       {"coarse_candidates": 1})
        self.assertEqual(ids.tolist(), [2])

    def test_prepare_and_valid_touch_only_allowed_valid_rows(self):
        arrays = np.stack([circle_wafer() for _ in range(8)])
        arrays[2:4, 20:27, 20:27] = 2
        arrays[4:6, 15, 10:42] = 2
        guarded = GuardedMaps(arrays, [2, 3, 4, 5])
        bundle = SimpleNamespace(maps=guarded, valid_ids=np.array([2, 3, 4, 5]),
                                 train_ids=np.array([0]), normal_train_ids=np.array([1]),
                                 normal_valid_ids=np.array([], dtype=np.int64),
                                 class_ids=np.array([0, -1, 0, 0, 1, 1, 0, -1]),
                                 signature="synthetic-wu-no-test")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            config = {"method": "wu", "seed": 2026, "output_dir": str(output),
                      "parameters": {}, "ks": [1, 2],
                      "valid_pool": {"queries": 4, "candidates": 3, "seed": 2026}}
            prepared = wu.prepare(bundle, config, output, workers=1)
            self.assertEqual(prepared["rows"], 4)
            self.assertEqual(guarded.accessed, [2, 3, 4, 5])
            # A matching cache is reused without touching the images again.
            wu.prepare(bundle, config, output, workers=1)
            self.assertEqual(guarded.accessed, [2, 3, 4, 5])
            result = wu.validate(bundle, config, output)
            self.assertEqual(result["queries"], 4)
            self.assertFalse(result["test_accessed"])
            self.assertAlmostEqual(result["metrics"]["metrics"]["Precision@1"], 1)
            self.assertEqual(guarded.accessed, [2, 3, 4, 5])
            self.assertTrue((output / "rankings.csv").exists())


if __name__ == "__main__":
    unittest.main()
