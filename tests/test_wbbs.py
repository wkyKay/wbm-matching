"""Mathematical and protocol-boundary checks for the WBBS reproduction."""
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.wbbs.wbbs import (WaferFeatures, best_buddies, encode_wafer, load_prepared,
                              parameters, prepare, rank_candidates, rank_from_pool,
                              retained_points, score_candidates, score_pair,
                              tensor_saliencies, tensor_vote)
from shared.retrieval_protocol import FullGalleryPool


class TensorVotingTests(unittest.TestCase):
    def test_votes_equal_literal_matrix_formula(self):
        # Independently evaluate Eqs. 2,6,7,8 with matrix products, not the
        # identity-tensor simplification used by the implementation.
        points = np.array([[0., 0.], [1., 0.], [2., 3.], [-1., 4.]])
        expected = np.zeros((len(points), 2, 2))
        for i, receiver in enumerate(points):
            for j, voter in enumerate(points):
                if i == j:
                    continue
                delta = receiver - voter
                distance2 = delta @ delta
                unit = delta / np.sqrt(distance2)
                outer = np.outer(unit, unit)
                reflection = np.eye(2) - 2 * outer
                expected[i] += np.exp(-distance2 / 9) * reflection @ np.eye(2) @ (np.eye(2) - 0.5 * outer) @ reflection.T
        np.testing.assert_allclose(tensor_vote(points, 9, 2), expected, atol=1e-14)
        np.testing.assert_allclose(tensor_vote(points, 9, 1), expected, atol=1e-14)
        self.assertTrue((np.linalg.eigvalsh(expected) >= 0).all())

    def test_analytic_two_point_saliency_and_self_exclusion(self):
        value = np.exp(-1 / 9)
        saliency = tensor_saliencies(tensor_vote(np.array([[0, 0], [0, 1]])))
        np.testing.assert_allclose(saliency, np.full((2, 2), value / 2))
        np.testing.assert_array_equal(tensor_vote(np.array([[0, 0]])), np.zeros((1, 2, 2)))
        self.assertEqual(tensor_vote(np.empty((0, 2))).shape, (0, 2, 2))

    def test_denoising_uses_and_and_strict_threshold(self):
        saliency = np.array([[1., 0.], [0., 1.], [.31, .31], [.32, .1], [.1, .32]])
        np.testing.assert_array_equal(retained_points(saliency, .32), [True, True, False, True, True])
        # If one channel is identically zero, its strict comparison to zero
        # is false, so the literal paper rule does not remove a site.
        np.testing.assert_array_equal(retained_points(np.array([[0., 0.], [0., 1.]])), [True, True])

    def test_input_codes_and_rotation(self):
        wafer = np.ones((9, 9), dtype=np.uint8)
        wafer[2:6, 4] = 2
        wafer[0, 1] = 3
        encoded = encode_wafer(wafer)
        self.assertEqual(encoded.original_count, 4)
        self.assertTrue((encoded.points[:, 0] >= 2).all())
        self.assertTrue(np.isfinite(encoded.descriptors).all())
        rotated = encode_wafer(np.rot90(wafer))
        # Rotation only permutes descriptors because Eq16 discards angle.
        np.testing.assert_allclose(np.sort(encoded.descriptors, axis=0),
                                   np.sort(rotated.descriptors, axis=0), atol=1e-14)
        self.assertEqual(encode_wafer(np.full((3, 3), 3)).descriptors.shape, (0, 3))

    def test_invalid_votes_and_parameters_fail(self):
        with self.assertRaises(ValueError):
            tensor_vote(np.array([[0, 0], [0, 0]]))
        with self.assertRaises(ValueError):
            tensor_saliencies(np.array([[[1, 2], [0, 1]]]))
        for change in ({'sigma_d': 0}, {'mu': -1}, {'total_dies': 1.5}, {'alpha': np.nan}, {'unknown': 1}):
            with self.assertRaises(ValueError):
                parameters(change)


def feature(rows):
    rows = np.asarray(rows, dtype=float).reshape(-1, 3)
    points = np.column_stack((np.arange(len(rows)), np.zeros(len(rows))))
    return WaferFeatures(points, rows, len(rows))


class BestBuddiesTests(unittest.TestCase):
    def test_minimum_count_normalization_and_penalty(self):
        query = feature([[0, 0, 0], [1, 1, 1]])
        candidate = feature([[0, 0, 0], [.5, .5, .5], [1, 1, 1]])
        settings = {'alpha': 5, 'total_dies': 10}
        self.assertAlmostEqual(score_pair(query, candidate, settings), np.exp(-.5))
        self.assertAlmostEqual(score_pair(candidate, query, settings), np.exp(-.5))
        self.assertAlmostEqual(score_pair(query, candidate, {'alpha': 0}), 1.)

    def test_repeated_descriptors_follow_single_argmin_rule(self):
        duplicate = feature([[.2, .3, .4], [.2, .3, .4], [.2, .3, .4]])
        np.testing.assert_array_equal(best_buddies(duplicate.descriptors, duplicate.descriptors), [[0, 0]])
        self.assertAlmostEqual(score_pair(duplicate, duplicate), 1 / 3)
        np.testing.assert_array_equal(best_buddies(duplicate.descriptors, duplicate.descriptors, block_size=1), [[0, 0]])

    def test_mu_is_squared_radius_distance_weight(self):
        query = np.array([[0., 0., 0.]])
        candidate = np.array([[.2, 0., 0.], [0., 0., 1.]])
        np.testing.assert_array_equal(best_buddies(query, candidate, mu=0), [[0, 1]])
        np.testing.assert_array_equal(best_buddies(query, candidate, mu=1), [[0, 0]])

    def test_empty_sets_bounds_symmetry_and_rank_ties(self):
        empty = feature([])
        one = feature([[0, 0, 0]])
        self.assertEqual(score_pair(empty, empty), 0)
        self.assertEqual(score_pair(one, empty), 0)
        rng = np.random.default_rng(27)
        q, c = feature(rng.random((7, 3))), feature(rng.random((11, 3)))
        self.assertGreaterEqual(score_pair(q, c), 0)
        self.assertLessEqual(score_pair(q, c), 1)
        self.assertAlmostEqual(score_pair(q, c), score_pair(c, q))
        ids, scores = rank_candidates(one, [one, one, one], np.array([9, 2, 4]), query_id=4)
        np.testing.assert_array_equal(ids, [2, 9])
        np.testing.assert_array_equal(scores, [1, 1])
        np.testing.assert_array_equal(score_candidates(one, [one, empty]), [1, 0])

    def test_duplicate_ids_and_length_mismatch_fail(self):
        one = feature([[0, 0, 0]])
        with self.assertRaises(ValueError):
            rank_candidates(one, [one, one], np.array([1, 1]))
        with self.assertRaises(ValueError):
            rank_candidates(one, [one], np.array([1, 2]))


class CacheTests(unittest.TestCase):
    def test_original_ids_cache_integrity_and_full_pool(self):
        maps = np.ones((9, 7, 7), dtype=np.uint8)
        maps[2, 2:5, 3] = 2
        maps[6, 3, 2:5] = 2
        maps[8, 2:4, 2:4] = 2
        bundle = SimpleNamespace(maps=maps, valid_ids=np.array([2, 6, 8]), signature='toy')
        with tempfile.TemporaryDirectory() as path:
            output = Path(path)
            meta = prepare(bundle, {'smoke': True}, output)
            self.assertEqual(meta['rows'], 3)
            self.assertFalse(meta['test_accessed'])
            self.assertFalse(meta['metrics_computed'])
            store = load_prepared(output)
            np.testing.assert_allclose(store.get(8).descriptors, encode_wafer(maps[8]).descriptors)
            with self.assertRaises(KeyError):
                store.get(0)  # CSV/cache row zero is not original sample ID 0.
            pool = FullGalleryPool(np.array([6]), np.array([2, 6, 8]), 'valid', 'toy')
            ids, scores = rank_from_pool(store, pool, 6)
            self.assertEqual(set(ids), {2, 8})
            self.assertEqual(len(scores), 2)
            self.assertEqual(prepare(bundle, {'smoke': True}, output), meta)
            with self.assertRaises(ValueError):
                prepare(bundle, {'parameters': {'mu': 2}}, output)
            with (output / 'features.npz').open('ab') as handle:
                handle.write(b'corruption')
            with self.assertRaises(ValueError):
                load_prepared(output)


if __name__ == '__main__':
    unittest.main()
