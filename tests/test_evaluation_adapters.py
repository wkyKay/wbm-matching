"""Boundary checks between unchanged methods and the unified evaluator."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.wbbs import wbbs
from methods.wmhd import wmhd
from methods.wu import wu
from shared import evaluation_adapters as adapters
from shared.exact_metrics import order_scores
from shared.retrieval_protocol import FullGalleryPool


class GuardedMaps:
    def __init__(self, values):
        self.values, self.accessed = values, []

    def __getitem__(self, index):
        if int(index) not in self.values:
            raise AssertionError(f"Unexpected image read: {index}")
        self.accessed.append(int(index))
        return self.values[int(index)]


def maps_and_pool():
    a = np.ones((5, 5), dtype=np.uint8)
    b, c = a.copy(), a.copy()
    a[1:3, 1:3], b[2:4, 2:4], c[2, 2] = 2, 2, 2
    ids = np.array([12, 17, 29], dtype=np.int64)
    return GuardedMaps(dict(zip(ids, (a, b, c)))), FullGalleryPool(ids, ids, "test", "frozen-toy")


def point_feature(points):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    return wmhd.WMHDFeatures(points, np.ones((len(points), 9)), 1., 1., len(points), 25)


class AdapterTests(unittest.TestCase):
    def test_parallel_feature_preparation_preserves_explicit_id_order(self):
        maps, _ = maps_and_pool()
        ids = np.array([29, 12, 17])
        bundle = SimpleNamespace(maps=maps)
        serial = adapters._extract_gallery("wbbs", bundle, ids, wbbs.parameters(), workers=1)
        parallel = adapters._extract_gallery("wbbs", bundle, ids, wbbs.parameters(), workers=2)
        for expected, actual in zip(serial, parallel):
            np.testing.assert_array_equal(actual.points, expected.points)
            np.testing.assert_array_equal(actual.descriptors, expected.descriptors)
        self.assertEqual(maps.accessed, ids.tolist() * 2)

    def test_wbbs_prepares_only_explicit_test_gallery_and_preserves_scores(self):
        maps, pool = maps_and_pool()
        bundle = SimpleNamespace(maps=maps, valid_ids=np.array([1]), train_ids=np.array([2]))
        with tempfile.TemporaryDirectory() as directory:
            config = {"method": "wbbs", "output_dir": str(Path(directory) / "source")}
            prepared = adapters.prepare_method(bundle, config, pool, Path(directory) / "eval")
            self.assertEqual(maps.accessed, pool.gallery.tolist())
            self.assertEqual(prepared.metadata["split"], "test")
            result = prepared.rank(12, pool[12])
            expected = wbbs.score_candidates(prepared.data[0], prepared.data[1:])
            order = order_scores(pool[12], expected, 12, seed=2026)
            np.testing.assert_array_equal(result.ids, pool[12][order])
            np.testing.assert_allclose(result.scores, expected[order])
            self.assertEqual(result.scored_count, 2)
            self.assertEqual(result.eligible_count, 2)
            with np.load(Path(directory) / "eval/features.npz") as cached:
                np.testing.assert_array_equal(cached["sample_ids"], pool.gallery)
            with self.assertRaises(FileExistsError):
                adapters.prepare_method(bundle, config, pool, Path(directory) / "eval")

    def test_wmhd_test_preparation_and_directional_infinite_ranking(self):
        maps, pool = maps_and_pool()
        bundle = SimpleNamespace(maps=maps)
        features = [point_feature([]), point_feature([[0, 0]]), point_feature([[0, 0], [1, 0]])]
        settings = {"weight_design": "W0", "matching_tolerance": .5}
        with tempfile.TemporaryDirectory() as directory, patch.object(wmhd, "extract_features", side_effect=features):
            config = {"method": "wmhd", "parameters": settings, "output_dir": str(Path(directory) / "source")}
            prepared = adapters.prepare_method(bundle, config, pool, Path(directory) / "eval")
            self.assertEqual(maps.accessed, pool.gallery.tolist())
            result = prepared.rank(12, pool[12])
            self.assertTrue(np.isneginf(result.scores).all())
            self.assertEqual(result.eligible_count, 2)
            self.assertEqual(prepared.metadata["empty_after_esrn"], 1)
            forward = prepared.rank(17, [29]).scores[0]
            reverse = prepared.rank(29, [17]).scores[0]
            self.assertEqual(forward, 0.)
            self.assertEqual(reverse, -.5)

    def test_wu_untied_order_and_scores_equal_legacy_ranker(self):
        rng = np.random.default_rng(11)
        ids = np.array([12, 17, 29, 40, 51])
        data = (rng.normal(size=(5, 116)), rng.normal(size=(5, 8)) / 8,
                rng.normal(size=(5, 8)) / 8)
        settings = wu.parameters({"coarse_candidates": 3})
        prepared = adapters.PreparedMethod("wu", ids, data, settings, {}, tie_seed=2026)
        result = prepared.rank(12, ids[1:])
        expected = wu.rank_candidates(data[0][0], data[0][1:], ids[1:], data[1][0], data[1][1:],
                                      data[2][0], data[2][1:], settings)
        np.testing.assert_array_equal(result.ids, expected[0])
        np.testing.assert_allclose(result.scores, expected[1])
        np.testing.assert_allclose(result.coarse_distances, expected[2])
        self.assertEqual((result.scored_count, result.eligible_count, result.coarse_scored_count), (3, 3, 4))

    def test_wu_hash_tie_precedes_shortlist_boundary_and_keeps_coarse_secondary(self):
        ids = np.array([1, 10, 20, 30, 40, 50])
        features = np.zeros((6, 116))
        full = np.zeros((6, 3))
        settings = wu.parameters({"coarse_candidates": 2})
        prepared = adapters.PreparedMethod("wu", ids, (features, full, full), settings, {}, tie_seed=2026)
        expected = ids[1:][order_scores(ids[1:], np.zeros(5), 1, seed=2026)[:2]]
        result = prepared.rank(1, ids[:0:-1])
        np.testing.assert_array_equal(result.ids, expected)
        # All NCC scores tie, but unequal coarse distances must still dominate the hash.
        features[1:, 0] = [5, 4, 3, 2, 1]
        result = prepared.rank(1, ids[1:])
        np.testing.assert_array_equal(result.ids, [50, 40])
        np.testing.assert_array_equal(result.coarse_distances, [1, 2])

    def test_wu_prepare_uses_pool_ids_and_does_not_touch_other_splits(self):
        maps, pool = maps_and_pool()
        features = [(np.full(116, j), np.zeros(4), np.zeros(4)) for j in range(3)]
        with tempfile.TemporaryDirectory() as directory, patch.object(wu, "encode_wafer", side_effect=features):
            prepared = adapters.prepare_method(SimpleNamespace(maps=maps),
                {"method": "wu", "output_dir": str(Path(directory) / "source")}, pool, Path(directory) / "eval")
            self.assertEqual(maps.accessed, pool.gallery.tolist())
            self.assertTrue(all(values.dtype == np.float32 for values in prepared.data))
            self.assertEqual(prepared.rank(12, pool[12]).coarse_scored_count, 2)

    def test_nakazawa_keeps_final_model_and_negative_hamming(self):
        from methods.nakazawa import nakazawa
        maps, pool = maps_and_pool()
        codes = np.zeros((3, 256), dtype=np.uint8)
        codes[1, :2], codes[2, 0] = 1, 1
        model = object()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            checkpoint = source / "checkpoint_last.pt"
            checkpoint.write_bytes(b"mock-completed-final-checkpoint")
            config = {"method": "nakazawa", "output_dir": str(source), "device": "cpu"}
            bundle = SimpleNamespace(maps=maps)
            with patch.object(nakazawa, "load_trained_model", return_value=(model, "cpu")) as load, \
                    patch.object(nakazawa, "encode_maps", return_value=(codes.astype(float), codes)) as encode:
                prepared = adapters.prepare_method(bundle, config, pool, Path(directory) / "eval")
            self.assertIs(load.call_args.args[1], config)
            self.assertIs(encode.call_args.args[0], model)
            np.testing.assert_array_equal(encode.call_args.args[2], pool.gallery)
            result = prepared.rank(12, pool[12])
            np.testing.assert_array_equal(result.ids, [29, 17])
            np.testing.assert_array_equal(result.scores, [-1, -2])
            self.assertEqual(prepared.metadata["model_selection"], "last_epoch")
            self.assertEqual(checkpoint.read_bytes(), b"mock-completed-final-checkpoint")

    def test_yu_fits_only_train_and_global_does_not_load_detector(self):
        self._check_yu("global")

    def test_yu_predicted_class_gates_query_and_candidates_without_labels(self):
        self._check_yu("predicted_class")

    def _check_yu(self, variant):
        from methods.yu import yu
        maps, pool = maps_and_pool()
        train_ids = np.array([2, 3, 5])
        bundle = SimpleNamespace(maps=maps, train_ids=train_ids)
        fit_calls, inference_calls = [], []
        codes = np.zeros((3, 256), dtype=np.uint8)
        codes[1, :2], codes[2, 0] = 1, 1

        class Encoder:
            def __init__(self, seed):
                self.seed = seed

            def fit(self, features, ids):
                fit_calls.append((features.copy(), ids.copy()))
                self.fit_ids = ids.copy()
                return self

            def transform(self, features):
                np.testing.assert_array_equal(features[:, 0], pool.gallery)
                return codes

            def save(self, path, signature):
                with Path(path).open("xb") as handle:
                    np.savez(handle, fit_ids=self.fit_ids, signature=signature)

        def infer(model, maps, ids, config, device, *, features):
            ids = np.asarray(ids)
            inference_calls.append((model, ids.copy(), features))
            embeddings = np.zeros((len(ids), 1024))
            embeddings[:, 0] = ids
            predictions = np.array([1, 1, 0]) if model == "detector" else np.array([4, 4, 9])
            return {"ids": ids, "features": embeddings, "predictions": predictions}

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            for task in ("classifier", "detector"):
                (source / task).mkdir(parents=True)
                (source / task / "best.pt").write_bytes(task.encode())
            config = {"method": "yu", "output_dir": str(source), "device": "cpu", "seed": 2026}
            with patch.object(yu, "_selected_model", side_effect=lambda task, *args: (task, {"task": task})) as load, \
                    patch.object(yu, "TenValueEncoder", Encoder), patch.object(adapters, "_yu_infer", side_effect=infer):
                prepared = adapters.prepare_method(bundle, config, pool, Path(directory) / "eval",
                                                   source_run_signature="validated-source", variant=variant)
            np.testing.assert_array_equal(fit_calls[0][1], train_ids)
            np.testing.assert_array_equal(fit_calls[0][0][:, 0], train_ids)
            np.testing.assert_array_equal(inference_calls[1][1], pool.gallery)
            self.assertEqual(load.call_args_list[0].args[2], "validated-source")
            self.assertFalse((source / "encoder.npz").exists())
            with np.load(Path(directory) / "eval/encoder.npz") as encoder:
                np.testing.assert_array_equal(encoder["fit_ids"], train_ids)
            result = prepared.rank(12, pool[12])
            self.assertEqual(result.scored_count, 2)
            if variant == "global":
                self.assertEqual(load.call_count, 1)
                np.testing.assert_array_equal(result.ids, [29, 17])
                np.testing.assert_array_equal(result.scores, [-1, -2])
            else:
                self.assertEqual(load.call_count, 2)
                np.testing.assert_array_equal(inference_calls[2][1], pool.gallery)
                np.testing.assert_array_equal(result.ids, [17])
                self.assertEqual(result.eligible_count, 1)
                rejected = prepared.rank(29, pool[29])
                self.assertEqual(len(rejected.ids), 0)
                self.assertEqual(rejected.scored_count, 2)

    def test_yu_inference_uses_original_preprocessing_and_needs_no_labels(self):
        import torch
        from methods.yu import yu

        class Model(torch.nn.Module):
            def forward_features(self, images):
                return images[:, :, 0, 0]

            def fc3(self, features):
                return features

        maps = GuardedMaps({12: np.array([[2]]), 29: np.array([[1]])})
        with patch.object(yu, "preprocess", side_effect=lambda raw: torch.tensor([[[float(raw[0, 0])]], [[0.]]])) as pre:
            result = adapters._yu_infer(Model(), maps, [29, 12], {"inference_batch_size": 1},
                                       torch.device("cpu"), features=True)
        self.assertEqual(maps.accessed, [29, 12])
        self.assertEqual(pre.call_count, 2)
        np.testing.assert_array_equal(result["features"], [[1, 0], [2, 0]])
        np.testing.assert_array_equal(result["ids"], [29, 12])

    def test_invalid_candidate_identity_rejected_and_empty_list_preserved(self):
        data = {"codes": np.zeros((2, 256), dtype=np.uint8)}
        prepared = adapters.PreparedMethod("nakazawa", np.array([12, 29]), data, {}, {}, tie_seed=2026)
        for values in ([12], [29, 29], [50], [29.0]):
            with self.assertRaises(ValueError):
                prepared.rank(12, values)
        result = prepared.rank(12, [])
        self.assertEqual((len(result.ids), result.scored_count, result.eligible_count), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
