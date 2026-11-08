"""Yu network, preprocessing, train-only encoder and exact resume regression checks."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))
from compared_methods.yu import yu

torch.set_num_threads(4)


def wafer():
    rows, cols = np.ogrid[:52, :52]
    return (((rows - 25.5) ** 2 + (cols - 25.5) ** 2) < 25 ** 2).astype(np.uint8)


class YuArchitectureTests(unittest.TestCase):
    def test_complete_models_have_expected_shapes_and_sigmoid_features(self):
        detector = yu.DetectionCNN().eval()
        classifier = yu.ClassificationCNN().eval()
        x = torch.zeros(1, 3, 224, 224)
        with torch.inference_mode():
            self.assertEqual(detector.features(x).shape, (1, 64, 8, 8))
            self.assertEqual(detector(x).shape, (1, 2))
            self.assertEqual(classifier.features(x).shape, (1, 128, 8, 8))
            feature = classifier.forward_features(x)
            self.assertEqual(feature.shape, (1, 1024))
            self.assertTrue(torch.all((feature > 0) & (feature < 1)))
            self.assertEqual(classifier(x).shape, (1, 37))
            self.assertTrue(torch.equal(classifier(x), classifier.fc3(feature)))
        self.assertEqual(sum(isinstance(m, nn.Conv2d) for m in classifier.modules()), 7)
        self.assertEqual(sum(isinstance(m, nn.MaxPool2d) for m in classifier.modules()), 3)
        self.assertEqual(classifier.fc1.in_features, 8192)
        self.assertEqual(classifier.fc1.out_features, 4096)
        with self.assertRaises(ValueError):
            yu.ClassificationCNN(38)

    def test_l2_is_explicit_and_excludes_bias(self):
        linear = nn.Linear(2, 1)
        with torch.no_grad():
            linear.weight.fill_(2)
            linear.bias.fill_(100)
        self.assertEqual(float(yu.explicit_l2(linear).detach()), 8.)


class YuPreprocessingTests(unittest.TestCase):
    def test_median_is_on_original_52_grid_before_resize(self):
        raw = wafer()
        raw[26, 10:42] = 2
        with patch.object(yu, "median_filter", wraps=yu.median_filter) as median:
            output = yu.preprocess(raw)
        self.assertEqual(median.call_args.args[0].shape, (52, 52))
        self.assertEqual(median.call_args.kwargs["size"], 9)
        self.assertEqual(output.shape, (3, 224, 224))
        self.assertEqual(int(np.count_nonzero(yu.filtered_map(raw) == 2)), 0)
        self.assertEqual(float(output.min()), -1.)
        self.assertEqual(float(output.max()), 1.)

    def test_code3_ignored_and_empty_inputs_remain_finite(self):
        zero = yu.preprocess(np.zeros((52, 52), dtype=np.uint8))
        ignored = yu.preprocess(np.full((52, 52), 3, dtype=np.uint8))
        self.assertTrue(torch.equal(zero, ignored))
        self.assertTrue(torch.isfinite(zero).all())
        self.assertTrue(torch.equal(zero, torch.ones_like(zero)))
        raw = wafer()
        raw[0, 0] = 3
        clean = raw.copy(); clean[0, 0] = 0
        self.assertTrue(torch.equal(yu.preprocess(raw), yu.preprocess(clean)))
        with self.assertRaises(ValueError):
            yu.preprocess(np.full((3, 3), 9))

    def test_augmentation_is_repeatable_without_global_rng(self):
        raw = wafer(); raw[15:32, 12:29] = 2
        first = yu.preprocess(raw, training=True, rng=np.random.default_rng(11))
        second = yu.preprocess(raw, training=True, rng=np.random.default_rng(11))
        self.assertTrue(torch.equal(first, second))

    def test_audit_touches_only_allowed_ids_and_reports_erased_scratch(self):
        raw = wafer(); raw[26, 10:42] = 2

        class Guard:
            def __getitem__(self, index):
                if index not in (1, 2):
                    raise AssertionError("Forbidden held-out access")
                return raw

        bundle = SimpleNamespace(maps=Guard(), class_ids=np.array([-1, 0, 0]),
                                 train_ids=np.array([1]), valid_ids=np.array([2]),
                                 normal_train_ids=np.array([], dtype=int), normal_valid_ids=np.array([], dtype=int))
        with tempfile.TemporaryDirectory() as directory:
            result = yu.preprocess_audit(bundle, directory, "fixed")
            self.assertEqual(result["groups"]["defect_train"]["nonempty_to_zero_defect_maps"], 1)
            self.assertFalse(result["test_processed"])
            self.assertTrue((Path(directory) / "preprocessing/per_sample.csv").exists())


class YuEncoderTests(unittest.TestCase):
    def test_categorical_hamming_is_mismatch_count_not_bitwise_xor(self):
        q = np.zeros(256, dtype=np.uint8)
        candidates = np.zeros((3, 256), dtype=np.uint8)
        candidates[0, 0], candidates[1, 0] = 1, 9
        candidates[2, :2] = [9, 9]
        np.testing.assert_array_equal(yu.categorical_hamming(q, candidates), [1, 1, 2])
        self.assertEqual(yu.categorical_hamming(q, candidates[:0]).shape, (0,))

    def test_pca256_roundtrip_fit_ids_and_rejection_of_small_training_set(self):
        rng = np.random.default_rng(8)
        features = rng.normal(size=(270, 1024))
        fit_ids = np.arange(1000, 1270)
        encoder = yu.TenValueEncoder(seed=17).fit(features, fit_ids)
        query = rng.normal(size=(4, 1024)) * 30
        encoded = encoder.transform(query)
        self.assertEqual(encoded.shape, (4, 256))
        self.assertTrue(((encoded >= 0) & (encoded <= 9)).all())
        np.testing.assert_array_equal(encoder.fit_ids, fit_ids)
        self.assertFalse(encoder.pca.whiten)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encoder.npz"
            encoder.save(path, "train-signature")
            loaded = yu.TenValueEncoder.load(path, "train-signature")
            np.testing.assert_array_equal(loaded.fit_ids, fit_ids)
            np.testing.assert_array_equal(loaded.transform(query), encoded)
            with self.assertRaises(ValueError):
                yu.TenValueEncoder.load(path, "other-fit")
        with self.assertRaises(ValueError):
            yu.TenValueEncoder().fit(features[:256], fit_ids[:256])
        with self.assertRaises(ValueError):
            yu.TenValueEncoder().fit(features, np.zeros(270, dtype=int))

    def test_minmax_is_train_fixed_clips_and_handles_constant_dimension(self):
        encoder = yu.TenValueEncoder()
        encoder.minimum = np.zeros(256)
        encoder.maximum = np.ones(256)
        encoder.maximum[0] = 0
        projected = np.full((1, 256), 0.55)
        projected[0, :4] = [99, -50, 90, 0.1]
        with patch.object(encoder.pca, "transform", return_value=projected):
            result = encoder.transform(np.zeros((1, 1024)))
        np.testing.assert_array_equal(result[0, :5], [0, 0, 9, 1, 5])


class TinyDetector(nn.Module):
    """Only for isolating resume behavior; production/smoke always use the complete CNN."""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        return self.layer(self.dropout(x.mean(dim=(2, 3))))


class YuResumeTests(unittest.TestCase):
    def test_mid_epoch_resume_matches_uninterrupted_optimizer_and_parameters(self):
        maps = np.stack([wafer() for _ in range(12)])
        maps[:6, 15:35, 15:35] = 2
        maps[8:10, 15:35, 15:35] = 2
        bundle = SimpleNamespace(maps=maps, class_ids=np.array([0]*6 + [-1]*2 + [0]*2 + [-1]*2),
                                 train_ids=np.arange(6), normal_train_ids=np.array([6, 7]),
                                 valid_ids=np.array([8, 9]), normal_valid_ids=np.array([10, 11]))
        config = {"seed": 2026, "device": "cpu", "num_workers": 0, "inference_batch_size": 2,
                  "augmentation": {"crop_scale": [0.95, 1.0], "rotation": True},
                  "training": {"detector": {"batch_size": 2, "epochs": 2},
                               "learning_rate": 1e-4, "classifier_l2": 1e-6, "checkpoint_every_steps": 2}}
        with tempfile.TemporaryDirectory() as directory, patch.object(yu, "_model", lambda _: TinyDetector()):
            continuous, resumed = Path(directory) / "continuous", Path(directory) / "resumed"
            yu.train_network("detector", bundle, config, continuous, "fixed", torch.device("cpu"), max_steps=6)
            first = yu.train_network("detector", bundle, config, resumed, "fixed", torch.device("cpu"), max_steps=3)
            self.assertTrue(first["truncated_epoch"])
            self.assertEqual(first["completed_epochs"], 0)
            yu.train_network("detector", bundle, config, resumed, "fixed", torch.device("cpu"),
                             resume=True, max_steps=6)
            a = yu._load_checkpoint(continuous / "detector/last.pt")
            b = yu._load_checkpoint(resumed / "detector/last.pt")
            self.assertEqual(a["state"]["steps"], b["state"]["steps"])
            self.assertEqual(a["state"]["next_batch"], b["state"]["next_batch"])
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
            for key in a["optimizer"]["state"]:
                for field in a["optimizer"]["state"][key]:
                    self.assertTrue(torch.equal(a["optimizer"]["state"][key][field],
                                                b["optimizer"]["state"][key][field]))
            self.assertTrue(torch.equal(a["rng"]["torch"], b["rng"]["torch"]))
            with self.assertRaises(ValueError):
                yu.train_network("detector", bundle, config, resumed, "wrong-signature", torch.device("cpu"),
                                 resume=True, max_steps=8)


if __name__ == "__main__":
    unittest.main()
