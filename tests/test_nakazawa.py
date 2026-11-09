"""Paper architecture, binary retrieval and training-isolation regressions."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.nakazawa.nakazawa import (
    CODE_BITS, NakazawaCNN, WaferDataset, binary_codes, classification_epoch,
    encode_maps, hamming_distances, parameters, preprocess_wafer,
    rank_candidates, score_candidates, train, training_parameters,
)
from shared.common import load_config


class RepresentationTests(unittest.TestCase):
    def test_only_failed_die_occupancy_without_invented_density(self):
        raw = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        result = preprocess_wafer(raw)
        self.assertEqual(tuple(result.shape), (1, 286, 400))
        self.assertEqual(set(result.unique().tolist()), {0.0, 1.0})
        self.assertEqual(result[0, 200, 50].item(), 1.0)
        self.assertEqual(result[0, 200, 350].item(), 0.0)
        self.assertFalse(result[:, :143].any())

    def test_background_and_invalid_code(self):
        self.assertFalse(preprocess_wafer(np.zeros((52, 52), dtype=np.uint8)).any())
        with self.assertRaises(ValueError):
            preprocess_wafer(np.array([[4]]))

    def test_encoding_equality_and_validation(self):
        values = np.full(256, 0.5)
        values[:2] = [0.4999, 0.5001]
        code = binary_codes(values)
        np.testing.assert_array_equal(code[:3], [0, 1, 1])
        self.assertEqual(code.dtype, np.uint8)
        with self.assertRaises(ValueError):
            binary_codes(np.full(256, np.nan))
        with self.assertRaises(ValueError):
            binary_codes(np.zeros(255))

    def test_hamming_max_distance_does_not_overflow(self):
        query = np.zeros(256, dtype=np.uint8)
        candidates = np.stack([query, np.ones_like(query)])
        np.testing.assert_array_equal(hamming_distances(query, candidates), [0, 256])
        np.testing.assert_array_equal(score_candidates(query, candidates), [0, -256])
        with self.assertRaises(ValueError):
            hamming_distances(query, np.full((2, 256), 0.2))

    def test_rank_ties_original_id_and_self_exclusion(self):
        query = np.zeros(256, dtype=np.uint8)
        candidates = np.zeros((4, 256), dtype=np.uint8)
        candidates[0, 0] = 1
        ids, distances = rank_candidates(query, candidates, np.array([9, 7, 5, 2]), query_id=5)
        np.testing.assert_array_equal(ids, [2, 7, 9])
        np.testing.assert_array_equal(distances, [0, 0, 1])
        self.assertEqual(len(hamming_distances(query, np.empty((0, 256), dtype=np.uint8))), 0)

    def test_dataset_uses_original_ids(self):
        maps = np.zeros((6, 2, 2), dtype=np.uint8)
        maps[4, 1, 0] = 2
        labels = np.array([-1, 9, 8, 7, 3, 6])
        dataset = WaferDataset(maps, np.array([4, 2]), labels)
        self.assertEqual(dataset[0][1], 3)
        self.assertTrue(dataset[0][0].any())
        self.assertFalse(dataset[1][0].any())
        with self.assertRaises(ValueError):
            WaferDataset(maps, [0], labels)[0]

    def test_configs_keep_source_protocol_and_no_metric_design(self):
        for filename in ("config.json", "smoke.json"):
            config = load_config(ROOT / "methods/nakazawa" / filename, require_ks=False)
            self.assertNotIn("ks", config)
            self.assertTrue(config["split_file"].endswith("protocol/deduplicated/split.csv"))
            self.assertTrue(config["retrieval_protocol_dir"].endswith("protocol/retrieval_v1"))
            self.assertEqual(parameters(config["parameters"])["input_shape"], [286, 400])
        with self.assertRaisesRegex(ValueError, "debug-only"):
            training_parameters({"max_train_batches": 1}, smoke=False)
        self.assertEqual(training_parameters()["epochs"], 10)
        self.assertEqual(training_parameters()["selection"], "last_epoch")
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            parameters({"drop_out": 0.2})
        with self.assertRaisesRegex(ValueError, "Unknown training"):
            training_parameters({"lr": 0.1})


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_microbatches_match_effective_batch_gradient(self):
        torch.manual_seed(4)
        model = nn.Linear(3, 2)
        reference = nn.Linear(3, 2)
        reference.load_state_dict(model.state_dict())
        inputs, labels = torch.randn(5, 3), torch.tensor([0, 1, 1, 0, 1])
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=5, shuffle=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        direct_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        record = classification_epoch(model, loader, torch.device("cpu"), optimizer=optimizer, micro_batch_size=2)
        direct_optimizer.zero_grad()
        nn.functional.cross_entropy(reference(inputs), labels).backward()
        direct_optimizer.step()
        self.assertEqual(record["optimizer_updates"], 1)
        self.assertEqual(record["samples"], 5)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)

    def test_validation_does_not_update_parameters(self):
        model = nn.Linear(3, 2)
        before = [value.detach().clone() for value in model.parameters()]
        loader = DataLoader(TensorDataset(torch.randn(3, 3), torch.tensor([1, 0, 1])), batch_size=2)
        result = classification_epoch(model, loader, torch.device("cpu"), micro_batch_size=1)
        self.assertEqual(result["optimizer_updates"], 0)
        for actual, expected in zip(model.parameters(), before):
            torch.testing.assert_close(actual, expected)

    def test_interrupted_epoch_resume_and_completed_run_preservation(self):
        class TinyCNN(nn.Module):
            def __init__(self, num_classes, settings):
                super().__init__()
                self.layers = nn.Sequential(nn.Linear(3, 5), nn.Dropout(0.3), nn.Linear(5, num_classes))

            def forward(self, inputs):
                return self.layers(inputs)

        inputs = torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
        loader = DataLoader(TensorDataset(inputs, torch.tensor([0, 1])), batch_size=2)
        bundle = SimpleNamespace(class_signatures=[[0, 1], [1, 0]], signature="fixture-data",
                                 train_ids=np.array([0, 1]), valid_ids=np.array([2, 3]))
        prefix = "methods.nakazawa.nakazawa."
        with tempfile.TemporaryDirectory() as temporary, \
                patch(prefix + "NakazawaCNN", TinyCNN), \
                patch(prefix + "_loader", return_value=loader), \
                patch(prefix + "_run_signature", return_value="fixture-run"):
            config = {"output_dir": temporary, "seed": 2026, "device": "cpu",
                      "training": {"epochs": 2, "batch_size": 2, "micro_batch_size": 1}}
            invocations = 0

            def interrupt_second_epoch(*args, **kwargs):
                nonlocal invocations
                invocations += 1
                if invocations == 3:
                    raise RuntimeError("simulated interruption")
                return classification_epoch(*args, **kwargs)

            with patch(prefix + "classification_epoch", side_effect=interrupt_second_epoch):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    train(bundle, config)
            self.assertFalse((Path(temporary) / "training_complete.json").exists())
            checkpoint = Path(temporary) / "checkpoint_last.pt"
            self.assertEqual(torch.load(checkpoint, weights_only=True)["epoch"], 1)
            with self.assertRaises(FileExistsError):
                train(bundle, config)
            result = train(bundle, config, resume=True)
            self.assertEqual(result["epochs"], 2)
            resumed = torch.load(checkpoint, weights_only=True)
            self.assertEqual([item["epoch"] for item in resumed["history"]], [1, 2])
            before = checkpoint.stat().st_mtime_ns
            train(bundle, config)
            self.assertEqual(checkpoint.stat().st_mtime_ns, before)
            # Compare resumed dropout + Adam state with uninterrupted training.
            with tempfile.TemporaryDirectory() as other:
                train(bundle, {**config, "output_dir": other})
                reference = torch.load(Path(other) / "checkpoint_last.pt", weights_only=True)
            for name in resumed["model"]:
                torch.testing.assert_close(resumed["model"][name], reference["model"][name], rtol=0, atol=0)

    def test_table_i_architecture_sigmoid_and_public_encoding(self):
        model = NakazawaCNN(37).eval()
        convolutions = [layer for layer in model.convolutions if isinstance(layer, nn.Conv2d)]
        pools = [layer for layer in model.convolutions if isinstance(layer, nn.MaxPool2d)]
        self.assertEqual([layer.out_channels for layer in convolutions], [32, 32, 64])
        self.assertEqual(len(pools), 3)
        self.assertEqual(model.feature_shape, (64, 34, 48))
        self.assertEqual((model.fc.in_features, model.fc.out_features), (104448, 256))
        self.assertEqual(model.classifier.out_features, 37)
        maps = np.zeros((2, 52, 52), dtype=np.uint8)
        maps[1, 10:20, 15:30] = 2
        features, codes = encode_maps(model, maps, np.array([1, 0]), batch_size=1)
        self.assertEqual(features.shape, (2, CODE_BITS))
        np.testing.assert_array_equal(codes, (features >= 0.5).astype(np.uint8))
        self.assertFalse(model.training)
        with torch.inference_mode():
            probabilities = model.predict_proba(preprocess_wafer(maps[1])[None])
        torch.testing.assert_close(probabilities.sum(1), torch.ones(1))
        with self.assertRaisesRegex(ValueError, "286"):
            model(torch.zeros((1, 1, 52, 52)))


if __name__ == "__main__":
    unittest.main()
