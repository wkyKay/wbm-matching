"""Unified evaluation isolation and checkpoint-provenance regression checks.

All datasets in this file are tiny synthetic fixtures; no real test evaluation
or CNN training is performed.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared import common, evaluation
from shared.evaluation_adapters import RankingResult
from shared.retrieval_protocol import build_retrieval_protocol


def snapshot(directory):
    """Include mtimes so rewriting identical source bytes is still detected."""
    return {str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in Path(directory).rglob("*") if path.is_file()}


class EvaluationFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        labels = np.zeros((17, 8), dtype=np.uint8)
        labels[:6, 0] = 1
        labels[6:12, 1] = 1
        maps = np.ones((17, 7, 7), dtype=np.uint8)
        maps[:12, 2:5, 2:5] = 2
        for sample_id in range(len(maps)):
            maps[sample_id, 0, :5] = [(sample_id >> bit) & 1 for bit in range(5)]
        source = self.root / "source.npz"
        np.savez(source, arr_0=maps, arr_1=labels)
        rows = []
        for start in (0, 6):
            for offset in range(6):
                sample_id = start + offset
                row = {"sample_id": sample_id, "original_index": sample_id,
                       "valid_index": sample_id,
                       "split": ("train", "valid", "test")[offset // 2]}
                row.update({f"label_{i}": int(labels[sample_id, i]) for i in range(8)})
                rows.append(row)
        split = self.root / "split.csv"
        common.save_csv(split, rows, list(rows[0]))
        protocol = self.root / "retrieval"
        build_retrieval_protocol(source, split, protocol, valid_queries=4, min_per_class=1)
        self.source_run = self.root / "source_run"
        self.config = {
            "method": "wu", "seed": 2026, "data_file": str(source),
            "split_file": str(split), "retrieval_protocol_dir": str(protocol),
            "output_dir": str(self.source_run), "expected_classes": 2,
            "normal_split": {"train": 2, "valid": 1, "holdout": 2},
            "subset": {}, "ks": [1, 2],
        }

    def tearDown(self):
        self.temporary.cleanup()

    def source_config(self, method="yu", **overrides):
        config = common.load_config(ROOT / "methods" / method / "config.json", require_ks=False)
        config.update(self.config)
        config["method"] = method
        config.update(overrides)
        common.ensure_run(config, [ROOT / "methods" / method / f"{method}.py"])
        return config


class RunBoundaryTests(EvaluationFixture):
    def test_parallel_full_evaluation_preserves_rankings_and_metrics(self):
        config = {**self.config, "method": "wbbs"}
        serial, parallel = self.root / "serial", self.root / "parallel"
        a = evaluation.run_evaluation(config, serial, ks=[1, 2], rank_workers=1)
        b = evaluation.run_evaluation(config, parallel, ks=[1, 2], rank_workers=2)
        for name in ("rankings.csv", "per_query.csv", "per_class.csv"):
            self.assertEqual((serial / "valid" / name).read_bytes(),
                             (parallel / "valid" / name).read_bytes())
        for key in ("metrics", "class_macro_metrics", "groups"):
            self.assertEqual(a[key], b[key])
        self.assertEqual(common.read_json(parallel / "evaluation_manifest.json")["execution"]["rank_workers"], 2)

    def test_invalid_rank_workers_rejected_before_writing(self):
        for value in (0, -1, True, 1.5):
            output = self.root / f"invalid_workers_{value}"
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "rank_workers"):
                evaluation.run_evaluation(self.config, output, ks=[1], rank_workers=value)
            self.assertFalse(output.exists())

    def fake_preparer(self, calls):
        def prepare(bundle, config, pool, destination, **kwargs):
            calls["gallery"] = list(map(int, pool.gallery))
            calls["queries"] = list(map(int, pool.queries))
            calls["split"] = pool.split_name
            calls["rank_calls"] = []

            def rank(query_id, candidate_ids):
                candidates = np.asarray(candidate_ids, dtype=np.int64)
                calls["rank_calls"].append((int(query_id), candidates.tolist()))
                return RankingResult(candidates, -np.arange(len(candidates), dtype=float),
                                     scored_count=len(candidates), eligible_count=len(candidates))

            return SimpleNamespace(metadata={"synthetic_fixture": True}, rank=rank)
        return prepare

    def test_valid_evaluation_preserves_source_and_uses_frozen_gallery(self):
        self.source_run.mkdir()
        (self.source_run / "keep.txt").write_text("source checkpoint placeholder", encoding="utf-8")
        config = {**self.config, "cache_dir": str(self.source_run / "old_cache")}
        original_config = copy.deepcopy(config)
        source_before = snapshot(self.source_run)
        protocol_before = snapshot(Path(config["retrieval_protocol_dir"]))
        output, calls = self.root / "evaluation", {}
        with mock.patch.object(evaluation, "prepare_method", side_effect=self.fake_preparer(calls)):
            evaluation.run_evaluation(config, output, ks=[1, 2])
        self.assertEqual(config, original_config)
        self.assertEqual(snapshot(self.source_run), source_before)
        self.assertEqual(snapshot(Path(config["retrieval_protocol_dir"])), protocol_before)
        self.assertEqual(calls["gallery"], [2, 3, 8, 9])
        self.assertEqual(calls["queries"], [2, 3, 8, 9])
        self.assertEqual(calls["split"], "valid")
        for query_id, candidates in calls["rank_calls"]:
            self.assertEqual(set(candidates), {2, 3, 8, 9} - {query_id})
        self.assertTrue((output / "raw_cache").is_dir())
        self.assertTrue((output / "protocol" / "valid_pool.json").is_file())

    def test_explicit_test_passes_only_synthetic_test_ids(self):
        output, calls = self.root / "synthetic_test_evaluation", {}
        with mock.patch.object(evaluation, "prepare_method", side_effect=self.fake_preparer(calls)):
            evaluation.run_evaluation(self.config, output, split_name="test", ks=[1, 2])
        self.assertEqual(calls["gallery"], [4, 5, 10, 11])
        self.assertEqual(calls["queries"], [4, 5, 10, 11])
        self.assertEqual(calls["split"], "test")
        for query_id, candidates in calls["rank_calls"]:
            self.assertEqual(set(candidates), {4, 5, 10, 11} - {query_id})
        self.assertFalse({0, 1, 2, 3, 6, 7, 8, 9} & set(calls["gallery"]))

    def test_existing_output_is_never_reused_even_if_empty(self):
        for populated in (False, True):
            output = self.root / f"existing_{populated}"
            output.mkdir()
            if populated:
                (output / "keep.txt").write_text("original output", encoding="utf-8")
            before = snapshot(output)
            with self.subTest(populated=populated), mock.patch.object(evaluation, "prepare_method") as prepare:
                with self.assertRaises(FileExistsError):
                    evaluation.run_evaluation(self.config, output, ks=[1, 2])
                prepare.assert_not_called()
                self.assertEqual(snapshot(output), before)

    def test_output_cannot_be_nested_in_source_run_or_protocol(self):
        for parent in (self.source_run, Path(self.config["retrieval_protocol_dir"])):
            with self.subTest(parent=parent), mock.patch.object(evaluation, "prepare_method") as prepare:
                with self.assertRaises(ValueError):
                    evaluation.run_evaluation(self.config, parent / "new_evaluation", ks=[1, 2])
                prepare.assert_not_called()
                self.assertFalse((parent / "new_evaluation").exists())

    def test_failed_ranking_has_no_completion_marker_and_cannot_be_reused(self):
        output = self.root / "failed_evaluation"
        ranker = SimpleNamespace(metadata={}, rank=mock.Mock(side_effect=RuntimeError("synthetic ranking failure")))
        with mock.patch.object(evaluation, "prepare_method", return_value=ranker):
            with self.assertRaisesRegex(RuntimeError, "synthetic ranking failure"):
                evaluation.run_evaluation(self.config, output, ks=[1, 2])
        self.assertTrue((output / "failed.json").exists())
        self.assertFalse((output / "completed.json").exists())
        with self.assertRaises(FileExistsError):
            evaluation.run_evaluation(self.config, output, ks=[1, 2])

    def test_source_change_during_evaluation_prevents_completion(self):
        output, calls = self.root / "changed_during_evaluation", {}
        original_prepare = self.fake_preparer(calls)

        def changing_prepare(*args, **kwargs):
            ranker = original_prepare(*args, **kwargs)
            path = Path(self.config["data_file"])
            path.write_bytes(path.read_bytes() + b"synthetic concurrent change")
            return ranker

        with mock.patch.object(evaluation, "prepare_method", side_effect=changing_prepare):
            with self.assertRaisesRegex(ValueError, "changed during evaluation"):
                evaluation.run_evaluation(self.config, output, ks=[1, 2])
        self.assertTrue((output / "failed.json").exists())
        self.assertFalse((output / "completed.json").exists())

    def test_smoke_and_subset_cannot_run_test(self):
        for changes in ({"smoke": True}, {"subset": {"train_per_class": 1}}):
            config = {**self.config, **changes}
            with self.subTest(changes=changes), mock.patch.object(evaluation, "prepare_method") as prepare:
                with self.assertRaises(ValueError):
                    evaluation.run_evaluation(config, self.root / f"forbidden_{len(changes)}_{'smoke' in changes}",
                                              split_name="test", ks=[1, 2])
                prepare.assert_not_called()


class SourceProvenanceTests(EvaluationFixture):
    def test_matching_source_can_be_verified_without_writing(self):
        config = self.source_config()
        before = snapshot(self.source_run)
        record = evaluation.verify_source_run(config)
        manifest = common.read_json(self.source_run / "run_manifest.json")
        self.assertEqual(record["base_signature"], manifest["signature"])
        self.assertEqual(record["manifest_sha256"], common.hash_file(self.source_run / "run_manifest.json"))
        self.assertEqual(snapshot(self.source_run), before)

    def test_changed_scientific_config_is_rejected(self):
        config = self.source_config()
        changed = copy.deepcopy(config)
        changed["training"]["learning_rate"] *= 2
        before = snapshot(self.source_run)
        with self.assertRaises(ValueError):
            evaluation.verify_source_run(changed)
        self.assertEqual(snapshot(self.source_run), before)

    def test_tampered_manifest_is_rejected_even_if_signature_is_self_consistent(self):
        config = self.source_config()
        path = self.source_run / "run_manifest.json"
        record = common.read_json(path)
        source_key = next(key for key in record["code"] if key.endswith("yu.py"))
        record["code"][source_key] = "0" * 64
        record["signature"] = common.signature({key: value for key, value in record.items() if key != "signature"})
        common.save_json(path, record)
        with self.assertRaises(ValueError):
            evaluation.verify_source_run(config)

    def test_missing_source_manifest_is_rejected(self):
        config = {**self.config, "method": "yu"}
        with self.assertRaises((FileNotFoundError, ValueError)):
            evaluation.verify_source_run(config)

    def test_changed_data_or_frozen_split_is_rejected(self):
        config = self.source_config()
        for key in ("data_file", "split_file"):
            path = Path(config[key])
            previous = path.read_bytes()
            with self.subTest(source=key):
                path.write_bytes(previous + b"\n")
                with self.assertRaises(ValueError):
                    evaluation.verify_source_run(config)
                path.write_bytes(previous)


class TrainingCompletionTests(EvaluationFixture):
    def yu_fixture(self, *, complete=True, selected_partial=False, smoke=False):
        from methods.yu import yu
        training = common.load_config(ROOT / "methods" / "yu" / "config.json")["training"]
        training["classifier"]["batch_size"] = 2
        config = self.source_config(smoke=smoke, training=training)
        bundle = common.prepare_data(config)
        record = evaluation.verify_source_run(config)
        checkpoint_signature = common.signature({"run": record["base_signature"],
                                                  "data": bundle.signature, "method": yu.METHOD_VERSION})
        epochs = config["training"]["classifier"]["epochs"]
        batches = math.ceil(len(bundle.train_ids) / config["training"]["classifier"]["batch_size"])
        state = {"epoch": epochs if complete else epochs - 1, "next_batch": 0,
                 "truncated_epoch": False, "steps": (epochs if complete else epochs - 1) * batches}
        last = {"signature": checkpoint_signature, "task": "classifier",
                "method_version": yu.METHOD_VERSION, "state": state}
        best = {**last, "state": {"epoch": 2, "next_batch": int(selected_partial),
                                  "steps": 2 * batches + int(selected_partial),
                                  "truncated_epoch": selected_partial}}
        folder = self.source_run / "classifier"
        folder.mkdir()
        (folder / "last.pt").write_bytes(b"synthetic last checkpoint, load is mocked")
        (folder / "best.pt").write_bytes(b"synthetic earlier selected checkpoint, load is mocked")
        common.save_json(folder / "summary.json", {
            "signature": checkpoint_signature, "task": "classifier", "configured_epochs": epochs,
            "completed_epochs": state["epoch"], "next_batch": state["next_batch"],
            "truncated_epoch": state["truncated_epoch"], "steps": state["steps"],
            "training_complete": complete,
        })
        return config, bundle, record, checkpoint_signature, last, best

    def test_yu_complete_run_accepts_earlier_selected_checkpoint(self):
        from methods.yu import yu
        config, bundle, record, expected, last, best = self.yu_fixture(selected_partial=True)
        before = snapshot(self.source_run)
        with mock.patch.object(yu, "_load_checkpoint", side_effect=[last, best]):
            signature, provenance = evaluation.validate_training_source(bundle, config, record)
        self.assertEqual(signature, expected)
        self.assertTrue(provenance["tasks"]["classifier"]["training_complete"])
        self.assertEqual(provenance["tasks"]["classifier"]["selected_completed_epochs"], 2)
        self.assertTrue(provenance["tasks"]["classifier"]["selected_truncated_epoch"])
        self.assertFalse(provenance["detector_used"])
        self.assertEqual(snapshot(self.source_run), before)

    def test_evaluation_cutoffs_do_not_change_cnn_training_signature(self):
        from methods.yu import yu
        config, _, _, _, last, best = self.yu_fixture()
        before = snapshot(self.source_run)
        calls = {}
        prepare = RunBoundaryTests.fake_preparer(self, calls)
        with mock.patch.object(yu, "_load_checkpoint", side_effect=[last, best]), \
                mock.patch.object(common, "seed_everything", wraps=common.seed_everything) as seed, \
                mock.patch.object(evaluation, "prepare_method", side_effect=prepare):
            result = evaluation.run_evaluation(config, self.root / "cnn_evaluation", ks=[1], device="cpu")
        seed.assert_called_once_with(config["seed"])
        inference = common.read_json(self.root / "cnn_evaluation/valid/inference_environment.json")
        self.assertFalse(inference["cudnn_benchmark"])
        self.assertTrue(inference["cudnn_deterministic"])
        self.assertEqual(result["ks"], [1])
        self.assertEqual(config["ks"], [1, 2])
        self.assertEqual(snapshot(self.source_run), before)

    def test_yu_formal_incomplete_training_is_rejected_before_best_load(self):
        from methods.yu import yu
        config, bundle, record, _, last, best = self.yu_fixture(complete=False)
        with mock.patch.object(yu, "_load_checkpoint", side_effect=[last, best]) as loader:
            with self.assertRaisesRegex(ValueError, "completed training"):
                evaluation.validate_training_source(bundle, config, record)
            self.assertEqual(loader.call_count, 1)

    def test_yu_smoke_remains_explicitly_incomplete(self):
        from methods.yu import yu
        config, bundle, record, _, last, best = self.yu_fixture(complete=False, smoke=True)
        with mock.patch.object(yu, "_load_checkpoint", side_effect=[last, best]):
            _, provenance = evaluation.validate_training_source(bundle, config, record)
        self.assertTrue(provenance["smoke_training"])
        self.assertFalse(provenance["tasks"]["classifier"]["training_complete"])

    def test_yu_checkpoint_summary_progress_mismatch_is_rejected(self):
        from methods.yu import yu
        config, bundle, record, _, last, best = self.yu_fixture()
        last["state"]["steps"] += 1
        with mock.patch.object(yu, "_load_checkpoint", return_value=last):
            with self.assertRaisesRegex(ValueError, "progress mismatch"):
                evaluation.validate_training_source(bundle, config, record)

    def nakazawa_fixture(self, *, epoch=None):
        import torch
        config = self.source_config("nakazawa")
        bundle = common.prepare_data(config)
        record = evaluation.verify_source_run(config)
        expected_epoch = config["training"]["epochs"]
        checkpoint_path = self.source_run / "checkpoint_last.pt"
        torch.save({"epoch": expected_epoch if epoch is None else epoch,
                    "run_signature": record["base_signature"], "data_signature": bundle.signature,
                    "class_signatures": bundle.class_signatures}, checkpoint_path)
        common.save_json(self.source_run / "training_complete.json", {
            "checkpoint_sha256": common.hash_file(checkpoint_path), "epochs": expected_epoch,
            "selection": "last_epoch", "validation_used_to_select": False, "smoke": False,
            "final_epoch": {"epoch": expected_epoch, "train_samples": len(bundle.train_ids),
                            "valid_samples": len(bundle.valid_ids)},
        })
        return config, bundle, record, checkpoint_path

    def test_nakazawa_final_checkpoint_is_verified_without_model_or_writes(self):
        config, bundle, record, _ = self.nakazawa_fixture()
        before = snapshot(self.source_run)
        signature, provenance = evaluation.validate_training_source(bundle, config, record)
        self.assertEqual(signature, record["base_signature"])
        self.assertTrue(provenance["training_complete"])
        self.assertEqual(provenance["selection"], "last_epoch")
        self.assertEqual(snapshot(self.source_run), before)

    def test_nakazawa_nonfinal_checkpoint_is_rejected(self):
        config, bundle, record, _ = self.nakazawa_fixture(epoch=1)
        with self.assertRaisesRegex(ValueError, "configured final epoch"):
            evaluation.validate_training_source(bundle, config, record)

    def test_nakazawa_changed_checkpoint_is_rejected(self):
        config, bundle, record, path = self.nakazawa_fixture()
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "completion record/hash/selection"):
            evaluation.validate_training_source(bundle, config, record)


if __name__ == "__main__":
    unittest.main()
