"""Exact-duplicate filtering and safe generation of a new frozen protocol."""
from __future__ import annotations

import csv
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.data_protocol import deduplicate_maps, stratified_split
from scripts.build_protocol import build_protocol


def unique_maps(count):
    maps = np.ones((count, 7, 7), dtype=np.uint8)
    for sample_id in range(count):
        maps[sample_id, 0, :4] = [(sample_id >> (2 * digit)) & 3 for digit in range(4)]
    return maps


class ExactDuplicateTests(unittest.TestCase):
    def test_lowest_original_id_is_kept_globally_including_normal(self):
        maps = unique_maps(6)
        labels = np.zeros((6, 8), dtype=np.uint8)
        labels[:4, 0] = 1
        maps = np.concatenate([maps, maps[[0, 4, 0]]])
        labels = np.concatenate([labels, labels[[0, 4, 0]]])
        before_maps, before_labels = maps.copy(), labels.copy()
        result = deduplicate_maps(maps, labels)
        np.testing.assert_array_equal(result.representative_ids, np.arange(6))
        np.testing.assert_array_equal(result.representative_of, [0, 1, 2, 3, 4, 5, 0, 4, 0])
        self.assertEqual([list(group) for group in result.duplicate_groups], [[0, 6, 8], [4, 7]])
        np.testing.assert_array_equal(maps, before_maps)
        np.testing.assert_array_equal(labels, before_labels)
        repeated = deduplicate_maps(maps.astype(np.int32), labels.astype(np.int32))
        np.testing.assert_array_equal(result.representative_of, repeated.representative_of)

    def test_all_raw_pixel_codes_distinguish_images(self):
        maps = np.ones((4, 7, 7), dtype=np.uint8)
        maps[:, 0, 0] = [0, 1, 2, 3]
        labels = np.zeros((4, 8), dtype=np.uint8)
        labels[:, 0] = 1
        result = deduplicate_maps(maps, labels)
        np.testing.assert_array_equal(result.representative_ids, np.arange(4))
        self.assertEqual(len(result.duplicate_groups), 0)

    def test_same_image_with_different_labels_is_an_error(self):
        maps = np.repeat(unique_maps(1), 2, axis=0)
        for second_label in (np.eye(8, dtype=np.uint8)[1], np.zeros(8, dtype=np.uint8)):
            labels = np.stack([np.eye(8, dtype=np.uint8)[0], second_label])
            with self.subTest(second_label=second_label.tolist()):
                with self.assertRaises(ValueError):
                    deduplicate_maps(maps, labels)

    def test_invalid_source_values_are_rejected_before_casting(self):
        maps = unique_maps(3).astype(np.int32)
        labels = np.zeros((3, 8), dtype=np.uint8)
        labels[:, 0] = 1
        maps[0, 0, 0] = 256  # Must not silently wrap to uint8 zero.
        with self.assertRaises(ValueError):
            deduplicate_maps(maps, labels)
        labels[0, 0] = 2
        with self.assertRaises(ValueError):
            deduplicate_maps(unique_maps(3), labels)


class StratifiedProtocolTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_covers_only_unique_defects(self):
        maps = unique_maps(33)
        labels = np.zeros((33, 8), dtype=np.uint8)
        labels[:20, 0] = 1
        labels[20:28, 1] = 1
        maps = np.concatenate([maps, maps[[0, 20, 28]]])
        labels = np.concatenate([labels, labels[[0, 20, 28]]])
        unique = deduplicate_maps(maps, labels)
        source_before = labels.copy()
        split = stratified_split(labels, unique.representative_ids, seed=2026)
        repeated = stratified_split(labels, unique.representative_ids, seed=2026)
        for name in ("train", "valid", "test"):
            np.testing.assert_array_equal(split[name], repeated[name])
            self.assertTrue(any(labels[i, 0] for i in split[name]))
            self.assertTrue(any(labels[i, 1] for i in split[name]))
        all_ids = [int(i) for name in ("train", "valid", "test") for i in split[name]]
        self.assertEqual(sorted(all_ids), list(range(28)))
        self.assertEqual(len({maps[i].tobytes() for i in all_ids}), len(all_ids))
        self.assertEqual(len(set(split["train"]) & set(split["valid"])), 0)
        self.assertEqual(len(set(split["train"]) & set(split["test"])), 0)
        self.assertEqual(len(set(split["valid"]) & set(split["test"])), 0)
        np.testing.assert_array_equal(labels, source_before)

    def test_too_small_class_and_invalid_ratios_fail_explicitly(self):
        labels = np.zeros((4, 8), dtype=np.uint8)
        labels[:2, 0] = 1
        with self.assertRaises(ValueError):
            stratified_split(labels, np.arange(4))
        labels[:, 0] = 1
        for ratios in ((0.8, 0.2, 0.2), (-0.1, 0.5, 0.6), (float("nan"), 0.1, 0.2)):
            with self.subTest(ratios=ratios):
                with self.assertRaises(ValueError):
                    stratified_split(labels, np.arange(4), ratios=ratios)

    def test_builder_writes_new_protocol_and_preserves_source_and_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            maps = unique_maps(45)
            labels = np.zeros((45, 8), dtype=np.uint8)
            labels[:20, 0] = 1
            labels[20:40, 1] = 1
            maps = np.concatenate([maps, maps[[0, 20, 40]]])
            labels = np.concatenate([labels, labels[[0, 20, 40]]])
            source = root / "source.npz"
            np.savez(source, arr_0=maps, arr_1=labels)
            source_before = hashlib.sha256(source.read_bytes()).digest()
            old_split = root / "split.csv"
            old_split.write_text("legacy split must survive\n")
            old_before = old_split.read_bytes()
            output = root / "deduplicated"
            build_protocol(source, output, seed=2026)
            with (output / "split.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            ids = [int(row["sample_id"]) for row in rows]
            self.assertEqual(sorted(ids), list(range(40)))
            self.assertEqual(len(set(ids)), 40)
            nonempty = np.flatnonzero(labels.sum(axis=1))
            for row in rows:
                sample_id = int(row["sample_id"])
                self.assertEqual(int(row["original_index"]), sample_id)
                self.assertEqual(nonempty[int(row["valid_index"])], sample_id)
                self.assertEqual([int(row[f"label_{k}"]) for k in range(8)], labels[sample_id].tolist())
            self.assertTrue((output / "duplicates.csv").is_file())
            self.assertTrue((output / "metadata.json").is_file())
            produced_before = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
            with self.assertRaises((FileExistsError, ValueError)):
                build_protocol(source, output, seed=2027)
            self.assertEqual(produced_before, {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()})
            self.assertEqual(source_before, hashlib.sha256(source.read_bytes()).digest())
            self.assertEqual(old_before, old_split.read_bytes())
            occupied = root / "occupied"
            occupied.mkdir()
            sentinel = occupied / "user.txt"
            sentinel.write_text("do not overwrite")
            with self.assertRaises((FileExistsError, ValueError)):
                build_protocol(source, occupied)
            self.assertEqual(sentinel.read_text(), "do not overwrite")


if __name__ == "__main__":
    unittest.main()
