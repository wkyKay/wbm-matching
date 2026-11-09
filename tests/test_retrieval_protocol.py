"""Frozen retrieval identity, stratification and complete-gallery boundaries."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_protocol import build_protocol
from shared.retrieval_protocol import (FullGalleryPool, allocate_query_counts, build_retrieval_protocol,
                                load_retrieval_protocol, protocol_file_hashes, stratified_queries)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(root):
    maps = np.ones((186, 7, 7), dtype=np.uint8)
    for i in range(len(maps)):
        maps[i, 0, :6] = [(i >> (2 * j)) & 3 for j in range(6)]
    labels = np.zeros((186, 8), dtype=np.uint8)
    labels[:100, 0] = 1
    labels[100:150, 1] = 1
    labels[150:180, 0] = labels[150:180, 2] = 1
    maps = np.concatenate([maps, maps[[0, 100, 180]]])
    labels = np.concatenate([labels, labels[[0, 100, 180]]])
    source = root / "source.npz"
    np.savez(source, arr_0=maps, arr_1=labels)
    split_dir = root / "split"
    build_protocol(source, split_dir)
    split = split_dir / "split.csv"
    with split.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = {name: np.asarray(sorted(int(r["sample_id"]) for r in rows if r["split"] == name))
           for name in ("train", "valid", "test")}
    return source, split, labels, ids


def write_ids(path, ids):
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id"])
        writer.writerows((int(i),) for i in ids)


def update_file_hash(directory, filename):
    path = directory / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["files"][filename] = file_hash(directory / filename)
    path.write_text(json.dumps(metadata))


class QueryAllocationTests(unittest.TestCase):
    def test_real_class_sizes_preserve_minorities_and_exact_budget(self):
        counts = allocate_query_counts([100] * 35 + [14, 85], 500, 5)
        self.assertEqual(counts.sum(), 500)
        self.assertEqual(counts[-2:].tolist(), [5, 11])
        self.assertEqual(set(counts[:35]), {13, 14})
        self.assertEqual(np.count_nonzero(counts[:35] == 14), 29)

    def test_tiny_class_is_fully_selected_and_impossible_budget_rejected(self):
        counts = allocate_query_counts([100, 2], 20, 5)
        self.assertEqual(counts.tolist(), [18, 2])
        for count in (6, 103):
            with self.assertRaises(ValueError):
                allocate_query_counts([100, 2], count, 5)
        for minimum in (0, True, -1):
            with self.assertRaises(ValueError):
                allocate_query_counts([100, 2], 20, minimum)

    def test_queries_deterministic_without_modifying_inputs(self):
        labels = np.zeros((40, 8), dtype=np.uint8)
        labels[:30, 0] = labels[30:, 1] = 1
        ids = np.arange(40)[::-1]
        before_labels, before_ids = labels.copy(), ids.copy()
        queries, allocation = stratified_queries(ids, labels, 12, seed=2026, min_per_class=3)
        repeated, other_allocation = stratified_queries(ids, labels, 12, seed=2026, min_per_class=3)
        np.testing.assert_array_equal(queries, repeated)
        self.assertEqual(allocation, other_allocation)
        self.assertEqual([row["query_count"] for row in allocation], [9, 3])
        np.testing.assert_array_equal(labels, before_labels)
        np.testing.assert_array_equal(ids, before_ids)


class FullGalleryTests(unittest.TestCase):
    def test_self_exclusion_and_unknown_queries(self):
        pool = FullGalleryPool([7, 3], [8, 3, 7, 1], "valid", "signature")
        self.assertEqual(list(pool), [3, 7])
        self.assertEqual(pool.candidates_per_query, 3)
        np.testing.assert_array_equal(pool[3], [1, 7, 8])
        np.testing.assert_array_equal(pool[7], [1, 3, 8])
        for bad_query in (1, 99, True, "3"):
            with self.assertRaises(KeyError):
                pool[bad_query]
        self.assertEqual(pool.split_name, "valid")
        self.assertEqual(pool.protocol_signature, "signature")

    def test_exhaustive_test_pool_keeps_only_linear_storage(self):
        ids = np.arange(7199)
        pool = FullGalleryPool(ids, ids, "test", "s")
        arrays = [value for value in vars(pool).values() if isinstance(value, np.ndarray)]
        self.assertEqual(sum(value.size for value in arrays), 2 * len(ids))
        first = pool[0]
        first[0] = -1
        self.assertEqual(pool[0][0], 1)  # Returned candidate arrays are not cached.
        self.assertFalse(pool.gallery.flags.writeable)
        self.assertEqual(sum(len(candidates) for _, candidates in pool.items()), 7199 * 7198)


class FrozenProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source, self.split, self.labels, self.ids = fixture(self.root)
        self.directory = self.root / "retrieval"
        self.kwargs = {"data_file": self.source, "split_file": self.split,
                       "valid_ids": self.ids["valid"], "test_ids": self.ids["test"], "labels": self.labels}

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, directory=None):
        return build_retrieval_protocol(self.source, self.split, directory or self.directory,
                                        seed=2026, valid_queries=10, min_per_class=2)

    def test_build_roundtrip_is_deterministic_preserves_sources_and_rejects_overwrite(self):
        before = {path: file_hash(path) for path in (self.source, self.split)}
        metadata = self.build()
        other = self.root / "other"
        self.build(other)
        self.assertEqual(protocol_file_hashes(self.directory), protocol_file_hashes(other))
        with patch("shared.retrieval_protocol.stratified_queries", side_effect=AssertionError("Loader must not resample")):
            protocol = load_retrieval_protocol(self.directory, **self.kwargs)
        self.assertEqual(len(protocol.valid), 10)
        self.assertEqual(protocol.valid.candidates_per_query, len(self.ids["valid"]) - 1)
        np.testing.assert_array_equal(protocol.test.queries, self.ids["test"])
        np.testing.assert_array_equal(protocol.test.gallery, self.ids["test"])
        for pool in (protocol.valid, protocol.test):
            for query, candidates in pool.items():
                self.assertNotIn(query, candidates)
                self.assertFalse(set(candidates) & set(self.ids["train"]))
        self.assertFalse(set(protocol.valid.gallery) & set(protocol.test.gallery))
        self.assertEqual(metadata["candidate_rule"], "all_gallery_except_query")
        self.assertEqual({p.name for p in self.directory.iterdir()},
                         {"metadata.json", "valid_queries.csv", "valid_gallery.csv", "test_queries.csv", "test_gallery.csv"})
        saved = {p.name: p.read_bytes() for p in self.directory.iterdir()}
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(saved, {p.name: p.read_bytes() for p in self.directory.iterdir()})
        self.assertEqual(before, {path: file_hash(path) for path in before})

    def test_loader_rejects_subset_and_other_source(self):
        self.build()
        for key in ("valid_ids", "test_ids"):
            with self.subTest(key=key):
                kwargs = {**self.kwargs, key: self.kwargs[key][1:]}
                with self.assertRaisesRegex(ValueError, "complete"):
                    load_retrieval_protocol(self.directory, **kwargs)
        other_source = self.root / "other.npz"
        np.savez(other_source, arr_0=np.zeros((3, 3, 3)), arr_1=np.zeros((3, 8)))
        with self.assertRaisesRegex(ValueError, "different source"):
            load_retrieval_protocol(self.directory, **{**self.kwargs, "data_file": other_source})

    def test_loader_rejects_tampered_csv_and_cross_split_gallery(self):
        self.build()
        path = self.directory / "valid_gallery.csv"
        altered = self.ids["valid"].copy()
        altered[0] = self.ids["train"][0]
        write_ids(path, sorted(altered))
        with self.assertRaisesRegex(ValueError, "hashes"):
            protocol_file_hashes(self.directory)
        update_file_hash(self.directory, path.name)
        with self.assertRaisesRegex(ValueError, "gallery does not equal"):
            load_retrieval_protocol(self.directory, **self.kwargs)

    def test_loader_rejects_nonexhaustive_test_and_wrong_query_class_allocation(self):
        self.build()
        test_path = self.directory / "test_queries.csv"
        write_ids(test_path, self.ids["test"][1:])
        update_file_hash(self.directory, test_path.name)
        with self.assertRaises(ValueError):
            load_retrieval_protocol(self.directory, **self.kwargs)
        write_ids(test_path, self.ids["test"])
        update_file_hash(self.directory, test_path.name)
        # All ten queries from one class fit in valid, but violate the quotas.
        majority = [i for i in self.ids["valid"] if self.labels[i, 0] == 1 and self.labels[i, 2] == 0]
        write_ids(self.directory / "valid_queries.csv", majority)
        update_file_hash(self.directory, "valid_queries.csv")
        with self.assertRaisesRegex(ValueError, "class allocation"):
            load_retrieval_protocol(self.directory, **self.kwargs)

    def test_builder_rejects_duplicate_or_incomplete_source_manifest_before_writing(self):
        with self.split.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        for mutated in (rows[:-1], rows + [rows[0]]):
            with self.split.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(mutated)
            with self.assertRaises(ValueError):
                self.build()
            self.assertFalse(self.directory.exists())


if __name__ == "__main__":
    unittest.main()
