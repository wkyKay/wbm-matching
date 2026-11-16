"""Derived candidate protocol; frozen query/gallery manifests remain unchanged."""
from collections.abc import Mapping
import hashlib
import json

import numpy as np

PROTOCOL_VERSION = "coverage_cross_combination_v1"
CANDIDATE_RULE = "all_gallery_except_query_and_exact_label_combination"


class CrossCombinationPool(Mapping):
    """Filter before any method scoring/coarse selection, never after Top-K."""

    def __init__(self, base, labels):
        self.base = base
        self.queries = base.queries
        self.gallery = base.gallery
        self.split_name = base.split_name
        self.labels = np.array(labels, copy=True)
        if self.labels.ndim != 2 or not np.isin(self.labels, [0, 1]).all():
            raise ValueError("labels must be a binary original-ID-indexed matrix")
        if not np.all(self.labels[self.gallery].sum(axis=1) > 0):
            raise ValueError("gallery must contain nonempty defect labels")
        self.labels.flags.writeable = False
        self.audit = []
        digest = hashlib.sha256()
        for q in self.queries:
            original = np.asarray(base[int(q)])
            eligible = self[int(q)]
            digest.update(json.dumps([int(q), eligible.tolist()], separators=(",", ":")).encode("ascii"))
            digest.update(b"\n")
            self.audit.append({"query_id": int(q), "original_candidates": len(original),
                               "excluded_exact": len(original) - len(eligible),
                               "eligible_candidates": len(eligible)})
        self.metadata = {
            "protocol_version": PROTOCOL_VERSION, "candidate_rule": CANDIDATE_RULE,
            "base_protocol_signature": base.protocol_signature,
            "candidate_identity_sha256": digest.hexdigest(),
            "filter_uses_ground_truth": True,
            "filter_scope": "benchmark eligibility only; method scores never receive labels",
            "candidate_count_min": min(r["eligible_candidates"] for r in self.audit),
            "candidate_count_max": max(r["eligible_candidates"] for r in self.audit),
        }
        self.protocol_signature = hashlib.sha256(
            json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()

    def __iter__(self):
        return iter(self.base)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, query):
        candidates = np.asarray(self.base[query])
        return candidates[np.any(self.labels[candidates] != self.labels[query], axis=1)]

    @property
    def candidates_per_query(self):
        """Minimum count, for validating common K; actual counts are per-query."""
        return self.metadata["candidate_count_min"]
