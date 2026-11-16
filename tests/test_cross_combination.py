"""Independent cross-combination protocol and pre-ranking filtering checks."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.cross_combination import CrossCombinationPool
from shared.retrieval_protocol import FullGalleryPool
from shared.coverage_metrics import query_metrics
from shared import evaluation
import test_evaluation as fixtures
from test_evaluation import snapshot


class PoolTests(unittest.TestCase):
    def test_exact_removed_superset_partial_and_zero_preserved(self):
        labels = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0], [0, 0, 1]])
        base = FullGalleryPool([0], np.arange(5), 'test', 'frozen')
        pool = CrossCombinationPool(base, labels)
        np.testing.assert_array_equal(pool[0], [2, 3, 4])
        np.testing.assert_array_equal(base[0], [1, 2, 3, 4])
        self.assertEqual(pool.audit[0]['excluded_exact'], 1)
        self.assertEqual(pool.candidates_per_query, 3)
        # Only the cross-combination superset counts in the full-coverage denominator.
        result = query_metrics(labels[0], labels[pool[0]], labels[[3, 2]], [2])
        self.assertEqual(result['full_coverage_in_pool'], 1)
        self.assertEqual(result['FullCoverageRecall@2'], 1)
        self.assertEqual(result['FullCoverageAP@2'], .5)
        self.assertEqual(result['MeanCoverage@2'], .75)
        self.assertAlmostEqual(result['Coverage-nDCG@2'], (.5 + 1/np.log2(3))/(1 + .5/np.log2(3)))
        self.assertEqual(pool.protocol_signature, CrossCombinationPool(base, labels).protocol_signature)
        self.assertNotEqual(pool.protocol_signature, base.protocol_signature)

    def test_zero_full_coverage_still_has_graded_metrics(self):
        labels = np.array([[1, 1], [1, 1], [1, 0], [0, 1]])
        pool = CrossCombinationPool(FullGalleryPool([0], np.arange(4), 'valid', 'x'), labels)
        result = query_metrics(labels[0], labels[pool[0]], labels[[2, 3]], [2])
        self.assertIsNone(result['FullCoverageRecall@2'])
        self.assertIsNone(result['FullCoverageAP@2'])
        self.assertEqual(result['Coverage-nDCG@2'], 1)
        self.assertEqual(result['MeanCoverage@2'], .5)


class RunnerTests(fixtures.EvaluationFixture):
    fake_preparer = fixtures.RunBoundaryTests.fake_preparer

    def test_filter_before_rank_and_preserve_original_protocol(self):
        before = snapshot(Path(self.config['retrieval_protocol_dir']))
        calls = {}
        output = self.root / 'cross'
        with mock.patch.object(evaluation, 'prepare_method', side_effect=self.fake_preparer(calls)):
            result = evaluation.run_evaluation(self.config, output, ks=[1, 2], exclude_exact=True)
        for q, candidates in calls['rank_calls']:
            self.assertEqual(set(candidates), {8, 9} if q in (2, 3) else {2, 3})
        self.assertEqual(result['metric_version'], 'coverage_v1')
        self.assertIn('derived_protocol', result['metadata'])
        self.assertEqual(snapshot(Path(self.config['retrieval_protocol_dir'])), before)
        self.assertFalse((output/'valid/metrics.json').exists())
        self.assertTrue((output/'completed.json').exists())
        manifest = json.loads((output/'evaluation_manifest.json').read_text())
        self.assertFalse(manifest['exact_reports'])
        self.assertEqual(manifest['metric_version'], 'coverage_v1')
        with (output/'protocol/cross_combination_candidates.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(all(int(r['excluded_exact']) == 1 for r in rows))

    def test_k_checked_after_filter(self):
        with self.assertRaisesRegex(ValueError, 'candidate count'):
            evaluation.run_evaluation(self.config, self.root/'too_large', ks=[3], exclude_exact=True)

    def test_real_wu_and_parallel_wbbs_filtered_rankings(self):
        for method, workers in [('wu', 1), ('wbbs', 2)]:
            output = self.root / method
            evaluation.run_evaluation({**self.config, 'method': method}, output,
                                      ks=[1, 2], exclude_exact=True, rank_workers=workers)
            with (output/'valid/rankings.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 8)
            for r in rows:
                q, c = int(r['query_id']), int(r['candidate_id'])
                self.assertNotEqual(q < 6, c < 6)

class FRunnerTests(unittest.TestCase):
    def test_read_only_f_source_and_filtered_scoring_end_to_end(self):
        project = ROOT.parent / 'wbm-region-retrieval'
        sys.path.insert(0, str(project))
        import torch
        from types import SimpleNamespace
        from scripts.evaluate_f_cross_combination import main
        from experiments.proposed._shared import ef_suite
        from experiments.proposed._shared.ef_training import run_signature
        from wbm_retrieval import data, training
        from wbm_retrieval.common import digest
        from shared import common
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source'
            cache = source/'cache'
            model_dir = source/'seed_2026/F'
            cache.mkdir(parents=True)
            model_dir.mkdir(parents=True)
            config = {'output_dir': str(source), 'seed': 2026, 'data_protocol': 'deduplicated_v1',
                      'evaluation': {'metric_version': 'exact_v1', 'tie_seed': 2026},
                      'local': {'epochs': 100}, 'matching': {'lambda_pos': 0, 'lambda_scale': 0,
                      'shape_threshold': 0, 'score_threshold': 0, 'weight': 'sqrt_area'},
                      'data_file': str(Path(tmp)/'data.npz'), 'split_manifest': str(Path(tmp)/'split.csv'),
                      'retrieval_protocol_dir': str(Path(tmp)/'protocol')}
            suite = {'suite_id': 'fixture', 'variants': {'F': {}}, 'valid_pool': {},
                     'selection_metric': 'mAP@10', 'evaluate_every': 5}
            settings = {k:v for k,v in config.items() if k not in (
                'output_dir', 'seed', 'data_file', 'split_manifest', 'retrieval_protocol_dir')}
            identity = {'sources': {}, 'settings': settings, 'suite': suite, 'code': {}}
            sig = digest(identity)
            common.save_json(source/'suite_manifest.json', {**identity, 'signature': sig})
            labels = np.array([[1,1], [1,1], [1,0], [0,1]], dtype=np.uint8)
            np.save(cache/'labels.npy', labels)
            np.savez(cache/'valid_regions.npz', map_ids=np.arange(4), offsets=np.arange(5), area=np.ones(4))
            np.save(cache/'valid_F.npy', np.zeros((4, 3, 96, 96), dtype=np.float32))
            common.save_json(cache/'prepared.json', {'signature': sig, 'files': {
                str(p.relative_to(source)): common.hash_file(p) for p in cache.iterdir()}})
            common.save_json(model_dir/'training_summary.json', {'completed': True, 'epochs_completed': 100, 'best_nce_epoch':100})
            (model_dir/'best_nce.pt').write_bytes(b'fixture')
            before = snapshot(source)
            pool = FullGalleryPool(np.arange(4), np.arange(4), 'valid', 'fixture')
            features = np.array([[1.,0.], [1.,0.], [.8,.6], [.6,.8]])
            checkpoint = {'signature':run_signature(sig,2026,'F'), 'epoch':100, 'model':{}}
            out = Path(tmp)/'result'
            with mock.patch.object(ef_suite, 'load_suite', return_value=(suite,config)), \
                 mock.patch.object(ef_suite, 'code_hashes', return_value={}), \
                 mock.patch.object(ef_suite, 'build_model', return_value=torch.nn.Identity()), \
                 mock.patch.object(ef_suite, 'embeddings', return_value=features), \
                 mock.patch.object(data, 'source_fingerprint', return_value={}), \
                 mock.patch.object(data, 'Protocol', return_value=SimpleNamespace(retrieval=SimpleNamespace(valid=pool))), \
                 mock.patch.object(training, 'load_checkpoint', return_value=checkpoint):
                self.assertEqual(main(['--source-run',str(source),'--output-dir',str(out),
                                       '--split','valid','--ks','1','2','--device','cpu']), 0)
            self.assertEqual(before, snapshot(source))
            self.assertTrue((out/'completed.json').exists())
            with (out/'valid/rankings.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 8)
            for row in rows:
                self.assertFalse(np.array_equal(labels[int(row['query_id'])],labels[int(row['candidate_id'])]))
            report = common.read_json(out/'valid/coverage_metrics.json')
            self.assertEqual(report['queries_without_full_coverage_candidates'],2)


if __name__ == '__main__':
    unittest.main()
