#!/usr/bin/env python3
"""Frozen radial+area F, excluding exact combinations before map scoring."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--f-root', type=Path, required=True)
    p.add_argument('--compared-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    root, compared, out = args.f_root, args.compared_root, args.output_dir
    sys.path[:0] = [str(compared), str(root)]
    from shared.common import hash_file, read_json, save_json, save_csv
    from shared.cross_combination import CrossCombinationPool
    from shared.coverage_metrics import evaluate_rankings_file
    from shared.exact_metrics import order_scores
    from wbm_retrieval.data import Protocol
    from wbm_retrieval.diagnostics import batched_map_scores
    area = root / 'artifacts/f_radial_area_test_20261111_v1'
    radial = root / 'artifacts/f_radial_beta1_test_20261111_v1'
    plan = read_json(area / 'selection_before_test.json')
    completed = read_json(area / 'completed.json')
    assert completed['status'] == 'completed'
    assert (plan['beta'], plan['sigma_scale'], plan['scale_weight'], plan['bins']) == (1, 1.5, .15, 16)
    source = Path(plan['config']['output_dir'])
    checked = {}
    def check(path, digest):
        path = Path(path)
        assert hash_file(path) == digest, str(path)
        checked[str(path)] = digest
    for name, digest in completed['files'].items():
        if name in ('selection_before_test.json', 'test_area_geometry.npz', 'test/rankings.csv'):
            check(area / name, digest)
    for name, state in plan['source_snapshot'].items():
        path = Path(name)
        if not path.exists() and path.suffix == '.py':
            path = root / 'scripts' / path.name
        check(path, state['sha256'])
    radial_done = read_json(radial / 'completed.json')
    assert radial_done['status'] == 'completed'
    for name in ('test_h512.npy', 'test_radial_histograms.npy'):
        check(radial / name, radial_done['files'][name])
    for name in ('test_queries.csv', 'test_gallery.csv', 'metadata.json'):
        assert hash_file(root/'protocol/retrieval_v1'/name) == hash_file(compared/'protocol/retrieval_v1'/name)
    with np.load(source/'cache/test_regions.npz') as z:
        meta = dict(z)
    labels = np.load(source/'cache/labels.npy')
    features = np.load(radial/'test_h512.npy')
    hist = np.load(radial/'test_radial_histograms.npy')
    with np.load(area/'test_area_geometry.npz') as z:
        relative = z['sqrt_relative_area']
        for key in ('map_ids', 'offsets', 'area'):
            np.testing.assert_array_equal(z[key], meta[key])
    protocol = Protocol(plan['config'])
    base = protocol.candidates
    assert len(base) == 7199 and base.candidates_per_query == 7198
    np.testing.assert_array_equal(base.gallery, meta['map_ids'])
    pool = CrossCombinationPool(base, labels)
    assert pool.metadata['candidate_identity_sha256'] == '68c6875330ed70841586c654da776f8d0bc72bcd4cd341b79cea29705543f0b8'
    indices = {int(q): np.arange(s,e) for q,s,e in zip(meta['map_ids'],meta['offsets'][:-1],meta['offsets'][1:])}
    positions = {int(q):i for i,q in enumerate(meta['map_ids'])}
    padded = np.full((7199, int(np.diff(meta['offsets']).max())), -1, dtype=int)
    for q, tokens in indices.items():
        padded[positions[q],:len(tokens)] = tokens
    cdf, log_area = np.cumsum(hist.astype(float), axis=-1), np.log(relative.astype(float))
    def score(q, candidates):
        qt = indices[int(q)]
        radial_dist = np.abs(cdf[qt,None]-cdf[None]).sum(-1)/16
        area_dist = np.abs(log_area[qt,None]-log_area[None])
        pair = np.clip(features[qt] @ features.T, 0, 1)*np.exp(-radial_dist)*(.85+.15*np.exp(-area_dist/1.5))
        return batched_map_scores(pair, padded[[positions[int(c)] for c in candidates]], meta['area'][qt], plan['config']['matching'])
    # Replay unfiltered full-pool rankings before using the new eligible pool.
    selected = set(map(int, base.queries[np.linspace(0,7198,16,dtype=int)]))
    saved = {q: [] for q in selected}
    with (area/'test/rankings.csv').open() as f:
        for row in csv.DictReader(f):
            q = int(row['query_id'])
            if q in selected:
                saved[q].append((int(row['candidate_id']),float(row['similarity_score'])))
    for q in sorted(selected):
        candidates = np.asarray(base[q]); scores = score(q, candidates)
        order = order_scores(candidates,scores,q,2026)[:500]
        ids, values = zip(*saved[q])
        np.testing.assert_array_equal(candidates[order],ids)
        np.testing.assert_allclose(scores[order],values,atol=1e-12,rtol=0)
        eligible = np.asarray(pool[q])
        np.testing.assert_allclose(score(q,eligible),scores[np.isin(candidates,eligible)],atol=1e-12,rtol=0)
    print('PREFLIGHT PASSED: source hashes, 16 original Top500 replays, filtered score equivalence, candidate identity',flush=True)
    if args.preflight_only:
        return
    out.mkdir(parents=True,exist_ok=False)
    target = out/'test'; target.mkdir()
    metadata = {'method':'F_radial_area','candidate_rule':pool.metadata['candidate_rule'],
                'derived_protocol':pool.metadata,'ranking_limit':500,'beta':1,'sigma_scale':1.5,'scale_weight':.15,
                'checkpoint_sha256':plan['checkpoint_sha256'],'source_files':checked,'preflight_replay_queries':16}
    save_json(out/'evaluation_manifest.json',metadata)
    save_csv(out/'cross_combination_candidates.csv',pool.audit,list(pool.audit[0]))
    start = time.monotonic()
    with (target/'rankings.csv').open('x',newline='') as f:
        writer=csv.writer(f); writer.writerow(['query_id','rank','candidate_id','score'])
        for n,q in enumerate(pool,1):
            candidates=np.asarray(pool[q]); scores=score(q,candidates)
            assert np.isfinite(scores).all() and not np.any(np.all(labels[candidates]==labels[q],axis=1))
            order=order_scores(candidates,scores,q,2026)[:500]
            writer.writerows((int(q),r,int(candidates[i]),float(scores[i])) for r,i in enumerate(order,1))
            if n%100==0: print(f'F area {n}/7199 {time.monotonic()-start:.1f}s',flush=True)
    taxonomy=sorted({tuple(labels[i]) for i in protocol.ids['train']})
    class_map={x:i for i,x in enumerate(taxonomy)}
    classes=np.full(len(labels),-1,dtype=int)
    for q in base.gallery: classes[q]=class_map[tuple(labels[q])]
    report=evaluate_rankings_file(SimpleNamespace(labels=labels,class_ids=classes),pool,target/'rankings.csv',
        [1,5,10,20,50,100,500],target,split_name='test',metadata=metadata)
    for name,digest in checked.items():
        assert hash_file(name)==digest,name
    save_json(out/'completed.json',{'status':'completed','seconds':time.monotonic()-start,
        'query_count':7199,'source_files_unchanged':True,'derived_protocol':pool.metadata,
        'files':{str(p.relative_to(out)):hash_file(p) for p in out.rglob('*') if p.is_file()}})
    print(json.dumps(report['metrics']),flush=True)

if __name__=='__main__':
    main()
