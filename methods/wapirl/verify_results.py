"""Independent post-run audit: frozen IDs, scores, exact and coverage metrics."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def rows(p):
    with p.open() as f:return list(csv.DictReader(f))

def audit(split):
    folder=ROOT/'artifacts'/f'wapirl_{split}_v1'
    done=json.loads((folder/'completed.json').read_text())
    assert done['smoke'] is False and done['split']==split
    for name,h in done['files'].items():assert digest(folder/name)==h,name
    manifest=json.loads((folder/'evaluation_manifest.json').read_text())
    assert manifest['candidate_rule']=='all_gallery_except_query'
    assert manifest['settings']['selection']=='minimum_valid_nce'
    queries=[int(r['sample_id']) for r in rows(ROOT/'protocol/retrieval_v1'/f'{split}_queries.csv')]
    gallery=np.array([int(r['sample_id']) for r in rows(ROOT/'protocol/retrieval_v1'/f'{split}_gallery.csv')])
    with np.load(ROOT.parent/'data/wm38k/Wafer_Map_Datasets.npz') as z: labels=z['arr_1'].astype(int)
    result=folder/split
    exact={int(r['query_id']):r for r in rows(result/'per_query.csv')}
    coverage={int(r['query_id']):r for r in rows(result/'coverage_per_query.csv')}
    assert set(exact)==set(coverage)==set(queries)
    diagnostics=rows(result/'query_diagnostics.csv')
    assert len(diagnostics)==len(queries)
    for r in diagnostics:
        for key in ['input_candidates','scored_count','eligible_count']:assert int(r[key])==len(gallery)-1
        assert int(r['returned'])==500
    features=np.load(result/'prepared/features.npz')
    np.testing.assert_array_equal(features['sample_ids'],gallery)
    feats=features['features'];lookup={i:j for j,i in enumerate(gallery)}
    checked=[]
    def check(row,key,value):
        if value is None:assert row[key]=='',key
        else:np.testing.assert_allclose(float(row[key]),value,rtol=1e-10,atol=1e-12,err_msg=key)
    with (result/'rankings.csv').open() as f:
        for q,group in itertools.groupby(csv.DictReader(f),key=lambda r:int(r['query_id'])):
            rank=list(group);ids=np.array([int(r['candidate_id']) for r in rank])
            assert len(rank)==500 and len(set(ids))==500 and q not in ids
            assert set(ids)<=set(gallery)
            assert [int(r['rank']) for r in rank]==list(range(1,501))
            scores=np.array([float(r['score']) for r in rank])
            assert np.all(scores[:-1]>=scores[1:])
            np.testing.assert_allclose(scores,np.clip(feats[[lookup[i] for i in ids]]@feats[lookup[q]],0,1),rtol=1e-5,atol=1e-6)
            candidates=gallery[gallery!=q]
            rel=np.all(labels[ids]==labels[q],axis=1)
            R=np.all(labels[candidates]==labels[q],axis=1).sum()
            cov=labels[ids]@labels[q]/labels[q].sum()
            full=labels[candidates]@labels[q]/labels[q].sum()
            ideal=np.sort(full)[::-1]
            flags=cov==1;full_R=(full==1).sum()
            for k in manifest['ks']:
                hits=rel[:k].sum();pr=np.cumsum(rel[:k])/np.arange(1,k+1)
                for key,value in [('Hit',float(hits>0)),('Precision',hits/k),('Recall',hits/R if R else None),('AP',(pr*rel[:k]).sum()/R if R else None)]:check(exact[q],f'{key}@{k}',value)
                discount=1/np.log2(np.arange(2,k+2));idcg=ideal[:k]@discount
                full_hits=flags[:k].sum();full_pr=np.cumsum(flags[:k])/np.arange(1,k+1)
                for key,value in [('Coverage-nDCG',cov[:k]@discount/idcg if idcg else None),('MeanCoverage',cov[:k].mean()),('FullCoverageHit',float(full_hits>0)),('FullCoveragePrecision',full_hits/k),('FullCoverageRecall',full_hits/full_R if full_R else None),('FullCoverageAP',(full_pr*flags[:k]).sum()/full_R if full_R else None)]:check(coverage[q],f'{key}@{k}',value)
            checked.append(q)
    assert checked==queries
    for file,records in [('metrics.json',exact),('coverage_metrics.json',coverage)]:
        report=json.loads((result/file).read_text())
        for key,value in report['metrics'].items():
            field=key.replace('FullCoverage_mAP','FullCoverageAP').replace('mAP','AP')
            vals=[float(r[field]) for r in records.values() if r[field]!='']
            np.testing.assert_allclose(value,np.mean(vals),rtol=1e-10,atol=1e-12)
    return {'split':split,'queries':len(checked),'gallery':len(gallery),'candidates_per_query':len(gallery)-1,'ranking_rows':len(checked)*500,'all_file_hashes_verified':True,'independent_exact_and_coverage_recalculation':True,'feature_scores_verified':True}

if __name__=='__main__':
    report=[audit(s) for s in ['valid','test']]
    path=ROOT/'artifacts/wapirl_pipeline_20261111/independent_verification.json'
    with path.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
