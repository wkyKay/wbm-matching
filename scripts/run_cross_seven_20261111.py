#!/usr/bin/env python3
"""Detached seven-method full test, followed by independent report validation."""
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path('/data/wangkaiyan2024/compared_methods')
FROOT=Path('/data/wangkaiyan2024/wbm-region-retrieval')
OLD=Path('/data/wangkaiyan2024/wbm-matching/artifacts/formal_exact_20261109_120738/evaluation')
OUT=ROOT/'artifacts/cross_coverage_seven_test_20261111_v1'
PYTHON='/home/wangkaiyan2024/miniconda3/envs/wbm/bin/python'
METHODS=['wu','yu','wbbs','nakazawa','wmhd','wapirl','f_radial_area']
EXPECTED='68c6875330ed70841586c654da776f8d0bc72bcd4cd341b79cea29705543f0b8'


def save(path,value):
    p=Path(path); tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); tmp.replace(p)


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()


def run_one(method):
    began=time.time()
    if method=='f_radial_area':
        cmd=[PYTHON,str(ROOT/'scripts/evaluate_f_area_cross_20261111.py'),'--f-root',str(FROOT),'--compared-root',str(ROOT),'--output-dir',str(OUT/method)]
    else:
        if method=='wapirl':
            config=ROOT/'artifacts/wapirl_test_v1/resolved_config.json'
        else:
            suffix='_test_r2' if method in ('wbbs','wmhd') else '_test'
            config=OLD/(method+suffix)/'resolved_config.json'
        cmd=[PYTHON,str(ROOT/'scripts/evaluate.py'),'--config',str(config),'--split','test','--exclude-exact','--output-dir',str(OUT/method),'--workers','4','--rank-workers',str({'wmhd':48,'wbbs':8,'wu':8}.get(method,1))]
        if method in ('yu','nakazawa','wapirl'): cmd+=['--device','cuda']
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES=str({'yu':0,'nakazawa':1,'wapirl':2}.get(method,3)))
    save(OUT/(method+'_status.json'),{'status':'running','command':cmd,'started_at':began})
    with (OUT/(method+'.log')).open('x') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        save(OUT/(method+'_status.json'),{'status':'running','pid':proc.pid,'command':cmd,'started_at':began})
        code=proc.wait()
    ok=code==0 and (OUT/method/'completed.json').is_file()
    save(OUT/(method+'_status.json'),{'status':'completed' if ok else 'failed','exit_code':code,'seconds':time.time()-began,'command':cmd})
    print(method,'completed' if ok else 'FAILED',flush=True)
    return ok


def validate_and_report():
    import numpy as np
    sys.path.insert(0,str(ROOT))
    from shared.cross_combination import CrossCombinationPool
    from shared.retrieval_protocol import FullGalleryPool
    labels=np.load(FROOT/'artifacts/formal_f_exact_20261109_132219/run/cache/labels.npy')
    with (ROOT/'protocol/retrieval_v1/test_queries.csv').open() as f: queries=np.array([int(r['sample_id']) for r in csv.DictReader(f)])
    pool=CrossCombinationPool(FullGalleryPool(queries,queries,'test','verification'),labels)
    assert pool.metadata['candidate_identity_sha256']==EXPECTED
    ks=[1,5,10,20,50,100,500]
    evidence={}; comparisons=[]; reports={}; fields=None
    for method in METHODS:
        folder=OUT/method/'test'; report=read(folder/'coverage_metrics.json'); reports[method]=report
        assert report['metadata']['derived_protocol']['candidate_identity_sha256']==EXPECTED
        with (folder/'coverage_per_query.csv').open() as f: per={int(r['query_id']):r for r in csv.DictReader(f)}
        assert set(per)==set(queries)
        accum={k:[] for k in report['metrics']}; nrows=0; maxerr=0.; seen=[]
        def verify(q,ids,scores):
            nonlocal maxerr
            ids=np.array(ids); scores=np.array(scores); seen.append(q)
            assert len(ids)==500 and len(set(ids))==500
            assert np.isin(ids,pool[q]).all()
            assert np.all(scores[:-1]>=scores[1:])
            gain=labels[ids]@labels[q]/labels[q].sum()
            full_gain=labels[pool[q]]@labels[q]/labels[q].sum()
            full=(gain==1); count=int((full_gain==1).sum())
            assert int(per[q]['full_coverage_in_pool'])==count
            precision=np.cumsum(full)/np.arange(1,501)
            for k in ks:
                discount=1/np.log2(np.arange(k)+2)
                idcg=float(np.sort(full_gain)[::-1][:k]@discount)
                vals={f'Coverage-nDCG@{k}':float(gain[:k]@discount/idcg) if idcg else None,
                      f'MeanCoverage@{k}':float(gain[:k].mean()),f'FullCoverageHit@{k}':float(full[:k].any()),
                      f'FullCoveragePrecision@{k}':float(full[:k].sum()/k),
                      f'FullCoverageRecall@{k}':float(full[:k].sum()/count) if count else None,
                      f'FullCoverage_mAP@{k}':float((precision[:k]*full[:k]).sum()/count) if count else None}
                for key,val in vals.items():
                    col=key.replace('FullCoverage_mAP','FullCoverageAP')
                    if val is None: assert per[q][col] in ('','None')
                    else:
                        err=abs(float(per[q][col])-val); assert err<1e-12,(method,q,key,err)
                        maxerr=max(maxerr,err); accum[key].append(val)
        with (folder/'rankings.csv').open() as f:
            current=None; ids=[]; scores=[]
            for row in csv.DictReader(f):
                q=int(row['query_id'])
                if current is not None and q!=current: verify(current,ids,scores); ids=[]; scores=[]
                current=q; assert int(row['rank'])==len(ids)+1
                ids.append(int(row['candidate_id'])); scores.append(float(row.get('score',row.get('similarity_score')))); nrows+=1
            verify(current,ids,scores)
        assert seen==queries.tolist() and nrows==7199*500
        for key,values in accum.items():
            value=float(np.mean(values)); assert abs(value-report['metrics'][key])<1e-12
            assert len(values)==report['metric_query_counts'][key]
            comparisons.append({'method':method,'metric':key,'value':value,'valid_queries':len(values)})
        assert report['queries_without_full_coverage_candidates']==999
        evidence[method]={'queries':7199,'ranking_rows':nrows,'same_combination_candidates':0,'max_per_query_metric_error':maxerr,
                          'ranking_sha256':sha(folder/'rankings.csv'),'coverage_sha256':sha(folder/'coverage_metrics.json')}
        print('Verified',method,flush=True)
    save(OUT/'independent_verification.json',evidence)
    save(OUT/'comparison.json',reports)
    with (OUT/'comparison.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(comparisons[0])); w.writeheader(); w.writerows(comparisons)
    keys=['Coverage-nDCG@10','MeanCoverage@10','FullCoveragePrecision@10','FullCoverage_mAP@10','FullCoverageRecall@500']
    lines=['# 排除同组合候选：六对比方法与径向＋面积 F','',
        'test 7199 queries；仅排除自身与完整八维标签组合相同候选。保留超集、部分和零覆盖。',
        '每 query 6999–7170 candidates；999 个 query 无完整覆盖正样本，Recall/AP 有效 query=6200。',
        '全部逐 query 指标和 Top500 候选资格已独立复算；完整 K 指标见 comparison.csv。','',
        '| 方法 | '+' | '.join(keys)+' |','|---|'+'---:|'*len(keys)]
    for method in sorted(METHODS,key=lambda m:reports[m]['metrics']['Coverage-nDCG@10'],reverse=True):
        lines.append('| '+method+' | '+' | '.join(f"{reports[method]['metrics'][k]:.9f}" for k in keys)+' |')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    OUT.mkdir(exist_ok=False)
    save(OUT/'launch.json',{'status':'running','started_at':time.time(),'pid':os.getpid(),'methods':METHODS,
        'candidate_identity_sha256':EXPECTED,'training':False,'test_parameter_search':False,'tmux_session':'coverage-seven-20261111'})
    try:
        with ThreadPoolExecutor(max_workers=7) as executor: ok=list(executor.map(run_one,METHODS))
        if not all(ok): raise RuntimeError('Some methods failed; inspect individual logs and status files')
        validate_and_report()
        save(OUT/'completed.json',{'status':'completed','all_seven_methods_verified':True,'finished_at':time.time(),
            'candidate_identity_sha256':EXPECTED,'files':{name:sha(OUT/name) for name in ['comparison.json','comparison.csv','REPORT.md','independent_verification.json']}})
        print('ALL SEVEN COMPLETED AND VERIFIED',flush=True)
    except Exception:
        save(OUT/'failed.json',{'status':'failed','traceback':traceback.format_exc(),'at':time.time()})
        raise

if __name__=='__main__': main()
