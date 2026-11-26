"""Independent metric and primary-bootstrap verification after locked evaluation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit,logit

from model_utils import rao_wu_multipliers
from prepare_updates import read_updating_pool
from verify_source import independent_ap,independent_auc
from evaluate_target import read_test

ROOT=Path(__file__).resolve().parents[1]


def newton_intercept(y,p,w):
    positive=w>0;y=y[positive];p=p[positive];w=w[positive]/w[positive].sum()
    z=logit(np.clip(p,1e-6,1-1e-6));a=0.
    for _ in range(100):
        q=expit(z+a);g=np.dot(w,q-y);h=np.dot(w,q*(1-q))
        if abs(g)<1e-12:return a
        if h<=0:raise ValueError('Degenerate independent Newton fit')
        a-=g/h
    raise ValueError('Independent Newton did not converge')


def main():
    folder=ROOT/'reports/final_v1'
    run=json.loads((folder/'run_manifest.json').read_text())
    freeze=json.loads((ROOT/'manifests/final_evaluation_freeze_v1.json').read_text())
    for name,expected in {**freeze['files'],**run['outputs']}.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError(f'Changed file {name}')
    archive=np.load(folder/'prediction_matrix.npz')
    metrics=pd.read_csv(folder/'draw_metrics.csv')
    manifest=pd.read_csv(ROOT/'manifests/participant_splits.csv')
    test_ids=set(manifest.loc[manifest.partition.eq('target_test_locked'),'participant_id'])
    if set(archive['participant_ids'])!=test_ids:raise ValueError('Final prediction IDs disagree')
    y=archive['y'];w=archive['weights'];ps=archive['probabilities'];keys=archive['keys']
    key_positions={name:i for i,name in enumerate(keys)}
    max_metric_difference=0.;primary_values=[]
    static=archive['static_probability']
    for _,row in metrics.loc[metrics.status.eq('success')].iterrows():
        key=f'rep{int(row.repetition):02d}_n{int(row.budget)}_{row.strategy}'
        p=ps[:,key_positions[key]]
        values=np.array([np.dot(w,(p-y)**2)/w.sum(),independent_auc(y,p,w),independent_ap(y,p,w)])
        expected=row[['brier','auroc','average_precision']].to_numpy(dtype=float)
        difference=float(np.max(np.abs(values-expected)));max_metric_difference=max(max_metric_difference,difference)
        if difference>1e-12:raise ValueError('Metric verification failed')
        if row.strategy=='no_update' and not np.array_equal(p,static):raise ValueError('Static prediction changed')
        if row.strategy=='intercept' and row.budget==500:
            # Algebraic loss-difference identity, independent of main metric function.
            delta=p-static
            primary_values.append(float(np.dot(w,2*(static-y)*delta+delta*delta)/w.sum()))
    result=json.loads((folder/'primary_result.json').read_text())
    point=float(np.mean(primary_values))
    if abs(point-result['mean_difference'])>1e-12:raise ValueError('Primary point mismatch')
    bootstrap=pd.read_csv(folder/'primary_bootstrap.csv')
    interval=np.quantile(bootstrap.mean_brier_difference.dropna(),[.025,.975])
    if not np.allclose(interval,result['percentile_95_interval'],atol=1e-12,rtol=0):raise ValueError('Primary interval mismatch')
    # Check initial survey replicates with a separately implemented Newton solver.
    pool=read_updating_pool(manifest);test=read_test(manifest)
    saved=pd.read_csv(ROOT/'reports/updating_v2/updating_pool_source_probabilities.csv').set_index('participant_id').loc[pool.index]
    up=saved.source_probability.to_numpy();uy=pool.phq9_positive.to_numpy(dtype=int)
    uw=pool.fitting_weight.to_numpy(dtype=float)
    samples=pd.read_csv(ROOT/'manifests/updating_samples_v1.csv')
    mapping={pid:i for i,pid in enumerate(pool.index)}
    draws=[np.array([mapping[pid] for pid in g.sort_values('draw_order').participant_id])
           for _,g in samples.loc[samples.budget.eq(500)].groupby('repetition',sort=True)]
    strata=np.r_[pool.survey_stratum.to_numpy(),test.survey_stratum.to_numpy()]
    psus=np.r_[pool.survey_psu.to_numpy(dtype=int),test.survey_psu.to_numpy(dtype=int)]
    rng=np.random.default_rng(20261017);errors=[]
    for rep in range(20):
        m=rao_wu_multipliers(strata,psus,rng);rw=uw*m[:len(pool)];tw=w*m[len(pool):]
        differences=[]
        for positions in draws:
            a=newton_intercept(uy[positions],up[positions],rw[positions])
            p=expit(logit(np.clip(static,1e-6,1-1e-6))+a)
            differences.append(float(np.dot(tw,(p-y)**2-(static-y)**2)/tw.sum()))
        error=abs(np.mean(differences)-bootstrap.loc[bootstrap.replicate.eq(rep),'mean_brier_difference'].iloc[0])
        errors.append(error)
    if max(errors)>1e-10:raise ValueError('Independent bootstrap solver mismatch')
    verified={'all_passed':True,'successful_draw_metrics_verified':int(metrics.status.eq('success').sum()),
              'final_test_ids_verified':len(test_ids),'maximum_metric_difference':max_metric_difference,
              'independent_primary_point':point,'independent_bootstrap_replicates_verified':20,
              'maximum_independent_bootstrap_difference':float(max(errors)),
              'final_artifact_and_freeze_hashes_verified':True}
    (folder/'verification.json').write_text(json.dumps(verified,indent=2))
    print(json.dumps(verified,indent=2))


if __name__=='__main__':main()
