"""Verify corrected preparation without evaluating final target outcomes."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from model_utils import predictor_frame
from prepare_updates import read_updating_pool

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'reports/updating_v2'
    run=json.loads((folder/'preparation_manifest.json').read_text())
    freeze=json.loads((ROOT/'manifests/updating_v2_preparation_freeze.json').read_text())
    for name,expected in {**freeze['files'],**run['output_hashes']}.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError(f'Changed identity {name}')
    calibration_path=ROOT/'models/updating_v2/calibration_fits.json'
    if hashlib.sha256(calibration_path.read_bytes()).hexdigest()!=run['calibration_fits_sha256']:raise ValueError('Calibration fits changed')
    manifest=pd.read_csv(ROOT/'manifests/participant_splits.csv')
    pool=read_updating_pool(manifest)
    source_spec=json.loads((ROOT/'config/experiment_v1.json').read_text())['feature_sets']['compact']
    features=source_spec['numeric']+source_spec['categorical']
    independent=pd.read_csv(ROOT/'data/processed/analytic_cohort.csv',usecols=['participant_id',*features]).set_index('participant_id').loc[pool.index]
    pd.testing.assert_frame_equal(predictor_frame(pool,source_spec),predictor_frame(independent,source_spec),check_dtype=False)
    model=joblib.load(ROOT/'models/source_v1/compact_logistic.joblib')
    saved=pd.read_csv(folder/'updating_pool_source_probabilities.csv').set_index('participant_id').loc[pool.index]
    with threadpool_limits(limits=2):
        p=model.predict_proba(predictor_frame(independent,source_spec))[:,1]
    maximum=float(np.max(np.abs(p-saved.source_probability.to_numpy())))
    if maximum>1e-12:raise ValueError('Independent reader predictions disagree')
    samples=pd.read_csv(ROOT/'manifests/updating_samples_v1.csv')
    if hashlib.sha256((ROOT/'manifests/updating_samples_v1.csv').read_bytes()).hexdigest()!=run['sample_sha256']:raise ValueError('Samples changed')
    test_ids=set(manifest.loc[manifest.partition.eq('target_test_locked'),'participant_id'])
    if set(samples.participant_id)&test_ids:raise ValueError('Target-test sample leakage')
    for _,rep in samples.groupby('repetition'):
        previous=[]
        for budget,group in rep.groupby('budget',sort=True):
            ids=group.sort_values('draw_order').participant_id.tolist()
            if len(ids)!=budget or len(set(ids))!=budget or ids[:len(previous)]!=previous:raise ValueError('Invalid nested sample')
            previous=ids
    status=pd.read_csv(folder/'fitting_status.csv')
    for _,row in status.loc[status.strategy.eq('refit') & status.status.eq('success')].iterrows():
        if hashlib.sha256((ROOT/row.artifact).read_bytes()).hexdigest()!=row.artifact_sha256:raise ValueError('Refit artifact changed')
    result={'all_passed':True,'independent_missing_category_handling_verified':True,
            'pool_prediction_max_difference':maximum,'unique_nested_sample_sets':120,
            'target_test_ids_in_updates':0,'target_test_predictions_generated':False,
            'prepared_refit_models_verified':int((status.strategy.eq('refit') & status.status.eq('success')).sum())}
    (folder/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
