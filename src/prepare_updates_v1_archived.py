"""Fit the primary updating experiment without target-test prediction."""
import csv
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from model_utils import build_pipeline,predictor_frame
from recalibration import fit_calibration,nested_sample_ids
from train_source import check_freeze

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read_updating_pool(manifest):
    ids=set(manifest.loc[manifest.partition.eq('target_updating'),'participant_id'])
    # Inspect the identifier before retaining any row fields for analysis.
    retained=[]
    with (ROOT/'data/processed/eligibility_2021-2023.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['participant_id'] in ids:
                retained.append(row)
    pool=pd.DataFrame(retained)
    numeric=['age','income_poverty_ratio','survey_weight','phq9_positive']
    for name in numeric:pool[name]=pd.to_numeric(pool[name],errors='raise' if name!='income_poverty_ratio' else 'coerce')
    pool=pool.merge(manifest[['participant_id','partition','partition_probability']],on='participant_id',validate='one_to_one')
    if set(pool.participant_id)!=ids or not pool.partition.eq('target_updating').all():
        raise ValueError('Updating-pool membership mismatch')
    pool['fitting_weight']=pool.survey_weight/pool.partition_probability
    return pool.set_index('participant_id',drop=False).sort_index()


def main():
    check_freeze()
    config=json.loads((ROOT/'config/updating_v1.json').read_text())
    source_config=json.loads((ROOT/'config/experiment_v1.json').read_text())
    if config['target_test_evaluation_enabled']:raise ValueError('Preparation cannot evaluate target test')
    source_model_path=ROOT/f"models/source_v1/{config['source_model']}.joblib"
    source_run=json.loads((ROOT/'reports/source_v1/run_manifest.json').read_text())
    if digest(source_model_path)!=source_run['model_hashes'][source_model_path.relative_to(ROOT).as_posix()]:
        raise ValueError('Frozen source classifier changed')
    output=ROOT/'reports/updating_v1';output.mkdir(exist_ok=True)
    if (output/'preparation_manifest.json').exists():raise ValueError('Completed preparation exists; use a versioned amendment')
    paths=[ROOT/'config/updating_v1.json',ROOT/'protocol_target_preparation_v1.md',ROOT/'src/recalibration.py',
           ROOT/'src/prepare_updates.py',ROOT/'src/model_utils.py',ROOT/'manifests/participant_splits.csv',
           source_model_path,ROOT/'data/processed/eligibility_2021-2023.csv']
    freeze_path=ROOT/'manifests/updating_preparation_freeze.json'
    files={p.relative_to(ROOT).as_posix():digest(p) for p in paths}
    if freeze_path.exists():
        if json.loads(freeze_path.read_text())['files']!=files:raise ValueError('Preparation specification changed after freeze')
    else:
        freeze_path.write_text(json.dumps({'frozen_utc':datetime.now(timezone.utc).isoformat(),'files':files,
                                          'target_test_predictions_generated':False,'external_preregistration':False},indent=2))
    manifest=pd.read_csv(ROOT/'manifests/participant_splits.csv')
    pool=read_updating_pool(manifest)
    draws=[]
    for repetition,budget,ids in nested_sample_ids(pool.index,config['budgets'],config['repetitions'],config['sample_seed']):
        draws.extend({'repetition':repetition,'budget':budget,'draw_order':position,'participant_id':pid}
                     for position,pid in enumerate(ids))
    draw_frame=pd.DataFrame(draws)
    sample_path=ROOT/'manifests/updating_samples_v1.csv'
    encoded=draw_frame.to_csv(index=False,lineterminator='\n').encode()
    if sample_path.exists() and sample_path.read_bytes()!=encoded:raise ValueError('Existing updating samples differ')
    sample_path.write_bytes(encoded)
    spec=source_config['feature_sets']['compact']
    source=joblib.load(source_model_path)
    selected=json.loads((ROOT/'reports/source_v1/selected_parameters.json').read_text())['compact_logistic']['parameters']
    directory=ROOT/'models/updating_v1';directory.mkdir(parents=True,exist_ok=True)
    fits=[];support=[]
    with threadpool_limits(limits=2):
        source_prob=source.predict_proba(predictor_frame(pool,spec))[:,1]
        pool_prob=pd.Series(source_prob,index=pool.index)
        for (repetition,budget),group in draw_frame.groupby(['repetition','budget'],sort=True):
            frame=pool.loc[group.participant_id]
            y=frame.phq9_positive.to_numpy(dtype=int);w=frame.fitting_weight.to_numpy(dtype=float)
            p=pool_prob.loc[frame.index].to_numpy()
            events=int(y.sum());nonevents=len(y)-events
            support.append({'repetition':int(repetition),'budget':int(budget),'events':events,'nonevents':nonevents,
                            'kish_effective_n':float(w.sum()**2/np.sum(w**2))})
            for method in ['no_update','intercept','intercept_slope']:
                fit=fit_calibration(y,p,w,method,config['minimum_events'],config['minimum_nonevents'],config['probability_clip'])
                fits.append({'repetition':int(repetition),'budget':int(budget),**fit.to_dict(),'artifact':None})
            record={'repetition':int(repetition),'budget':int(budget),'strategy':'refit','events':events,
                    'nonevents':nonevents,'status':'insufficient_events','artifact':None}
            if events>=config['minimum_events'] and nonevents>=config['minimum_nonevents']:
                try:
                    model=build_pipeline('logistic',spec,selected,source_config)
                    with warnings.catch_warnings():
                        warnings.simplefilter('error',ConvergenceWarning)
                        model.fit(predictor_frame(frame,spec),y,model__sample_weight=w/w.mean())
                    path=directory/f'refit_rep{repetition:02d}_n{budget}.joblib'
                    joblib.dump(model,path)
                    record.update(status='success',artifact=path.relative_to(ROOT).as_posix(),artifact_sha256=digest(path))
                except (ValueError,ConvergenceWarning,FloatingPointError) as exc:
                    record.update(status='fit_failed',error=str(exc))
            fits.append(record)
    pd.DataFrame(fits).to_csv(output/'fitting_status.csv',index=False)
    pd.DataFrame(support).to_csv(output/'sample_support.csv',index=False)
    (directory/'calibration_fits.json').write_text(json.dumps([f for f in fits if f['strategy']!='refit'],indent=2))
    pool_predictions=pool[['participant_id']].copy().reset_index(drop=True)
    pool_predictions['source_probability']=source_prob
    pool_predictions.to_csv(output/'updating_pool_source_probabilities.csv',index=False)
    status=pd.DataFrame(fits).groupby(['budget','strategy','status']).size().rename('draws').reset_index()
    status.to_csv(output/'status_counts.csv',index=False)
    result={'completed_utc':datetime.now(timezone.utc).isoformat(),'pool_n':len(pool),'repetitions':config['repetitions'],
            'budgets':config['budgets'],'samples':len(support),'fitting_records':len(fits),'sample_sha256':digest(sample_path),
            'target_test_predictions_generated':False,'target_test_metrics_computed':False,
            'freeze_sha256':digest(freeze_path),'output_hashes':{p.relative_to(ROOT).as_posix():digest(p) for p in output.glob('*') if p.is_file()},
            'calibration_fits_sha256':digest(directory/'calibration_fits.json')}
    (output/'preparation_manifest.json').write_text(json.dumps(result,indent=2))
    lines=['# Target updating preparation','',f"Prepared {len(support)} fixed updating samples from {len(pool)} participants; four strategies yield {len(fits)} fitting records. No final target-test probabilities or metrics were computed.",'',
           '| Budget | Method | Status | Draws |','|---:|---|---|---:|']
    for _,row in status.iterrows():lines.append(f"| {row.budget} | {row.strategy} | {row.status} | {row.draws} |")
    lines+=['','Failures are preserved without redraws. These are fitting/support results, not evidence that a method improves final-test predictions. The next step is end-to-end inference verification before the locked evaluation.']
    (output/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    print(status.to_string(index=False))


if __name__=='__main__':main()
