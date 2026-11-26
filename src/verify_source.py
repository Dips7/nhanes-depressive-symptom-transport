"""Verify source outputs independently from the main metric implementation."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from model_utils import predictor_frame
from train_source import check_freeze

ROOT=Path(__file__).resolve().parents[1]


def independent_auc(y,p,w):
    table=pd.DataFrame({'p':p,'positive':w*y,'negative':w*(1-y)}).groupby('p',sort=True).sum()
    below=table.negative.cumsum()-table.negative
    return float((table.positive*(below+.5*table.negative)).sum()/(table.positive.sum()*table.negative.sum()))


def independent_ap(y,p,w):
    table=pd.DataFrame({'p':p,'positive':w*y,'negative':w*(1-y)}).groupby('p',sort=True).sum().sort_index(ascending=False)
    tp=table.positive.cumsum(); all_selected=(table.positive+table.negative).cumsum()
    return float(((tp/all_selected)*(table.positive/table.positive.sum())).sum())


def main():
    check_freeze()
    folder=ROOT/'reports/source_v1'
    run=json.loads((folder/'run_manifest.json').read_text())
    for relative,expected in {**run['input_files'],**run['output_hashes'],**run['model_hashes']}.items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Output or input identity mismatch {relative}')
    predictions=pd.read_csv(folder/'validation_predictions.csv')
    manifest=pd.read_csv(ROOT/'manifests/participant_splits.csv')
    expected_ids=set(manifest.loc[manifest.partition.eq('source_validation'),'participant_id'])
    if set(predictions.participant_id)!=expected_ids or not predictions.participant_id.is_unique:
        raise ValueError('Validation IDs mismatch')
    if not predictions.cycle.eq('2017-2018').all():raise ValueError('Unexpected prediction cycle')
    metrics=pd.read_csv(folder/'validation_metrics.csv').set_index('model')
    val=pd.read_csv(ROOT/'data/processed/eligibility_2017-2018.csv')
    val=val.loc[val.eligible].set_index('participant_id').loc[predictions.participant_id].reset_index()
    y=predictions.phq9_positive.to_numpy();w=predictions.survey_weight.to_numpy()
    config=json.loads((ROOT/'config/experiment_v1.json').read_text())
    checks=[]
    with threadpool_limits(limits=2):
        for name in run['models']:
            p=predictions[name].to_numpy()
            brier=float(np.dot(w,(p-y)**2)/w.sum())
            auc=independent_auc(y,p,w);ap=independent_ap(y,p,w)
            if not np.allclose([brier,auc,ap],metrics.loc[name,['brier','auroc','average_precision']].to_numpy(dtype=float),atol=1e-12,rtol=0):
                raise ValueError(f'Independent metric mismatch {name}')
            difference=0.
            if name!='prevalence_reference':
                feature_set='expanded_bmi' if name.startswith('expanded_bmi') else 'compact'
                pipeline=joblib.load(ROOT/f'models/source_v1/{name}.joblib')
                reproduced=pipeline.predict_proba(predictor_frame(val,config['feature_sets'][feature_set]))[:,1]
                difference=float(np.max(np.abs(reproduced-p)))
                if difference>1e-12:raise ValueError(f'Prediction reproduction mismatch {name}')
            checks.append({'model':name,'brier_verified':brier,'weighted_auc_verified':auc,'weighted_ap_verified':ap,'max_reproduction_difference':difference})
    result={'checks':checks,'all_passed':True,'target_test_predictions_generated':False,
            'validation_ids_verified':len(expected_ids),'freeze_and_artifact_hashes_verified':True,
            'method':'direct weighted squared error; independent tied-rank AUC and grouped precision AP; reload saved pipelines'}
    (folder/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
