"""Execute the frozen target evaluation and primary conditional survey interval."""
import csv
import hashlib
import json
from dataclasses import fields
from datetime import datetime,timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from model_utils import (calibration_diagnostics,confusion_metrics,predictor_frame,
                         weighted_metrics,workload_selection)
from prepare_updates import read_updating_pool
from recalibration import CalibrationFit,apply_calibration
from target_inference import primary_bootstrap
from train_source import check_freeze

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read_test(manifest):
    ids=set(manifest.loc[manifest.partition.eq('target_test_locked'),'participant_id'])
    retained=[]
    with (ROOT/'data/processed/eligibility_2021-2023.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['participant_id'] in ids:retained.append(row)
    frame=pd.DataFrame(retained).replace('',np.nan)
    for name in ['age','income_poverty_ratio','survey_weight','phq9_positive','survey_psu']:
        frame[name]=pd.to_numeric(frame[name],errors='coerce' if name=='income_poverty_ratio' else 'raise')
    frame=frame.merge(manifest[['participant_id','partition','partition_probability']],on='participant_id',validate='one_to_one')
    if set(frame.participant_id)!=ids or not frame.partition.eq('target_test_locked').all():raise ValueError('Test membership mismatch')
    frame['analysis_weight']=frame.survey_weight/frame.partition_probability
    return frame.set_index('participant_id',drop=False).sort_index()


def main():
    check_freeze()
    output=ROOT/'reports/final_v1';output.mkdir(exist_ok=True)
    if (output/'run_manifest.json').exists():raise ValueError('Final evaluation completed; do not silently overwrite it')
    verification=json.loads((ROOT/'reports/updating_v2/verification.json').read_text())
    if not verification['all_passed']:raise ValueError('Preparation verification did not pass')
    config=json.loads((ROOT/'config/updating_v2.json').read_text())
    spec=json.loads((ROOT/'config/experiment_v1.json').read_text())['feature_sets']['compact']
    paths=[ROOT/'src/evaluate_target.py',ROOT/'src/target_inference.py',ROOT/'src/recalibration.py',
           ROOT/'src/model_utils.py',ROOT/'protocol_target_preparation_v2.md',ROOT/'config/updating_v2.json',
           ROOT/'manifests/participant_splits.csv',ROOT/'manifests/updating_samples_v1.csv',
           ROOT/'reports/updating_v2/preparation_manifest.json',ROOT/'reports/updating_v2/verification.json',
           ROOT/'reports/updating_v2/fitting_status.csv',ROOT/'models/updating_v2/calibration_fits.json',
           ROOT/'models/source_v1/compact_logistic.joblib',ROOT/'data/processed/eligibility_2021-2023.csv']
    status=pd.read_csv(ROOT/'reports/updating_v2/fitting_status.csv')
    paths.extend(ROOT/p for p in status.loc[status.strategy.eq('refit') & status.status.eq('success'),'artifact'])
    files={p.relative_to(ROOT).as_posix():digest(p) for p in paths}
    freeze_path=ROOT/'manifests/final_evaluation_freeze_v1.json'
    if freeze_path.exists():
        if json.loads(freeze_path.read_text())['files']!=files:raise ValueError('Final specification changed')
    else:
        freeze_path.write_text(json.dumps({'frozen_utc':datetime.now(timezone.utc).isoformat(),'files':files,
                                          'primary_budget':500,'bootstrap_replicates':1000,'bootstrap_seed':20261017,
                                          'target_test_predictions_at_freeze':False,'external_preregistration':False},indent=2))
    opened_path=ROOT/'manifests/target_test_opened.json'
    if not opened_path.exists():
        opened_path.write_text(json.dumps({'opened_utc':datetime.now(timezone.utc).isoformat(),
                                          'freeze_sha256':digest(freeze_path),'reason':'User instructed continue after preparation verification'},indent=2))
    manifest=pd.read_csv(ROOT/'manifests/participant_splits.csv')
    pool=read_updating_pool(manifest);test=read_test(manifest)
    if set(pool.index)&set(test.index):raise ValueError('Updating/test overlap')
    source=joblib.load(ROOT/'models/source_v1/compact_logistic.joblib')
    calfits=json.loads((ROOT/'models/updating_v2/calibration_fits.json').read_text())
    by_key={(r['repetition'],r['budget'],r['strategy']):r for r in calfits}
    names={field.name for field in fields(CalibrationFit)}
    y=test.phq9_positive.to_numpy(dtype=int);w=test.analysis_weight.to_numpy(dtype=float)
    rows=[];operating=[];matrices=[];keys=[];rank_checks=[]
    with threadpool_limits(limits=2):
        static=source.predict_proba(predictor_frame(test,spec))[:,1]
        pool_p=source.predict_proba(predictor_frame(pool,spec))[:,1]
        static_metrics=weighted_metrics(y,static,w)
        static_selection=workload_selection(static,w,test.participant_id,.1)
        for _,row in status.iterrows():
            rep,budget,method=int(row.repetition),int(row.budget),row.strategy
            result={'repetition':rep,'budget':budget,'strategy':method,'status':row.status}
            if row.status!='success':rows.append(result);continue
            if method=='refit':
                if digest(ROOT/row.artifact)!=row.artifact_sha256:raise ValueError('Refit identity changed')
                model=joblib.load(ROOT/row.artifact)
                p=model.predict_proba(predictor_frame(test,spec))[:,1]
            else:
                fit=CalibrationFit(**{k:v for k,v in by_key[(rep,budget,method)].items() if k in names})
                p=apply_calibration(static,fit,config['probability_clip'])
            result.update(weighted_metrics(y,p,w))
            result.update(calibration_diagnostics(y,p,w))
            result['brier_minus_no_update']=result['brier']-static_metrics['brier']
            rows.append(result)
            matrices.append(p);keys.append(f'rep{rep:02d}_n{budget}_{method}')
            selection=workload_selection(p,w,test.participant_id,.1)
            if method=='intercept':rank_checks.append(bool(np.array_equal(selection,static_selection)))
            for rule,selected in [('fixed_probability_0.1',p>=.1),('weighted_top_10_percent',selection)]:
                operating.append({'repetition':rep,'budget':budget,'strategy':method,'rule':rule,**confusion_metrics(y,selected,w)})
        metrics=pd.DataFrame(rows)
        metrics.to_csv(output/'draw_metrics.csv',index=False)
        pd.DataFrame(operating).to_csv(output/'operating_points.csv',index=False)
        np.savez_compressed(output/'prediction_matrix.npz',probabilities=np.column_stack(matrices),
                            keys=np.asarray(keys),participant_ids=test.index.to_numpy(dtype=str),
                            y=y,weights=w,static_probability=static)
        summaries=[]
        for (budget,method),group in metrics.groupby(['budget','strategy'],sort=True):
            valid=group.loc[group.status.eq('success')]
            entry={'budget':int(budget),'strategy':method,'successful_draws':len(valid),'failed_draws':len(group)-len(valid)}
            for endpoint in ['brier','brier_minus_no_update','log_loss','auroc','average_precision','calibration_intercept','calibration_slope']:
                values=valid[endpoint].dropna()
                entry.update({f'{endpoint}_mean':float(values.mean()),f'{endpoint}_median':float(values.median()),
                              f'{endpoint}_draw_p05':float(values.quantile(.05)),f'{endpoint}_draw_p95':float(values.quantile(.95))})
            summaries.append(entry)
        pd.DataFrame(summaries).to_csv(output/'strategy_summary.csv',index=False)
        primary=metrics.loc[metrics.budget.eq(500) & metrics.strategy.eq('intercept') & metrics.status.eq('success')]
        point=float(primary.brier_minus_no_update.mean())
        samples=pd.read_csv(ROOT/'manifests/updating_samples_v1.csv')
        index={pid:i for i,pid in enumerate(pool.index)}
        draws=[np.asarray([index[p] for p in group.sort_values('draw_order').participant_id])
               for _,group in samples.loc[samples.budget.eq(500)].groupby('repetition',sort=True)]
        updating={'y':pool.phq9_positive.to_numpy(dtype=int),'probability':pool_p,
                  'weight':pool.fitting_weight.to_numpy(dtype=float),'stratum':pool.survey_stratum.to_numpy(),
                  'psu':pool.survey_psu.to_numpy(dtype=int)}
        evaluation={'y':y,'probability':static,'weight':w,'stratum':test.survey_stratum.to_numpy(),
                    'psu':test.survey_psu.to_numpy(dtype=int)}
        print('Target point estimates saved; computing primary survey interval',flush=True)
        bootstrap=primary_bootstrap(updating,evaluation,draws,replicates=1000,seed=20261017,
                                    progress=lambda n:print(f'Primary survey replicate {n}/1000',flush=True))
    boot=pd.DataFrame(bootstrap);boot.to_csv(output/'primary_bootstrap.csv',index=False)
    values=boot.mean_brier_difference.dropna();interval=values.quantile([.025,.975]).tolist()
    failure_fraction=float(boot.failed_draws.sum()/(1000*len(draws)))
    primary_result={'endpoint':'500-label compact-logistic intercept update minus no update weighted Brier',
                    'mean_difference':point,'percentile_95_interval':interval,'successful_point_draws':len(primary),
                    'planned_point_draws':30,'bootstrap_replicates':1000,'valid_bootstrap_replicates':len(values),
                    'bootstrap_failed_draw_fraction':failure_fraction,'conditional_on_source_fit_and_target_split':True,
                    'test_n':len(test),'test_positive':int(y.sum()),'static_metrics':static_metrics,
                    'intercept_equal_workload_selection_preserved':all(rank_checks)}
    (output/'primary_result.json').write_text(json.dumps(primary_result,indent=2))
    lines=['# NHANES locked target results','',
           f"Final evaluation of {len(test):,} adults in 2021–2023, separated from {len(pool):,} updating participants. The source model and method choices were frozen before evaluation.",'',
           '## Primary result','',f"At 500 updating labels, mean weighted Brier difference for intercept-only updating minus no update was {point:.6f}, with conditional 95 percent survey-bootstrap interval [{interval[0]:.6f}, {interval[1]:.6f}]. Negative favors updating.",'',
           f"Contributing draws: {len(primary)}/30. Valid survey replicates: {len(values)}/1000. Bootstrap draw failure fraction: {failure_fraction:.2%}.",'',
           '## Strategy comparison','',
           '| Labels | Method | Successful draws | Mean Brier | Mean AUROC |','|---:|---|---:|---:|---:|']
    for r in summaries:lines.append(f"| {r['budget']} | {r['strategy']} | {r['successful_draws']}/30 | {r['brier_mean']:.5f} | {r['auroc_mean']:.3f} |")
    lines+=['','Secondary differences are descriptive. Repeated-draw percentiles describe updating-sample variability, not population confidence intervals. Seven 100-person draws lacked the fixed event support; they were not replaced. Preparation v1 was superseded before test prediction because of corrected missing-category parsing; all final inputs use verified v2.',
            '',f"No-update mean predicted probability is {static_metrics['mean_probability']:.2%}; weighted observed symptom prevalence in this test domain is {static_metrics['observed_prevalence']:.2%}. Intercept updating preserves ranking and matched-workload selection: {all(rank_checks)}.",'',
            '## Interpretation limits','',
            'This is concurrent PHQ-defined symptom screening, not confirmed depression diagnosis or future illness prediction. The interval is conditional on the source model, fixed target split and draw design. Original survey weights do not fully repair PHQ outcome nonresponse. Population, questionnaire administration and sample-design changes remain inseparable. A Brier improvement alone does not establish clinically useful screening. Methodological novelty remains provisional given close prior national-survey maintenance work.',
            '', 'The final test is now evaluated and must not be reused to choose new primary methods, variables or operating thresholds. Any additional analyses are exploratory or require a new independent evaluation cohort.']
    (output/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    result={'completed_utc':datetime.now(timezone.utc).isoformat(),'final_test_evaluated':True,
            'freeze_sha256':digest(freeze_path),'primary_result':primary_result,
            'outputs':{p.relative_to(ROOT).as_posix():digest(p) for p in sorted(output.glob('*')) if p.is_file()}}
    (output/'run_manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(primary_result,indent=2))


if __name__=='__main__':main()
