"""Fit source models only; do not load any target eligibility table."""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from model_utils import (build_pipeline, calibration_diagnostics, confusion_metrics,
                         fit_weighted, predictor_frame, rao_wu_multipliers,
                         require_source_only, weighted_metrics, workload_selection)

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources(study: dict) -> tuple[pd.DataFrame, list[Path]]:
    inputs = []
    frames = []
    for cycle in study["cycles"]:
        if cycle["role"] not in {"source_train", "source_validation"}:
            continue
        path = ROOT / f"data/processed/eligibility_{cycle['name']}.csv"
        frame = pd.read_csv(path)
        frame = frame.loc[frame.eligible].copy()
        require_source_only(frame)
        if not frame.role.eq(cycle["role"]).all():
            raise ValueError("Cycle role mismatch")
        inputs.append(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), inputs


def check_freeze() -> dict:
    freeze = json.loads((ROOT / "manifests/source_freeze.json").read_text())
    for relative, expected in freeze["files"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen input or specification changed: {relative}")
    return freeze


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start = time.monotonic()
    freeze = check_freeze()
    study = json.loads((ROOT / "config/study.json").read_text())
    config = json.loads((ROOT / "config/experiment_v1.json").read_text())
    if config["target_test_evaluation_enabled"]:
        raise ValueError("Source executable cannot enable target evaluation")
    sources, inputs = load_sources(study)
    train = sources.loc[sources.role.eq("source_train")].copy()
    val = sources.loc[sources.role.eq("source_validation")].copy()
    if set(train.participant_id) & set(val.participant_id):
        raise ValueError("Participant leakage")
    output = ROOT / "reports/source_v1"
    output.mkdir(exist_ok=True)
    models_dir = ROOT / "models/source_v1"
    models_dir.mkdir(parents=True, exist_ok=True)
    if (output / "run_manifest.json").exists():
        raise ValueError("Completed source run exists; use a versioned amendment for another run")
    fold_rows, candidates, chosen, probabilities = [], [], {}, {}
    y = val.phq9_positive.to_numpy(dtype=int)
    w = val.survey_weight.to_numpy(dtype=float)
    reference = float(np.average(train.phq9_positive, weights=train.survey_weight))
    probabilities["prevalence_reference"] = np.full(len(val),reference)
    with threadpool_limits(limits=config["thread_limit"]):
        for feature_set, spec in config["feature_sets"].items():
            for kind in ["logistic", "boosting"]:
                grid = config[f"{kind}_grid"]
                scores = []
                for index, params in enumerate(grid):
                    briers = []
                    for held_cycle in sorted(train.cycle.unique()):
                        fit = train.loc[train.cycle.ne(held_cycle)]
                        held = train.loc[train.cycle.eq(held_cycle)]
                        pipeline = fit_weighted(build_pipeline(kind,spec,params,config), fit,spec)
                        p = pipeline.predict_proba(predictor_frame(held,spec))[:,1]
                        metrics = weighted_metrics(held.phq9_positive.to_numpy(),p,held.survey_weight.to_numpy())
                        briers.append(metrics["brier"])
                        fold_rows.append({"feature_set":feature_set,"kind":kind,"candidate":index,"held_cycle":held_cycle,**params,**metrics})
                    score = float(np.mean(briers))
                    scores.append(score)
                    candidates.append({"feature_set":feature_set,"kind":kind,"candidate":index,"parameters":json.dumps(params,sort_keys=True),"mean_cycle_brier":score})
                    LOGGER.info("%s %s candidate %d: source CV Brier %.6f",feature_set,kind,index,score)
                selected = int(np.argmin(scores))
                model_name = f"{feature_set}_{kind}"
                pipeline = fit_weighted(build_pipeline(kind,spec,grid[selected],config),train,spec)
                joblib.dump(pipeline,models_dir / f"{model_name}.joblib")
                probabilities[model_name] = pipeline.predict_proba(predictor_frame(val,spec))[:,1]
                chosen[model_name] = {"parameters":grid[selected],"cv_brier":scores[selected],"n_train":len(train),
                                      "transformed_features":pipeline.named_steps["preprocess"].get_feature_names_out().tolist()}
        metric_rows, operation_rows, deciles = [], [], []
        for name,p in probabilities.items():
            metrics = weighted_metrics(y,p,w)
            metrics.update(calibration_diagnostics(y,p,w))
            metric_rows.append({"model":name,**metrics})
            for rule, selected in [("fixed_probability_0.1",p>=config["fixed_probability_threshold"]),
                                   ("weighted_top_10_percent",workload_selection(p,w,val.participant_id,config["workload_fraction"]))]:
                operation_rows.append({"model":name,"rule":rule,**confusion_metrics(y,selected,w)})
            if np.ptp(p)>0:
                groups = pd.qcut(pd.Series(p),10,duplicates="drop")
                for interval in groups.cat.categories:
                    mask = (groups==interval).to_numpy()
                    deciles.append({"model":name,"bin":str(interval),"n":int(mask.sum()),
                                    "mean_probability":float(np.average(p[mask],weights=w[mask])),
                                    "observed_fraction":float(np.average(y[mask],weights=w[mask])),
                                    "weight_sum":float(w[mask].sum())})
        LOGGER.info("Computing prespecified source-validation intervals")
        bootstrap_rows = []
        rng = np.random.default_rng(config["source_validation_bootstrap"]["seed"])
        for rep in range(config["source_validation_bootstrap"]["replicates"]):
            rw = w*rao_wu_multipliers(val.survey_stratum,val.survey_psu,rng)
            ref_brier = np.average((probabilities["prevalence_reference"]-y)**2,weights=rw)
            for name,p in probabilities.items():
                m = weighted_metrics(y,p,rw)
                bootstrap_rows.append({"replicate":rep,"model":name,**m,"brier_minus_prevalence":m["brier"]-ref_brier})
            if (rep+1)%250==0: LOGGER.info("Source interval replicate %d",rep+1)
    boot=pd.DataFrame(bootstrap_rows)
    intervals=[]
    for name, group in boot.groupby("model"):
        for endpoint in ["brier","log_loss","auroc","average_precision","brier_minus_prevalence"]:
            low,high=group[endpoint].quantile([.025,.975]).tolist()
            intervals.append({"model":name,"endpoint":endpoint,"lower":low,"upper":high,
                              "replicates":len(group),"conditional_on_source_fit":True})
    sensitivity=[]
    mask=val.age.ge(20).to_numpy()
    for name,p in probabilities.items():
        sensitivity.append({"model":name,"restriction":"age_at_least20","n":int(mask.sum()),
                            **weighted_metrics(y[mask],p[mask],w[mask])})
    pd.DataFrame(fold_rows).to_csv(output/"cv_folds.csv",index=False)
    pd.DataFrame(candidates).to_csv(output/"cv_candidates.csv",index=False)
    pd.DataFrame(metric_rows).to_csv(output/"validation_metrics.csv",index=False)
    pd.DataFrame(operation_rows).to_csv(output/"validation_operating_points.csv",index=False)
    pd.DataFrame(deciles).to_csv(output/"calibration_bins.csv",index=False)
    pd.DataFrame(intervals).to_csv(output/"validation_intervals.csv",index=False)
    pd.DataFrame(sensitivity).to_csv(output/"age20_sensitivity.csv",index=False)
    boot.to_csv(output/"bootstrap_metrics.csv",index=False)
    predictions=val[["participant_id","cycle","survey_weight","survey_stratum","survey_psu","phq9_positive"]].copy()
    for name,p in probabilities.items(): predictions[name]=p
    predictions.to_csv(output/"validation_predictions.csv",index=False)
    (output/"selected_parameters.json").write_text(json.dumps(chosen,indent=2))
    run={"completed_utc":datetime.now(timezone.utc).isoformat(),"duration_seconds":time.monotonic()-start,
         "n_train":len(train),"n_validation":len(val),"validation_positive":int(y.sum()),
         "source_freeze_sha256":sha(ROOT/"manifests/source_freeze.json"),
         "input_files":{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in inputs},
         "target_data_loaded":False,"target_predictions_generated":False,
         "models":list(probabilities),"python":sys.version,"sklearn":sklearn.__version__,
         "numpy":np.__version__,"pandas":pd.__version__,"platform":platform.platform()}
    run["output_hashes"]={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(output.glob('*')) if p.is_file()}
    run["model_hashes"]={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(models_dir.glob('*.joblib'))}
    (output/"run_manifest.json").write_text(json.dumps(run,indent=2))
    report=["# NHANES source baseline results","","2017–2018 source validation only. No target eligibility table was loaded and no target predictions were generated.","",
            f"Models fitted on {len(train):,} adults from 2009–2016; evaluated on {len(val):,} adults in 2017–2018. Hyperparameters selected only by source-training cycle cross-validation.","",
            "| Model | Weighted Brier | AUROC | Average precision | Calibration intercept | Calibration slope |",
            "|---|---:|---:|---:|---:|---:|"]
    for r in metric_rows:
        slope='undefined' if r['calibration_slope'] is None else f"{r['calibration_slope']:.3f}"
        report.append(f"| {r['model']} | {r['brier']:.5f} | {r['auroc']:.3f} | {r['average_precision']:.3f} | {r['calibration_intercept']:.3f} | {slope} |")
    report += ["","Lower Brier is better. A constant prevalence predictor has AUROC 0.5 and average precision equal to the weighted outcome prevalence. Calibration intercept targets zero and slope targets one. These are symptom-screening benchmarks, not evidence of diagnostic or prospective clinical utility.","",
               "The interval CSV uses 1000 paired stratified PSU bootstrap replicates conditional on source model fitting. It does not include training uncertainty or fully correct outcome nonresponse. No significance or superiority claim should be based on rounded table values.","",
               "See validation_operating_points.csv for fixed 0.1 and matched-workload results; age20_sensitivity.csv for the planned age restriction; cv_folds.csv for source variability; and the run manifest for hashes and package versions.","",
               "The compact logistic model remains primary regardless of the comparator outcome. No source-validation model refitting or target-test opening occurs in this script."]
    (output/"summary.md").write_text('\n'.join(report),encoding='utf-8')
    LOGGER.info("Completed source models in %.1f seconds",run["duration_seconds"])
    print(pd.DataFrame(metric_rows).to_string(index=False))


if __name__ == "__main__":
    main()
