"""Source-only modeling utilities and weighted probability diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from audit_data import assert_features_allowed


def require_source_only(frame: pd.DataFrame) -> None:
    if not frame.role.isin(["source_train", "source_validation"]).all():
        raise ValueError("Target records prohibited in source development")


def build_pipeline(kind: str, spec: dict, parameters: dict, config: dict) -> Pipeline:
    assert_features_allowed(spec["numeric"] + spec["categorical"])
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocess = ColumnTransformer([("numeric", numeric, spec["numeric"]),
                                    ("categorical", categorical, spec["categorical"])])
    if kind == "logistic":
        model = LogisticRegression(**parameters, solver="lbfgs", max_iter=2000, random_state=config["seed"])
    elif kind == "boosting":
        model = HistGradientBoostingClassifier(**parameters, **config["boosting_fixed"], random_state=config["seed"])
    else:
        raise ValueError(f"Unknown model type {kind}")
    return Pipeline([("preprocess", preprocess), ("model", model)])


def predictor_frame(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    names = spec["numeric"] + spec["categorical"]
    assert_features_allowed(names)
    result = frame[names].copy()
    for name in spec["categorical"]:
        result[name] = result[name].astype(object).where(result[name].notna(), np.nan)
    return result


def fit_weighted(pipeline: Pipeline, frame: pd.DataFrame, spec: dict) -> Pipeline:
    require_source_only(frame)
    if not frame.role.eq("source_train").all():
        raise ValueError("Only source-training cycles may fit classifiers")
    weights = frame.survey_weight.to_numpy(dtype=float)
    weights = weights / weights.mean()
    return pipeline.fit(predictor_frame(frame, spec), frame.phq9_positive.astype(int), model__sample_weight=weights)


def weighted_metrics(y: np.ndarray, probability: np.ndarray, weights: np.ndarray) -> dict:
    y, probability, weights = np.asarray(y, dtype=float), np.asarray(probability, dtype=float), np.asarray(weights, dtype=float)
    if not (np.isfinite(y).all() and np.isfinite(probability).all() and np.isfinite(weights).all()):
        raise ValueError("Nonfinite metric input")
    if not np.isin(y, [0,1]).all() or (weights < 0).any() or weights.sum() <= 0 or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid outcome, weight or probability")
    p = np.clip(probability, 1e-6, 1-1e-6)
    result = {"brier": float(np.average((probability-y)**2, weights=weights)),
              "log_loss": float(np.average(-y*np.log(p)-(1-y)*np.log1p(-p), weights=weights)),
              "observed_prevalence": float(np.average(y, weights=weights)),
              "mean_probability": float(np.average(probability, weights=weights))}
    if len(np.unique(y[weights > 0])) == 2:
        result["auroc"] = float(roc_auc_score(y, probability, sample_weight=weights))
        result["average_precision"] = float(average_precision_score(y, probability, sample_weight=weights))
    else:
        result.update(auroc=None, average_precision=None)
    return result


def calibration_diagnostics(y: np.ndarray, probability: np.ndarray, weights: np.ndarray) -> dict:
    z = logit(np.clip(np.asarray(probability), 1e-6, 1-1e-6))
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float) / np.sum(weights)
    intercept = brentq(lambda a: np.dot(w, expit(z+a)-y), -40, 40)
    if np.ptp(z) < 1e-10:
        return {"calibration_intercept": float(intercept), "calibration_slope": None,
                "joint_calibration_intercept": None, "calibration_fit_status": "slope_undefined_constant"}
    design = np.column_stack([np.ones(len(z)), z])
    def loss(beta):
        eta = design @ beta
        return float(np.dot(w, np.logaddexp(0,eta)-y*eta))
    def grad(beta):
        return design.T @ (w*(expit(design @ beta)-y))
    result = minimize(loss, [0.0,1.0], jac=grad, method="BFGS", options={"gtol":1e-9,"maxiter":1000})
    okay = bool(np.isfinite(result.x).all() and np.linalg.norm(grad(result.x),ord=np.inf) < 1e-6)
    return {"calibration_intercept": float(intercept),
            "calibration_slope": float(result.x[1]) if okay else None,
            "joint_calibration_intercept": float(result.x[0]) if okay else None,
            "calibration_fit_status": "converged" if okay else "failed"}


def confusion_metrics(y, accepted_positive, weights) -> dict:
    y, predicted, w = np.asarray(y).astype(bool), np.asarray(accepted_positive).astype(bool), np.asarray(weights,dtype=float)
    tp, fp = w[y & predicted].sum(), w[~y & predicted].sum()
    fn, tn = w[y & ~predicted].sum(), w[~y & ~predicted].sum()
    return {"sensitivity": float(tp/(tp+fn)) if tp+fn else None,
            "specificity": float(tn/(tn+fp)) if tn+fp else None,
            "ppv": float(tp/(tp+fp)) if tp+fp else None,
            "workload": float((tp+fp)/w.sum()), "selected_n": int(predicted.sum())}


def workload_selection(probability, weights, ids, fraction=.1) -> np.ndarray:
    order = np.lexsort((np.asarray(ids).astype(str), -np.asarray(probability)))
    cumulative = np.cumsum(np.asarray(weights)[order])
    target = fraction*cumulative[-1]
    count = int(np.argmin(np.abs(np.r_[0,cumulative]-target)))
    selected = np.zeros(len(order), dtype=bool)
    selected[order[:count]] = True
    return selected


def rao_wu_multipliers(strata, psus, rng: np.random.Generator) -> np.ndarray:
    """Rescaled with-replacement PSU weights for a fixed observed domain."""
    strata, psus = np.asarray(strata), np.asarray(psus)
    result = np.zeros(len(strata))
    for stratum in np.unique(strata):
        mask = strata == stratum
        units = np.unique(psus[mask])
        m = len(units)
        if m < 2:
            raise ValueError(f"Singleton design stratum {stratum}")
        sampled = rng.choice(units, size=m-1, replace=True)
        for unit in units:
            result[mask & (psus == unit)] = m/(m-1)*np.sum(sampled == unit)
    return result
