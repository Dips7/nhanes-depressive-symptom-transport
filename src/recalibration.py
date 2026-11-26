"""Probability updating with explicit support and numerical failure rules."""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit


@dataclass
class CalibrationFit:
    strategy: str
    status: str
    intercept: float | None = None
    slope: float | None = None
    events: int = 0
    nonevents: int = 0
    gradient_norm: float | None = None

    def to_dict(self):
        return asdict(self)


def fit_calibration(y, probability, weights, strategy, minimum_events=10,
                    minimum_nonevents=10, clip=1e-6) -> CalibrationFit:
    if strategy not in {"no_update", "intercept", "intercept_slope"}:
        raise ValueError("Unknown calibration strategy")
    y=np.asarray(y,dtype=float);p=np.asarray(probability,dtype=float);w=np.asarray(weights,dtype=float)
    if y.shape!=p.shape or y.shape!=w.shape or y.ndim!=1:
        raise ValueError("Mismatched input dimensions")
    if not (np.isfinite(y).all() and np.isfinite(p).all() and np.isfinite(w).all()):
        raise ValueError("Nonfinite input")
    if not np.isin(y,[0,1]).all() or ((p<0)|(p>1)).any() or (w<0).any() or w.sum()<=0:
        raise ValueError("Invalid outcome, probability or weight")
    positive=w>0
    events=int((y[positive]==1).sum());nonevents=int((y[positive]==0).sum())
    base=dict(strategy=strategy,events=events,nonevents=nonevents)
    if strategy=="no_update":
        return CalibrationFit(**base,status="success",intercept=0.,slope=1.)
    if events<minimum_events or nonevents<minimum_nonevents:
        return CalibrationFit(**base,status="insufficient_events")
    y=y[positive];p=p[positive];w=w[positive];w=w/w.sum()
    z=logit(np.clip(p,clip,1-clip))
    if strategy=="intercept":
        try:
            a=brentq(lambda a:np.dot(w,expit(z+a)-y),-40,40,xtol=1e-12)
        except ValueError:
            return CalibrationFit(**base,status="root_failure")
        gradient=float(abs(np.dot(w,expit(z+a)-y)))
        return CalibrationFit(**base,status="success",intercept=float(a),slope=1.,gradient_norm=gradient)
    if np.ptp(z)<1e-10:
        return CalibrationFit(**base,status="constant_score")
    design=np.column_stack([np.ones(len(z)),z])
    def objective(beta):
        eta=design@beta
        return float(np.dot(w,np.logaddexp(0,eta)-y*eta))
    def gradient(beta):
        return design.T@(w*(expit(design@beta)-y))
    result=minimize(objective,[0.,1.],jac=gradient,method="BFGS",options={"gtol":1e-9,"maxiter":1000})
    norm=float(np.linalg.norm(gradient(result.x),ord=np.inf))
    q=expit(design@result.x)
    hessian=design.T@((w*q*(1-q))[:,None]*design)
    eigenvalues=np.linalg.eigvalsh(hessian)
    okay=(np.isfinite(result.x).all() and norm<1e-6 and np.max(np.abs(result.x))<40
          and eigenvalues[0]>1e-8 and eigenvalues[-1]/eigenvalues[0]<1e10)
    if not okay:
        return CalibrationFit(**base,status="nonconvergence_or_separation",gradient_norm=norm)
    return CalibrationFit(**base,status="success",intercept=float(result.x[0]),slope=float(result.x[1]),gradient_norm=norm)


def apply_calibration(probability,fit: CalibrationFit,clip=1e-6):
    if fit.status!="success":raise ValueError("Failed calibration cannot issue predictions")
    p=np.asarray(probability,dtype=float)
    if fit.strategy=="no_update":return p.copy()
    return expit(fit.intercept+fit.slope*logit(np.clip(p,clip,1-clip)))


def nested_sample_ids(ids,budgets,repetitions,seed):
    """Use outcome-blind full-pool permutations; smaller budgets are prefixes."""
    ids=np.asarray(sorted(ids),dtype=str)
    if len(set(ids))!=len(ids) or max(budgets)>len(ids):raise ValueError("Invalid pool or budget")
    rng=np.random.default_rng(seed)
    for repetition in range(repetitions):
        order=rng.permutation(ids)
        for budget in budgets:
            yield repetition,budget,order[:budget].copy()
