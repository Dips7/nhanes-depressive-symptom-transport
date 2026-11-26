"""Conditional paired survey inference for the prespecified primary contrast."""
import numpy as np

from model_utils import rao_wu_multipliers
from recalibration import apply_calibration,fit_calibration


def primary_bootstrap(updating,test,draw_positions,replicates=1000,seed=20261017,
                      minimum_events=10,minimum_nonevents=10,progress=None):
    """Joint PSU resampling of fixed updating/test partitions and sample memberships.

    Dictionaries contain y, probability, weight, stratum and psu arrays. No model
    selection or source fitting occurs here; failed draws are counted explicitly.
    """
    nu=len(updating['y'])
    strata=np.r_[updating['stratum'],test['stratum']]
    psus=np.r_[updating['psu'],test['psu']]
    rng=np.random.default_rng(seed)
    rows=[]
    for rep in range(replicates):
        multipliers=rao_wu_multipliers(strata,psus,rng)
        uw=np.asarray(updating['weight'])*multipliers[:nu]
        tw=np.asarray(test['weight'])*multipliers[nu:]
        static=float(np.average((test['probability']-test['y'])**2,weights=tw))
        differences=[];failures=0
        for positions in draw_positions:
            if uw[positions].sum()<=0:
                failures+=1
                continue
            fit=fit_calibration(updating['y'][positions],updating['probability'][positions],uw[positions],
                                'intercept',minimum_events,minimum_nonevents)
            if fit.status!='success':failures+=1;continue
            updated=apply_calibration(test['probability'],fit)
            value=float(np.average((updated-test['y'])**2,weights=tw))
            differences.append(value-static)
        rows.append({'replicate':rep,'mean_brier_difference':float(np.mean(differences)) if differences else None,
                     'successful_draws':len(differences),'failed_draws':failures})
        if progress and (rep+1)%100==0:progress(rep+1)
    return rows
