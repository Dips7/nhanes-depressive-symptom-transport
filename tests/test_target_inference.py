import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.special import expit,logit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from target_inference import primary_bootstrap


def balanced_domain(shift=0):
    y=[];p=[];strata=[];psus=[]
    for s in ['a','b']:
        for u in [1,2]:
            for probability in [.2,.5,.8]:
                y.extend([1]*int(100*probability)+[0]*(100-int(100*probability)))
                p.extend([expit(logit(probability)-shift)]*100)
                strata.extend([s]*100);psus.extend([u]*100)
    return {'y':np.array(y),'probability':np.array(p),'weight':np.ones(len(y)),
            'stratum':np.array(strata),'psu':np.array(psus)}


class EndToEndTests(unittest.TestCase):
    def test_already_calibrated_has_zero_effect(self):
        d=balanced_domain()
        rows=primary_bootstrap(d,d,[np.arange(len(d['y']))],replicates=8)
        self.assertTrue(all(abs(r['mean_brier_difference'])<1e-12 for r in rows))
        self.assertTrue(all(r['failed_draws']==0 for r in rows))

    def test_known_shift_recovers_brier_improvement(self):
        d=balanced_domain(.8)
        rows=primary_bootstrap(d,d,[np.arange(len(d['y']))],replicates=8)
        self.assertTrue(all(r['mean_brier_difference']<-.005 for r in rows))

    def test_failed_draws_are_not_zero_effects(self):
        d=balanced_domain()
        rows=primary_bootstrap(d,d,[np.arange(3)],replicates=3)
        self.assertTrue(all(r['mean_brier_difference'] is None for r in rows))
        self.assertTrue(all(r['failed_draws']==1 for r in rows))


if __name__=='__main__':unittest.main()
