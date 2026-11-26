import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.special import expit,logit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from recalibration import apply_calibration,fit_calibration,nested_sample_ids
from model_utils import rao_wu_multipliers


def population():
    true=np.array([.2,.5,.8])
    y=np.concatenate([np.r_[np.ones(int(p*1000)),np.zeros(1000-int(p*1000))] for p in true])
    return y,np.repeat(logit(true),1000)


class CalibrationTests(unittest.TestCase):
    def test_recovers_known_intercept(self):
        y,z=population();p=expit(z-.7)
        fit=fit_calibration(y,p,np.ones(len(y)),"intercept")
        self.assertEqual(fit.status,"success")
        self.assertAlmostEqual(fit.intercept,.7,places=8)
        self.assertTrue(np.array_equal(np.argsort(p),np.argsort(apply_calibration(p,fit))))

    def test_recovers_known_intercept_and_slope(self):
        y,z=population();p=expit((z-.4)/.8)
        fit=fit_calibration(y,p,np.ones(len(y)),"intercept_slope")
        self.assertEqual(fit.status,"success")
        self.assertAlmostEqual(fit.intercept,.4,places=5)
        self.assertAlmostEqual(fit.slope,.8,places=5)

    def test_weight_scale_invariance(self):
        y,z=population();p=expit(z-.2);w=np.linspace(.1,5,len(y))
        a=fit_calibration(y,p,w,"intercept");b=fit_calibration(y,p,w*1000,"intercept")
        self.assertAlmostEqual(a.intercept,b.intercept,places=10)

    def test_support_failure_not_silently_repaired(self):
        y=np.r_[np.ones(3),np.zeros(97)]
        fit=fit_calibration(y,np.full(100,.1),np.ones(100),"intercept")
        self.assertEqual(fit.status,"insufficient_events")
        with self.assertRaises(ValueError):apply_calibration([.1],fit)

    def test_constant_and_separated_scores_fail_slope(self):
        y=np.r_[np.ones(50),np.zeros(50)]
        self.assertEqual(fit_calibration(y,np.full(100,.5),np.ones(100),"intercept_slope").status,"constant_score")
        self.assertEqual(fit_calibration(y,np.where(y==1,.9,.1),np.ones(100),"intercept_slope").status,"nonconvergence_or_separation")

    def test_no_update_preserves_unclipped_probabilities(self):
        p=np.array([0.,.5,1.]);y=np.array([0,0,1]);fit=fit_calibration(y,p,np.ones(3),"no_update")
        np.testing.assert_array_equal(apply_calibration(p,fit),p)


class SamplingTests(unittest.TestCase):
    def test_nested_unique_and_deterministic(self):
        a=list(nested_sample_ids([str(i) for i in range(50)],[10,20],3,42))
        b=list(nested_sample_ids([str(i) for i in reversed(range(50))],[10,20],3,42))
        for x,y in zip(a,b):np.testing.assert_array_equal(x[2],y[2])
        for i in range(0,len(a),2):
            self.assertEqual(len(set(a[i][2])),10)
            np.testing.assert_array_equal(a[i][2],a[i+1][2][:10])

    def test_joint_psu_multiplier_same_across_roles(self):
        # Alternating positions stand for updating and test records within each PSU.
        m=rao_wu_multipliers(['a']*4+['b']*4,[1,1,2,2,1,1,2,2],np.random.default_rng(42))
        self.assertEqual(m[0],m[1]);self.assertEqual(m[2],m[3])
        self.assertEqual(m[4],m[5]);self.assertEqual(m[6],m[7])


if __name__=='__main__':unittest.main()
