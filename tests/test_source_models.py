import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from model_utils import (build_pipeline, calibration_diagnostics, fit_weighted,
                         predictor_frame, rao_wu_multipliers, require_source_only,
                         weighted_metrics, workload_selection)


class WeightedMetricsTests(unittest.TestCase):
    def test_constant_prevalence_reference(self):
        y=np.array([0,0,1]); w=np.array([1.,2.,3.]); p=np.full(3,.5)
        m=weighted_metrics(y,p,w)
        self.assertAlmostEqual(m["brier"],.25)
        self.assertAlmostEqual(m["auroc"],.5)
        self.assertAlmostEqual(m["average_precision"],.5)
        self.assertIsNone(calibration_diagnostics(y,p,w)["calibration_slope"])

    def test_weight_scale_invariance(self):
        y=np.array([0,0,1,1]); p=np.array([.1,.3,.4,.9]); w=np.array([1.,3.,2.,5.])
        self.assertEqual(weighted_metrics(y,p,w),weighted_metrics(y,p,w*10))

    def test_equal_workload_has_deterministic_ties(self):
        selected=workload_selection([.5]*10,[1]*10,[str(i) for i in range(10)],.1)
        self.assertEqual(np.flatnonzero(selected).tolist(),[0])

    def test_invalid_metrics_rejected(self):
        with self.assertRaises(ValueError):weighted_metrics([0,1],[.1,.9],[0,0])


class SurveyTests(unittest.TestCase):
    def test_psu_draws_stay_together(self):
        strata=np.array(["a"]*4+["b"]*4); psus=np.array([1,1,2,2]*2)
        m=rao_wu_multipliers(strata,psus,np.random.default_rng(42))
        self.assertTrue(np.isin(m,[0,2]).all())
        self.assertEqual(m[0],m[1]);self.assertEqual(m[2],m[3])
        self.assertEqual(m[:4].sum(),4)
        self.assertEqual(m[4:].sum(),4)

    def test_singleton_rejected(self):
        with self.assertRaises(ValueError):rao_wu_multipliers(["a","a"],[1,1],np.random.default_rng(42))


class IsolationTests(unittest.TestCase):
    def test_target_records_rejected(self):
        with self.assertRaises(ValueError):require_source_only(pd.DataFrame({"role":["target"]}))

    def test_preprocessing_fits_training_only_and_unknown_level_runs(self):
        spec={"numeric":["age"],"categorical":["sex_recorded"]}
        config={"seed":42}
        data=pd.DataFrame({"age":[20.,30.,np.nan,50.,60.,70.,40.,80.],
                           "sex_recorded":["male","female"]*4,"role":"source_train",
                           "survey_weight":[1.]*8,"phq9_positive":[0,0,0,1,0,1,1,1]})
        pipe=fit_weighted(build_pipeline("logistic",spec,{"C":1},config),data,spec)
        median=pipe.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"].statistics_[0]
        self.assertEqual(median,50.)
        held=pd.DataFrame({"age":[np.nan],"sex_recorded":["unseen"]})
        p=pipe.predict_proba(predictor_frame(held,spec))[:,1]
        self.assertTrue(np.isfinite(p).all())
        self.assertEqual(pipe.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"].statistics_[0],50.)

    def test_validation_cannot_fit_classifier(self):
        data=pd.DataFrame({"role":["source_validation"]})
        with self.assertRaises(ValueError):fit_weighted(None,data,{})


if __name__=="__main__":unittest.main()
