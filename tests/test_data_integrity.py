import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_data import (PHQ_ITEMS, assert_features_allowed, assign_partitions,
                        harmonize_marital, harmonize_smoking, score_phq, canonicalize_xport_zeros)


class OutcomeTests(unittest.TestCase):
    def test_xport_zero_is_restored(self):
        frame = pd.DataFrame([[5.397605346934028e-79]*9], columns=PHQ_ITEMS)
        self.assertEqual(score_phq(canonicalize_xport_zeros(frame)).phq9_total.iloc[0],0)

    def test_official_target_response_frequency(self):
        path=Path(__file__).resolve().parents[1]/'data/raw/2021-2023/DPQ_L.xpt'
        if not path.exists(): self.skipTest('Official data not downloaded')
        frame=canonicalize_xport_zeros(pd.read_sas(path,format='xport'))
        # Published DPQ_L DPQ010 frequencies, retrieved 15 October 2026.
        self.assertEqual(int(frame.DPQ010.eq(0).sum()),3703)
        self.assertEqual(int(frame.DPQ010.eq(1).sum()),1205)
        self.assertEqual(int(frame.DPQ010.eq(2).sum()),334)
        self.assertEqual(int(frame.DPQ010.eq(3).sum()),256)

    def test_zero_and_threshold(self):
        frame = pd.DataFrame([[0]*9, [3,3,3,1,0,0,0,0,0]], columns=PHQ_ITEMS)
        result = score_phq(frame)
        self.assertEqual(result.phq9_total.tolist(), [0,10])
        self.assertEqual(result.phq9_positive.tolist(), [0,1])

    def test_incomplete_never_becomes_negative(self):
        frame = pd.DataFrame([[np.nan]*9, [0]*8+[7], [0]*8+[9]], columns=PHQ_ITEMS)
        result = score_phq(frame)
        self.assertTrue(result.phq9_total.isna().all())
        self.assertTrue(result.phq9_positive.isna().all())

    def test_out_of_range_rejected(self):
        result = score_phq(pd.DataFrame([[0]*8+[4]], columns=PHQ_ITEMS))
        self.assertFalse(result.phq_complete.iloc[0])


class HarmonizationTests(unittest.TestCase):
    def test_marital_old_to_new(self):
        old = harmonize_marital(pd.Series([1,6,2,3,4,5,77]), "J", pd.Series([40]*7))
        self.assertEqual(old.iloc[:6].tolist(), ["partnered","partnered","previously_partnered","previously_partnered","previously_partnered","never_married"])
        self.assertTrue(pd.isna(old.iloc[6]))

    def test_marital_age_ineligible(self):
        result = harmonize_marital(pd.Series([np.nan,3]), "L", pd.Series([19,40]))
        self.assertEqual(result.tolist(), ["not_available_under20","never_married"])

    def test_smoking_skip_and_unknown(self):
        result = harmonize_smoking(pd.Series([2,1,1,1,9,2]), pd.Series([np.nan,1,2,3,np.nan,np.nan]), pd.Series([40,40,40,40,40,19]))
        self.assertEqual(result.iloc[:4].tolist(), ["never_100_cigarettes","current","current","former"])
        self.assertTrue(pd.isna(result.iloc[4]))
        self.assertEqual(result.iloc[5], "not_available_under20")


class PartitionTests(unittest.TestCase):
    def test_deterministic_disjoint_and_weight_adjusted(self):
        frame = pd.DataFrame({"participant_id":[str(i) for i in range(20)], "cycle":"2021-2023", "survey_stratum":"L:1", "survey_psu":[1]*10+[2]*10, "role":"target"})
        result = assign_partitions(frame, .4, 42)
        shuffled = assign_partitions(frame.sample(frac=1,random_state=3), .4,42)
        pd.testing.assert_frame_equal(result.sort_index(),shuffled.sort_index())
        self.assertEqual(result.partition.eq("target_updating").sum(),8)
        self.assertTrue(result.participant_id.is_unique)
        self.assertEqual(result.loc[result.partition.eq("target_test_locked"),"partition_probability"].unique().tolist(),[.6])

    def test_outcome_cannot_enter_features(self):
        with self.assertRaises(ValueError): assert_features_allowed(["age","phq9_total"])
        with self.assertRaises(ValueError): assert_features_allowed(["DPQ010"])


if __name__ == "__main__": unittest.main()
