# NHANES source baseline results

2017–2018 source validation only. No target eligibility table was loaded and no target predictions were generated.

Models fitted on 20,977 adults from 2009–2016; evaluated on 5,068 adults in 2017–2018. Hyperparameters selected only by source-training cycle cross-validation.

| Model | Weighted Brier | AUROC | Average precision | Calibration intercept | Calibration slope |
|---|---:|---:|---:|---:|---:|
| prevalence_reference | 0.07807 | 0.500 | 0.085 | 0.082 | undefined |
| compact_logistic | 0.07452 | 0.691 | 0.184 | 0.145 | 0.865 |
| compact_boosting | 0.07481 | 0.684 | 0.179 | 0.123 | 0.791 |
| expanded_bmi_logistic | 0.07448 | 0.693 | 0.188 | 0.122 | 0.837 |
| expanded_bmi_boosting | 0.07479 | 0.682 | 0.185 | 0.101 | 0.769 |

Lower Brier is better. A constant prevalence predictor has AUROC 0.5 and average precision equal to the weighted outcome prevalence. Calibration intercept targets zero and slope targets one. These are symptom-screening benchmarks, not evidence of diagnostic or prospective clinical utility.

The interval CSV uses 1000 paired stratified PSU bootstrap replicates conditional on source model fitting. It does not include training uncertainty or fully correct outcome nonresponse. No significance or superiority claim should be based on rounded table values.

See validation_operating_points.csv for fixed 0.1 and matched-workload results; age20_sensitivity.csv for the planned age restriction; cv_folds.csv for source variability; and the run manifest for hashes and package versions.

The compact logistic model remains primary regardless of the comparator outcome. No source-validation model refitting or target-test opening occurs in this script.