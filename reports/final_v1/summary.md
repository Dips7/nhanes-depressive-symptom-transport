# NHANES locked target results

Final evaluation of 3,277 adults in 2021–2023, separated from 2,178 updating participants. The source model and method choices were frozen before evaluation.

## Primary result

At 500 updating labels, mean weighted Brier difference for intercept-only updating minus no update was -0.002374, with conditional 95 percent survey-bootstrap interval [-0.003635, -0.000737]. Negative favors updating.

Contributing draws: 30/30. Valid survey replicates: 1000/1000. Bootstrap draw failure fraction: 0.00%.

## Strategy comparison

| Labels | Method | Successful draws | Mean Brier | Mean AUROC |
|---:|---|---:|---:|---:|
| 100 | intercept | 23/30 | 0.11223 | 0.645 |
| 100 | intercept_slope | 23/30 | 0.11311 | 0.633 |
| 100 | no_update | 30/30 | 0.11356 | 0.645 |
| 100 | refit | 23/30 | 0.11156 | 0.642 |
| 250 | intercept | 30/30 | 0.11156 | 0.645 |
| 250 | intercept_slope | 30/30 | 0.11181 | 0.645 |
| 250 | no_update | 30/30 | 0.11356 | 0.645 |
| 250 | refit | 30/30 | 0.11117 | 0.655 |
| 500 | intercept | 30/30 | 0.11119 | 0.645 |
| 500 | intercept_slope | 30/30 | 0.11100 | 0.645 |
| 500 | no_update | 30/30 | 0.11356 | 0.645 |
| 500 | refit | 30/30 | 0.11036 | 0.664 |
| 1000 | intercept | 30/30 | 0.11105 | 0.645 |
| 1000 | intercept_slope | 30/30 | 0.11047 | 0.645 |
| 1000 | no_update | 30/30 | 0.11356 | 0.645 |
| 1000 | refit | 30/30 | 0.11000 | 0.670 |

Secondary differences are descriptive. Repeated-draw percentiles describe updating-sample variability, not population confidence intervals. Seven 100-person draws lacked the fixed event support; they were not replaced. Preparation v1 was superseded before test prediction because of corrected missing-category parsing; all final inputs use verified v2.

No-update mean predicted probability is 7.18%; weighted observed symptom prevalence in this test domain is 13.04%. Intercept updating preserves ranking and matched-workload selection: True.

## Interpretation limits

This is concurrent PHQ-defined symptom screening, not confirmed depression diagnosis or future illness prediction. The interval is conditional on the source model, fixed target split and draw design. Original survey weights do not fully repair PHQ outcome nonresponse. Population, questionnaire administration and sample-design changes remain inseparable. A Brier improvement alone does not establish clinically useful screening. Methodological novelty remains provisional given close prior national-survey maintenance work.

The final test is now evaluated and must not be reused to choose new primary methods, variables or operating thresholds. Any additional analyses are exploratory or require a new independent evaluation cohort.