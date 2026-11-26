# Exploratory target-prevalence comparison

Added after the final test was evaluated. Descriptive only; no new confirmatory inference.

For each archived updating sample, estimate a constant probability using its partition-adjusted survey weights. Evaluate on the same saved test outcomes and weights. No test labels estimate the baseline. All 120 samples contribute to baseline results; comparisons use each model's successful draws, including only 23 supported draws at budget 100.

| Labels | Method | Paired draws | Model Brier | Constant Brier | Difference |
|---:|---|---:|---:|---:|---:|
| 100 | intercept | 23 | 0.112228 | 0.114316 | -0.002088 |
| 100 | intercept_slope | 23 | 0.113112 | 0.114316 | -0.001205 |
| 100 | no_update | 30 | 0.113560 | 0.115148 | -0.001588 |
| 100 | refit | 23 | 0.111561 | 0.114316 | -0.002756 |
| 250 | intercept | 30 | 0.111559 | 0.114151 | -0.002592 |
| 250 | intercept_slope | 30 | 0.111808 | 0.114151 | -0.002343 |
| 250 | no_update | 30 | 0.113560 | 0.114151 | -0.000591 |
| 250 | refit | 30 | 0.111167 | 0.114151 | -0.002984 |
| 500 | intercept | 30 | 0.111186 | 0.113739 | -0.002553 |
| 500 | intercept_slope | 30 | 0.111001 | 0.113739 | -0.002738 |
| 500 | no_update | 30 | 0.113560 | 0.113739 | -0.000179 |
| 500 | refit | 30 | 0.110361 | 0.113739 | -0.003378 |
| 1000 | intercept | 30 | 0.111053 | 0.113531 | -0.002479 |
| 1000 | intercept_slope | 30 | 0.110465 | 0.113531 | -0.003066 |
| 1000 | no_update | 30 | 0.113560 | 0.113531 | +0.000029 |
| 1000 | refit | 30 | 0.110000 | 0.113531 | -0.003532 |

Negative differences favor the feature-based model. Draw win fractions are not probabilities of superiority or statistical significance. The constant predictor has AUROC 0.5 and cannot rank individuals. These comparisons assess probability accuracy and do not establish clinical benefit. No threshold, feature or model selection is performed.