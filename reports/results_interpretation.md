# Interpretation of the locked NHANES experiment

15 October 2026. The final 3277-person test is evaluated; use new independent data for any new confirmatory model choices.

## Prespecified result

At 500 target labels, intercept-only updating reduced mean survey-weighted Brier score from 0.113560 to 0.111186. Updated minus unchanged difference is -0.002374, with conditional 95 percent joint-PSU bootstrap interval [-0.003635, -0.000737]. All 30 primary updating draws and all 1000 bootstrap replicates contributed, with zero primary replicate-draw failures. The interval conditions on source fit, target split and archived draw membership.

## Secondary comparisons and limits

At that same budget, intercept-and-slope mean Brier was 0.111001; full refit mean Brier was 0.110361. These secondary differences were not supplied with new confirmatory tests. Full refitting had a descriptively higher average AUROC, approximately 0.664 compared with the frozen/intercept models' 0.645. The results do not support a claim that intercept-only updating is the best strategy. It is the simplest primary probability repair and achieved a modest improvement over no updating.

The frozen classifier's target ranking was modest. Predicted prevalence averaged 7.18 percent versus 13.04 percent weighted observed symptom prevalence in the test domain. Intercept-only updating preserves AUROC and matched-workload selection. Better probability calibration does not imply better identification at the same referral budget or clinical utility.

At budget 100, seven of 30 draws had fewer than ten positive participants. They were retained as failures and not replaced. Secondary draw percentiles quantify algorithmic sample variation, not survey population confidence intervals.

## Verification and reporting

### Exploratory prevalence-only comparison

After the locked analysis, we added a missing simple reference: a constant probability estimated separately in each archived updating sample using partition-adjusted survey weights. Test labels did not estimate these probabilities. At 500 labels, mean constant-predictor Brier was 0.113739, compared with 0.111186 for intercept updating (difference -0.002553, about 2.24 percent relative improvement) and 0.110361 for full refitting (difference -0.003378, about 2.97 percent). Both methods beat their corresponding constant predictor in all 30 archived 500-label draws. These are descriptive comparisons on one shared test cohort; 30 wins are not 30 independent external replications or a significance test.

The unchanged source model's mean advantage over the same constant reference was only 0.000179 at 500 labels. This strengthens the motivation for probability maintenance, while the small absolute gains and modest discrimination still limit clinical claims. Full refitting remains descriptively strongest. The new comparison supports a modest feature contribution beyond estimating target prevalence, not a high-impact or clinically validated screening claim. See exploratory_prevalence_v1/summary.md and its input-hash manifest. The prespecified analysis and outputs remain unchanged.

Thirty automated checks passed. Independent verification reproduced all 459 successful draw metric records using tied-rank AUC and grouped precision calculations, validated all final participant IDs and hashes, and recomputed the primary contrast and 20 bootstrap replicates with a separately implemented Newton solver. Numerical differences were at floating-point scale.

The six-empty-category updating-reader correction occurred before final test access, is documented, and superseded preparation v1 with v2 using unchanged participants. Retain this amendment in reproducibility records rather than presenting the pipeline as error-free development.

Close prior work already studies temporal calibration and label-budget maintenance in national surveys. Frame publication as a focused adult PHQ-defined application or replication unless full-method review establishes a more specific contribution. Questionnaire mode, outcome nonresponse and sampling changes limit causal and clinical interpretation.
