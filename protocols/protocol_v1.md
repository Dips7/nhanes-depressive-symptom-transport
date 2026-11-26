# NHANES source development protocol

Frozen source-development specification, 15 October 2026, before source model training. No target predictions or predictive results exist at this freeze. The freeze is a local timestamped and hashed record, not an external preregistration.

## Scope and positioning

Use existing public data to test probability updating for concurrent PHQ-9 symptom screening status. The source phase trains inexpensive comparators and examines their 2017–2018 performance. It does not open the 2021–2023 final test or establish clinical usefulness. Broad temporal validation, weighted screening models and label-budget recalibration have prior precedent; this is a focused application or replication until a narrower new finding is demonstrated.

The initial overlap review is sufficient to proceed without first-of-kind claims. A close conference paper's primary full methods remain inaccessible, so final publication novelty clearance is unfinished. See reports/literature_overlap.csv and reports/novelty_review.md. No inaccessible source is treated as full-text verified.

## Data and features

Retain the audited adult, complete-outcome cohort, fixed label rule, harmonization and deterministic target assignments. Preserve participant ID checks and outcome separation. The compact feature set is age, poverty-income ratio, released sex, education, harmonized marital status and cigarette status. The deliberately limited expanded set adds measured BMI only. This replaces the earlier open-ended expanded-health-variable proposal; do not add features based on validation or target results. Education, marital and smoking information below 20 are consistently marked unavailable. Include an age-20-plus secondary source-validation analysis.

All preprocessing is fitted within each training fold. Numeric features use unweighted training median imputation and standard scaling. Categorical missing values receive an explicit missing level, and a training-fitted one-hot encoder ignores unknown levels. No missingness indicators, symptom proxies, class rebalancing or synthetic augmentation are used in the primary phase. Weighted fitting does not imply weighted imputation; this difference is explicitly recorded.

## Training and selection

Pool the four ordinary 2009–2016 cycles with WTMEC2YR divided by four. For optimization normalize positive fitting weights to mean one within each fit so regularization is defined consistently; retain unnormalized survey weights for descriptive metrics. No weight clipping. Tune using four leave-one-cycle-out folds entirely inside 2009–2016. Fit preprocessing anew for every fit. Select the minimum arithmetic mean of within-cycle weighted Brier scores, with configured candidate order as deterministic tie-breaker. Report fold variation rather than treating this as prospective forward-only validation.

Models: weighted training-prevalence reference; L2 logistic regression with C in 0.1, 1 and 10; histogram gradient boosting with 150 iterations, 40 minimum leaf records, L2 regularization 1, no early stopping, and the four configured leaf-count/learning-rate combinations. Use random seed 42 and two computation threads. Logistic regression is primary independently of the comparison outcome. Save fold results, fitted preprocessing, selected parameters and 2017–2018 predictions. Do not refit on 2017–2018 after viewing these results.

## Source validation outcomes

Evaluate both feature sets on the identical 2017–2018 participants with their MEC weights: Brier, binary log loss, AUROC, weighted average precision (the implementation of AUPRC), calibration-in-the-large with slope fixed at one, and joint calibration slope. Calibration fits are diagnostics and do not alter predicted probabilities. A constant predictor has undefined calibration slope. Report weighted confusion measures at the prespecified probability cutoff 0.1 and at a score-only rule selecting approximately 10 percent of weighted population. Ties are resolved by participant ID; report realized workload because records are indivisible.

For source uncertainty use 1000 Rao–Wu rescaled bootstrap replicates: within each stratum with m observed PSUs draw m minus one PSUs with replacement; multiply each selected PSU weight by m divided by m minus one times its selection count. Preserve all within-PSU observations and apply identical replicate weights across models. With two PSUs this selects one with multiplicity multiplier two. Verify source-validation design support. These percentile intervals are an approximation conditional on the trained source pipeline and complete-outcome domain; they omit source-training variability and do not cure nonresponse or household dependence.

## Future target experiment choices

The planned primary comparison remains compact logistic intercept-only updating versus no update at 500 labels, averaged across 30 nested random updating draws; budgets 100, 250, 500 and 1000 remain fixed. Draw from target updating pool without outcome balancing or replacement within a draw. Require at least ten events and ten nonevents for each fitted strategy, record failures without redraws and give failure rates. Use a fixed numerical probability clip of 1e-6. Penalize the calibration slope weakly toward one only if that exact method is specified in a subsequent pre-target amendment; default unpenalized intercept and slope fitting with recorded separation/nonconvergence failure. No slope-sign correction or model switching after outcomes.

Before executing target updates, freeze solver, coefficient bounds, refit penalty and fallback implementation and test on synthetic/source data only. This source-development freeze does not falsely claim those implementation details are already finalized.

## Target survey inference specification

The estimand is performance in the observed-outcome adult target domain, conditional on source fit and target participant assignment. Original weights adjust for survey sampling, not fully for PHQ item nonresponse. Use original MEC weights divided by the saved partition inclusion probability for test and updating-pool domain calculations. A label-budget draw is a simple random subset of the updating pool; normalize its pool-adjusted fitting weights to mean one, with the common budget-selection factor cancelling in fitting.

For later target bootstrap, draw PSU multipliers jointly across the full target design so updating and test participants in the same PSU receive the same multiplier. Preserve excluded design records for domain calculations, retain the original disjoint split, and refit each updating strategy within replicate using fixed archived draw membership and multiplier weights. Do not count repeated updating draws as independent test participants. Require paired successful primary fits; publish failure frequencies. Implement and verify this procedure before final target evaluation. It is a conditional design-based approximation; the original participant split is not a household/PSU-independent split.

## Changes and limits

The fixed source threshold is 0.1, a research reporting point rather than a clinical recommendation. Positively monotone intercept updating preserves ranking and score-budget selections. Brier improvement is the primary probability outcome; no ranking improvement is expected from intercept updating alone. Report all prespecified comparisons and retain null findings. The 2021–2023 administration-mode and sampling changes remain inseparable from population changes. Any source-result-informed future change must be versioned and labeled development; target-test choices cannot be revised after opening that test.
