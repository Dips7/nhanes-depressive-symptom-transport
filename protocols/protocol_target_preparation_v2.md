# Corrected updating reader and locked evaluation specification

15 October 2026, before target-test prediction. This amends preparation v1 without changing the hypotheses, methods, budgets, sample identities, source model or primary outcome.

## Correction and provenance

The first updating reader retained six empty-string categorical entries in the 2178-person updating pool. The source reader interprets these as missing, which are explicitly encoded by the source preprocessing pipeline. Empty strings instead became unknown categories. Verification found an updating-pool probability discrepancy, so preparation v1 is superseded and must not enter any final results. Preserve all v1 artifacts and its original executable as prepare_updates_v1_archived.py. Normalize empty strings to missing before numeric/categorical preprocessing and run preparation v2 on the exact same sampled participants. No final target-test probabilities had been generated when this was corrected.

## Final evaluation authorization and scope

The user instructed us to continue through the study. Following verification of v2 preprocessing, sample nesting and saved fits, evaluate the fixed final target test once. Include all four strategies and all budgets; preserve method failures. The compact source logistic classifier remains primary. Do not use results to alter feature choice, update method, sample budget, solver or split.

Evaluate every successful saved fit on the same 3277-person target test, with partition-adjusted MEC weights. Report per-draw Brier, log loss, AUROC, average precision, predicted prevalence, fixed 0.1 operating measures and approximate weighted top-10-percent selection. The no-update reference uses original probabilities; recalibration uses the fixed clip for logits. Summaries give median and 5th/95th percentiles across updating draws as algorithmic variation, not national population confidence intervals. The primary point estimate is the mean paired Brier difference for 500-label intercept updating minus no update.

## Primary survey interval

Use 1000 fixed-seed (20261017) jointly sampled Rao–Wu stratified PSU replicates over the target updating and test population; the same PSU multiplier applies in both partitions. Retain the archived 500-label draw memberships in every replicate, apply multiplier-adjusted updating weights, refit the intercept and evaluate on the multiplier-weighted test records. Average successful paired primary contrasts across the 30 draws per replicate, and report all fit failures and contributing-draw counts. No-update remains fixed. Retain replicates with at least one paired successful draw and report this conditional rule; if more than 5 percent of planned draws fail in the primary replicates, qualify inference as failure-sensitive rather than altering the analysis.

The primary interval is the percentile interval of these mean contrasts. It conditions on the source fit, target split and archived draw design, and uses a with-replacement PSU approximation without finite-population correction. Outcome-nonresponse and household dependence are not resolved. Secondary strategies receive descriptive repeated-draw summaries only in this minimum run; no post hoc superiority P values or multiple primary endpoints are introduced.

## End-to-end checks before test access

Verify that the updating pool's numeric and categorical inputs match an independent pandas read of the harmonized features; recover saved updating-pool probabilities; validate all saved fit and sample hashes; and ensure samples are unique, nested, outcome-blind and disjoint from test IDs. Use synthetic data for the complete primary bootstrap to recover near-zero update effect when the score is already calibrated and a negative Brier contrast for a deliberately shifted score. Test paired weighting and common PSU multipliers. The final evaluation executable and settings must be hashed before target-test prediction.

Report null or adverse findings, failures at the 100-label budget and the provisional publication novelty position. Do not equate probability improvement with useful screening or disease diagnosis.
