# NHANES novelty review in progress

Search date 15 October 2026. This is an overlap register, not novelty clearance.

## Closest original study inspected

Salimi et al. Interpretable machine learning for depression symptom classification in NHANES: Performance in a curated high-confidence corpus and the full survey population. 2026. DOI 10.1016/j.ibneur.2026.08.009.

Original full article: https://pmc.ncbi.nlm.nih.gov/articles/PMC13563247/

The paper includes ordinary ML comparisons, calibration and temporal evaluation. Generic NHANES depression classification or temporal validation is not a contribution by itself. Our remaining candidate distinction is a label-budget comparison of intercept-only and slope updating on the later release, using survey-aware evaluation and a predictor set excluding symptom proxies. Inspect the supplement for updating-budget experiments before freezing.

## Additional close citation requiring primary full text

Rawat, Kirola and Joshi. Predicting PHQ-9-Derived Depressive Symptom Screening Proxy Labels in NHANES. ICSADL 2026. DOI 10.1109/ICSADL67539.2026.11452096.

Publisher identifier: https://ieeexplore.ieee.org/document/11452096

A secondary indexed abstract suggests compact demographic/socioeconomic covariates, survey-weighted evaluation, cross-cycle transfer and calibration transport. This substantially overlaps a broad framing of our study. Publisher full text was not available in the initial web fetch. These details remain unverified against the primary source and must not be presented as an established full-method comparison. Do not make a first-calibration or first-cross-cycle claim.

## Other original studies to inspect

- Temporal validation in an obstructive-sleep-apnea subgroup: https://pubmed.ncbi.nlm.nih.gov/42473119/ . Different population, but reinforces temporal-validation precedent.
- Prediction-model transport methodology using NHANES as a target population: https://doi.org/10.1093/aje/kwac128 . General transportability methods are established; our work is an application/evaluation, not a new theory.

## Queries logged

- NHANES depression prediction recalibration temporal validation sample size label efficient 2025 2026
- NHANES depressive symptoms prediction 2021 2023 calibration transportability 2026
- Exact title and DOI searches for the close conference paper

## Before source model fitting

1. Inspect closest full methods and supplements, with special attention to calibration sample size and later-cycle transfer.
2. Complete a matrix of outcome, cycles, feature restrictions, weights, updating methods and label budgets.
3. Freeze a narrowly stated question, or explicitly reframe as replication if overlap is too strong.

Data integrity work can proceed regardless of the novelty outcome. No model benefit or publication impact is assumed.

## Source phase decision after follow-up search

Proceed with source development as a focused application/replication, with provisional novelty. Ma, Jia and Zhang (2026), *Temporal maintenance of an adolescent suicide attempt screening score in national surveys*, DOI 10.1097/MD.0000000000050513, already study survey-weighted frozen scores, intercept and slope updates, refitting and target label budgets in YRBS. The original abstract and indexed full-method sections were inspected: https://pubmed.ncbi.nlm.nih.gov/42675777/ and https://pmc.ncbi.nlm.nih.gov/articles/PMC13529099/ . This is strong methodological overlap despite the different outcome and population. We must not claim the label-budget updating framework as a new method.

The source-development protocol now fixes compact predictors and a BMI-only extension, cycle-grouped Brier selection, source fitting weights and descriptive survey intervals. Full final publication novelty clearance is not marked complete: the close conference paper's primary full methods remain unavailable. This limitation does not prevent baseline replication or feasibility work, and no paid access or author contact was initiated.
