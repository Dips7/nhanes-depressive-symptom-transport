# Replication Package: Limited-Label Updating of an Adult Depressive-Symptom Model Across NHANES Survey Periods

This repository contains the reviewer replication package for:

> **Limited-label updating of an adult depressive-symptom model across NHANES survey periods: a survey-weighted transport study**
> Binod Bhattarai, Dipesh Gurung*, R. N. Thakur

Raw NHANES survey files are publicly hosted by the U.S. Centers for Disease Control and Prevention (CDC / NCHS) and are **not** redistributed here; `src/download_data.py` fetches and checksum-verifies them directly from official CDC URLs.

## Repository Contents

- `src/` — Reproducible acquisition, codebook audit, source training, updating-pool preparation, target evaluation, Rao–Wu PSU bootstrap inference, and plotting scripts.
- `tests/` — Unit and integrity tests (`test_data_integrity.py`, `test_recalibration.py`, `test_source_models.py`, `test_target_inference.py`).
- `config/` — Frozen study, experiment, and updating configurations (`study.json`, `experiment_v1.json`, `updating_v2.json`).
- `manifests/` — Download metadata, SHA-256 freeze manifests, deterministic participant splits (`participant_splits.csv`), and archived updating-sample memberships (`updating_samples_v1.csv`).
- `protocols/` — Locally timestamped analysis and target-preparation protocols (`protocol_v1.md`, `protocol_target_preparation_v1.md`, `protocol_target_preparation_v2.md`).
- `reports/` — Audit summaries, decision log, source/updating/final evaluation manifests, and exploratory baseline reports.
- `table_data/` — Aggregate summary CSVs backing all manuscript tables.
- `requirements-lock.txt` & `requirements-reporting-lock.txt` — Pinned Python dependencies (**Python >= 3.12** required).

## Quick Verification (No Download Required)

Run the offline unit and integrity tests using `uv` (or a Python 3.12+ virtual environment):

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
```

*(Note: `test_official_target_response_frequency` is skipped until official CDC NHANES data are downloaded.)*

## Full End-to-End Replication (With Official NHANES Download)

```bash
python src/download_data.py
python src/audit_data.py
python -m unittest discover -s tests -v
python src/verify_source.py
python src/verify_updates.py
python src/verify_final.py
```

For plotting and table figure generation, install `requirements-reporting-lock.txt` and run `python src/plot_manuscript.py`.

## Primary Result Summary

- **500-label intercept updating vs. no updating:** Mean survey-weighted Brier error reduced from `0.113560` to `0.111186` (paired difference `−0.002374`, conditional 95% joint-PSU Rao–Wu bootstrap interval `−0.003635` to `−0.000737`).
- **Scope note:** Outcomes represent concurrent PHQ-9 (≥ 10) symptom status in the complete-outcome NHANES adult domain, not clinical depression diagnoses.

## License & Citation

Released under the [MIT License](LICENSE). See [CITATION.cff](CITATION.cff) and [.zenodo.json](.zenodo.json) for citation and Zenodo archival metadata.
