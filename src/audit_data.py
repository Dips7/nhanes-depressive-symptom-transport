"""Build an audited, unimputed cohort and protocol partitions; fit no models."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import html

ROOT = Path(__file__).resolve().parents[1]
PHQ_ITEMS = [f"DPQ{i:03d}" for i in range(10, 100, 10)]


def score_phq(frame: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
    """Require nine valid 0..3 responses, including a legitimate zero total."""
    values = frame[PHQ_ITEMS].where(frame[PHQ_ITEMS].isin([0, 1, 2, 3]))
    complete = values.notna().all(axis=1)
    total = values.sum(axis=1, min_count=9)
    positive = (total >= threshold).astype("Int64").where(complete)
    return pd.DataFrame({"phq_complete": complete, "phq9_total": total,
                         "phq9_positive": positive}, index=frame.index)


def harmonize_marital(values: pd.Series, suffix: str, age: pd.Series) -> pd.Series:
    mapping = {1: "partnered", 2: "previously_partnered", 3: "never_married"}
    if suffix != "L":
        mapping = {1: "partnered", 6: "partnered", 2: "previously_partnered",
                   3: "previously_partnered", 4: "previously_partnered", 5: "never_married"}
    result = values.map(mapping).astype("string")
    return result.mask(age < 20, "not_available_under20")


def harmonize_smoking(ever: pd.Series, current: pd.Series, age: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=ever.index, dtype="string")
    result.loc[ever.eq(2)] = "never_100_cigarettes"
    result.loc[ever.eq(1) & current.isin([1, 2])] = "current"
    result.loc[ever.eq(1) & current.eq(3)] = "former"
    # Earlier cycles use a different youth instrument. Do not combine incompatible answers.
    return result.mask(age < 20, "not_available_under20")


def assign_partitions(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    result = frame[["participant_id", "cycle", "survey_stratum", "survey_psu", "role"]].copy()
    result["partition"] = result.role
    result["partition_probability"] = 1.0
    target = result.loc[result.role.eq("target")]
    for _, group in target.groupby(["survey_stratum", "survey_psu"], sort=True):
        if len(group) < 3:
            raise ValueError("Too few target participants in a design cell for both partitions")
        ordered = sorted(group.index, key=lambda i: hashlib.sha256(
            f"{seed}:{result.at[i, 'participant_id']}".encode()).hexdigest())
        count = max(1, min(len(group) - 1, int(round(fraction * len(group)))))
        result.loc[ordered[:count], "partition"] = "target_updating"
        result.loc[ordered[count:], "partition"] = "target_test_locked"
        result.loc[ordered[:count], "partition_probability"] = count / len(group)
        result.loc[ordered[count:], "partition_probability"] = (len(group) - count) / len(group)
    return result.drop(columns="role")


def assert_features_allowed(names: list[str]) -> None:
    allow = {"age", "income_poverty_ratio", "sex_recorded", "education",
             "marital_status", "smoking_status", "bmi"}
    if set(names) - allow:
        raise ValueError(f"Unexpected predictor names: {set(names) - allow}")


def canonicalize_xport_zeros(frame: pd.DataFrame) -> pd.DataFrame:
    """Restore SAS IBM floating-point zero decoded as approximately 5.4e-79.

    Restricted to numeric columns and magnitudes impossible for these questionnaire,
    demographic, weight and anthropometry fields. Preserve ordinary values/NaN.
    """
    result = frame.copy()
    for name in result.select_dtypes(include="number").columns:
        result[name] = result[name].mask(result[name].abs().lt(1e-70), 0.0)
    return result


def read_module(cycle: dict, module: str, hashes: dict) -> pd.DataFrame:
    path = ROOT / f"data/raw/{cycle['name']}/{module}_{cycle['suffix']}.xpt"
    data = path.read_bytes()
    relative = path.relative_to(ROOT).as_posix()
    if hashlib.sha256(data).hexdigest() != hashes[relative]:
        raise ValueError(f"File hash mismatch: {relative}")
    frame = canonicalize_xport_zeros(pd.read_sas(path, format="xport", encoding="utf-8"))
    if frame.SEQN.duplicated().any() or frame.SEQN.isna().any():
        raise ValueError(f"Invalid identifiers in {relative}")
    return frame


def codebook_variables(cycle: dict, module: str) -> list[dict]:
    path = ROOT / f"data/raw/{cycle['name']}/{module}_{cycle['suffix']}.htm"
    tree = html.fromstring(path.read_bytes())
    rows = []
    for heading in tree.xpath("//h3"):
        title = " ".join(heading.text_content().split())
        if " - " not in title:
            continue
        name, label = title.split(" - ", 1)
        sections = []
        for sibling in heading.itersiblings():
            if sibling.tag in {"h2", "h3"}:
                break
            sections.append(" ".join(sibling.text_content().split()))
        rows.append({"cycle": cycle["name"], "module": module, "variable": name,
                     "label": label, "documentation": " ".join(sections),
                     "local_codebook": path.relative_to(ROOT).as_posix()})
    return rows


def verify_phq_frequencies(cycle: dict, frame: pd.DataFrame) -> list[dict]:
    """Check parsed responses against the independently published HTML frequencies."""
    path = ROOT / f"data/raw/{cycle['name']}/DPQ_{cycle['suffix']}.htm"
    tree = html.fromstring(path.read_bytes())
    checks = []
    for variable in PHQ_ITEMS:
        heading = tree.xpath(f"//h3[@id='{variable}']")
        if len(heading) != 1:
            raise ValueError(f"Missing codebook heading for {variable}")
        table = next((e for e in heading[0].itersiblings() if e.tag == "table"), None)
        if table is None:
            raise ValueError(f"Missing codebook frequencies for {variable}")
        found_codes = set()
        for row in table.xpath(".//tr"):
            cells = [" ".join(c.text_content().split()) for c in row.xpath("./td")]
            if len(cells) < 3 or cells[0] not in {"0", "1", "2", "3", "7", "9", "."}:
                continue
            code = cells[0]
            expected = int(cells[2].replace(",", ""))
            observed = int(frame[variable].isna().sum() if code == "." else frame[variable].eq(int(code)).sum())
            if observed != expected:
                raise ValueError(f"Frequency mismatch {cycle['name']} {variable} code {code}: {observed} != {expected}")
            checks.append({"cycle": cycle["name"], "variable": variable, "code": code,
                           "published": expected, "observed": observed, "match": True})
            found_codes.add(code)
        if not {"0", "1", "2", "3"}.issubset(found_codes):
            raise ValueError(f"Incomplete frequency validation {cycle['name']} {variable}")
    return checks


def main() -> None:
    config = json.loads((ROOT / "config/study.json").read_text())
    manifest = json.loads((ROOT / "manifests/downloads.json").read_text())
    expected = len(config["cycles"]) * len(config["modules"]) * 2
    if manifest["errors"] or len(manifest["files"]) != expected:
        raise ValueError("Acquisition incomplete; do not construct partial cohort")
    hashes = {row["path"]: row["sha256"] for row in manifest["files"]}
    cohorts, flows, missingness, quality, codebooks, frequency_checks = [], [], [], [], [], []
    for cycle in config["cycles"]:
        modules = {m: read_module(cycle, m, hashes) for m in config["modules"]}
        frequency_checks.extend(verify_phq_frequencies(cycle, modules["DPQ"]))
        demo = modules["DEMO"]
        ids = set(demo.SEQN)
        for module, frame in modules.items():
            if not set(frame.SEQN).issubset(ids):
                raise ValueError(f"{cycle['name']} {module} contains IDs outside DEMO")
            codebooks.extend(codebook_variables(cycle, module))
        marital_name = "DMDMARTZ" if cycle["suffix"] == "L" else "DMDMARTL"
        columns = ["SEQN", "RIDAGEYR", "RIAGENDR", "DMDEDUC2", marital_name,
                   "INDFMPIR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU", "RIDSTATR"]
        merged = demo[columns].merge(modules["DPQ"][["SEQN", *PHQ_ITEMS]], how="left", on="SEQN", validate="one_to_one")
        merged = merged.merge(modules["SMQ"][["SEQN", "SMQ020", "SMQ040"]], how="left", on="SEQN", validate="one_to_one")
        merged = merged.merge(modules["BMX"][["SEQN", "BMXBMI"]], how="left", on="SEQN", validate="one_to_one")
        scores = score_phq(merged, config["outcome_threshold"])
        if not merged[PHQ_ITEMS].stack().dropna().isin([0, 1, 2, 3, 7, 9]).all():
            raise ValueError("Unexpected depression-response code after XPORT decoding")
        adults = merged.RIDAGEYR >= config["minimum_age"]
        examined = adults & merged.RIDSTATR.eq(2) & merged.WTMEC2YR.gt(0)
        eligible = examined & scores.phq_complete
        if merged.loc[eligible, ["SDMVSTRA", "SDMVPSU"]].isna().any().any():
            raise ValueError("Eligible records lack survey-design identifiers")
        out = pd.DataFrame({
            "participant_id": cycle["name"] + ":" + merged.SEQN.astype(int).astype(str),
            "SEQN": merged.SEQN.astype(int), "cycle": cycle["name"], "role": cycle["role"],
            "age": merged.RIDAGEYR,
            "sex_recorded": merged.RIAGENDR.map({1: "male", 2: "female"}).astype("string"),
            "education": merged.DMDEDUC2.map({1: "less_than_9th", 2: "9th_to_11th", 3: "high_school_GED", 4: "some_college_AA", 5: "college_or_above"}).astype("string").mask(merged.RIDAGEYR < 20, "not_available_under20"),
            "marital_status": harmonize_marital(merged[marital_name], cycle["suffix"], merged.RIDAGEYR),
            "smoking_status": harmonize_smoking(merged.SMQ020, merged.SMQ040, merged.RIDAGEYR),
            "income_poverty_ratio": merged.INDFMPIR.where(merged.INDFMPIR.between(0, 5)),
            "bmi": merged.BMXBMI,
            "survey_weight": merged.WTMEC2YR,
            "survey_stratum": cycle["name"] + ":" + merged.SDMVSTRA.astype("Int64").astype(str),
            "survey_psu": merged.SDMVPSU.astype("Int64"),
            "eligible": eligible,
            **{c: scores[c] for c in scores},
        })
        analytic = out.loc[eligible].copy()
        cohorts.append(analytic)
        out.to_csv(ROOT / f"data/processed/eligibility_{cycle['name']}.csv", index=False)
        weights = analytic.survey_weight.to_numpy()
        prevalence = float(np.average(analytic.phq9_positive.astype(float), weights=weights))
        flows.append({"cycle": cycle["name"], "role": cycle["role"], "demographic_records": len(merged),
                      "adults": int(adults.sum()), "adult_examined_positive_weight": int(examined.sum()),
                      "excluded_outcome_incomplete": int((examined & ~scores.phq_complete).sum()),
                      "eligible": len(analytic), "positive": int(analytic.phq9_positive.sum()),
                      "weighted_complete_outcome_prevalence": prevalence,
                      "kish_weight_effective_n": float(weights.sum()**2 / (weights**2).sum()),
                      "weight_sum": float(weights.sum()), "n_design_strata": analytic.survey_stratum.nunique(),
                      "n_design_cells": analytic.groupby(["survey_stratum", "survey_psu"]).ngroups})
        features = config["compact_numeric"] + config["compact_categorical"] + ["bmi"]
        for name in features:
            structural = analytic[name].eq("not_available_under20").sum() if name in config["compact_categorical"] else 0
            missingness.append({"cycle": cycle["name"], "feature": name,
                                "missing_n": int(analytic[name].isna().sum()),
                                "missing_fraction": float(analytic[name].isna().mean()),
                                "structurally_unavailable_n": int(structural)})
        # Describe outcome nonresponse without manufacturing a score for excluded people.
        for status, mask in [("complete", eligible), ("incomplete", examined & ~scores.phq_complete)]:
            group = merged.loc[mask]
            quality.append({"cycle": cycle["name"], "outcome_status": status, "n": len(group),
                            "mean_age_unweighted": float(group.RIDAGEYR.mean()),
                            "female_fraction_unweighted": float(group.RIAGENDR.eq(2).mean()),
                            "income_missing_fraction": float(group.INDFMPIR.isna().mean())})
    cohort = pd.concat(cohorts, ignore_index=True)
    if cohort.participant_id.duplicated().any():
        raise ValueError("Duplicate global participant IDs")
    feature_names = config["compact_numeric"] + config["compact_categorical"]
    assert_features_allowed(feature_names)
    splits = assign_partitions(cohort, config["target_updating_fraction"], config["seed"])
    split_path = ROOT / "manifests/participant_splits.csv"
    split_text = splits.to_csv(index=False, lineterminator="\n")
    if split_path.exists():
        existing = pd.read_csv(split_path)
        try:
            pd.testing.assert_frame_equal(existing, splits.reset_index(drop=True), check_dtype=False)
        except AssertionError as exc:
            raise ValueError("Existing participant assignments differ; protocol amendment required") from exc
        old_bytes = split_path.read_bytes()
        if old_bytes != split_text.encode():
            archive = ROOT / "manifests/archive"
            archive.mkdir(exist_ok=True)
            (archive / "participant_splits_initial_serialization.csv").write_bytes(old_bytes)
    split_path.write_bytes(split_text.encode())
    cohort = cohort.merge(splits[["participant_id", "partition", "partition_probability"]], on="participant_id", validate="one_to_one")
    cohort["partition_analysis_weight"] = cohort.survey_weight / cohort.partition_probability
    cohort["source_pooled_weight"] = cohort.survey_weight.where(cohort.role.ne("source_train"), cohort.survey_weight / 4)
    cohort.to_csv(ROOT / "data/processed/analytic_cohort.csv", index=False)
    cohort[["participant_id", *feature_names]].to_csv(ROOT / "data/processed/compact_features.csv", index=False)
    pd.DataFrame(flows).to_csv(ROOT / "reports/cohort_flow.csv", index=False)
    pd.DataFrame(missingness).to_csv(ROOT / "reports/predictor_missingness.csv", index=False)
    pd.DataFrame(quality).to_csv(ROOT / "reports/outcome_nonresponse.csv", index=False)
    pd.DataFrame(codebooks).to_csv(ROOT / "reports/codebook_variables.csv", index=False)
    pd.DataFrame(frequency_checks).to_csv(ROOT / "reports/phq_frequency_validation.csv", index=False)
    partition_rows = []
    for name, group in cohort.groupby("partition"):
        partition_rows.append({"partition": name, "n": len(group), "positive": int(group.phq9_positive.sum()),
                               "negative": int(group.phq9_positive.eq(0).sum()),
                               "design_cells": group.groupby(["survey_stratum", "survey_psu"]).ngroups})
    pd.DataFrame(partition_rows).to_csv(ROOT / "reports/partition_counts.csv", index=False)
    pool = cohort.loc[cohort.partition.eq("target_updating")]
    budgets = [{"n": n, "fits_pool": n <= len(pool)} for n in config["updating_budgets"]]
    manifest_hash = hashlib.sha256(split_text.encode()).hexdigest()
    status = {"stage": "data_audit_complete_pending_protocol_review", "cohort_n": len(cohort),
              "partitions": partition_rows, "budgets": budgets, "split_sha256": manifest_hash,
              "source_files": len(manifest["files"]), "published_frequency_checks_passed": len(frequency_checks),
              "target_test_predictions_generated": False,
              "models_trained": False, "python": sys.version, "platform": platform.platform(),
              "pandas": pd.__version__, "numpy": np.__version__}
    (ROOT / "reports/startup_status.json").write_text(json.dumps(status, indent=2))
    lines = ["# NHANES data audit", "", "15 October 2026. No model has been trained and no target-test predictions have been generated.", "",
             f"Validated {len(manifest['files'])} official files (24 XPT datasets and 24 HTML codebooks). Local hashes are recorded, not publisher-verified checksums.", "",
             "## Cohort flow", "", "| Cycle | Eligible | PHQ positive | Incomplete outcome exclusions | Weighted complete-outcome prevalence |", "|---|---:|---:|---:|---:|"]
    for row in flows:
        lines.append(f"| {row['cycle']} | {row['eligible']} | {row['positive']} | {row['excluded_outcome_incomplete']} | {row['weighted_complete_outcome_prevalence']:.2%} |")
    lines += ["", "Prevalence is descriptive for the eligible outcome-complete domain; these are not fully nonresponse-adjusted all-adult estimates. Kish effective sample size in the CSV describes unequal weights only, not the full design effect.", "", "## Saved partitions", ""]
    lines += [f"- {r['partition']}: {r['n']} participants, {r['positive']} positive, {r['negative']} negative." for r in partition_rows]
    lines += ["", f"Split SHA256: `{manifest_hash}`. Partitions were constructed without using outcome labels. The target split has a known cell-specific inclusion probability, retained for weight adjustment.", "", "## Harmonization decisions", "",
              "- Education and marital status are structurally unavailable below age 20. Preserve an explicit age-ineligible category.",
              "- Harmonize older six-way marital codes into the later three-way release.",
              "- Adult smoking instruments change eligibility. For comparability, mark smoking below age 20 unavailable in every cycle rather than mixing youth and adult definitions.",
              "- A skipped current-smoking question after fewer than 100 lifetime cigarettes implies the never-100-cigarettes category, not ordinary missingness.",
              "- DPQ refused/unknown/missing values never contribute to a valid score. Outcome items and DPQ100 are excluded from features.",
              "- Correct the XPORT parser representation of numeric zero (approximately 5.4e-79) to exact zero before response validation; validate against official codebook counts.",
              "- Use original MEC weights per cycle; divide by four only for the four ordinary source-training cycles when pooling. Do not pool the later target for national trend inference.",
              "- Keep all eligible/excluded design records for later survey-domain variance estimation. Target interval code is not yet implemented.", "", "## Next steps", "",
              "Complete the full-method novelty matrix, specify updating event/failure rules and survey resampling, then freeze the protocol before fitting source models or opening final target performance."]
    (ROOT / "reports/data_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
