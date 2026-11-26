"""Post-test exploratory target-prevalence comparator; never changes locked outputs."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_updates import read_updating_pool

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'reports/exploratory_prevalence_v1'
    output.mkdir(exist_ok=True)
    if (output / 'manifest.json').exists():
        raise ValueError('Completed exploratory analysis exists')
    paths = ['src/explore_prevalence_baseline.py',
             'manifests/participant_splits.csv', 'manifests/updating_samples_v1.csv',
             'reports/final_v1/prediction_matrix.npz',
             'reports/final_v1/draw_metrics.csv',
             'data/processed/eligibility_2021-2023.csv']
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    pool = read_updating_pool(pd.read_csv(ROOT / paths[1]))
    samples = pd.read_csv(ROOT / paths[2])
    saved = np.load(ROOT / paths[3])
    y, w = saved['y'], saved['weights']
    w = w / w.sum()
    if set(pool.index) & set(saved['participant_ids']):
        raise ValueError('Pool/test overlap')
    existing = pd.read_csv(ROOT / paths[4])
    rows = []
    for (rep, budget), group in samples.groupby(['repetition', 'budget']):
        draw = pool.loc[group.sort_values('draw_order').participant_id]
        probability = float(np.average(draw.phq9_positive, weights=draw.fitting_weight))
        brier = float(np.dot(w, (y - probability) ** 2))
        # Independently check the constant-predictor Brier identity.
        prevalence = float(np.dot(w, y))
        identity = prevalence * (1 - prevalence) + (probability - prevalence) ** 2
        if not np.isclose(brier, identity, atol=1e-14, rtol=0):
            raise ValueError('Brier identity failed')
        rows.append({'repetition': int(rep), 'budget': int(budget),
                     'estimated_prevalence': probability, 'brier': brier})
    baseline = pd.DataFrame(rows)
    baseline.to_csv(output / 'prevalence_draws.csv', index=False)
    paired = existing.loc[existing.status.eq('success')].merge(
        baseline.rename(columns={'brier': 'prevalence_brier'}),
        on=['repetition', 'budget'], validate='many_to_one')
    paired['brier_minus_target_prevalence'] = paired.brier - paired.prevalence_brier
    paired.to_csv(output / 'paired_comparisons.csv', index=False)
    summary = paired.groupby(['budget', 'strategy']).agg(
        successful_draws=('brier', 'size'), model_brier=('brier', 'mean'),
        matched_prevalence_brier=('prevalence_brier', 'mean'),
        mean_difference=('brier_minus_target_prevalence', 'mean'),
        fraction_draws_model_better=('brier_minus_target_prevalence', lambda x: float((x < 0).mean()))
    ).reset_index()
    summary.to_csv(output / 'summary.csv', index=False)
    lines = ['# Exploratory target-prevalence comparison', '',
             'Added after the final test was evaluated. Descriptive only; no new confirmatory inference.', '',
             'For each archived updating sample, estimate a constant probability using its partition-adjusted survey weights. Evaluate on the same saved test outcomes and weights. No test labels estimate the baseline. All 120 samples contribute to baseline results; comparisons use each model\'s successful draws, including only 23 supported draws at budget 100.', '',
             '| Labels | Method | Paired draws | Model Brier | Constant Brier | Difference |',
             '|---:|---|---:|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.budget} | {r.strategy} | {r.successful_draws} | {r.model_brier:.6f} | {r.matched_prevalence_brier:.6f} | {r.mean_difference:+.6f} |')
    lines += ['', 'Negative differences favor the feature-based model. Draw win fractions are not probabilities of superiority or statistical significance. The constant predictor has AUROC 0.5 and cannot rank individuals. These comparisons assess probability accuracy and do not establish clinical benefit. No threshold, feature or model selection is performed.']
    (output / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    (output / 'manifest.json').write_text(json.dumps({
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'exploratory_after_final_test': True, 'input_sha256': hashes,
        'baseline_draws': len(baseline), 'paired_records': len(paired),
        'checks': {'pool_test_disjoint': True, 'independent_brier_identity': True},
    }, indent=2))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
