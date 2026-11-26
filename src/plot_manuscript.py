"""Reporting figures from archived results; no fitting or method selection."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'


def save(fig, name):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    flow = pd.read_csv(ROOT / 'reports/cohort_flow.csv')
    splits = pd.read_csv(ROOT / 'manifests/participant_splits.csv')
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.8))
    groups = [('source_train', 'Source development\n2009–2016'),
              ('source_validation', 'Source validation\n2017–2018'),
              ('target', 'Target period\n2021–2023')]
    for ax, (role, title) in zip(axes, groups):
        f = flow.loc[flow.role.eq(role)]
        totals = f[['adults', 'adult_examined_positive_weight',
                    'excluded_outcome_incomplete', 'eligible', 'positive']].sum()
        texts = [f"Adults aged ≥18\nn = {totals.adults:,}",
                 f"Examined with positive weight\nn = {totals.adult_examined_positive_weight:,}",
                 f"Complete nine-item PHQ\nn = {totals.eligible:,}\nPHQ ≥10: {totals.positive:,}"]
        ax.set(xlim=(0, 1), ylim=(0, 1), title=title)
        ax.axis('off')
        for y, text in zip([.83, .59, .35], texts):
            ax.text(.5, y, text, ha='center', va='center', fontsize=10,
                    bbox={'boxstyle': 'round,pad=.6', 'facecolor': '#eef4fa', 'edgecolor': '#6b88a5'})
        for upper, lower in [(.75, .67), (.51, .44)]:
            ax.annotate('', xy=(.5, lower), xytext=(.5, upper), arrowprops={'arrowstyle': '->'})
        ax.text(.5, .19, f"Incomplete outcome excluded: {totals.excluded_outcome_incomplete:,}",
                ha='center', fontsize=9)
        if role == 'target':
            counts = splits.partition.value_counts()
            ax.text(.5, .06, f"Updating: {counts['target_updating']:,}\nLocked test: {counts['target_test_locked']:,}",
                    ha='center', fontsize=10)
    fig.suptitle('NHANES analytic cohort flow', fontsize=14)
    fig.text(.02, .01, 'Counts are unweighted. The complete-outcome domain excludes missing PHQ responses; survey weights do not fully correct this selection.', fontsize=8)
    fig.tight_layout(rect=[0, .05, 1, .94])
    save(fig, 'cohort_flow')

    data = np.load(ROOT / 'reports/final_v1/prediction_matrix.npz')
    y, w = data['y'], data['weights']
    keys = data['keys'].astype(str)
    edges = np.array([0, .05, .10, .15, .20, .30, .50, 1.000001])
    colors = ['#747b85', '#2563a9', '#ca7a24', '#33866b']
    methods = ['no_update', 'intercept', 'intercept_slope', 'refit']
    labels = ['No update', 'Intercept only', 'Intercept and slope', 'Full refit']
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
    for ax, method, label, color in zip(axes.flat, methods, labels, colors):
        if method == 'no_update':
            predictions = data['static_probability'][:, None]
        else:
            selected = [i for i, key in enumerate(keys) if key.endswith('_n500_' + method)]
            if len(selected) != 30:
                raise ValueError(f'Expected 30 archived fits: {method}')
            predictions = data['probabilities'][:, selected]
        # Average each participant's bin contribution across draws, not their predictions.
        for b in range(len(edges) - 1):
            mask = (predictions >= edges[b]) & (predictions < edges[b + 1])
            mass = mask * w[:, None] / predictions.shape[1]
            total = mass.sum()
            if total <= 0:
                continue
            predicted = float((mass * predictions).sum() / total)
            observed = float((mass * y[:, None]).sum() / total)
            share = float(total / w.sum())
            rows.append({'strategy': method, 'bin_lower': edges[b],
                         'bin_upper': min(edges[b + 1], 1), 'mean_predicted': predicted,
                         'observed_fraction': observed, 'weighted_share': share})
            ax.scatter(predicted, observed, s=25 + 400 * share, color=color, alpha=.8)
        ax.plot([0, 1], [0, 1], '--', color='#aaaaaa', linewidth=1)
        ax.set(title=label, xlim=(0, .6), ylim=(0, .6))
        ax.grid(alpha=.15)
    for ax in axes[1]:
        ax.set_xlabel('Mean predicted probability')
    for ax in axes[:, 0]:
        ax.set_ylabel('Observed symptom fraction')
    fig.suptitle('Descriptive calibration · 500-label updating', fontsize=14)
    fig.text(.02, .015, 'Fixed probability bins; marker area reflects survey-weighted bin mass. Updated curves average bin contributions across 30 draws.\nThe same 3,277 test adults recur across draws; these are not independent observations. No confidence intervals shown.\nDisplay limited to 0–0.6; the companion table retains every nonempty bin through probability 1.', fontsize=8)
    fig.tight_layout(rect=[0, .10, 1, .95])
    save(fig, 'calibration_500')
    pd.DataFrame(rows).to_csv(ROOT / 'reports/final_calibration_descriptive.csv', index=False)
    print('Created cohort_flow and calibration_500 in PNG, PDF and SVG.')


if __name__ == '__main__':
    main()
