"""Descriptive label-budget figure and prespecified primary interval."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'reports/final_v1';out=ROOT/'figures';out.mkdir(exist_ok=True)
    summary=pd.read_csv(folder/'strategy_summary.csv')
    primary=json.loads((folder/'primary_result.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,(left,right)=plt.subplots(1,2,figsize=(11.5,4.5),gridspec_kw={'width_ratios':[1.6,1]})
    colors={'no_update':'#747b85','intercept':'#2563a9','intercept_slope':'#ca7a24','refit':'#33866b'}
    labels={'no_update':'No update','intercept':'Intercept only','intercept_slope':'Intercept and slope','refit':'Full refit'}
    for method in ['no_update','intercept','intercept_slope','refit']:
        rows=summary.loc[summary.strategy.eq(method)].sort_values('budget')
        x=rows.budget.to_numpy()
        left.plot(x,rows.brier_mean,marker='o',label=labels[method],color=colors[method],linewidth=1.8)
        if method!='no_update':left.fill_between(x,rows.brier_draw_p05,rows.brier_draw_p95,color=colors[method],alpha=.10)
    left.set(xlabel='Labeled updating participants',ylabel='Weighted Brier error (lower is better)',title='A  Label-budget comparison')
    left.set_xticks([100,250,500,1000]);left.grid(axis='y',alpha=.2);left.legend(frameon=False,fontsize=9)
    point=primary['mean_difference'];low,high=primary['percentile_95_interval']
    right.axvline(0,color='#747b85',linestyle='--',linewidth=1)
    right.errorbar(point,0,xerr=np.array([[point-low],[high-point]]),fmt='o',capsize=6,color=colors['intercept'],markersize=7)
    right.set_yticks([0],['500-label\nintercept update'])
    right.set_xticks([-.004,-.002,0.])
    right.set(xlabel='Brier difference versus no update',title='B  Primary conditional 95% interval',ylim=(-.7,.7))
    right.grid(axis='x',alpha=.2);right.ticklabel_format(axis='x',style='plain',useOffset=False)
    fig.suptitle('NHANES 2021–2023 locked evaluation · 3,277 adults',fontsize=13,y=.99)
    fig.text(.02,.015,'A: shading is the 5th–95th percentile across updating draws, not a population CI. 100 labels: 23/30 fitted draws; other budgets: 30/30.\nB: 1,000 joint PSU bootstrap replicates, conditional on source fit and target split. Symptom-screening research; no clinical utility established.',fontsize=8,color='#444')
    fig.tight_layout(rect=[0,.12,1,.94])
    for extension in ['png','pdf','svg']:fig.savefig(out/f'updating_results.{extension}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    print(out/'updating_results.png')


if __name__=='__main__':main()
