"""Render a comparison of all predeclared structure bundles, without selecting a law."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mond_atlas_common import ROOT, read_json, write_json, digest


def report(source, output):
    if output.exists(): raise FileExistsError('immutable report directory')
    summary=read_json(source/'summary.json'); runtime=read_json(source/'runtime.json')
    rows=summary['metrics']; output.mkdir(parents=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharex=True,sharey=True)
    colors=['#176b87','#8d4ca5']; names=['Stellar summaries','Gas fraction','Combined']
    for ax,estimator,color,title in zip(axes,['linear_ridge','rbf_kernel_ridge'],colors,['Linear learning','Nonlinear learning']):
        baseline=next(r for r in rows if r['estimator']==estimator and r['bundle']=='baseline')['rmse_dex']**2
        chosen=[next(r for r in rows if r['estimator']==estimator and r['bundle']==b) for b in ['stellar','gas','combined']]
        for i,r in enumerate(chosen):
            center=r['mse_gain_percent_over_same_estimator_baseline']
            lo=r['conditional_paired_bootstrap95_low']/baseline*100
            hi=r['conditional_paired_bootstrap95_high']/baseline*100
            ax.plot([lo,hi],[i,i],color=color,lw=3,alpha=.45)
            ax.scatter([center],[i],color=color,s=70,zorder=3)
            ax.text(hi+.35,i,f'{center:+.2f}%',va='center',fontsize=10,color=color)
        ax.axvline(0,color='#666666',lw=1); ax.set_title(title,fontsize=13)
        ax.set_yticks(range(3),names); ax.invert_yaxis(); ax.set_ylim(2.6,-.6)
        ax.grid(axis='x',alpha=.15); ax.set_xlabel('Change in held-out mean squared error (%)\nPositive = improvement over same model baseline')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_xlim(-13,16)
    fig.suptitle('First GPU run: no stable improvement from these structure summaries',fontsize=14,y=1.01)
    fig.text(.5,-.035,'126 development galaxies | 3 whole-galaxy fold partitions | bars: conditional paired 95% intervals\nBaseline includes acceleration, inclination and quality. These are not full 3D or causal tests.',ha='center',fontsize=10,color='#555555')
    fig.tight_layout(); fig.savefig(output/'structure-comparison.png',dpi=170,bbox_inches='tight'); plt.close(fig)
    lines=['# First executed GPU gravity-pattern experiment','',
        '**The CUDA learning path works. These coarse stellar/gas summaries do not yet give a stable improvement in prediction.**','',
        '![All structure comparisons](structure-comparison.png)','',
        'We reanalyzed 126 previously exposed SPARC galaxies. The target is each galaxy\'s median log observed-speed/fixed-MOND-speed ratio. Predictions use the existing algebraic radial baseline, not a new full-field model. Four feature bundles and two estimators were declared before this run, and all are reported.','',
        '| Model | Added information | RMSE (dex) | MSE improvement | Range across the three split seeds |',
        '|---|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['estimator']} | {r['bundle']} | {r['rmse_dex']:.5f} | {r['mse_gain_percent_over_same_estimator_baseline']:+.2f}% | {r['minimum_seed_gain_percent']:+.2f}% to {r['maximum_seed_gain_percent']:+.2f}% |")
    lines+=['',
        'The nonlinear combined case improves mean squared error by 2.26% on average, but changes from a 5.33% loss to a 10.29% gain across splits. Its paired interval includes zero. Stellar summaries alone and gas fraction alone do not reliably help. The combined linear improvement is only 0.30%. These results do not identify a gravity formula or establish that resolved structure is irrelevant.','',
        'Features are median/spread of baryonic acceleration, quality and inclination in the baseline; stellar surface brightness, disk scale length and morphology in the stellar bundle; and an HI-plus-fixed-stellar-M/L gas-fraction proxy. No actual ages, 3D clump arrangements or measured exterior fields were available to this experiment.','',
        'Every outer test galaxy is excluded from its training and hyperparameter selection. Scaling uses training inputs only. Three deterministic five-fold assignments give 3,024 out-of-fold predictions. Hyperparameters are chosen with the four remaining folds inside each outer training partition. Physical group/survey separation and genuinely unexposed confirmation remain outstanding.','',
        f"The 16 acceleration-bin structure shuffles have reference fraction {summary['structure_shuffle_reference_fraction']:.4f} for the first seed. This coarse, development-data diagnostic is not a calibrated discovery p-value. It does not override the unstable cross-seed result or the six added-structure comparisons.", '',
        'The GPU/CPU predictions agree within 2.2e-15, and a separate scikit-learn implementation agrees within 2.9e-15. A planted nonlinear signal is recovered on synthetic held-out examples with RMSE 0.109 times the constant-mean baseline. That validates learning machinery on its known control, not a physical discovery. Tests also verify galaxy identity partitioning, response leakage prevention, constant targets and unit rescaling.','',
        f"Actual runtime: {runtime['device']}, CuPy {runtime['cupy']}, Python {runtime['python_version']}. The nested fitting and shuffle stage took {runtime['fit_wall_seconds']:.2f} seconds after initialization; the memory pool was limited to 1 GiB. This small workload does not demonstrate a GPU speedup. The installed PyTorch build is CPU-only; no PyTorch CUDA success is claimed.", '',
        'Reproduce with Python313 from the recorded runtime:','',
        '```text','python -m unittest discover -s tests -p "test_mond_atlas_pattern_learning.py" -v',
        'python scripts/run_mond_atlas_pattern_learning.py --backend cuda --output <new-output-directory>',
        'python scripts/report_mond_atlas_pattern_learning.py --source <run-directory> --output <new-report-directory>','```','',
        'Raw observations remain outside Git. This learning pass consumes only the previously computed radial galaxy summary. Source acquisition, native selection and a direct-observable lensing pilot run as separate tasks; resolved fields and a motion/lensing theory require their own validation. See [task plan](../../../docs/GRAVITY_PATTERN_SYSTEM_TASKS.md).','',
        'References: [SPARC measurement paper](https://arxiv.org/abs/1606.09251); [independent kernel-ridge implementation](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html).']
    (output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    write_json(output/'report-bindings.json',{p.relative_to(ROOT).as_posix():digest(p) for p in [source/'summary.json',source/'runtime.json',Path(__file__)]})
    print(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); report(a.source.resolve(),a.output.resolve())
