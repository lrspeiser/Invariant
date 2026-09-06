"""Report the frozen fixed-image source refinement without selecting a height."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mond_atlas_common import ROOT, read_json, write_json, write_csv, digest


def report(summary_path, output):
    if output.exists():
        raise FileExistsError('immutable report output exists')
    output.mkdir(parents=True)
    summary = read_json(summary_path)
    for asset in summary['assets']:
        if digest(ROOT/asset['path']) != asset['sha256']:
            raise ValueError('changed private fit: '+asset['id'])
    rows = summary['rows']
    labels = {'stellar_luminosity':'Cleaned stellar light','atomic_helium':'Atomic gas + helium','co21':'CO emission'}
    comparisons = []
    for component in labels:
        for h in summary['config']['height_grid_kpc']:
            selected = [r for r in rows if r['component']==component and r['height_kpc']==h]
            a,c = selected[0],selected[-1]
            comparisons.append(dict(component=component,height_kpc=h,
                coarse_relative_rms=a['refitted_source_image_rms'],fine_relative_rms=c['refitted_source_image_rms'],
                fine_prediction_change_from_factor2=c['prediction_change_relative_to_target_rms'],
                nonnegative_image_floor=c['nonnegative_image_floor'],
                fine_squared_residual_fraction_due_to_floor=(c['nonnegative_image_floor']/c['refitted_source_image_rms'])**2,
                source_integral_fine_over_coarse=c['conditional_source_integral']/a['conditional_source_integral']))
    write_csv(output/'coarse-fine-comparison.csv',comparisons)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(1,3,figsize=(13.8,5.8),sharey=True)
    colors = ['#116466','#3b73b9','#cc7a24','#a43b4a']
    for ax,(component,label) in zip(axes,labels.items()):
        for h,color in zip(summary['config']['height_grid_kpc'],colors):
            selected = [r for r in rows if r['component']==component and r['height_kpc']==h]
            ax.plot([r['refinement_factor'] for r in selected],
                    [100*r['refitted_source_image_rms'] for r in selected],'-o',color=color,lw=2,
                    label='Thin sheet' if h==0 else f'Assumed height {h:g} kpc')
        if component == 'co21':
            floor = 100*selected[0]['nonnegative_image_floor']
            ax.axhline(floor,color='#555555',ls='--',lw=1.3)
            ax.annotate('Signed-data floor: 9.23%',(3.96,floor),xytext=(0,-19),textcoords='offset points',
                        ha='right',fontsize=9,color='#444444')
        ax.set_title(label,pad=12)
        ax.set_yscale('log')
        ax.set_ylim(.006,45)
        ax.set_xticks([1,2,4],['125','62.5','31.25'])
        ax.set_xlabel('Latent source-node spacing (pc)')
        ax.grid(axis='y',alpha=.2,which='major')
    axes[0].set_ylabel('Same-image relative RMS mismatch (%)')
    fig.suptitle('A coarse source grid can create a mismatch',x=.08,y=.98,ha='left',fontsize=20,fontweight='bold')
    fig.text(.08,.916,'NGC2976: the same 129 × 129 measured cells and weights in every fit',fontsize=12,color='#444444')
    handles,legends = axes[0].get_legend_handles_labels()
    fig.legend(handles,legends,loc='lower center',bbox_to_anchor=(.52,.12),ncol=4,frameon=False,fontsize=10)
    fig.text(.08,.063,'36 CUDA fits; all converge. Greater source freedom reduces mismatch; it does not add observations.',fontsize=10)
    fig.text(.08,.022,'Illustrative thicknesses, native beam differences and missing source noise remain. No measured depth or gravity test.',fontsize=10,color='#555555')
    fig.subplots_adjust(left=.08,right=.985,top=.82,bottom=.29,wspace=.10)
    fig.savefig(output/'source-resolution.png',dpi=170)
    plt.close(fig)
    write_json(output/'summary.json',dict(summary_sha256=digest(summary_path),
        report_script_sha256=digest(Path(__file__)),source_fit_count=len(rows),
        all_optimizers_converged=summary['all_optimizers_converged'],comparisons=comparisons,
        latent_refinement_is_not_new_observed_resolution=True,
        fine_grid_changes_do_not_prove_continuum_inverse_convergence=True,
        new_observed_gravity_scores=0,new_lensing_scores=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,default=ROOT/'work/gravity-first-principles/mond-atlas-source-resolution-001/run-001/summary.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report(args.summary.resolve(),args.output.resolve())
