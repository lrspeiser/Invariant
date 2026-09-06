"""Retain source-closure failures and distinguish the nonnegative noise floor."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest


def run(source,output):
    if output.exists():raise FileExistsError('immutable output exists')
    output.mkdir(parents=True)
    summary=read_json(source/'summary.json')
    for path,expected in read_json(source/'artifact-hashes.json').items():
        if digest(ROOT/path)!=expected:raise ValueError('changed source-closure artifact')
    assets={r['id']:r for r in summary['assets']};floors=[];radial=[]
    for component in summary['config']['components']:
        row=assets[component+'-h0p0'];path=ROOT/row['path']
        if digest(path)!=row['sha256']:raise ValueError('changed source packet')
        with np.load(path) as f:
            axis=f['axis'];xx,yy=np.meshgrid(axis,axis,indexing='ij');r=np.hypot(xx,yy)
            t=np.where(np.isfinite(f['source_mean']),f['source_mean'],0);w=f['evaluation_weight'];prediction=f['projected_surface']
            assert np.min(prediction)>=-1e-12
            for radius,weight in [('all',w)]+[(float(low),np.where((r>=low)&(r<low+.5),w,0)) for low in np.arange(0,3,.5)]:
                den=float(np.sum(weight*t*t));err=float(np.sum(weight*(prediction-t)**2));neg=float(np.sum(weight*np.minimum(t,0)**2))
                if den<=0:continue
                record=dict(component=component,radius_inner_kpc=radius,
                            nonnegative_prediction_lower_bound_rms=float(np.sqrt(neg/den)),thin_source_rms=float(np.sqrt(err/den)),
                            negative_measurement_pixels=int(np.sum((weight>0)&(t<0))),evaluated_pixels=int(np.sum(weight>0)),
                            fraction_of_thin_squared_error_at_least_from_negatives=neg/err if err else 0.)
                assert record['thin_source_rms']+1e-12>=record['nonnegative_prediction_lower_bound_rms']
                (floors if radius=='all' else radial).append(record)
    write_csv(output/'nonnegative-source-floor.csv',floors);write_csv(output/'nonnegative-source-floor-annuli.csv',radial)
    fig,axes=plt.subplots(1,3,figsize=(12,4.3),constrained_layout=True)
    names={'stellar_luminosity':'Stellar light','atomic_helium':'Atomic gas','co21':'CO molecular tracer'}
    for ax,component in zip(axes,summary['config']['components']):
        rows=[r for r in summary['rows'] if r['component']==component]
        height=[r['height_kpc'] for r in rows]
        ax.plot(height,[100*r['refitted_source_image_rms'] for r in rows],'-o',label='Refitted source',color='#286b9c')
        ax.plot(height,[100*r['unchanged_source_image_rms'] for r in rows],':s',label='Unchanged source',color='#b97e3f')
        floor=next(r['nonnegative_prediction_lower_bound_rms'] for r in floors if r['component']==component)*100
        ax.axhline(floor,color='#7f437f',ls='--',label='Nonnegative-model floor')
        ax.axhline(5,color='#777777',ls=':',label='5% diagnostic flag')
        ax.set(title=names[component],xlabel='Assumed exponential height (kpc)',ylabel='Coverage-weighted source RMS (%)',ylim=(0,48))
        ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=8)
    fig.suptitle('NGC2976: can a separable flat disk reproduce the tracer image?\nSame-source closure only; this is not a noise likelihood or a thickness measurement',fontsize=12)
    fig.savefig(output/'source-closure.png',dpi=150);plt.close(fig)
    write_json(output/'summary.json',dict(status='SOURCE_CLOSURE_FAILURES_RETAINED',object_id=summary['object_id'],
               source_models=len(summary['rows']),all_optimizers_converged=summary['all_optimizers_converged'],
               models_over_descriptive_threshold=sum(r['gross_refitted_mismatch'] for r in summary['rows']),
               nonnegative_floor=floors,noise_calibrated_height_rejections=0,new_gravity_scores=0,
               bindings={str(p.relative_to(ROOT)):digest(p) for p in (source/'summary.json',Path(__file__))}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.source.resolve(),args.output.resolve())
