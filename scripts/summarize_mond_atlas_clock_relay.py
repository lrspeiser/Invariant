"""Read completed results; no fit changes or target-based cohort changes."""
import csv,json
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT,read_json,write_json,write_csv
from mond_atlas_clock_relay import predict_logv
from run_mond_atlas_clock_relay import load_sources

def main():
    package=ROOT/'work/gravity-first-principles/mond-atlas-clock-relay-001';out=package/'interpretation'
    out.mkdir(exist_ok=False)
    run=package/'run001';summary=read_json(run/'summary.json');selections=read_json(run/'selections.json')
    config=read_json(ROOT/'configs/mond_atlas_clock_relay_v1.json')
    sources,y,errors,ids,radial_ids,names,meta,exclusions,members=load_sources(config)
    gindex=np.array([names.index(n) for n in ids]);candidates=read_json(run/'candidate-formulas.json')
    folds={seed:{r['galaxy']:int(r['fold']) for r in csv.DictReader((run/f'folds-{seed}.csv').open(encoding='utf-8'))} for seed in config['fold_seeds']}
    groups={'inner':sources['r']/sources['rd']<1,'middle':(sources['r']/sources['rd']>=1)&(sources['r']/sources['rd']<3),'outer':sources['r']/sources['rd']>=3}
    rows=[]
    for family in config['families']:
        for seed in config['fold_seeds']:
            lookup={v['fold']:v['candidate_index'] for v in selections if v['seed']==seed and v['family']==family}
            pred=np.empty_like(y)
            for fold,ix in lookup.items():
                mask=np.array([folds[seed][name]==fold for name in ids]);pred[mask]=predict_logv({k:v[mask] for k,v in sources.items()},candidates[ix])
            for group,mask in groups.items():
                galbias=[float(np.mean(pred[mask&(gindex==i)]-y[mask&(gindex==i)])) for i in range(len(names)) if np.any(mask&(gindex==i))]
                rows.append(dict(family=family,seed=seed,region=group,galaxies=len(galbias),radii=int(mask.sum()),equal_galaxy_mean_log10_pred_over_obs=float(np.mean(galbias)),galaxies_mean_overpredicted=sum(v>1e-12 for v in galbias)))
    write_csv(out/'signed-region-residuals.csv',rows)
    # Mathematical baseline equality is not an improvement. The raw summary retained roundoff signs.
    write_json(out/'reporting-corrections.json',dict(original_run_unchanged=True,
        source_only_parameters='The original flag means source-only predictor inputs. Free global parameters were trained using other galaxies observed velocities; they were not derived independently from source data.',
        fixed_mond_comparison='Identical to itself. Raw count19 is floating-point sign noise at ~1e-19 MSE, not19 improved galaxies. Interpret as zero.',
        held_predictions='All-family held prediction replay and boundary frequencies supplied by source-audit/replay; original run saves radial residuals for overall selector only.',
        conditional_uncertainty='Bootstrap intervals conditional on this previously exposed sample; overlap zero for adjusted MOND gain. No confirmed improvement or causal time claim.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9})
    families=['newton_fixed','mond_fixed','mond_adjusted','surface_relay','clock_potential','kernel_point','finite_p2','finite_p3']
    labels=['Newton, ordinary matter','MOND, fixed','MOND, train adjusted','Surface-density proxy','Clock potential','Point-kernel approximation','Finite response, p2','Finite response, p3']
    metrics={v['family']:v for v in summary['metrics']}
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].barh(labels,[metrics[f]['rmse_dex'] for f in families],color=['#888888','#386cb0','#1b9e77','#7570b3','#e6ab02','#d95f02','#a6761d','#66a61e'])
    axes[0].invert_yaxis();axes[0].set_xlabel('Held-galaxy log-speed RMSE (dex); lower is better');axes[0].set_title('102 galaxies / 2,212 radii / 713 frozen settings')
    strata=list(csv.DictReader((run/'strata.csv').open(encoding='utf-8')))
    for f,label in zip(['mond_fixed','mond_adjusted','clock_potential','finite_p3'],['MOND fixed','MOND adjusted','Clock potential','Finite p3']):
        vals=[np.sqrt(np.mean([float(v['logspeed_rmse'])**2 for v in strata if v['family']==f and v['group']==g])) for g in ['inner_r_over_Rd_lt1','middle_1to3','outer_ge3']]
        axes[1].plot(['Inner < Rd','Middle 1–3 Rd','Outer >= 3 Rd'],vals,'o-',label=label)
    axes[1].set_ylabel('Log-speed RMSE (dex)');axes[1].set_title('Where the mismatches occur');axes[1].legend()
    fig.suptitle('Real SPARC development data: radial empirical tests, not evidence of time-energy transfer',fontsize=11)
    fig.tight_layout();fig.savefig(out/'real-data-comparison.png',dpi=160);plt.close(fig)
    print('Interpretation artifacts complete')

if __name__=='__main__':main()
